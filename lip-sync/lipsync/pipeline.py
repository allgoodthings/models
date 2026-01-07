"""
Multi-face lip-sync pipeline.

Orchestrates the complete lip-sync workflow:
1. Align faces (optionally with target bbox)
2. Neutralize expressions with LivePortrait
3. Generate lip-sync with MuseTalk
4. Enhance with CodeFormer
5. Composite back into original video
"""

import logging
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from .alignment import FaceAligner, align_video, unalign_video, AlignmentMetadata
from .musetalk import MuseTalk, MuseTalkConfig
from .liveportrait import LivePortrait, LivePortraitConfig
from .codeformer import CodeFormer, CodeFormerConfig
from .compositor import FaceCompositor, FaceRegion
from .cache import FaceCache
from .utils import (
    get_video_info,
    get_audio_duration,
    extract_audio,
    combine_audio_video,
    concat_videos,
    trim_video,
    generate_timestamp_chunks,
)

# Configure logging
logger = logging.getLogger('lipsync.pipeline')


@dataclass
class FaceJob:
    """Definition of a face to process."""
    character_id: str
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    audio_path: str
    start_time_ms: int
    end_time_ms: int

    @property
    def start_time(self) -> float:
        return self.start_time_ms / 1000.0

    @property
    def end_time(self) -> float:
        return self.end_time_ms / 1000.0


@dataclass
class PipelineConfig:
    """Configuration for lip-sync pipeline."""
    # Model paths
    musetalk_path: str = "models/musetalk"
    liveportrait_path: str = "models/liveportrait"
    codeformer_path: str = "models/codeformer"

    # Processing options
    device: str = "cuda"
    fp16: bool = True
    aligned_size: int = 512

    # Pipeline stage toggles
    # ----------------------
    # use_neutralization: Run LivePortrait to neutralize lips before lip-sync
    #   - REQUIRED if source faces are actively speaking (prevents ghost lips)
    #   - SKIP if source faces are already neutral/still (e.g., portraits)
    use_neutralization: bool = True

    # use_enhancement: Run CodeFormer to enhance face quality after lip-sync
    #   - KEEP for final export quality (reduces VAE artifacts)
    #   - SKIP for preview/draft/real-time (saves ~15ms/frame)
    use_enhancement: bool = True

    # enhancement_mode: Which enhancement method to use
    #   - "codeformer": Full CodeFormer restoration (~15ms/frame, best quality)
    #   - "fast": Simple sharpening + color matching (~3ms/frame, good quality)
    #   - "none": No enhancement (fastest, acceptable quality)
    enhancement_mode: str = "codeformer"

    # Quality settings
    fidelity_weight: float = 0.7
    feather_radius: int = 15
    boundary_blend_frames: int = 5

    # Parallel processing
    chunk_duration: float = 5.0
    max_workers: int = 4

    # Caching (for repeated processing of same face)
    # When enabled, caches VAE latents and masks per face identity
    use_face_caching: bool = True


class LipSyncPipeline:
    """
    Complete multi-face lip-sync pipeline.

    Processes one or more faces in a video, each with their own
    audio track and bounding box region.
    """

    def __init__(self, config: Optional[PipelineConfig] = None):
        """
        Initialize pipeline.

        Args:
            config: Pipeline configuration
        """
        self.config = config or PipelineConfig()
        logger.info("Initializing LipSyncPipeline")
        logger.debug(f"  Device: {self.config.device}")
        logger.debug(f"  FP16: {self.config.fp16}")
        logger.debug(f"  Aligned size: {self.config.aligned_size}")
        logger.debug(f"  Neutralization: {self.config.use_neutralization}")
        logger.debug(f"  Enhancement: {self.config.use_enhancement}")
        logger.debug(f"  Face caching: {self.config.use_face_caching}")
        logger.debug(f"  MuseTalk path: {self.config.musetalk_path}")
        logger.debug(f"  LivePortrait path: {self.config.liveportrait_path}")
        logger.debug(f"  CodeFormer path: {self.config.codeformer_path}")

        # Initialize face cache if enabled
        self.cache = FaceCache(device=self.config.device) if self.config.use_face_caching else None
        if self.cache:
            logger.debug("  Face cache initialized")

        # Initialize models (lazy loading)
        logger.debug("Creating model wrappers (lazy loading)...")
        self.musetalk = MuseTalk(
            config=MuseTalkConfig(
                model_dir=self.config.musetalk_path,
                device=self.config.device,
                fp16=self.config.fp16,
            ),
            cache=self.cache,
        )

        self.liveportrait = LivePortrait(LivePortraitConfig(
            model_dir=self.config.liveportrait_path,
            device=self.config.device,
            fp16=self.config.fp16,
        ))

        self.codeformer = CodeFormer(CodeFormerConfig(
            model_path=self.config.codeformer_path,
            device=self.config.device,
            fp16=self.config.fp16,
            fidelity_weight=self.config.fidelity_weight,
        ))

        self.compositor = FaceCompositor(
            feather_radius=self.config.feather_radius,
            boundary_blend_frames=self.config.boundary_blend_frames,
        )

        self._models_loaded = False
        logger.info("LipSyncPipeline initialized successfully")

    def load_models(self) -> None:
        """Load all models to GPU."""
        if self._models_loaded:
            logger.debug("Models already loaded, skipping")
            return

        logger.info("Loading lip-sync pipeline models to GPU...")
        logger.info(f"  Pipeline config: neutralization={self.config.use_neutralization}, enhancement={self.config.use_enhancement}")
        start_time = time.time()

        # MuseTalk is always required (core lip-sync engine)
        logger.info("  Loading MuseTalk...")
        musetalk_start = time.time()
        self.musetalk.load()
        logger.info(f"  MuseTalk loaded in {time.time() - musetalk_start:.2f}s")

        # LivePortrait is optional (lip neutralization)
        if self.config.use_neutralization:
            logger.info("  Loading LivePortrait...")
            liveportrait_start = time.time()
            self.liveportrait.load()
            logger.info(f"  LivePortrait loaded in {time.time() - liveportrait_start:.2f}s")
        else:
            logger.info("  Skipping LivePortrait (use_neutralization=False)")

        # CodeFormer is optional (face enhancement) - only load if using codeformer mode
        if self.config.use_enhancement and self.config.enhancement_mode == "codeformer":
            logger.info("  Loading CodeFormer...")
            codeformer_start = time.time()
            self.codeformer.load()
            logger.info(f"  CodeFormer loaded in {time.time() - codeformer_start:.2f}s")
        elif self.config.use_enhancement and self.config.enhancement_mode == "fast":
            logger.info("  Using fast enhancement (no model loading required)")
        else:
            logger.info("  Skipping enhancement (use_enhancement=False or mode=none)")

        self._models_loaded = True
        total_time = time.time() - start_time
        logger.info(f"All models loaded in {total_time:.2f}s")

    def unload_models(self) -> None:
        """Unload all models from GPU."""
        logger.info("Unloading models from GPU...")
        self.musetalk.unload()
        self.liveportrait.unload()
        self.codeformer.unload()
        self._models_loaded = False
        logger.info("All models unloaded")

    def process_single_face(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
        target_bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> str:
        """
        Process lip-sync for a single face.

        This is the simpler case where we process the entire video
        with one face and one audio track.

        Args:
            video_path: Path to input video
            audio_path: Path to audio file
            output_path: Path for output video
            target_bbox: Optional bbox to focus on specific face

        Returns:
            Path to output video
        """
        logger.info("=" * 60)
        logger.info("STARTING SINGLE FACE LIP-SYNC")
        logger.info("=" * 60)
        logger.info(f"  Video: {video_path}")
        logger.info(f"  Audio: {audio_path}")
        logger.info(f"  Output: {output_path}")
        logger.info(f"  Target bbox: {target_bbox}")

        pipeline_start = time.time()

        # Validate inputs
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        video_size = os.path.getsize(video_path) / (1024 * 1024)
        audio_size = os.path.getsize(audio_path) / (1024 * 1024)
        logger.debug(f"  Video size: {video_size:.2f} MB")
        logger.debug(f"  Audio size: {audio_size:.2f} MB")

        self.load_models()

        with tempfile.TemporaryDirectory() as tmpdir:
            logger.debug(f"Using temp directory: {tmpdir}")

            # Step 1: Align face
            logger.info("-" * 40)
            logger.info("STEP 1: Face Alignment")
            logger.info("-" * 40)
            aligned_path = os.path.join(tmpdir, "aligned.mp4")
            step_start = time.time()

            try:
                logger.debug(f"  Calling align_video with target_bbox={target_bbox}")
                align_metadata = align_video(
                    video_path,
                    aligned_path,
                    target_bbox=target_bbox,
                    output_size=self.config.aligned_size,
                )
                has_alignment = True
                logger.info(f"  Alignment successful in {time.time() - step_start:.2f}s")
                logger.debug(f"  Avg face size: {align_metadata.avg_face_size:.2f}%")
                logger.debug(f"  Output: {aligned_path}")
            except Exception as e:
                logger.warning(f"  Alignment failed: {e}")
                logger.warning("  Continuing without alignment...")
                aligned_path = video_path
                align_metadata = None
                has_alignment = False

            # Step 2: Neutralize with LivePortrait (OPTIONAL)
            if self.config.use_neutralization:
                logger.info("-" * 40)
                logger.info("STEP 2: Expression Neutralization (LivePortrait)")
                logger.info("-" * 40)
                neutral_path = os.path.join(tmpdir, "neutral.mp4")
                step_start = time.time()

                try:
                    logger.debug(f"  Input: {aligned_path}")
                    logger.debug(f"  lip_ratio: 0.0")
                    self.liveportrait.neutralize(
                        aligned_path,
                        neutral_path,
                        lip_ratio=0.0,
                    )
                    logger.info(f"  Neutralization successful in {time.time() - step_start:.2f}s")
                    logger.debug(f"  Output: {neutral_path}")
                except Exception as e:
                    logger.warning(f"  Neutralization failed: {e}")
                    logger.warning("  Continuing with aligned video...")
                    neutral_path = aligned_path
            else:
                logger.info("-" * 40)
                logger.info("STEP 2: Skipping Neutralization (use_neutralization=False)")
                logger.info("-" * 40)
                logger.info("  Source assumed to be neutral (no active lip movement)")
                neutral_path = aligned_path

            # Step 3: Lip-sync with MuseTalk
            logger.info("-" * 40)
            logger.info("STEP 3: Lip-Sync Generation (MuseTalk)")
            logger.info("-" * 40)
            lipsync_path = os.path.join(tmpdir, "lipsync.mp4")
            step_start = time.time()

            logger.debug(f"  Video input: {neutral_path}")
            logger.debug(f"  Audio input: {audio_path}")
            self.musetalk.process(
                neutral_path,
                audio_path,
                lipsync_path,
            )
            logger.info(f"  Lip-sync generation successful in {time.time() - step_start:.2f}s")
            logger.debug(f"  Output: {lipsync_path}")

            # Step 4: Enhancement (OPTIONAL - multiple modes)
            enhancement_mode = self.config.enhancement_mode if self.config.use_enhancement else "none"

            if enhancement_mode == "codeformer":
                logger.info("-" * 40)
                logger.info("STEP 4: Quality Enhancement (CodeFormer)")
                logger.info("-" * 40)
                enhanced_path = os.path.join(tmpdir, "enhanced.mp4")
                step_start = time.time()

                blend_ratio = CodeFormer.compute_blend_ratio(
                    align_metadata.avg_face_size if align_metadata else 5.0
                )
                logger.debug(f"  Input: {lipsync_path}")
                logger.debug(f"  Blend ratio: {blend_ratio:.2f}")
                logger.debug(f"  Has aligned: {has_alignment}")

                self.codeformer.enhance_video(
                    lipsync_path,
                    enhanced_path,
                    blend_ratio=blend_ratio,
                    has_aligned=has_alignment,
                )
                logger.info(f"  Enhancement successful in {time.time() - step_start:.2f}s")
                logger.debug(f"  Output: {enhanced_path}")

            elif enhancement_mode == "fast":
                logger.info("-" * 40)
                logger.info("STEP 4: Fast Enhancement (bilateral + sharpen)")
                logger.info("-" * 40)
                enhanced_path = os.path.join(tmpdir, "enhanced.mp4")
                step_start = time.time()

                self._fast_enhance_video(lipsync_path, enhanced_path)
                logger.info(f"  Fast enhancement done in {time.time() - step_start:.2f}s")
                logger.debug(f"  Output: {enhanced_path}")

            else:  # "none" or disabled
                logger.info("-" * 40)
                logger.info("STEP 4: Skipping Enhancement (mode=none)")
                logger.info("-" * 40)
                enhanced_path = lipsync_path

            # Step 5: Unalign back to original video
            if has_alignment and align_metadata:
                logger.info("-" * 40)
                logger.info("STEP 5: Unalignment (back to original frame)")
                logger.info("-" * 40)
                unaligned_path = os.path.join(tmpdir, "unaligned.mp4")
                step_start = time.time()

                logger.debug(f"  Input: {enhanced_path}")
                logger.debug(f"  Original video: {video_path}")
                unalign_video(
                    enhanced_path,
                    video_path,
                    align_metadata,
                    unaligned_path,
                )
                logger.info(f"  Unalignment successful in {time.time() - step_start:.2f}s")
                logger.debug(f"  Output: {unaligned_path}")
            else:
                logger.info("-" * 40)
                logger.info("STEP 5: Skipping Unalignment (no alignment metadata)")
                logger.info("-" * 40)
                unaligned_path = enhanced_path

            # Step 6: Add audio
            logger.info("-" * 40)
            logger.info("STEP 6: Combining Audio and Video")
            logger.info("-" * 40)
            step_start = time.time()

            logger.debug(f"  Video: {unaligned_path}")
            logger.debug(f"  Audio: {audio_path}")
            logger.debug(f"  Output: {output_path}")
            combine_audio_video(unaligned_path, audio_path, output_path)
            logger.info(f"  Audio combined in {time.time() - step_start:.2f}s")

        # Done
        total_time = time.time() - pipeline_start
        output_size = os.path.getsize(output_path) / (1024 * 1024)

        logger.info("=" * 60)
        logger.info("SINGLE FACE LIP-SYNC COMPLETE")
        logger.info("=" * 60)
        logger.info(f"  Output: {output_path}")
        logger.info(f"  Output size: {output_size:.2f} MB")
        logger.info(f"  Total time: {total_time:.2f}s")

        return output_path

    def process_multi_face(
        self,
        video_path: str,
        face_jobs: List[FaceJob],
        output_path: str,
    ) -> str:
        """
        Process lip-sync for multiple faces.

        Each face has its own bbox, audio track, and time range.
        Faces are processed independently and composited together.

        Args:
            video_path: Path to input video
            face_jobs: List of face processing jobs
            output_path: Path for output video

        Returns:
            Path to output video
        """
        logger.info("=" * 60)
        logger.info("STARTING MULTI-FACE LIP-SYNC")
        logger.info("=" * 60)
        logger.info(f"  Video: {video_path}")
        logger.info(f"  Output: {output_path}")
        logger.info(f"  Number of faces: {len(face_jobs)}")

        for i, job in enumerate(face_jobs):
            logger.info(f"  Face {i+1}: {job.character_id}")
            logger.debug(f"    bbox: {job.bbox}")
            logger.debug(f"    time: {job.start_time_ms}ms - {job.end_time_ms}ms")
            logger.debug(f"    audio: {job.audio_path}")

        pipeline_start = time.time()

        if not face_jobs:
            raise ValueError("No face jobs provided")

        if len(face_jobs) == 1:
            logger.info("Single face detected, using simpler path")
            job = face_jobs[0]
            return self.process_single_face(
                video_path,
                job.audio_path,
                output_path,
                target_bbox=job.bbox,
            )

        self.load_models()

        logger.debug("Getting video info...")
        video_info = get_video_info(video_path)
        logger.debug(f"  FPS: {video_info.fps}")
        logger.debug(f"  Duration: {video_info.duration}s")
        logger.debug(f"  Resolution: {video_info.width}x{video_info.height}")

        face_regions: List[FaceRegion] = []

        with tempfile.TemporaryDirectory() as tmpdir:
            logger.debug(f"Using temp directory: {tmpdir}")

            # Process each face independently
            for i, job in enumerate(face_jobs):
                logger.info("-" * 40)
                logger.info(f"PROCESSING FACE {i+1}/{len(face_jobs)}: {job.character_id}")
                logger.info("-" * 40)
                face_start = time.time()

                try:
                    # Trim video to face time range
                    logger.debug(f"  Trimming video: {job.start_time}s - {job.end_time}s")
                    trimmed_path = os.path.join(tmpdir, f"trimmed_{i}.mp4")
                    trim_video(video_path, trimmed_path, job.start_time, job.end_time)
                    logger.debug(f"  Trimmed: {trimmed_path}")

                    # Process the trimmed video with target bbox
                    processed_path = os.path.join(tmpdir, f"processed_{i}.mp4")
                    logger.debug(f"  Processing segment with bbox {job.bbox}...")
                    self._process_face_segment(
                        trimmed_path,
                        job.audio_path,
                        processed_path,
                        job.bbox,
                        tmpdir,
                    )
                    logger.debug(f"  Processed: {processed_path}")

                    # Extract processed frames
                    logger.debug("  Extracting frames from processed video...")
                    processed_frames = self._extract_frames(processed_path)
                    logger.debug(f"  Extracted {len(processed_frames)} frames")

                    # Calculate frame range
                    start_frame = int(job.start_time * video_info.fps)
                    end_frame = int(job.end_time * video_info.fps)
                    logger.debug(f"  Frame range: {start_frame} - {end_frame}")

                    face_regions.append(FaceRegion(
                        bbox=job.bbox,
                        frames=processed_frames,
                        start_frame=start_frame,
                        end_frame=end_frame,
                        character_id=job.character_id,
                    ))

                    face_time = time.time() - face_start
                    logger.info(f"  Face {job.character_id} processed in {face_time:.2f}s")

                except Exception as e:
                    logger.error(f"  Failed to process face {job.character_id}: {e}")
                    logger.exception("  Full traceback:")
                    continue

            if not face_regions:
                raise RuntimeError("All face processing failed")

            logger.info(f"Successfully processed {len(face_regions)}/{len(face_jobs)} faces")

            # Composite all faces back into original video
            logger.info("-" * 40)
            logger.info("COMPOSITING FACES")
            logger.info("-" * 40)
            composited_path = os.path.join(tmpdir, "composited.mp4")
            step_start = time.time()

            logger.debug(f"  Compositing {len(face_regions)} face regions...")
            self.compositor.composite_video_from_paths(
                video_path,
                face_regions,
                composited_path,
            )
            logger.info(f"  Compositing done in {time.time() - step_start:.2f}s")

            # Mix audio tracks
            logger.info("-" * 40)
            logger.info("MIXING AUDIO TRACKS")
            logger.info("-" * 40)
            combined_audio_path = os.path.join(tmpdir, "combined_audio.wav")
            step_start = time.time()

            logger.debug(f"  Mixing {len(face_jobs)} audio tracks...")
            self._mix_audio_tracks(
                [job.audio_path for job in face_jobs],
                [(job.start_time, job.end_time) for job in face_jobs],
                video_info.duration,
                combined_audio_path,
            )
            logger.info(f"  Audio mixed in {time.time() - step_start:.2f}s")

            # Combine final video
            logger.info("-" * 40)
            logger.info("COMBINING FINAL VIDEO")
            logger.info("-" * 40)
            step_start = time.time()
            combine_audio_video(composited_path, combined_audio_path, output_path)
            logger.info(f"  Final video created in {time.time() - step_start:.2f}s")

        # Done
        total_time = time.time() - pipeline_start
        output_size = os.path.getsize(output_path) / (1024 * 1024)

        logger.info("=" * 60)
        logger.info("MULTI-FACE LIP-SYNC COMPLETE")
        logger.info("=" * 60)
        logger.info(f"  Output: {output_path}")
        logger.info(f"  Output size: {output_size:.2f} MB")
        logger.info(f"  Faces processed: {len(face_regions)}/{len(face_jobs)}")
        logger.info(f"  Total time: {total_time:.2f}s")

        return output_path

    def _process_face_segment(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
        bbox: Tuple[int, int, int, int],
        tmpdir: str,
    ) -> None:
        """Process a single face segment."""
        logger.debug(f"    _process_face_segment: bbox={bbox}")

        # Align with target bbox
        aligned_path = os.path.join(tmpdir, f"aligned_{uuid.uuid4()}.mp4")
        logger.debug(f"    Aligning face...")
        align_metadata = align_video(
            video_path,
            aligned_path,
            target_bbox=bbox,
            output_size=self.config.aligned_size,
        )
        logger.debug(f"    Aligned: {aligned_path}")

        # Neutralize (optional)
        if self.config.use_neutralization:
            neutral_path = os.path.join(tmpdir, f"neutral_{uuid.uuid4()}.mp4")
            logger.debug(f"    Neutralizing...")
            self.liveportrait.neutralize(aligned_path, neutral_path, lip_ratio=0.0)
            logger.debug(f"    Neutralized: {neutral_path}")
        else:
            neutral_path = aligned_path
            logger.debug(f"    Skipping neutralization (use_neutralization=False)")

        # Lip-sync
        lipsync_path = os.path.join(tmpdir, f"lipsync_{uuid.uuid4()}.mp4")
        logger.debug(f"    Generating lip-sync...")
        self.musetalk.process(neutral_path, audio_path, lipsync_path)
        logger.debug(f"    Lip-synced: {lipsync_path}")

        # Enhance
        enhancement_mode = self.config.enhancement_mode if self.config.use_enhancement else "none"

        if enhancement_mode == "codeformer":
            enhanced_path = os.path.join(tmpdir, f"enhanced_{uuid.uuid4()}.mp4")
            blend_ratio = CodeFormer.compute_blend_ratio(align_metadata.avg_face_size)
            logger.debug(f"    Enhancing with CodeFormer (blend_ratio={blend_ratio:.2f})...")
            self.codeformer.enhance_video(
                lipsync_path,
                enhanced_path,
                blend_ratio=blend_ratio,
                has_aligned=True,
            )
            logger.debug(f"    Enhanced: {enhanced_path}")
        elif enhancement_mode == "fast":
            enhanced_path = os.path.join(tmpdir, f"enhanced_{uuid.uuid4()}.mp4")
            logger.debug(f"    Enhancing with fast mode...")
            self._fast_enhance_video(lipsync_path, enhanced_path)
            logger.debug(f"    Fast enhanced: {enhanced_path}")
        else:
            enhanced_path = lipsync_path

        # Copy to output
        logger.debug(f"    Copying to output: {output_path}")
        shutil.copy(enhanced_path, output_path)

    def _extract_frames(self, video_path: str) -> List[np.ndarray]:
        """Extract all frames from video."""
        frames = []
        cap = cv2.VideoCapture(video_path)

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.debug(f"    Extracting {frame_count} frames from {video_path}")

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        cap.release()
        logger.debug(f"    Extracted {len(frames)} frames")
        return frames

    def _fast_enhance_video(self, input_path: str, output_path: str) -> None:
        """
        Apply fast enhancement to video using simple CV2 operations.

        This is ~4x faster than CodeFormer while providing acceptable quality.
        Uses bilateral filtering + unsharp mask sharpening.
        """
        from .compositor import fast_enhance_face

        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Read first frame as reference for color matching
        ret, first_frame = cap.read()
        if not ret:
            cap.release()
            out.release()
            return

        reference = first_frame.copy()
        enhanced = fast_enhance_face(first_frame, reference=None, sharpen_amount=0.3)
        out.write(enhanced)

        processed = 1
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            enhanced = fast_enhance_face(frame, reference=reference, sharpen_amount=0.3)
            out.write(enhanced)
            processed += 1

            if processed % 100 == 0:
                logger.debug(f"    Fast enhanced {processed}/{frame_count} frames")

        cap.release()
        out.release()
        logger.debug(f"    Fast enhanced {processed} frames total")

    def _mix_audio_tracks(
        self,
        audio_paths: List[str],
        time_ranges: List[Tuple[float, float]],
        total_duration: float,
        output_path: str,
    ) -> str:
        """Mix multiple audio tracks at specified time offsets."""
        logger.debug(f"  Mixing {len(audio_paths)} audio tracks")

        # Build FFmpeg filter for mixing
        inputs = []
        delays = []

        for i, (audio_path, (start, end)) in enumerate(zip(audio_paths, time_ranges)):
            inputs.extend(['-i', audio_path])
            delay_ms = int(start * 1000)
            delays.append(f'[{i}:a]adelay={delay_ms}|{delay_ms}[a{i}]')
            logger.debug(f"    Track {i}: {audio_path} (delay={delay_ms}ms)")

        # Mix all delayed tracks
        mix_inputs = ''.join(f'[a{i}]' for i in range(len(audio_paths)))
        filter_complex = ';'.join(delays) + f';{mix_inputs}amix=inputs={len(audio_paths)}:normalize=0[out]'

        cmd = [
            'ffmpeg', '-y',
            *inputs,
            '-filter_complex', filter_complex,
            '-map', '[out]',
            '-t', str(total_duration),
            output_path,
        ]

        logger.debug(f"  FFmpeg command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, check=True)
        logger.debug(f"  Audio mixed: {output_path}")

        return output_path

    @property
    def is_loaded(self) -> bool:
        """Check if models are loaded."""
        return self._models_loaded


def process_lipsync(
    video_path: str,
    audio_path: str,
    output_path: str,
    face_bbox: Optional[Tuple[int, int, int, int]] = None,
    config: Optional[PipelineConfig] = None,
) -> str:
    """
    Convenience function for single-face lip-sync.

    Args:
        video_path: Path to input video
        audio_path: Path to audio file
        output_path: Path for output video
        face_bbox: Optional bbox for target face
        config: Pipeline configuration

    Returns:
        Path to output video
    """
    logger.info("process_lipsync() called")
    pipeline = LipSyncPipeline(config)
    return pipeline.process_single_face(
        video_path,
        audio_path,
        output_path,
        target_bbox=face_bbox,
    )
