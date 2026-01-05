"""
Multi-face lip-sync pipeline.

Orchestrates the complete lip-sync workflow:
1. Align faces (optionally with target bbox)
2. Neutralize expressions with LivePortrait
3. Generate lip-sync with MuseTalk
4. Enhance with CodeFormer
5. Composite back into original video
"""

import os
import tempfile
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
from .utils import (
    get_video_info,
    get_audio_duration,
    extract_audio,
    combine_audio_video,
    concat_videos,
    trim_video,
    generate_timestamp_chunks,
)


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

    # Quality settings
    enhance_quality: bool = True
    fidelity_weight: float = 0.7
    feather_radius: int = 15
    boundary_blend_frames: int = 5

    # Parallel processing
    chunk_duration: float = 5.0
    max_workers: int = 4


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

        # Initialize models (lazy loading)
        self.musetalk = MuseTalk(MuseTalkConfig(
            model_path=self.config.musetalk_path,
            device=self.config.device,
            fp16=self.config.fp16,
        ))

        self.liveportrait = LivePortrait(LivePortraitConfig(
            model_path=self.config.liveportrait_path,
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

    def load_models(self) -> None:
        """Load all models to GPU."""
        if self._models_loaded:
            return

        print("Loading lip-sync pipeline models...")
        self.musetalk.load()
        self.liveportrait.load()
        if self.config.enhance_quality:
            self.codeformer.load()

        self._models_loaded = True
        print("All models loaded")

    def unload_models(self) -> None:
        """Unload all models from GPU."""
        self.musetalk.unload()
        self.liveportrait.unload()
        self.codeformer.unload()
        self._models_loaded = False

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
        self.load_models()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Step 1: Align face
            aligned_path = os.path.join(tmpdir, "aligned.mp4")
            print("Aligning face...")
            try:
                align_metadata = align_video(
                    video_path,
                    aligned_path,
                    target_bbox=target_bbox,
                    output_size=self.config.aligned_size,
                )
                has_alignment = True
            except Exception as e:
                print(f"Alignment failed: {e}, continuing without alignment")
                aligned_path = video_path
                align_metadata = None
                has_alignment = False

            # Step 2: Neutralize with LivePortrait
            neutral_path = os.path.join(tmpdir, "neutral.mp4")
            print("Neutralizing expressions...")
            try:
                self.liveportrait.neutralize(
                    aligned_path,
                    neutral_path,
                    lip_ratio=0.0,
                )
            except Exception as e:
                print(f"Neutralization failed: {e}, continuing with aligned video")
                neutral_path = aligned_path

            # Step 3: Lip-sync with MuseTalk
            lipsync_path = os.path.join(tmpdir, "lipsync.mp4")
            print("Generating lip-sync...")
            self.musetalk.process(
                neutral_path,
                audio_path,
                lipsync_path,
            )

            # Step 4: Enhance with CodeFormer
            if self.config.enhance_quality:
                enhanced_path = os.path.join(tmpdir, "enhanced.mp4")
                print("Enhancing quality...")
                blend_ratio = CodeFormer.compute_blend_ratio(
                    align_metadata.avg_face_size if align_metadata else 5.0
                )
                self.codeformer.enhance_video(
                    lipsync_path,
                    enhanced_path,
                    blend_ratio=blend_ratio,
                    has_aligned=has_alignment,
                )
            else:
                enhanced_path = lipsync_path

            # Step 5: Unalign back to original video
            if has_alignment and align_metadata:
                unaligned_path = os.path.join(tmpdir, "unaligned.mp4")
                print("Unaligning video...")
                unalign_video(
                    enhanced_path,
                    video_path,
                    align_metadata,
                    unaligned_path,
                )
            else:
                unaligned_path = enhanced_path

            # Step 6: Add audio
            print("Adding audio...")
            combine_audio_video(unaligned_path, audio_path, output_path)

        print(f"Lip-sync complete: {output_path}")
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
        if not face_jobs:
            raise ValueError("No face jobs provided")

        if len(face_jobs) == 1:
            # Single face - use simpler path
            job = face_jobs[0]
            return self.process_single_face(
                video_path,
                job.audio_path,
                output_path,
                target_bbox=job.bbox,
            )

        self.load_models()

        video_info = get_video_info(video_path)
        face_regions: List[FaceRegion] = []

        with tempfile.TemporaryDirectory() as tmpdir:
            # Process each face independently
            for i, job in enumerate(face_jobs):
                print(f"Processing face {i+1}/{len(face_jobs)}: {job.character_id}")

                try:
                    # Trim video to face time range
                    trimmed_path = os.path.join(tmpdir, f"trimmed_{i}.mp4")
                    trim_video(video_path, trimmed_path, job.start_time, job.end_time)

                    # Process the trimmed video with target bbox
                    processed_path = os.path.join(tmpdir, f"processed_{i}.mp4")
                    self._process_face_segment(
                        trimmed_path,
                        job.audio_path,
                        processed_path,
                        job.bbox,
                        tmpdir,
                    )

                    # Extract processed frames
                    processed_frames = self._extract_frames(processed_path)

                    # Calculate frame range
                    start_frame = int(job.start_time * video_info.fps)
                    end_frame = int(job.end_time * video_info.fps)

                    face_regions.append(FaceRegion(
                        bbox=job.bbox,
                        frames=processed_frames,
                        start_frame=start_frame,
                        end_frame=end_frame,
                        character_id=job.character_id,
                    ))

                except Exception as e:
                    print(f"Failed to process face {job.character_id}: {e}")
                    continue

            if not face_regions:
                raise RuntimeError("All face processing failed")

            # Composite all faces back into original video
            print("Compositing faces...")
            composited_path = os.path.join(tmpdir, "composited.mp4")
            self.compositor.composite_video_from_paths(
                video_path,
                face_regions,
                composited_path,
            )

            # Extract and add combined audio
            # For multi-face, we need to mix the audio tracks
            combined_audio_path = os.path.join(tmpdir, "combined_audio.wav")
            self._mix_audio_tracks(
                [job.audio_path for job in face_jobs],
                [(job.start_time, job.end_time) for job in face_jobs],
                video_info.duration,
                combined_audio_path,
            )

            combine_audio_video(composited_path, combined_audio_path, output_path)

        print(f"Multi-face lip-sync complete: {output_path}")
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
        # Align with target bbox
        aligned_path = os.path.join(tmpdir, f"aligned_{uuid.uuid4()}.mp4")
        align_metadata = align_video(
            video_path,
            aligned_path,
            target_bbox=bbox,
            output_size=self.config.aligned_size,
        )

        # Neutralize
        neutral_path = os.path.join(tmpdir, f"neutral_{uuid.uuid4()}.mp4")
        self.liveportrait.neutralize(aligned_path, neutral_path, lip_ratio=0.0)

        # Lip-sync
        lipsync_path = os.path.join(tmpdir, f"lipsync_{uuid.uuid4()}.mp4")
        self.musetalk.process(neutral_path, audio_path, lipsync_path)

        # Enhance
        if self.config.enhance_quality:
            enhanced_path = os.path.join(tmpdir, f"enhanced_{uuid.uuid4()}.mp4")
            blend_ratio = CodeFormer.compute_blend_ratio(align_metadata.avg_face_size)
            self.codeformer.enhance_video(
                lipsync_path,
                enhanced_path,
                blend_ratio=blend_ratio,
                has_aligned=True,
            )
        else:
            enhanced_path = lipsync_path

        # Copy to output
        import shutil
        shutil.copy(enhanced_path, output_path)

    def _extract_frames(self, video_path: str) -> List[np.ndarray]:
        """Extract all frames from video."""
        frames = []
        cap = cv2.VideoCapture(video_path)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        cap.release()
        return frames

    def _mix_audio_tracks(
        self,
        audio_paths: List[str],
        time_ranges: List[Tuple[float, float]],
        total_duration: float,
        output_path: str,
    ) -> str:
        """Mix multiple audio tracks at specified time offsets."""
        import subprocess

        # Build FFmpeg filter for mixing
        inputs = []
        delays = []

        for i, (audio_path, (start, end)) in enumerate(zip(audio_paths, time_ranges)):
            inputs.extend(['-i', audio_path])
            delay_ms = int(start * 1000)
            delays.append(f'[{i}:a]adelay={delay_ms}|{delay_ms}[a{i}]')

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

        subprocess.run(cmd, capture_output=True, check=True)
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
    pipeline = LipSyncPipeline(config)
    return pipeline.process_single_face(
        video_path,
        audio_path,
        output_path,
        target_bbox=face_bbox,
    )
