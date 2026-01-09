"""
Wav2Lip-HD wrapper for lip-sync processing.

This module provides a Python interface to the Wav2Lip-HD inference script,
handling bounding box injection, video processing, and GFPGAN enhancement.
"""

import json
import logging
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger("lipsync.wav2lip")

# Path to Wav2Lip-HD vendor directory
WAV2LIP_DIR = Path(__file__).parent.parent / "vendors" / "wav2lip-hd" / "Wav2Lip-master"
GFPGAN_DIR = Path(__file__).parent.parent / "vendors" / "wav2lip-hd" / "GFPGAN-master"


@dataclass
class Wav2LipConfig:
    """Configuration for Wav2Lip-HD processing."""

    checkpoint_path: str = "/app/models/wav2lip_gan.pth"
    gfpgan_checkpoint: str = "/app/models/GFPGANv1.4.pth"
    face_det_batch_size: int = 16
    wav2lip_batch_size: int = 128
    resize_factor: int = 1
    pads: List[int] = None  # [top, bottom, left, right]
    nosmooth: bool = False
    enhance: bool = True

    def __post_init__(self):
        if self.pads is None:
            # Padding: [top, bottom, left, right]
            # Increased bottom padding to capture more chin area
            # and reduce visible edge artifacts
            self.pads = [0, 20, 0, 0]


class Wav2LipHD:
    """
    Wrapper for Wav2Lip-HD lip-sync processing.

    Supports per-frame bounding boxes for multi-face tracking.
    """

    def __init__(self, config: Optional[Wav2LipConfig] = None):
        """
        Initialize Wav2Lip-HD wrapper.

        Args:
            config: Optional configuration override
        """
        self.config = config or Wav2LipConfig()
        self._validate_paths()

    def _validate_paths(self):
        """Validate that required files exist."""
        if not WAV2LIP_DIR.exists():
            raise FileNotFoundError(f"Wav2Lip directory not found: {WAV2LIP_DIR}")

        if not Path(self.config.checkpoint_path).exists():
            logger.warning(f"Wav2Lip checkpoint not found: {self.config.checkpoint_path}")

        if self.config.enhance and not Path(self.config.gfpgan_checkpoint).exists():
            logger.warning(f"GFPGAN checkpoint not found: {self.config.gfpgan_checkpoint}")

    def process(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
        bboxes: Optional[List[Optional[List[int]]]] = None,
        enhance: Optional[bool] = None,
        loop_mode: str = "crossfade",
        crossfade_frames: int = 10,
    ) -> str:
        """
        Process video with Wav2Lip-HD lip-sync.

        Args:
            video_path: Path to input video
            audio_path: Path to audio file
            output_path: Path for output video
            bboxes: Optional list of per-frame bounding boxes [[y1,y2,x1,x2], ...]
                    Use None for frames that need face detection
            enhance: Override config enhance setting
            loop_mode: How to handle audio longer than video:
                       'none' = trim to shortest
                       'repeat' = loop with hard cut
                       'pingpong' = forward-backward loop
                       'crossfade' = loop with smooth blend (default)
            crossfade_frames: Number of frames to blend at loop boundary

        Returns:
            Path to output video
        """
        start_time = time.time()
        enhance = enhance if enhance is not None else self.config.enhance

        logger.info(f"Processing video with Wav2Lip-HD:")
        logger.info(f"  Video: {video_path}")
        logger.info(f"  Audio: {audio_path}")
        logger.info(f"  Output: {output_path}")
        logger.info(f"  Enhance: {enhance}")
        logger.info(f"  Loop mode: {loop_mode}")
        if bboxes:
            non_null = sum(1 for b in bboxes if b is not None)
            logger.info(f"  Bboxes: {non_null}/{len(bboxes)} frames with tracked boxes")

        with tempfile.TemporaryDirectory() as tmpdir:
            # Write bboxes to temp file if provided
            bbox_file = None
            if bboxes:
                bbox_file = os.path.join(tmpdir, "bboxes.json")
                with open(bbox_file, "w") as f:
                    json.dump(bboxes, f)
                logger.debug(f"  Bbox file: {bbox_file}")

            # Run Wav2Lip inference
            wav2lip_output = os.path.join(tmpdir, "wav2lip_output.mp4")
            self._run_wav2lip(
                video_path, audio_path, wav2lip_output, bbox_file,
                loop_mode=loop_mode, crossfade_frames=crossfade_frames
            )

            # Run GFPGAN enhancement if enabled
            if enhance:
                logger.info("Running GFPGAN enhancement...")
                self._run_gfpgan(wav2lip_output, output_path, bboxes)
            else:
                # Copy output directly
                import shutil
                shutil.copy(wav2lip_output, output_path)

        elapsed = time.time() - start_time
        logger.info(f"  Completed in {elapsed:.2f}s")

        return output_path

    def _run_wav2lip(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
        bbox_file: Optional[str] = None,
        loop_mode: str = "crossfade",
        crossfade_frames: int = 10,
    ):
        """Run Wav2Lip inference.

        Args:
            video_path: Input video path
            audio_path: Input audio path
            output_path: Output video path
            bbox_file: Optional JSON file with per-frame bounding boxes
            loop_mode: How to handle audio > video length
            crossfade_frames: Frames to blend at loop boundary
        """
        cmd = [
            "python",
            str(WAV2LIP_DIR / "inference.py"),
            "--checkpoint_path", self.config.checkpoint_path,
            "--face", video_path,
            "--audio", audio_path,
            "--outfile", output_path,
            "--face_det_batch_size", str(self.config.face_det_batch_size),
            "--wav2lip_batch_size", str(self.config.wav2lip_batch_size),
            "--resize_factor", str(self.config.resize_factor),
            "--pads", *[str(p) for p in self.config.pads],
        ]

        # Use per-frame bboxes if provided (our patched inference.py)
        if bbox_file:
            cmd.extend(["--bbox_file", bbox_file])

        # Loop mode for handling audio longer than video
        cmd.extend(["--loop_mode", loop_mode])
        cmd.extend(["--crossfade_frames", str(crossfade_frames)])

        if self.config.nosmooth:
            cmd.append("--nosmooth")

        logger.debug(f"  Running: {' '.join(cmd)}")

        # Set up environment with Wav2Lip in Python path
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{WAV2LIP_DIR}:{env.get('PYTHONPATH', '')}"

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(WAV2LIP_DIR),
            env=env,
        )

        if result.returncode != 0:
            logger.error(f"Wav2Lip failed: {result.stderr}")
            raise RuntimeError(f"Wav2Lip inference failed: {result.stderr}")

        if not os.path.exists(output_path):
            raise RuntimeError(f"Wav2Lip did not produce output: {output_path}")

    def _run_gfpgan(
        self,
        input_path: str,
        output_path: str,
        bboxes: Optional[List[Optional[List[int]]]] = None,
    ):
        """
        Run optimized GFPGAN face enhancement on video.

        Uses face-region-only processing with Poisson blending for speed.

        Args:
            input_path: Input video path
            output_path: Output video path
            bboxes: Per-frame face bounding boxes for targeted enhancement
        """
        import cv2
        from .enhancer import FaceEnhancer

        # Read video frames
        logger.info("  Reading video frames...")
        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        if not frames:
            logger.warning("No frames read from video")
            import shutil
            shutil.copy(input_path, output_path)
            return

        logger.info(f"  Read {len(frames)} frames at {width}x{height} @ {fps:.1f}fps")

        # Prepare bboxes - if not provided or wrong length, use None (full frame)
        if bboxes is None or len(bboxes) != len(frames):
            logger.info("  No bboxes provided, enhancing full frames")
            frame_bboxes = [None] * len(frames)
        else:
            frame_bboxes = bboxes

        # Run optimized face enhancement
        enhancer = FaceEnhancer(model_path=self.config.gfpgan_checkpoint)
        try:
            enhanced_frames = enhancer.enhance_frames_batch(
                frames, frame_bboxes, batch_size=8
            )
        finally:
            enhancer.unload()

        # Write enhanced video
        logger.info("  Writing enhanced video...")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path + ".tmp.mp4", fourcc, fps, (width, height))

        for frame in enhanced_frames:
            out.write(frame)
        out.release()

        # Re-encode with ffmpeg to add audio and proper codec
        logger.info("  Re-encoding with audio...")
        encode_cmd = [
            "ffmpeg", "-y",
            "-i", output_path + ".tmp.mp4",
            "-i", input_path,
            "-map", "0:v", "-map", "1:a?",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-c:a", "aac", "-b:a", "128k",
            "-pix_fmt", "yuv420p",
            "-shortest",
            output_path
        ]
        subprocess.run(encode_cmd, capture_output=True, check=True)

        # Clean up temp file
        if os.path.exists(output_path + ".tmp.mp4"):
            os.remove(output_path + ".tmp.mp4")


def download_models(model_dir: str = "/models"):
    """
    Download required model weights.

    Args:
        model_dir: Directory to save models
    """
    import urllib.request

    os.makedirs(model_dir, exist_ok=True)

    models = {
        "wav2lip_gan.pth": "https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/Eb3LEzbfuKlJiR600lQWRxgBIY27JZg80f7V9ber7mQwcQ?e=TBFBVW&download=1",
        "GFPGANv1.4.pth": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth",
    }

    for name, url in models.items():
        path = os.path.join(model_dir, name)
        if not os.path.exists(path):
            logger.info(f"Downloading {name}...")
            urllib.request.urlretrieve(url, path)
            logger.info(f"  Saved to {path}")
