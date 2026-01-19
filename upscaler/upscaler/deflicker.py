"""
Temporal Deflicker Post-Processing.

Uses All-In-One-Deflicker (CVPR 2023) to reduce frame-to-frame flickering
in AI-upscaled videos.

Repository: https://github.com/ChenyangLEI/All-In-One-Deflicker
Paper: "Blind Video Deflickering by Neural Filtering with a Flawed Atlas"
"""

import gc
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger("upscaler.deflicker")

# Path to cloned All-In-One-Deflicker repo
DEFLICKER_REPO_PATH = Path("/app/vendors/All-In-One-Deflicker")


class Deflicker:
    """
    Temporal deflicker post-processor.

    Reduces frame-to-frame inconsistencies from AI upscaling.
    """

    def __init__(self):
        self._available = self._check_availability()

    @property
    def is_available(self) -> bool:
        """Check if deflicker is available."""
        return self._available

    def _check_availability(self) -> bool:
        """Check if All-In-One-Deflicker is installed."""
        if not DEFLICKER_REPO_PATH.exists():
            logger.warning(f"Deflicker repo not found at {DEFLICKER_REPO_PATH}")
            return False

        test_script = DEFLICKER_REPO_PATH / "test.py"
        if not test_script.exists():
            logger.warning("Deflicker test.py not found")
            return False

        return True

    def process_video(
        self,
        input_video: str,
        output_video: str,
        original_video: Optional[str] = None,
    ) -> bool:
        """
        Apply temporal deflicker to a video.

        Args:
            input_video: Path to processed (upscaled) video
            output_video: Path to save deflickered video
            original_video: Optional path to original video (improves quality)

        Returns:
            True if successful, False otherwise
        """
        if not self.is_available:
            logger.error("Deflicker is not available")
            return False

        try:
            return self._run_deflicker(input_video, output_video, original_video)
        except Exception as e:
            logger.error(f"Deflicker failed: {e}")
            return False

    def _run_deflicker(
        self,
        input_video: str,
        output_video: str,
        original_video: Optional[str],
    ) -> bool:
        """Run the deflicker subprocess."""
        # Build command
        cmd = [
            sys.executable,
            str(DEFLICKER_REPO_PATH / "test.py"),
            "--video_name", input_video,
        ]

        # Add original video if provided (helps deflicker quality)
        if original_video and os.path.exists(original_video):
            cmd.extend(["--input_video", original_video])

        # Set environment
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0")

        logger.info(f"Running deflicker on {input_video}...")

        result = subprocess.run(
            cmd,
            cwd=str(DEFLICKER_REPO_PATH),
            capture_output=True,
            text=True,
            env=env,
            timeout=3600,  # 1 hour timeout
        )

        if result.returncode != 0:
            logger.error(f"Deflicker stderr: {result.stderr}")
            return False

        # Find output file (deflicker saves to results/ directory)
        video_name = Path(input_video).stem
        deflicker_output = DEFLICKER_REPO_PATH / "results" / video_name / "final" / "output.mp4"

        if not deflicker_output.exists():
            logger.error(f"Deflicker output not found: {deflicker_output}")
            return False

        # Move to requested output path
        import shutil
        shutil.move(str(deflicker_output), output_video)

        logger.info(f"Deflicker complete: {output_video}")
        return True


class SimpleDeflicker:
    """
    Simple frame-averaging deflicker (fallback when All-In-One-Deflicker unavailable).

    Uses temporal weighted average to reduce flickering.
    Less effective than neural method but works without dependencies.
    """

    def __init__(self, window_size: int = 3, weights: Optional[list] = None):
        """
        Args:
            window_size: Number of frames to average (must be odd)
            weights: Optional custom weights (default: gaussian-like)
        """
        if window_size % 2 == 0:
            window_size += 1

        self.window_size = window_size
        self.half_window = window_size // 2

        if weights is None:
            # Gaussian-like weights
            import numpy as np
            x = np.arange(window_size) - self.half_window
            self.weights = np.exp(-x ** 2 / (2 * (window_size / 4) ** 2))
            self.weights /= self.weights.sum()
        else:
            self.weights = weights

    def process_frames(self, input_dir: str, output_dir: str) -> int:
        """
        Apply temporal smoothing to frames.

        Args:
            input_dir: Directory with input frames
            output_dir: Directory for output frames

        Returns:
            Number of frames processed
        """
        import cv2
        import numpy as np
        from tqdm import tqdm

        os.makedirs(output_dir, exist_ok=True)

        frame_paths = sorted(Path(input_dir).glob("frame_*.png"))
        total_frames = len(frame_paths)

        if total_frames == 0:
            return 0

        # Load all frames into memory (for small videos)
        # For large videos, use a sliding window approach
        logger.info(f"Loading {total_frames} frames for temporal smoothing...")

        frames = []
        for path in frame_paths:
            frame = cv2.imread(str(path))
            if frame is not None:
                frames.append(frame.astype(np.float32))

        logger.info(f"Applying temporal smoothing (window={self.window_size})...")

        processed = 0
        for i in tqdm(range(total_frames), desc="Deflickering", unit="frame"):
            # Get window of frames
            start_idx = max(0, i - self.half_window)
            end_idx = min(total_frames, i + self.half_window + 1)

            # Build weighted average
            window_frames = frames[start_idx:end_idx]
            window_weights = self.weights[
                (start_idx - i + self.half_window):(end_idx - i + self.half_window)
            ]

            # Normalize weights for partial windows at edges
            window_weights = np.array(window_weights)
            window_weights /= window_weights.sum()

            # Weighted average
            result = np.zeros_like(frames[i])
            for j, (frame, weight) in enumerate(zip(window_frames, window_weights)):
                result += frame * weight

            result = np.clip(result, 0, 255).astype(np.uint8)

            output_path = Path(output_dir) / frame_paths[i].name
            cv2.imwrite(str(output_path), result)
            processed += 1

        return processed


def get_deflicker() -> Optional[Deflicker]:
    """Get deflicker instance if available."""
    deflicker = Deflicker()
    if deflicker.is_available:
        return deflicker
    return None


def get_simple_deflicker(window_size: int = 5) -> SimpleDeflicker:
    """Get simple fallback deflicker."""
    return SimpleDeflicker(window_size=window_size)


def clear_memory() -> None:
    """Clear GPU memory after deflicker."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
