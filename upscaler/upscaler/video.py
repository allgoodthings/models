"""
Video utilities for frame extraction and reconstruction.

Uses ffmpeg for video I/O operations.
"""

import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("upscaler.video")


@dataclass
class VideoInfo:
    """Video metadata."""

    width: int
    height: int
    fps: float
    frame_count: int
    duration_ms: int
    has_audio: bool

    @property
    def resolution(self) -> str:
        return f"{self.width}x{self.height}"


def get_video_info(video_path: str) -> VideoInfo:
    """Get video metadata using OpenCV and ffprobe."""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_ms = int((frame_count / fps) * 1000) if fps > 0 else 0

    cap.release()

    # Check for audio track with ffprobe
    has_audio = _check_audio_track(video_path)

    return VideoInfo(
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration_ms=duration_ms,
        has_audio=has_audio,
    )


def _check_audio_track(video_path: str) -> bool:
    """Check if video has an audio track."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=codec_type",
        "-of", "csv=p=0",
        video_path,
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return "audio" in result.stdout.lower()
    except Exception:
        return False


def extract_frames(video_path: str, output_dir: str) -> Tuple[int, float]:
    """
    Extract all frames from video as PNG files.

    Args:
        video_path: Path to input video
        output_dir: Directory to save frames (frame_00001.png, frame_00002.png, ...)

    Returns:
        Tuple of (frame_count, fps)
    """
    os.makedirs(output_dir, exist_ok=True)

    info = get_video_info(video_path)

    # Use ffmpeg for fast extraction
    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-start_number", "1",
        "-q:v", "1",  # High quality PNG
        os.path.join(output_dir, "frame_%05d.png"),
    ]

    logger.info(f"Extracting {info.frame_count} frames at {info.fps:.2f} fps...")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Frame extraction failed: {result.stderr}")

    # Count actual extracted frames
    frame_count = len(list(Path(output_dir).glob("frame_*.png")))
    logger.info(f"Extracted {frame_count} frames")

    return frame_count, info.fps


def extract_audio(video_path: str, output_path: str) -> bool:
    """
    Extract audio track from video.

    Args:
        video_path: Path to input video
        output_path: Path to save audio (typically .aac or .mp3)

    Returns:
        True if audio was extracted, False if no audio track
    """
    if not _check_audio_track(video_path):
        logger.info("No audio track found in video")
        return False

    cmd = [
        "ffmpeg",
        "-y",
        "-i", video_path,
        "-vn",  # No video
        "-acodec", "copy",  # Copy audio codec
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning(f"Audio extraction failed: {result.stderr}")
        return False

    logger.info(f"Extracted audio to {output_path}")
    return True


def frames_to_video(
    frames_dir: str,
    output_path: str,
    fps: float,
    audio_path: Optional[str] = None,
    crf: int = 18,
    preset: str = "medium",
) -> None:
    """
    Reconstruct video from frame images.

    Args:
        frames_dir: Directory containing frame_00001.png, frame_00002.png, ...
        output_path: Path for output video
        fps: Frame rate
        audio_path: Optional audio file to mux in
        crf: Quality (lower = better, 18-23 typical)
        preset: Encoding speed preset
    """
    # Build ffmpeg command
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate", str(fps),
        "-i", os.path.join(frames_dir, "frame_%05d.png"),
    ]

    # Add audio if provided
    if audio_path and os.path.exists(audio_path):
        cmd.extend(["-i", audio_path])
        cmd.extend(["-map", "0:v", "-map", "1:a"])
        cmd.extend(["-c:a", "aac", "-b:a", "192k"])
        cmd.extend(["-shortest"])  # Trim to shortest stream

    # Video encoding settings
    cmd.extend([
        "-c:v", "libx264",
        "-preset", preset,
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",  # Compatibility
        output_path,
    ])

    logger.info(f"Encoding video at {fps:.2f} fps, CRF {crf}...")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Video encoding failed: {result.stderr}")

    output_size = os.path.getsize(output_path) / (1024 * 1024)
    logger.info(f"Encoded video: {output_size:.1f}MB")


def iterate_frames(frames_dir: str) -> Iterator[Tuple[int, np.ndarray]]:
    """
    Iterate over frames in a directory.

    Yields:
        Tuple of (frame_number, frame_array) where frame_array is BGR
    """
    frame_paths = sorted(Path(frames_dir).glob("frame_*.png"))

    for i, frame_path in enumerate(frame_paths, start=1):
        frame = cv2.imread(str(frame_path))
        if frame is not None:
            yield i, frame


def save_frame(frame: np.ndarray, output_path: str) -> None:
    """Save a frame (BGR numpy array) as PNG."""
    cv2.imwrite(output_path, frame)


def calculate_output_resolution(
    input_width: int,
    input_height: int,
    target: str,
) -> Tuple[int, int]:
    """
    Calculate output resolution maintaining aspect ratio.

    Args:
        input_width: Input video width
        input_height: Input video height
        target: "1080p" or "4k"

    Returns:
        Tuple of (output_width, output_height)
    """
    target_heights = {
        "1080p": 1080,
        "4k": 2160,
    }

    target_height = target_heights.get(target, 1080)

    # Calculate scale factor based on height
    scale = target_height / input_height

    # Calculate width maintaining aspect ratio
    output_width = int(input_width * scale)
    output_height = target_height

    # Ensure even dimensions (required for video encoding)
    output_width = output_width + (output_width % 2)
    output_height = output_height + (output_height % 2)

    return output_width, output_height


def calculate_scale_factor(
    input_width: int,
    input_height: int,
    output_width: int,
    output_height: int,
) -> float:
    """Calculate the scale factor between input and output resolutions."""
    width_scale = output_width / input_width
    height_scale = output_height / input_height
    return max(width_scale, height_scale)
