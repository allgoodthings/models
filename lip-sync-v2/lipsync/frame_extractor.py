"""
Frame extraction using ffmpeg.

Provides batch frame extraction with sampling support,
much faster than sequential cv2.VideoCapture.read() calls.
"""

import logging
import subprocess
from dataclasses import dataclass
from typing import Generator, List, Tuple

import numpy as np

logger = logging.getLogger("lipsync.frame_extractor")

# Maximum frames to hold in memory at once
MAX_FRAMES_IN_MEMORY = 500


@dataclass
class VideoInfo:
    """Video metadata."""

    width: int
    height: int
    fps: float
    total_frames: int
    duration_ms: int


def get_video_info(video_path: str) -> VideoInfo:
    """
    Get video metadata using ffprobe.

    Args:
        video_path: Path to video file

    Returns:
        VideoInfo with dimensions, fps, frame count, duration
    """
    cmd = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,r_frame_rate,nb_frames,duration",
        "-of", "csv=p=0",
        video_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    output = result.stdout.strip()

    # Handle multiple lines (some formats output extra info)
    lines = output.split("\n")
    parts = lines[0].split(",")

    # Parse width and height
    width = int(parts[0])
    height = int(parts[1])

    # Parse frame rate (e.g., "30/1" or "30000/1001")
    fps_str = parts[2]
    if "/" in fps_str:
        fps_parts = fps_str.split("/")
        fps = float(fps_parts[0]) / float(fps_parts[1])
    else:
        fps = float(fps_str)

    # Parse frame count - try nb_frames first
    total_frames = 0
    if len(parts) > 3 and parts[3] and parts[3] not in ("N/A", ""):
        try:
            total_frames = int(parts[3])
        except ValueError:
            pass

    # Fallback: calculate from duration
    if total_frames == 0 and len(parts) > 4 and parts[4] and parts[4] not in ("N/A", ""):
        try:
            duration = float(parts[4])
            total_frames = int(duration * fps)
        except ValueError:
            pass

    # Last resort: use cv2 to count
    if total_frames == 0:
        import cv2
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

    duration_ms = int((total_frames / fps) * 1000) if fps > 0 else 0

    return VideoInfo(
        width=width,
        height=height,
        fps=fps,
        total_frames=total_frames,
        duration_ms=duration_ms,
    )


def extract_sampled_frames(
    video_path: str,
    sample_interval: int = 5,
) -> Tuple[List[np.ndarray], List[int], VideoInfo]:
    """
    Extract sampled frames using ffmpeg.

    Uses ffmpeg to extract every Nth frame, streaming data to avoid
    loading everything into memory at once.

    Args:
        video_path: Path to video file
        sample_interval: Extract every Nth frame (default: 5)

    Returns:
        Tuple of:
        - frames: List of BGR numpy arrays (H, W, 3)
        - frame_indices: Original frame indices [0, 5, 10, ...]
        - video_info: Video metadata
    """
    logger.info(f"Extracting frames from {video_path}")
    logger.info(f"  Sample interval: every {sample_interval} frames")

    # Get video info first
    info = get_video_info(video_path)
    logger.info(f"  Video: {info.width}x{info.height} @ {info.fps:.2f}fps, {info.total_frames} frames")

    # Calculate expected sampled frames
    expected_frames = (info.total_frames + sample_interval - 1) // sample_interval
    logger.info(f"  Expected sampled frames: {expected_frames}")

    # Warn if large video
    frame_size_mb = (info.width * info.height * 3) / (1024 * 1024)
    estimated_memory_mb = expected_frames * frame_size_mb
    if estimated_memory_mb > 2000:  # > 2GB
        logger.warning(
            f"  Large video: estimated {estimated_memory_mb:.0f}MB for {expected_frames} frames. "
            "Consider reducing sample_interval or video resolution."
        )

    # Build ffmpeg command with pipe output
    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vf", f"select=not(mod(n\\,{sample_interval}))",
        "-vsync", "vfr",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-loglevel", "error",
        "pipe:1",
    ]

    # Stream frames from ffmpeg
    frame_size = info.width * info.height * 3
    frames = []
    frame_indices = []

    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    frame_num = 0
    while True:
        # Read one frame at a time
        raw_data = process.stdout.read(frame_size)
        if len(raw_data) < frame_size:
            break

        frame = np.frombuffer(raw_data, dtype=np.uint8).reshape(
            (info.height, info.width, 3)
        ).copy()  # Copy to avoid buffer reuse issues

        frames.append(frame)
        frame_indices.append(frame_num * sample_interval)
        frame_num += 1

    process.wait()

    if process.returncode != 0:
        stderr = process.stderr.read().decode()
        logger.error(f"ffmpeg error: {stderr}")
        raise RuntimeError(f"ffmpeg failed: {stderr}")

    logger.info(f"  Extracted {len(frames)} frames")

    return frames, frame_indices, info


def iter_sampled_frames(
    video_path: str,
    sample_interval: int = 5,
) -> Generator[Tuple[np.ndarray, int], None, VideoInfo]:
    """
    Iterator for sampled frames - memory efficient for large videos.

    Yields frames one at a time instead of loading all into memory.

    Args:
        video_path: Path to video file
        sample_interval: Extract every Nth frame (default: 5)

    Yields:
        Tuple of (frame, frame_index) for each sampled frame

    Returns:
        VideoInfo after iteration completes (accessible via generator.value)
    """
    logger.info(f"Streaming frames from {video_path}")
    logger.info(f"  Sample interval: every {sample_interval} frames")

    info = get_video_info(video_path)
    logger.info(f"  Video: {info.width}x{info.height} @ {info.fps:.2f}fps")

    cmd = [
        "ffmpeg",
        "-i", video_path,
        "-vf", f"select=not(mod(n\\,{sample_interval}))",
        "-vsync", "vfr",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-loglevel", "error",
        "pipe:1",
    ]

    frame_size = info.width * info.height * 3
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    frame_num = 0
    while True:
        raw_data = process.stdout.read(frame_size)
        if len(raw_data) < frame_size:
            break

        frame = np.frombuffer(raw_data, dtype=np.uint8).reshape(
            (info.height, info.width, 3)
        ).copy()

        yield frame, frame_num * sample_interval
        frame_num += 1

    process.wait()
    return info


def extract_all_frames(video_path: str) -> Tuple[List[np.ndarray], VideoInfo]:
    """
    Extract all frames from video using ffmpeg.

    For cases where every frame is needed (e.g., visualization output).
    WARNING: Can use significant memory for long videos.

    Args:
        video_path: Path to video file

    Returns:
        Tuple of:
        - frames: List of BGR numpy arrays
        - video_info: Video metadata
    """
    frames, _, info = extract_sampled_frames(video_path, sample_interval=1)
    return frames, info
