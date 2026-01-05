"""
Utility functions for video/audio processing.

Includes FFmpeg wrappers, frame extraction, and timestamp handling.
"""

import logging
import os
import re
import subprocess
import tempfile
import time
import uuid
from pathlib import Path
from typing import List, Optional, Tuple, Generator
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger('lipsync.utils')


@dataclass
class VideoInfo:
    """Video metadata."""
    width: int
    height: int
    fps: float
    duration: float
    frame_count: int


def get_video_info(video_path: str) -> VideoInfo:
    """
    Get video metadata using FFprobe.

    Args:
        video_path: Path to video file

    Returns:
        VideoInfo with metadata
    """
    logger.debug(f"Getting video info for: {video_path}")
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    cap.release()

    logger.debug(f"  -> {width}x{height} @ {fps:.2f}fps, {frame_count} frames, {duration:.2f}s")

    return VideoInfo(
        width=width,
        height=height,
        fps=fps,
        duration=duration,
        frame_count=frame_count,
    )


def get_audio_duration(audio_path: str) -> float:
    """
    Get audio duration using FFprobe.

    Args:
        audio_path: Path to audio file

    Returns:
        Duration in seconds
    """
    cmd = [
        'ffprobe', '-i', audio_path,
        '-show_entries', 'format=duration',
        '-v', 'quiet', '-of', 'csv=p=0'
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())


def extract_frames(
    video_path: str,
    fps: Optional[float] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
) -> Generator[Tuple[int, np.ndarray], None, None]:
    """
    Extract frames from video.

    Args:
        video_path: Path to video
        fps: Target FPS (None = original)
        start_time: Start time in seconds
        end_time: End time in seconds

    Yields:
        Tuples of (frame_number, frame_bgr)
    """
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if start_time is not None:
        start_frame = int(start_time * video_fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    else:
        start_frame = 0

    if end_time is not None:
        end_frame = int(end_time * video_fps)
    else:
        end_frame = total_frames

    # Calculate frame skip for target FPS
    if fps is not None and fps < video_fps:
        skip = int(video_fps / fps)
    else:
        skip = 1

    frame_num = start_frame
    while frame_num < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if (frame_num - start_frame) % skip == 0:
            yield frame_num, frame

        frame_num += 1

    cap.release()


def sample_frames_at_fps(
    video_path: str,
    target_fps: float = 5.0,
) -> List[Tuple[float, np.ndarray]]:
    """
    Sample frames at specified FPS.

    Args:
        video_path: Path to video
        target_fps: Frames per second to sample

    Returns:
        List of (timestamp_seconds, frame_bgr)
    """
    info = get_video_info(video_path)
    interval = 1.0 / target_fps
    frames = []

    cap = cv2.VideoCapture(video_path)

    for t in np.arange(0, info.duration, interval):
        frame_num = int(t * info.fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if ret:
            frames.append((t, frame))

    cap.release()
    return frames


def extract_audio(video_path: str, output_path: str) -> str:
    """
    Extract audio from video.

    Args:
        video_path: Path to video
        output_path: Path for output audio

    Returns:
        Path to extracted audio
    """
    cmd = [
        'ffmpeg', '-y', '-i', video_path,
        '-vn', '-acodec', 'pcm_s16le',
        '-ar', '16000', '-ac', '1',
        output_path
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return output_path


def combine_audio_video(
    video_path: str,
    audio_path: str,
    output_path: str,
) -> str:
    """
    Combine video with audio track.

    Args:
        video_path: Path to video
        audio_path: Path to audio
        output_path: Path for output

    Returns:
        Path to combined video
    """
    logger.info(f"Combining audio and video:")
    logger.info(f"  Video: {video_path}")
    logger.info(f"  Audio: {audio_path}")
    logger.info(f"  Output: {output_path}")

    start_time = time.time()

    cmd = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-i', audio_path,
        '-c:v', 'libx264', '-preset', 'ultrafast', '-crf', '23',
        '-c:a', 'aac', '-b:a', '128k',
        '-map', '0:v:0', '-map', '1:a:0',
        '-shortest',
        output_path
    ]
    logger.debug(f"  Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=True, check=True)

    elapsed = time.time() - start_time
    logger.info(f"  Completed in {elapsed:.2f}s")

    return output_path


def concat_videos(
    video_paths: List[str],
    output_path: str,
    method: str = 'demuxer',
) -> str:
    """
    Concatenate multiple videos.

    Args:
        video_paths: List of video paths
        output_path: Path for output
        method: 'demuxer' (fast, same codec) or 'filter' (re-encode)

    Returns:
        Path to concatenated video
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        if method == 'demuxer':
            # Fast concat for same-codec videos
            list_file = os.path.join(tmpdir, 'concat.txt')
            with open(list_file, 'w') as f:
                for path in video_paths:
                    f.write(f"file '{path}'\n")

            cmd = [
                'ffmpeg', '-y',
                '-f', 'concat', '-safe', '0',
                '-i', list_file,
                '-c', 'copy',
                output_path
            ]
        else:
            # Re-encode concat for different codecs
            inputs = []
            for path in video_paths:
                inputs.extend(['-i', path])

            filter_parts = ''.join(f'[{i}:v][{i}:a]' for i in range(len(video_paths)))
            filter_complex = f'{filter_parts}concat=n={len(video_paths)}:v=1:a=1[outv][outa]'

            cmd = [
                'ffmpeg', '-y',
                *inputs,
                '-filter_complex', filter_complex,
                '-map', '[outv]', '-map', '[outa]',
                '-c:v', 'libx264', '-preset', 'fast',
                '-c:a', 'aac',
                output_path
            ]

        subprocess.run(cmd, capture_output=True, check=True)
        return output_path


def trim_video(
    video_path: str,
    output_path: str,
    start_time: float,
    end_time: float,
) -> str:
    """
    Trim video to time range.

    Args:
        video_path: Path to video
        output_path: Path for output
        start_time: Start time in seconds
        end_time: End time in seconds

    Returns:
        Path to trimmed video
    """
    duration = end_time - start_time
    cmd = [
        'ffmpeg', '-y',
        '-ss', str(start_time),
        '-i', video_path,
        '-t', str(duration),
        '-c', 'copy',
        output_path
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return output_path


def resize_video(
    video_path: str,
    output_path: str,
    width: int,
    height: int,
) -> str:
    """
    Resize video to specified dimensions.

    Args:
        video_path: Path to video
        output_path: Path for output
        width: Target width
        height: Target height

    Returns:
        Path to resized video
    """
    cmd = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-vf', f'scale={width}:{height}',
        '-c:v', 'libx264', '-preset', 'fast',
        '-c:a', 'copy',
        output_path
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return output_path


def frames_to_video(
    frames: List[np.ndarray],
    output_path: str,
    fps: float = 25.0,
) -> str:
    """
    Create video from frame list.

    Args:
        frames: List of BGR frames
        output_path: Path for output
        fps: Output FPS

    Returns:
        Path to output video
    """
    if not frames:
        raise ValueError("No frames provided")

    height, width = frames[0].shape[:2]
    logger.info(f"Creating video from {len(frames)} frames:")
    logger.info(f"  Size: {width}x{height}")
    logger.info(f"  FPS: {fps}")
    logger.info(f"  Output: {output_path}")

    start_time = time.time()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)

    out.release()

    elapsed = time.time() - start_time
    logger.info(f"  Completed in {elapsed:.2f}s ({len(frames)/elapsed:.1f} fps)")

    return output_path


def generate_timestamp_chunks(
    duration: float,
    chunk_duration: float = 5.0,
) -> List[Tuple[float, float]]:
    """
    Generate timestamp chunks for parallel processing.

    Args:
        duration: Total duration in seconds
        chunk_duration: Duration of each chunk

    Returns:
        List of (start_time, end_time) tuples
    """
    chunks = []
    start = 0.0

    while start < duration:
        end = min(start + chunk_duration, duration)
        chunks.append((start, end))
        start = end

    return chunks


def interpolate_bboxes(
    bboxes: List[Tuple[float, Tuple[int, int, int, int]]],
    target_timestamps: List[float],
    method: str = 'cosine',
) -> List[Tuple[int, int, int, int]]:
    """
    Interpolate bounding boxes between sampled frames.

    Args:
        bboxes: List of (timestamp, bbox) tuples
        target_timestamps: Timestamps to interpolate to
        method: 'linear' or 'cosine'

    Returns:
        List of interpolated bboxes
    """
    if not bboxes:
        raise ValueError("No bboxes provided")

    if len(bboxes) == 1:
        return [bboxes[0][1]] * len(target_timestamps)

    # Sort by timestamp
    bboxes = sorted(bboxes, key=lambda x: x[0])
    timestamps = np.array([b[0] for b in bboxes])
    boxes = np.array([b[1] for b in bboxes])

    result = []
    for t in target_timestamps:
        # Find surrounding keyframes
        idx = np.searchsorted(timestamps, t)

        if idx == 0:
            result.append(tuple(boxes[0]))
        elif idx >= len(timestamps):
            result.append(tuple(boxes[-1]))
        else:
            t0, t1 = timestamps[idx - 1], timestamps[idx]
            b0, b1 = boxes[idx - 1], boxes[idx]

            # Interpolation factor
            alpha = (t - t0) / (t1 - t0) if t1 != t0 else 0

            if method == 'cosine':
                alpha = (1 - np.cos(alpha * np.pi)) / 2

            bbox = b0 * (1 - alpha) + b1 * alpha
            result.append(tuple(bbox.astype(int)))

    return result


def detect_scene_cuts(
    video_path: str,
    threshold: float = 0.3,
) -> List[float]:
    """
    Detect scene cuts in video.

    Args:
        video_path: Path to video
        threshold: Detection threshold (0-1)

    Returns:
        List of scene cut timestamps
    """
    cmd = [
        'ffprobe', '-v', 'error',
        '-show_entries', 'frame=pkt_pts_time',
        '-select_streams', 'v',
        '-of', 'csv=p=0',
        '-f', 'lavfi',
        f"movie={video_path},select='gt(scene,{threshold})'"
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        timestamps = [float(t.strip()) for t in result.stdout.strip().split('\n') if t.strip()]
        return timestamps
    except:
        return []
