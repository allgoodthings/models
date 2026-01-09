"""
Smart Loop Handler with RIFE Interpolation.

This module provides seamless video looping by:
1. Finding the best loop point (frame most similar to frame 0)
2. Using RIFE to interpolate smooth transition frames
3. Assembling a seamless loop that can repeat infinitely
"""

import logging
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

logger = logging.getLogger("lipsync.loop_handler")

# RIFE model path
RIFE_DIR = Path("/app/vendors/rife")
RIFE_MODEL_PATH = RIFE_DIR / "train_log"


@dataclass
class LoopConfig:
    """Configuration for loop handling."""

    # Smart loop detection
    min_loop_frames: int = 30  # Minimum frames before considering loop point (1s at 30fps)
    similarity_threshold: float = 0.85  # Minimum similarity to consider a match

    # RIFE interpolation
    transition_frames: int = 10  # Number of frames to interpolate at loop boundary
    rife_scale: float = 1.0  # RIFE scale factor (higher = better quality, slower)

    # Fallback
    use_crossfade_fallback: bool = True  # If no good loop point found, use crossfade


class RIFEInterpolator:
    """
    RIFE (Real-Time Intermediate Flow Estimation) wrapper for frame interpolation.
    """

    def __init__(self, model_path: Optional[str] = None, device: str = "cuda"):
        """
        Initialize RIFE interpolator.

        Args:
            model_path: Path to RIFE model weights
            device: Device to run on ("cuda" or "cpu")
        """
        self.model_path = Path(model_path) if model_path else RIFE_MODEL_PATH
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = None
        self._loaded = False

    def load(self):
        """Load RIFE model."""
        if self._loaded:
            return

        logger.info("Loading RIFE model...")
        start = time.time()

        # Add RIFE to path for imports
        rife_path = str(RIFE_DIR)
        if rife_path not in sys.path:
            sys.path.insert(0, rife_path)

        try:
            from model.RIFE import Model

            self.model = Model()
            self.model.load_model(str(self.model_path), -1)
            self.model.eval()
            self.model.device()

            self._loaded = True
            logger.info(f"RIFE loaded in {time.time() - start:.2f}s on {self.device}")

        except Exception as e:
            logger.error(f"Failed to load RIFE: {e}")
            raise

    def interpolate(
        self,
        frame_a: np.ndarray,
        frame_b: np.ndarray,
        num_frames: int = 10,
    ) -> List[np.ndarray]:
        """
        Interpolate frames between two images.

        Args:
            frame_a: Start frame (BGR numpy array)
            frame_b: End frame (BGR numpy array)
            num_frames: Number of intermediate frames to generate

        Returns:
            List of interpolated frames (excluding frame_a and frame_b)
        """
        if not self._loaded:
            self.load()

        # Convert BGR to RGB and normalize
        img0 = cv2.cvtColor(frame_a, cv2.COLOR_BGR2RGB)
        img1 = cv2.cvtColor(frame_b, cv2.COLOR_BGR2RGB)

        # Ensure dimensions are divisible by 32
        h, w = img0.shape[:2]
        ph = ((h - 1) // 32 + 1) * 32
        pw = ((w - 1) // 32 + 1) * 32

        padding = (0, pw - w, 0, ph - h)

        # Convert to tensors
        img0_t = torch.from_numpy(img0.transpose(2, 0, 1)).float() / 255.0
        img1_t = torch.from_numpy(img1.transpose(2, 0, 1)).float() / 255.0

        img0_t = img0_t.unsqueeze(0).to(self.device)
        img1_t = img1_t.unsqueeze(0).to(self.device)

        # Pad if needed
        if padding != (0, 0, 0, 0):
            img0_t = torch.nn.functional.pad(img0_t, padding)
            img1_t = torch.nn.functional.pad(img1_t, padding)

        interpolated = []

        with torch.no_grad():
            for i in range(1, num_frames + 1):
                # Calculate timestep (0 to 1)
                t = i / (num_frames + 1)

                # RIFE inference
                mid = self.model.inference(img0_t, img1_t, timestep=t)

                # Convert back to numpy
                mid = mid[0].cpu().numpy().transpose(1, 2, 0)
                mid = (mid * 255).clip(0, 255).astype(np.uint8)

                # Remove padding
                mid = mid[:h, :w]

                # Convert RGB back to BGR
                mid = cv2.cvtColor(mid, cv2.COLOR_RGB2BGR)
                interpolated.append(mid)

        return interpolated

    def unload(self):
        """Unload model from GPU."""
        if self.model is not None:
            del self.model
            self.model = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._loaded = False
        logger.info("RIFE model unloaded")


class SmartLoopDetector:
    """
    Detects optimal loop points in video using face landmark similarity.
    """

    def __init__(self, config: Optional[LoopConfig] = None):
        """
        Initialize loop detector.

        Args:
            config: Loop configuration
        """
        self.config = config or LoopConfig()

    def compute_landmark_similarity(
        self,
        landmarks_a: np.ndarray,
        landmarks_b: np.ndarray,
    ) -> float:
        """
        Compute similarity between two sets of face landmarks.

        Args:
            landmarks_a: First landmark array (Nx2 or Nx3)
            landmarks_b: Second landmark array (Nx2 or Nx3)

        Returns:
            Similarity score (0-1, higher is more similar)
        """
        if landmarks_a is None or landmarks_b is None:
            return 0.0

        # Normalize landmarks to [0, 1] range
        a = landmarks_a[:, :2].copy()
        b = landmarks_b[:, :2].copy()

        # Center and scale normalize
        a_center = a.mean(axis=0)
        b_center = b.mean(axis=0)

        a_centered = a - a_center
        b_centered = b - b_center

        a_scale = np.sqrt((a_centered ** 2).sum(axis=1).mean())
        b_scale = np.sqrt((b_centered ** 2).sum(axis=1).mean())

        if a_scale < 1e-6 or b_scale < 1e-6:
            return 0.0

        a_norm = a_centered / a_scale
        b_norm = b_centered / b_scale

        # Compute distance (lower is more similar)
        dist = np.sqrt(((a_norm - b_norm) ** 2).sum(axis=1).mean())

        # Convert to similarity (0-1)
        similarity = np.exp(-dist * 2)  # Exponential decay

        return float(similarity)

    def compute_bbox_similarity(
        self,
        bbox_a: List[int],
        bbox_b: List[int],
        frame_size: Tuple[int, int],
    ) -> float:
        """
        Compute similarity between two bounding boxes.

        Args:
            bbox_a: First bbox [y1, y2, x1, x2]
            bbox_b: Second bbox [y1, y2, x1, x2]
            frame_size: (height, width) of frame

        Returns:
            Similarity score (0-1, higher is more similar)
        """
        if bbox_a is None or bbox_b is None:
            return 0.0

        h, w = frame_size

        # Normalize to [0, 1]
        a = np.array(bbox_a, dtype=float)
        b = np.array(bbox_b, dtype=float)

        a[[0, 1]] /= h  # y coordinates
        a[[2, 3]] /= w  # x coordinates
        b[[0, 1]] /= h
        b[[2, 3]] /= w

        # Compute center and size
        a_center = np.array([(a[0] + a[1]) / 2, (a[2] + a[3]) / 2])
        b_center = np.array([(b[0] + b[1]) / 2, (b[2] + b[3]) / 2])

        a_size = np.array([a[1] - a[0], a[3] - a[2]])
        b_size = np.array([b[1] - b[0], b[3] - b[2]])

        # Distance-based similarity
        center_dist = np.sqrt(((a_center - b_center) ** 2).sum())
        size_dist = np.sqrt(((a_size - b_size) ** 2).sum())

        # Weighted combination
        dist = center_dist * 0.7 + size_dist * 0.3
        similarity = np.exp(-dist * 5)

        return float(similarity)

    def find_best_loop_point(
        self,
        bboxes: List[Optional[List[int]]],
        frame_size: Tuple[int, int],
        landmarks: Optional[List[Optional[np.ndarray]]] = None,
    ) -> Tuple[int, float]:
        """
        Find the best frame to loop back to frame 0.

        Args:
            bboxes: List of bounding boxes per frame
            frame_size: (height, width) of frames
            landmarks: Optional list of landmarks per frame

        Returns:
            Tuple of (best_frame_index, similarity_score)
        """
        if len(bboxes) < self.config.min_loop_frames:
            logger.warning(f"Video too short for smart loop ({len(bboxes)} frames)")
            return len(bboxes) - 1, 0.0

        first_bbox = bboxes[0]
        first_landmarks = landmarks[0] if landmarks else None

        if first_bbox is None:
            logger.warning("No face detected in first frame, falling back to last frame")
            return len(bboxes) - 1, 0.0

        best_frame = len(bboxes) - 1
        best_similarity = 0.0

        # Search from min_loop_frames to end
        for i in range(self.config.min_loop_frames, len(bboxes)):
            bbox = bboxes[i]
            if bbox is None:
                continue

            # Compute bbox similarity
            bbox_sim = self.compute_bbox_similarity(first_bbox, bbox, frame_size)

            # Compute landmark similarity if available
            if landmarks and landmarks[i] is not None and first_landmarks is not None:
                landmark_sim = self.compute_landmark_similarity(first_landmarks, landmarks[i])
                # Weighted combination
                similarity = bbox_sim * 0.4 + landmark_sim * 0.6
            else:
                similarity = bbox_sim

            if similarity > best_similarity:
                best_similarity = similarity
                best_frame = i

        logger.info(f"Best loop point: frame {best_frame} (similarity: {best_similarity:.3f})")

        return best_frame, best_similarity


class LoopHandler:
    """
    Handles video looping with smart detection and RIFE interpolation.
    """

    def __init__(self, config: Optional[LoopConfig] = None):
        """
        Initialize loop handler.

        Args:
            config: Loop configuration
        """
        self.config = config or LoopConfig()
        self.detector = SmartLoopDetector(config)
        self.interpolator = RIFEInterpolator()

    def create_seamless_loop(
        self,
        video_path: str,
        output_path: str,
        target_duration: float,
        bboxes: List[Optional[List[int]]],
        fps: float = 25.0,
        landmarks: Optional[List[Optional[np.ndarray]]] = None,
    ) -> str:
        """
        Create a seamlessly looping video of target duration.

        Args:
            video_path: Input video path
            output_path: Output video path
            target_duration: Target duration in seconds
            bboxes: Per-frame bounding boxes
            fps: Video frame rate
            landmarks: Optional per-frame landmarks

        Returns:
            Path to output video
        """
        logger.info("=" * 50)
        logger.info("LOOP HANDLER - Creating seamless loop")
        logger.info("=" * 50)
        logger.info(f"  Input: {video_path}")
        logger.info(f"  Target duration: {target_duration:.2f}s")

        start_time = time.time()

        # Read video frames
        cap = cv2.VideoCapture(video_path)
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
            raise ValueError("No frames read from video")

        logger.info(f"  Read {len(frames)} frames at {width}x{height}")

        # Find best loop point
        best_frame, similarity = self.detector.find_best_loop_point(
            bboxes, (height, width), landmarks
        )

        # Decide strategy based on similarity
        if similarity >= self.config.similarity_threshold:
            logger.info(f"  Good loop point found (sim={similarity:.3f}), using RIFE")
            loop_frames = self._create_rife_loop(frames, best_frame)
        elif self.config.use_crossfade_fallback:
            logger.info(f"  No good loop point (sim={similarity:.3f}), using crossfade fallback")
            loop_frames = self._create_crossfade_loop(frames)
        else:
            logger.info(f"  No good loop point, using hard cut")
            loop_frames = frames

        # Extend to target duration
        target_frames = int(target_duration * fps)
        final_frames = self._extend_to_duration(loop_frames, target_frames)

        logger.info(f"  Extended to {len(final_frames)} frames ({len(final_frames)/fps:.2f}s)")

        # Write output
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for frame in final_frames:
            out.write(frame)
        out.release()

        elapsed = time.time() - start_time
        logger.info(f"  Complete in {elapsed:.2f}s")
        logger.info("LOOP HANDLER - Done")

        return output_path

    def _create_rife_loop(
        self,
        frames: List[np.ndarray],
        loop_point: int,
    ) -> List[np.ndarray]:
        """
        Create a seamless loop using RIFE interpolation.

        Args:
            frames: List of video frames
            loop_point: Frame index to loop back from

        Returns:
            List of frames forming a seamless loop
        """
        # Trim to loop point
        loop_frames = frames[:loop_point + 1]

        # Interpolate from last frame back to first frame
        last_frame = loop_frames[-1]
        first_frame = loop_frames[0]

        logger.info(f"  Interpolating {self.config.transition_frames} transition frames...")
        transition = self.interpolator.interpolate(
            last_frame,
            first_frame,
            num_frames=self.config.transition_frames,
        )

        # Combine: original frames + transition (excluding endpoints)
        result = loop_frames + transition

        return result

    def _create_crossfade_loop(
        self,
        frames: List[np.ndarray],
        crossfade_frames: int = 10,
    ) -> List[np.ndarray]:
        """
        Create a loop with crossfade blending (fallback).

        Args:
            frames: List of video frames
            crossfade_frames: Number of frames to crossfade

        Returns:
            List of frames with crossfade at boundaries
        """
        if len(frames) <= crossfade_frames * 2:
            return frames

        result = frames.copy()

        # Blend last N frames with first N frames
        for i in range(crossfade_frames):
            alpha = i / crossfade_frames
            idx = len(result) - crossfade_frames + i

            blended = cv2.addWeighted(
                result[idx].astype(float),
                1 - alpha,
                result[i].astype(float),
                alpha,
                0,
            ).astype(np.uint8)

            result[idx] = blended

        return result

    def _extend_to_duration(
        self,
        loop_frames: List[np.ndarray],
        target_frames: int,
    ) -> List[np.ndarray]:
        """
        Extend loop to target frame count.

        Args:
            loop_frames: Seamless loop frames
            target_frames: Target total frames

        Returns:
            Extended frame list
        """
        if len(loop_frames) >= target_frames:
            return loop_frames[:target_frames]

        result = []
        loop_len = len(loop_frames)

        for i in range(target_frames):
            result.append(loop_frames[i % loop_len])

        return result

    def unload(self):
        """Unload models."""
        self.interpolator.unload()


# Convenience function
def create_seamless_loop(
    video_path: str,
    output_path: str,
    target_duration: float,
    bboxes: List[Optional[List[int]]],
    fps: float = 25.0,
    config: Optional[LoopConfig] = None,
) -> str:
    """
    Create a seamlessly looping video.

    Args:
        video_path: Input video path
        output_path: Output video path
        target_duration: Target duration in seconds
        bboxes: Per-frame bounding boxes from face tracking
        fps: Video frame rate
        config: Loop configuration

    Returns:
        Path to output video
    """
    handler = LoopHandler(config)
    try:
        return handler.create_seamless_loop(
            video_path, output_path, target_duration, bboxes, fps
        )
    finally:
        handler.unload()
