"""
Smart Loop Handler with RIFE Interpolation.

This module provides seamless video looping by:
1. Finding the best loop point (frame most similar to frame 0)
2. Using RIFE to interpolate smooth transition frames
3. Assembling a seamless loop that can repeat infinitely
"""

import logging
import math
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

        # Add RIFE paths for imports - train_log first for v3.6 model files
        train_log_path = str(RIFE_MODEL_PATH)
        rife_path = str(RIFE_DIR)
        for path in [train_log_path, rife_path]:
            if path not in sys.path:
                sys.path.insert(0, path)

        try:
            # Use v3.6 model from train_log (matches flownet.pkl weights)
            from RIFE_HDv3 import Model

            self.model = Model()
            self.model.load_model(str(self.model_path), -1)
            self.model.eval()
            self.model.device()

            self._loaded = True
            logger.info(f"RIFE v3.6 loaded in {time.time() - start:.2f}s on {self.device}")

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
        Interpolate frames between two images using recursive binary subdivision.

        Args:
            frame_a: Start frame (BGR numpy array)
            frame_b: End frame (BGR numpy array)
            num_frames: Number of intermediate frames to generate

        Returns:
            List of interpolated frames (excluding frame_a and frame_b)
        """
        if not self._loaded:
            self.load()

        h, w = frame_a.shape[:2]

        def to_tensor(frame):
            """Convert BGR numpy to normalized tensor."""
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
            t = t.unsqueeze(0).to(self.device)
            # Pad to multiple of 32
            ph = ((h - 1) // 32 + 1) * 32
            pw = ((w - 1) // 32 + 1) * 32
            if ph != h or pw != w:
                t = torch.nn.functional.pad(t, (0, pw - w, 0, ph - h))
            return t

        def from_tensor(t):
            """Convert tensor back to BGR numpy."""
            t = t[0].cpu().numpy().transpose(1, 2, 0)
            t = (t * 255).clip(0, 255).astype(np.uint8)
            t = t[:h, :w]
            return cv2.cvtColor(t, cv2.COLOR_RGB2BGR)

        def get_midpoint(img0_t, img1_t):
            """Get midpoint frame using RIFE inference."""
            with torch.no_grad():
                return self.model.inference(img0_t, img1_t, scale=1.0)

        # Convert endpoints to tensors
        img0_t = to_tensor(frame_a)
        img1_t = to_tensor(frame_b)

        # Use binary subdivision to generate frames
        # For num_frames intermediate frames, we need log2(num_frames+1) levels
        # But for simplicity, just recursively subdivide until we have enough

        # Start with just the midpoint
        frames_dict = {}  # position (0-1) -> tensor

        def subdivide(left_pos, right_pos, left_t, right_t, depth):
            """Recursively subdivide to generate frames."""
            if len(frames_dict) >= num_frames:
                return
            mid_pos = (left_pos + right_pos) / 2
            mid_t = get_midpoint(left_t, right_t)
            frames_dict[mid_pos] = mid_t

            if depth > 0 and len(frames_dict) < num_frames:
                subdivide(left_pos, mid_pos, left_t, mid_t, depth - 1)
                subdivide(mid_pos, right_pos, mid_t, right_t, depth - 1)

        # Calculate depth needed
        depth = max(1, int(math.ceil(math.log2(num_frames + 1))))
        subdivide(0.0, 1.0, img0_t, img1_t, depth)

        # Sort by position and convert to numpy
        sorted_frames = sorted(frames_dict.items(), key=lambda x: x[0])
        interpolated = [from_tensor(t) for _, t in sorted_frames[:num_frames]]

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
        video_fps = cap.get(cv2.CAP_PROP_FPS)

        # Use video's actual fps if available, otherwise use provided fps
        if video_fps > 0:
            fps = video_fps

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        if not frames:
            raise ValueError("No frames read from video")

        logger.info(f"  Read {len(frames)} frames at {width}x{height} @ {fps:.1f}fps")

        # Validate and adjust bboxes to match frame count
        # This handles cases where original tracking has different frame count than processed video
        if len(bboxes) != len(frames):
            logger.warning(f"  Bbox count ({len(bboxes)}) != frame count ({len(frames)}), adjusting...")
            if len(bboxes) > len(frames):
                # Truncate bboxes to match frames
                bboxes = bboxes[:len(frames)]
            else:
                # Extend bboxes with None for extra frames
                bboxes = bboxes + [None] * (len(frames) - len(bboxes))

        # Also adjust landmarks if provided
        if landmarks is not None and len(landmarks) != len(frames):
            if len(landmarks) > len(frames):
                landmarks = landmarks[:len(frames)]
            else:
                landmarks = landmarks + [None] * (len(frames) - len(landmarks))

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
        # Ensure loop_point is within bounds
        if loop_point < 0:
            loop_point = 0
        if loop_point >= len(frames):
            loop_point = len(frames) - 1

        # Trim to loop point
        loop_frames = frames[:loop_point + 1]

        if len(loop_frames) < 2:
            logger.warning(f"  Too few frames for RIFE loop ({len(loop_frames)}), returning all frames")
            return frames

        # Interpolate from last frame back to first frame
        last_frame = loop_frames[-1]
        first_frame = loop_frames[0]

        logger.info(f"  Creating RIFE loop: {len(loop_frames)} frames + {self.config.transition_frames} transition frames")

        try:
            transition = self.interpolator.interpolate(
                last_frame,
                first_frame,
                num_frames=self.config.transition_frames,
            )
            logger.info(f"  RIFE generated {len(transition)} transition frames")
        except Exception as e:
            logger.warning(f"  RIFE interpolation failed: {e}, using crossfade fallback")
            return self._create_crossfade_loop(frames)

        # Combine: original frames + transition (excluding endpoints)
        result = loop_frames + transition

        logger.info(f"  Total loop frames: {len(result)}")
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
        loop_len = len(loop_frames)

        if loop_len == 0:
            logger.error("  No loop frames to extend!")
            return []

        if loop_len >= target_frames:
            logger.info(f"  Loop already meets target ({loop_len} >= {target_frames}), trimming")
            return loop_frames[:target_frames]

        num_loops = (target_frames + loop_len - 1) // loop_len
        logger.info(f"  Extending {loop_len} frames to {target_frames} frames ({num_loops} loops)")

        result = []
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
