"""
Face tracking with sampling, interpolation, and smoothing.

Provides efficient face tracking by:
1. Sampling frames at intervals (not every frame)
2. Detecting faces only on sampled frames
3. Interpolating bboxes for frames between samples
4. Applying EMA smoothing for temporal stability
"""

import logging
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2

from .frame_extractor import extract_sampled_frames
from .server.insightface_detector import InsightFaceDetector

logger = logging.getLogger("lipsync.tracker")


@dataclass
class CharacterTrack:
    """Tracking data for a single character."""

    character_id: str
    name: str
    reference_image_url: str
    color: Tuple[int, int, int]  # BGR color for visualization


@dataclass
class TrackingResult:
    """Result of face tracking across all frames."""

    # {character_id: [bbox_per_frame]} where bbox is [y1, y2, x1, x2] or None
    tracks: Dict[str, List[Optional[List[int]]]]
    frame_count: int
    fps: float
    width: int
    height: int
    characters: List[CharacterTrack]


class FaceTracker:
    """
    Face tracker with sampling, interpolation, and EMA smoothing.

    Uses InsightFace for detection and identity matching.
    Samples frames at intervals, interpolates between samples,
    and applies EMA smoothing for stable bounding boxes.
    """

    # Distinct colors for visualization (BGR format)
    COLORS = [
        (0, 0, 255),    # Red
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 255, 255),  # Yellow
        (255, 0, 255),  # Magenta
        (255, 255, 0),  # Cyan
        (128, 0, 255),  # Orange
        (255, 128, 0),  # Light Blue
    ]

    def __init__(
        self,
        detector: InsightFaceDetector,
        sample_interval: int = 5,
        smoothing_factor: float = 0.3,
        similarity_threshold: float = 0.5,
    ):
        """
        Initialize face tracker.

        Args:
            detector: InsightFace detector instance
            sample_interval: Detect every Nth frame (default: 5)
            smoothing_factor: EMA alpha (0 = no smoothing, 1 = no history)
            similarity_threshold: Cosine similarity threshold for matching
        """
        self.detector = detector
        self.sample_interval = sample_interval
        self.alpha = smoothing_factor
        self.similarity_threshold = similarity_threshold

    def track_video(
        self,
        video_path: str,
        characters: List[Dict],
    ) -> TrackingResult:
        """
        Track all characters across video with sampling + interpolation.

        Args:
            video_path: Path to video file
            characters: List of character dicts with id, name, reference_image_url

        Returns:
            TrackingResult with per-frame bboxes for each character
        """
        logger.info(f"Starting face tracking for {video_path}")
        logger.info(f"  Characters: {[c['id'] for c in characters]}")
        logger.info(f"  Sample interval: {self.sample_interval}")
        logger.info(f"  Smoothing factor: {self.alpha}")

        # Extract sampled frames
        frames, frame_indices, info = extract_sampled_frames(
            video_path, self.sample_interval
        )

        char_ids = [c["id"] for c in characters]

        # Build character tracks with colors
        char_tracks = [
            CharacterTrack(
                character_id=c["id"],
                name=c.get("name", c["id"]),
                reference_image_url=c.get("reference_image_url", ""),
                color=self.COLORS[i % len(self.COLORS)],
            )
            for i, c in enumerate(characters)
        ]

        # Detect faces on sampled frames
        sampled_detections: Dict[int, Dict[str, List[int]]] = {}
        detected_count = {cid: 0 for cid in char_ids}

        for frame, frame_idx in zip(frames, frame_indices):
            faces = self.detector.detect_faces(frame, self.similarity_threshold)

            for face in faces:
                if face.character_id in char_ids:
                    # Convert bbox from (x, y, w, h) to Wav2Lip format [y1, y2, x1, x2]
                    x, y, w, h = face.bbox
                    wav2lip_bbox = [y, y + h, x, x + w]

                    sampled_detections.setdefault(frame_idx, {})[face.character_id] = wav2lip_bbox
                    detected_count[face.character_id] += 1

        # Log detection stats
        for char_id, count in detected_count.items():
            pct = (count / len(frames)) * 100 if frames else 0
            logger.info(f"  {char_id}: detected in {count}/{len(frames)} sampled frames ({pct:.1f}%)")

        # Interpolate to all frames
        tracks = self._interpolate_tracks(
            sampled_detections, info.total_frames, char_ids
        )

        # Apply EMA smoothing
        tracks = self._smooth_tracks(tracks, char_ids)

        return TrackingResult(
            tracks=tracks,
            frame_count=info.total_frames,
            fps=info.fps,
            width=info.width,
            height=info.height,
            characters=char_tracks,
        )

    def _interpolate_tracks(
        self,
        sampled: Dict[int, Dict[str, List[int]]],
        total_frames: int,
        char_ids: List[str],
    ) -> Dict[str, List[Optional[List[int]]]]:
        """
        Linear interpolation between sampled detections.

        Args:
            sampled: {frame_idx: {char_id: bbox}} for sampled frames
            total_frames: Total number of frames in video
            char_ids: List of character IDs to track

        Returns:
            {char_id: [bbox_per_frame]} with interpolated values
        """
        tracks: Dict[str, List[Optional[List[int]]]] = {
            cid: [None] * total_frames for cid in char_ids
        }

        for char_id in char_ids:
            # Get sorted frame indices where this character was detected
            detected_indices = sorted(
                idx for idx, faces in sampled.items()
                if char_id in faces
            )

            if not detected_indices:
                continue

            # Interpolate between consecutive detections
            for i in range(len(detected_indices) - 1):
                start_idx = detected_indices[i]
                end_idx = detected_indices[i + 1]
                start_bbox = sampled[start_idx][char_id]
                end_bbox = sampled[end_idx][char_id]

                for frame_idx in range(start_idx, end_idx + 1):
                    if end_idx == start_idx:
                        t = 0
                    else:
                        t = (frame_idx - start_idx) / (end_idx - start_idx)

                    tracks[char_id][frame_idx] = [
                        int(start_bbox[k] * (1 - t) + end_bbox[k] * t)
                        for k in range(4)
                    ]

            # Extend first detection to start of video
            first_idx = detected_indices[0]
            first_bbox = sampled[first_idx][char_id]
            for frame_idx in range(0, first_idx):
                tracks[char_id][frame_idx] = first_bbox.copy()

            # Extend last detection to end of video
            last_idx = detected_indices[-1]
            last_bbox = sampled[last_idx][char_id]
            for frame_idx in range(last_idx + 1, total_frames):
                tracks[char_id][frame_idx] = last_bbox.copy()

        return tracks

    def _smooth_tracks(
        self,
        tracks: Dict[str, List[Optional[List[int]]]],
        char_ids: List[str],
    ) -> Dict[str, List[Optional[List[int]]]]:
        """
        Apply EMA smoothing to tracks.

        Args:
            tracks: {char_id: [bbox_per_frame]}
            char_ids: List of character IDs

        Returns:
            Smoothed tracks
        """
        if self.alpha >= 1.0:
            # No smoothing needed
            return tracks

        for char_id in char_ids:
            prev_bbox: Optional[List[int]] = None

            for i, bbox in enumerate(tracks[char_id]):
                if bbox is None:
                    prev_bbox = None
                    continue

                if prev_bbox is None:
                    prev_bbox = bbox
                    continue

                # EMA: smoothed = alpha * current + (1 - alpha) * previous
                smoothed = [
                    int(self.alpha * bbox[k] + (1 - self.alpha) * prev_bbox[k])
                    for k in range(4)
                ]
                tracks[char_id][i] = smoothed
                prev_bbox = smoothed

        return tracks


def generate_tracking_video(
    video_path: str,
    tracking_result: TrackingResult,
    output_path: str,
) -> str:
    """
    Generate visualization video with colored bboxes and labels.

    Args:
        video_path: Input video path
        tracking_result: TrackingResult from FaceTracker
        output_path: Output video path

    Returns:
        Path to output video
    """
    start_time = time.time()
    logger.info(f"Generating tracking visualization: {output_path}")

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Build color map
    color_map = {c.character_id: c.color for c in tracking_result.characters}
    name_map = {c.character_id: c.name for c in tracking_result.characters}

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Draw bboxes for each character
        for char_id, bboxes in tracking_result.tracks.items():
            if frame_idx < len(bboxes) and bboxes[frame_idx] is not None:
                # Bbox is in Wav2Lip format [y1, y2, x1, x2]
                y1, y2, x1, x2 = bboxes[frame_idx]
                color = color_map.get(char_id, (255, 255, 255))
                name = name_map.get(char_id, char_id)

                # Draw rectangle
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Draw label background
                label = name
                (label_w, label_h), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                cv2.rectangle(
                    frame,
                    (x1, y1 - label_h - 10),
                    (x1 + label_w + 4, y1),
                    color,
                    -1,
                )

                # Draw label text
                cv2.putText(
                    frame,
                    label,
                    (x1 + 2, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()

    elapsed = time.time() - start_time
    logger.info(f"  Completed in {elapsed:.2f}s")

    return output_path


def bbox_to_hex_color(color_bgr: Tuple[int, int, int]) -> str:
    """Convert BGR color tuple to hex string."""
    b, g, r = color_bgr
    return f"#{r:02x}{g:02x}{b:02x}"
