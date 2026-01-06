"""
Segment builder for collapsing per-frame face data into contiguous time segments.

Converts frame-by-frame detection results into segments where each segment
represents a contiguous time range with the same sync state.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class FrameFaceData:
    """Per-frame face data for segment building."""

    timestamp_ms: int
    character_id: str
    bbox: Tuple[int, int, int, int]
    head_pose: Tuple[float, float, float]
    confidence: float
    syncable: bool
    sync_quality: float
    skip_reason: Optional[str] = None


@dataclass
class Segment:
    """A contiguous time segment with accumulated data."""

    start_ms: int
    end_ms: int
    synced: bool
    skip_reason: Optional[str]

    # Accumulated values for averaging
    bbox_sum: List[int] = field(default_factory=lambda: [0, 0, 0, 0])
    pose_sum: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    quality_sum: float = 0.0
    frame_count: int = 0

    def add_frame(
        self,
        bbox: Tuple[int, int, int, int],
        head_pose: Tuple[float, float, float],
        sync_quality: float,
    ) -> None:
        """Add a frame's data to this segment."""
        self.bbox_sum[0] += bbox[0]
        self.bbox_sum[1] += bbox[1]
        self.bbox_sum[2] += bbox[2]
        self.bbox_sum[3] += bbox[3]
        self.pose_sum[0] += head_pose[0]
        self.pose_sum[1] += head_pose[1]
        self.pose_sum[2] += head_pose[2]
        self.quality_sum += sync_quality
        self.frame_count += 1

    @property
    def avg_bbox(self) -> Tuple[int, int, int, int]:
        """Get averaged bounding box."""
        if self.frame_count == 0:
            return (0, 0, 0, 0)
        return (
            self.bbox_sum[0] // self.frame_count,
            self.bbox_sum[1] // self.frame_count,
            self.bbox_sum[2] // self.frame_count,
            self.bbox_sum[3] // self.frame_count,
        )

    @property
    def avg_head_pose(self) -> Tuple[float, float, float]:
        """Get averaged head pose."""
        if self.frame_count == 0:
            return (0.0, 0.0, 0.0)
        return (
            round(self.pose_sum[0] / self.frame_count, 2),
            round(self.pose_sum[1] / self.frame_count, 2),
            round(self.pose_sum[2] / self.frame_count, 2),
        )

    @property
    def avg_quality(self) -> Optional[float]:
        """Get averaged sync quality (only for synced segments)."""
        if not self.synced or self.frame_count == 0:
            return None
        return round(self.quality_sum / self.frame_count, 3)

    @property
    def duration_ms(self) -> int:
        """Get segment duration."""
        return self.end_ms - self.start_ms


def build_segments_for_face(
    frames: List[FrameFaceData],
) -> List[Segment]:
    """
    Build segments from per-frame data for a single face.

    A new segment is created when:
    - syncable state changes (True -> False or vice versa)
    - skip_reason changes (different reason for being unsyncable)

    Args:
        frames: List of per-frame face data, sorted by timestamp

    Returns:
        List of segments with averaged values
    """
    if not frames:
        return []

    # Sort by timestamp
    sorted_frames = sorted(frames, key=lambda f: f.timestamp_ms)

    segments: List[Segment] = []
    current_segment: Optional[Segment] = None

    for frame in sorted_frames:
        # Determine segment key (what defines a segment boundary)
        is_synced = frame.syncable
        skip_reason = frame.skip_reason if not is_synced else None

        # Check if we need a new segment
        need_new_segment = (
            current_segment is None
            or current_segment.synced != is_synced
            or current_segment.skip_reason != skip_reason
        )

        if need_new_segment:
            # Close current segment
            if current_segment is not None:
                segments.append(current_segment)

            # Start new segment
            current_segment = Segment(
                start_ms=frame.timestamp_ms,
                end_ms=frame.timestamp_ms,
                synced=is_synced,
                skip_reason=skip_reason,
            )

        # Update current segment
        current_segment.end_ms = frame.timestamp_ms
        current_segment.add_frame(
            bbox=frame.bbox,
            head_pose=frame.head_pose,
            sync_quality=frame.sync_quality,
        )

    # Don't forget the last segment
    if current_segment is not None:
        segments.append(current_segment)

    return segments


def build_segments_for_all_faces(
    frame_data: Dict[str, List[FrameFaceData]],
) -> Dict[str, List[Segment]]:
    """
    Build segments for multiple faces.

    Args:
        frame_data: Dict mapping character_id to list of per-frame data

    Returns:
        Dict mapping character_id to list of segments
    """
    result = {}
    for character_id, frames in frame_data.items():
        result[character_id] = build_segments_for_face(frames)
    return result


def compute_summary(segments: List[Segment]) -> Tuple[int, int, int]:
    """
    Compute summary statistics from segments.

    Args:
        segments: List of segments for a face

    Returns:
        Tuple of (total_ms, synced_ms, skipped_ms)
    """
    if not segments:
        return (0, 0, 0)

    total_ms = 0
    synced_ms = 0
    skipped_ms = 0

    for segment in segments:
        duration = segment.duration_ms
        total_ms += duration
        if segment.synced:
            synced_ms += duration
        else:
            skipped_ms += duration

    return (total_ms, synced_ms, skipped_ms)


def extend_segment_end_times(
    segments: List[Segment],
    frame_interval_ms: int,
) -> None:
    """
    Extend segment end times to fill gaps between frames.

    Since we sample at discrete intervals, the end_ms of each segment
    should extend to just before the next segment starts (or to the
    next frame interval for the last segment).

    Args:
        segments: List of segments to modify in place
        frame_interval_ms: Time between frames in milliseconds
    """
    for i, segment in enumerate(segments):
        if i < len(segments) - 1:
            # Extend to just before next segment
            next_start = segments[i + 1].start_ms
            segment.end_ms = next_start
        else:
            # Last segment - extend by one frame interval
            segment.end_ms += frame_interval_ms
