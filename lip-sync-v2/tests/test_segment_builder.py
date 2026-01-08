"""
Unit tests for segment builder logic.

These tests validate the segment collapsing algorithm without any models.
Run with: pytest tests/test_segment_builder.py -v
"""

import importlib.util
import sys
from pathlib import Path

import pytest

# Load segment_builder module directly from file (avoid package __init__.py)
_module_path = Path(__file__).parent.parent / "lipsync" / "server" / "segment_builder.py"
_spec = importlib.util.spec_from_file_location("segment_builder", _module_path)
_segment_builder = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_segment_builder)

# Import the classes and functions we need
FrameFaceData = _segment_builder.FrameFaceData
Segment = _segment_builder.Segment
build_segments_for_face = _segment_builder.build_segments_for_face
build_segments_for_all_faces = _segment_builder.build_segments_for_all_faces
compute_summary = _segment_builder.compute_summary
extend_segment_end_times = _segment_builder.extend_segment_end_times


class TestSegment:
    """Test Segment dataclass methods."""

    def test_add_frame_accumulates_values(self):
        seg = Segment(start_ms=0, end_ms=100, synced=True, skip_reason=None)

        seg.add_frame(bbox=(10, 20, 100, 150), head_pose=(5.0, -10.0, 2.0), sync_quality=0.9)
        seg.add_frame(bbox=(12, 22, 98, 148), head_pose=(6.0, -8.0, 1.0), sync_quality=0.8)

        assert seg.frame_count == 2
        assert seg.bbox_sum == [22, 42, 198, 298]
        assert seg.pose_sum == [11.0, -18.0, 3.0]
        assert abs(seg.quality_sum - 1.7) < 0.0001  # Float comparison

    def test_avg_bbox_computes_correctly(self):
        seg = Segment(start_ms=0, end_ms=100, synced=True, skip_reason=None)
        seg.add_frame(bbox=(10, 20, 100, 150), head_pose=(0, 0, 0), sync_quality=0.9)
        seg.add_frame(bbox=(20, 30, 100, 150), head_pose=(0, 0, 0), sync_quality=0.8)

        assert seg.avg_bbox == (15, 25, 100, 150)

    def test_avg_bbox_empty_segment(self):
        seg = Segment(start_ms=0, end_ms=100, synced=True, skip_reason=None)
        assert seg.avg_bbox == (0, 0, 0, 0)

    def test_avg_head_pose_computes_correctly(self):
        seg = Segment(start_ms=0, end_ms=100, synced=True, skip_reason=None)
        seg.add_frame(bbox=(0, 0, 0, 0), head_pose=(5.0, -10.0, 2.0), sync_quality=0.9)
        seg.add_frame(bbox=(0, 0, 0, 0), head_pose=(7.0, -12.0, 4.0), sync_quality=0.8)

        assert seg.avg_head_pose == (6.0, -11.0, 3.0)

    def test_avg_quality_for_synced_segment(self):
        seg = Segment(start_ms=0, end_ms=100, synced=True, skip_reason=None)
        seg.add_frame(bbox=(0, 0, 0, 0), head_pose=(0, 0, 0), sync_quality=0.9)
        seg.add_frame(bbox=(0, 0, 0, 0), head_pose=(0, 0, 0), sync_quality=0.7)

        assert seg.avg_quality == 0.8

    def test_avg_quality_for_skipped_segment_is_none(self):
        seg = Segment(start_ms=0, end_ms=100, synced=False, skip_reason="profile_view")
        seg.add_frame(bbox=(0, 0, 0, 0), head_pose=(0, 0, 0), sync_quality=0.0)

        assert seg.avg_quality is None

    def test_duration_ms(self):
        seg = Segment(start_ms=100, end_ms=500, synced=True, skip_reason=None)
        assert seg.duration_ms == 400


class TestBuildSegmentsForFace:
    """Test segment building from per-frame data."""

    def test_empty_frames_returns_empty(self):
        result = build_segments_for_face([])
        assert result == []

    def test_single_syncable_frame(self):
        frames = [
            FrameFaceData(
                timestamp_ms=0,
                character_id="alice",
                bbox=(100, 50, 200, 250),
                head_pose=(5.0, -10.0, 2.0),
                confidence=0.95,
                syncable=True,
                sync_quality=0.9,
            )
        ]

        result = build_segments_for_face(frames)

        assert len(result) == 1
        assert result[0].synced is True
        assert result[0].skip_reason is None
        assert result[0].start_ms == 0
        assert result[0].end_ms == 0

    def test_continuous_syncable_frames_merge(self):
        frames = [
            FrameFaceData(
                timestamp_ms=0, character_id="alice", bbox=(100, 50, 200, 250),
                head_pose=(5.0, -10.0, 2.0), confidence=0.95, syncable=True, sync_quality=0.9,
            ),
            FrameFaceData(
                timestamp_ms=200, character_id="alice", bbox=(102, 52, 198, 248),
                head_pose=(6.0, -9.0, 1.5), confidence=0.94, syncable=True, sync_quality=0.88,
            ),
            FrameFaceData(
                timestamp_ms=400, character_id="alice", bbox=(101, 51, 199, 249),
                head_pose=(5.5, -9.5, 1.8), confidence=0.96, syncable=True, sync_quality=0.92,
            ),
        ]

        result = build_segments_for_face(frames)

        assert len(result) == 1
        assert result[0].synced is True
        assert result[0].start_ms == 0
        assert result[0].end_ms == 400
        assert result[0].frame_count == 3

    def test_syncable_to_unsyncable_creates_new_segment(self):
        frames = [
            FrameFaceData(
                timestamp_ms=0, character_id="alice", bbox=(100, 50, 200, 250),
                head_pose=(5.0, -10.0, 2.0), confidence=0.95, syncable=True, sync_quality=0.9,
            ),
            FrameFaceData(
                timestamp_ms=200, character_id="alice", bbox=(95, 48, 190, 240),
                head_pose=(8.0, 52.0, -3.0), confidence=0.90, syncable=False, sync_quality=0.0,
                skip_reason="profile_view",
            ),
        ]

        result = build_segments_for_face(frames)

        assert len(result) == 2
        assert result[0].synced is True
        assert result[0].skip_reason is None
        assert result[1].synced is False
        assert result[1].skip_reason == "profile_view"

    def test_different_skip_reasons_create_new_segments(self):
        frames = [
            FrameFaceData(
                timestamp_ms=0, character_id="alice", bbox=(100, 50, 200, 250),
                head_pose=(8.0, 52.0, -3.0), confidence=0.90, syncable=False, sync_quality=0.0,
                skip_reason="profile_view",
            ),
            FrameFaceData(
                timestamp_ms=200, character_id="alice", bbox=(30, 20, 50, 60),
                head_pose=(5.0, -10.0, 2.0), confidence=0.85, syncable=False, sync_quality=0.0,
                skip_reason="face_too_small",
            ),
        ]

        result = build_segments_for_face(frames)

        assert len(result) == 2
        assert result[0].skip_reason == "profile_view"
        assert result[1].skip_reason == "face_too_small"

    def test_same_skip_reason_merges(self):
        frames = [
            FrameFaceData(
                timestamp_ms=0, character_id="alice", bbox=(100, 50, 200, 250),
                head_pose=(8.0, 52.0, -3.0), confidence=0.90, syncable=False, sync_quality=0.0,
                skip_reason="profile_view",
            ),
            FrameFaceData(
                timestamp_ms=200, character_id="alice", bbox=(98, 48, 195, 245),
                head_pose=(9.0, 55.0, -2.0), confidence=0.88, syncable=False, sync_quality=0.0,
                skip_reason="profile_view",
            ),
        ]

        result = build_segments_for_face(frames)

        assert len(result) == 1
        assert result[0].skip_reason == "profile_view"
        assert result[0].frame_count == 2

    def test_out_of_order_frames_sorted(self):
        frames = [
            FrameFaceData(
                timestamp_ms=400, character_id="alice", bbox=(100, 50, 200, 250),
                head_pose=(5.0, -10.0, 2.0), confidence=0.95, syncable=True, sync_quality=0.9,
            ),
            FrameFaceData(
                timestamp_ms=0, character_id="alice", bbox=(100, 50, 200, 250),
                head_pose=(5.0, -10.0, 2.0), confidence=0.95, syncable=True, sync_quality=0.9,
            ),
            FrameFaceData(
                timestamp_ms=200, character_id="alice", bbox=(100, 50, 200, 250),
                head_pose=(5.0, -10.0, 2.0), confidence=0.95, syncable=True, sync_quality=0.9,
            ),
        ]

        result = build_segments_for_face(frames)

        assert len(result) == 1
        assert result[0].start_ms == 0
        assert result[0].end_ms == 400

    def test_complex_sequence(self):
        """Test a realistic sequence: synced -> profile -> synced -> small face."""
        frames = [
            # Synced segment (0-400ms)
            FrameFaceData(
                timestamp_ms=0, character_id="alice", bbox=(100, 50, 200, 250),
                head_pose=(5.0, -10.0, 2.0), confidence=0.95, syncable=True, sync_quality=0.9,
            ),
            FrameFaceData(
                timestamp_ms=200, character_id="alice", bbox=(102, 52, 198, 248),
                head_pose=(6.0, -9.0, 1.5), confidence=0.94, syncable=True, sync_quality=0.88,
            ),
            # Profile segment (400-600ms)
            FrameFaceData(
                timestamp_ms=400, character_id="alice", bbox=(95, 48, 190, 240),
                head_pose=(8.0, 52.0, -3.0), confidence=0.90, syncable=False, sync_quality=0.0,
                skip_reason="profile_view",
            ),
            # Synced again (600-800ms)
            FrameFaceData(
                timestamp_ms=600, character_id="alice", bbox=(105, 52, 195, 245),
                head_pose=(4.0, -8.0, 1.0), confidence=0.96, syncable=True, sync_quality=0.92,
            ),
            # Too small (800-1000ms)
            FrameFaceData(
                timestamp_ms=800, character_id="alice", bbox=(30, 20, 50, 60),
                head_pose=(5.0, -10.0, 2.0), confidence=0.85, syncable=False, sync_quality=0.0,
                skip_reason="face_too_small",
            ),
            FrameFaceData(
                timestamp_ms=1000, character_id="alice", bbox=(32, 22, 48, 58),
                head_pose=(4.5, -9.5, 1.5), confidence=0.84, syncable=False, sync_quality=0.0,
                skip_reason="face_too_small",
            ),
        ]

        result = build_segments_for_face(frames)

        assert len(result) == 4

        # First synced segment
        assert result[0].synced is True
        assert result[0].start_ms == 0
        assert result[0].end_ms == 200
        assert result[0].frame_count == 2

        # Profile segment
        assert result[1].synced is False
        assert result[1].skip_reason == "profile_view"
        assert result[1].start_ms == 400
        assert result[1].end_ms == 400

        # Second synced segment
        assert result[2].synced is True
        assert result[2].start_ms == 600
        assert result[2].end_ms == 600

        # Too small segment
        assert result[3].synced is False
        assert result[3].skip_reason == "face_too_small"
        assert result[3].start_ms == 800
        assert result[3].end_ms == 1000
        assert result[3].frame_count == 2


class TestBuildSegmentsForAllFaces:
    """Test building segments for multiple faces."""

    def test_multiple_faces(self):
        frame_data = {
            "alice": [
                FrameFaceData(
                    timestamp_ms=0, character_id="alice", bbox=(100, 50, 200, 250),
                    head_pose=(5.0, -10.0, 2.0), confidence=0.95, syncable=True, sync_quality=0.9,
                ),
            ],
            "bob": [
                FrameFaceData(
                    timestamp_ms=0, character_id="bob", bbox=(400, 60, 180, 220),
                    head_pose=(3.0, 15.0, -1.0), confidence=0.92, syncable=True, sync_quality=0.85,
                ),
                FrameFaceData(
                    timestamp_ms=200, character_id="bob", bbox=(405, 62, 175, 218),
                    head_pose=(4.0, 48.0, -2.0), confidence=0.90, syncable=False, sync_quality=0.0,
                    skip_reason="profile_view",
                ),
            ],
        }

        result = build_segments_for_all_faces(frame_data)

        assert "alice" in result
        assert "bob" in result
        assert len(result["alice"]) == 1
        assert len(result["bob"]) == 2


class TestComputeSummary:
    """Test summary computation from segments."""

    def test_empty_segments(self):
        total, synced, skipped = compute_summary([])
        assert total == 0
        assert synced == 0
        assert skipped == 0

    def test_all_synced(self):
        segments = [
            Segment(start_ms=0, end_ms=1000, synced=True, skip_reason=None),
            Segment(start_ms=1000, end_ms=2000, synced=True, skip_reason=None),
        ]

        total, synced, skipped = compute_summary(segments)

        assert total == 2000
        assert synced == 2000
        assert skipped == 0

    def test_all_skipped(self):
        segments = [
            Segment(start_ms=0, end_ms=500, synced=False, skip_reason="profile_view"),
            Segment(start_ms=500, end_ms=1000, synced=False, skip_reason="face_too_small"),
        ]

        total, synced, skipped = compute_summary(segments)

        assert total == 1000
        assert synced == 0
        assert skipped == 1000

    def test_mixed(self):
        segments = [
            Segment(start_ms=0, end_ms=1000, synced=True, skip_reason=None),
            Segment(start_ms=1000, end_ms=1500, synced=False, skip_reason="profile_view"),
            Segment(start_ms=1500, end_ms=2500, synced=True, skip_reason=None),
        ]

        total, synced, skipped = compute_summary(segments)

        assert total == 2500
        assert synced == 2000
        assert skipped == 500


class TestExtendSegmentEndTimes:
    """Test segment end time extension."""

    def test_extends_intermediate_segments_to_next_start(self):
        segments = [
            Segment(start_ms=0, end_ms=0, synced=True, skip_reason=None),
            Segment(start_ms=200, end_ms=200, synced=True, skip_reason=None),
            Segment(start_ms=400, end_ms=400, synced=True, skip_reason=None),
        ]

        extend_segment_end_times(segments, frame_interval_ms=200)

        assert segments[0].end_ms == 200  # Extended to start of next
        assert segments[1].end_ms == 400  # Extended to start of next
        assert segments[2].end_ms == 600  # Last segment extended by interval

    def test_single_segment(self):
        segments = [
            Segment(start_ms=0, end_ms=0, synced=True, skip_reason=None),
        ]

        extend_segment_end_times(segments, frame_interval_ms=200)

        assert segments[0].end_ms == 200

    def test_empty_segments(self):
        segments = []
        extend_segment_end_times(segments, frame_interval_ms=200)  # Should not raise
        assert segments == []
