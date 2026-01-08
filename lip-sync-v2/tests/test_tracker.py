"""
Unit tests for face tracker.
"""

import pytest
from unittest.mock import MagicMock, patch
from typing import List, Optional


class TestInterpolation:
    """Tests for bbox interpolation logic."""

    def test_interpolate_between_two_points(self):
        """Test linear interpolation between two detections."""
        from lipsync.tracker import FaceTracker

        # Create mock detector
        mock_detector = MagicMock()

        tracker = FaceTracker(
            detector=mock_detector,
            sample_interval=5,
            smoothing_factor=1.0,  # No smoothing for this test
        )

        # Simulate sampled detections at frames 0 and 10
        sampled = {
            0: {"char_1": [100, 200, 50, 150]},  # [y1, y2, x1, x2]
            10: {"char_1": [110, 210, 60, 160]},  # Moved by 10 pixels
        }

        tracks = tracker._interpolate_tracks(sampled, total_frames=11, char_ids=["char_1"])

        # Frame 0 should be exact
        assert tracks["char_1"][0] == [100, 200, 50, 150]

        # Frame 5 should be midpoint
        assert tracks["char_1"][5] == [105, 205, 55, 155]

        # Frame 10 should be exact
        assert tracks["char_1"][10] == [110, 210, 60, 160]

    def test_extend_to_boundaries(self):
        """Test that first/last detection extends to video boundaries."""
        from lipsync.tracker import FaceTracker

        mock_detector = MagicMock()
        tracker = FaceTracker(
            detector=mock_detector,
            sample_interval=5,
            smoothing_factor=1.0,
        )

        # Detection only at frame 5
        sampled = {
            5: {"char_1": [100, 200, 50, 150]},
        }

        tracks = tracker._interpolate_tracks(sampled, total_frames=10, char_ids=["char_1"])

        # Frames 0-4 should extend from frame 5
        for i in range(5):
            assert tracks["char_1"][i] == [100, 200, 50, 150]

        # Frames 5-9 should extend from frame 5
        for i in range(5, 10):
            assert tracks["char_1"][i] == [100, 200, 50, 150]

    def test_no_detections_returns_none(self):
        """Test that no detections results in None for all frames."""
        from lipsync.tracker import FaceTracker

        mock_detector = MagicMock()
        tracker = FaceTracker(
            detector=mock_detector,
            sample_interval=5,
            smoothing_factor=1.0,
        )

        sampled = {}  # No detections

        tracks = tracker._interpolate_tracks(sampled, total_frames=10, char_ids=["char_1"])

        # All frames should be None
        assert all(tracks["char_1"][i] is None for i in range(10))


class TestEMASmoothing:
    """Tests for EMA smoothing logic."""

    def test_no_smoothing_when_alpha_is_one(self):
        """Test that alpha=1.0 means no smoothing."""
        from lipsync.tracker import FaceTracker

        mock_detector = MagicMock()
        tracker = FaceTracker(
            detector=mock_detector,
            sample_interval=5,
            smoothing_factor=1.0,  # No smoothing
        )

        tracks = {
            "char_1": [
                [100, 200, 50, 150],
                [200, 300, 100, 200],  # Big jump
                [100, 200, 50, 150],
            ]
        }

        smoothed = tracker._smooth_tracks(tracks, char_ids=["char_1"])

        # Should be unchanged
        assert smoothed["char_1"][0] == [100, 200, 50, 150]
        assert smoothed["char_1"][1] == [200, 300, 100, 200]
        assert smoothed["char_1"][2] == [100, 200, 50, 150]

    def test_smoothing_reduces_jumps(self):
        """Test that smoothing reduces large bbox jumps."""
        from lipsync.tracker import FaceTracker

        mock_detector = MagicMock()
        tracker = FaceTracker(
            detector=mock_detector,
            sample_interval=5,
            smoothing_factor=0.5,  # 50% smoothing
        )

        tracks = {
            "char_1": [
                [100, 200, 50, 150],
                [200, 300, 100, 200],  # Big jump of 100
            ]
        }

        smoothed = tracker._smooth_tracks(tracks, char_ids=["char_1"])

        # First frame unchanged (no previous)
        assert smoothed["char_1"][0] == [100, 200, 50, 150]

        # Second frame should be smoothed (0.5 * 200 + 0.5 * 100 = 150)
        assert smoothed["char_1"][1] == [150, 250, 75, 175]


class TestBboxToHexColor:
    """Tests for color conversion."""

    def test_red_conversion(self):
        """Test BGR red to hex."""
        from lipsync.tracker import bbox_to_hex_color

        # BGR red = (0, 0, 255)
        assert bbox_to_hex_color((0, 0, 255)) == "#ff0000"

    def test_green_conversion(self):
        """Test BGR green to hex."""
        from lipsync.tracker import bbox_to_hex_color

        # BGR green = (0, 255, 0)
        assert bbox_to_hex_color((0, 255, 0)) == "#00ff00"

    def test_blue_conversion(self):
        """Test BGR blue to hex."""
        from lipsync.tracker import bbox_to_hex_color

        # BGR blue = (255, 0, 0)
        assert bbox_to_hex_color((255, 0, 0)) == "#0000ff"
