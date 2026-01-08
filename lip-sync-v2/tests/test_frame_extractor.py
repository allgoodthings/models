"""
Unit tests for frame extractor.
"""

import pytest
import numpy as np
import tempfile
import subprocess
import os


class TestVideoInfo:
    """Tests for VideoInfo dataclass."""

    def test_video_info_creation(self):
        """Test VideoInfo can be created."""
        from lipsync.frame_extractor import VideoInfo

        info = VideoInfo(
            width=1920,
            height=1080,
            fps=30.0,
            total_frames=300,
            duration_ms=10000,
        )
        assert info.width == 1920
        assert info.height == 1080
        assert info.fps == 30.0
        assert info.total_frames == 300
        assert info.duration_ms == 10000


class TestGetVideoInfo:
    """Tests for get_video_info function."""

    @pytest.fixture
    def sample_video(self):
        """Create a sample video for testing."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            video_path = f.name

        # Create a simple test video with ffmpeg
        cmd = [
            "ffmpeg",
            "-y",
            "-f", "lavfi",
            "-i", "color=c=red:s=320x240:d=1",
            "-c:v", "libx264",
            "-t", "1",
            video_path,
        ]
        subprocess.run(cmd, capture_output=True, check=True)

        yield video_path

        # Cleanup
        os.unlink(video_path)

    def test_get_video_info(self, sample_video):
        """Test getting video info from a real video."""
        from lipsync.frame_extractor import get_video_info

        info = get_video_info(sample_video)

        assert info.width == 320
        assert info.height == 240
        assert info.fps > 0
        assert info.total_frames > 0
        assert info.duration_ms > 0


class TestExtractSampledFrames:
    """Tests for extract_sampled_frames function."""

    @pytest.fixture
    def sample_video(self):
        """Create a sample video for testing."""
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            video_path = f.name

        # Create a 2-second test video at 30fps = 60 frames
        cmd = [
            "ffmpeg",
            "-y",
            "-f", "lavfi",
            "-i", "color=c=blue:s=160x120:d=2:r=30",
            "-c:v", "libx264",
            video_path,
        ]
        subprocess.run(cmd, capture_output=True, check=True)

        yield video_path

        os.unlink(video_path)

    def test_extract_all_frames(self, sample_video):
        """Test extracting all frames (sample_interval=1)."""
        from lipsync.frame_extractor import extract_sampled_frames

        frames, indices, info = extract_sampled_frames(sample_video, sample_interval=1)

        assert len(frames) > 0
        assert len(frames) == len(indices)
        assert all(isinstance(f, np.ndarray) for f in frames)
        assert all(f.shape == (120, 160, 3) for f in frames)  # BGR format

    def test_extract_sampled_frames(self, sample_video):
        """Test extracting sampled frames."""
        from lipsync.frame_extractor import extract_sampled_frames

        frames, indices, info = extract_sampled_frames(sample_video, sample_interval=5)

        # With 60 frames and interval 5, expect ~12 frames
        assert len(frames) > 0
        assert len(frames) < 60  # Should be sampled
        assert indices[0] == 0
        assert indices[1] == 5 if len(indices) > 1 else True

    def test_frame_indices_match_interval(self, sample_video):
        """Test that frame indices match the sample interval."""
        from lipsync.frame_extractor import extract_sampled_frames

        frames, indices, info = extract_sampled_frames(sample_video, sample_interval=10)

        for i, idx in enumerate(indices):
            assert idx == i * 10
