"""Pytest configuration and fixtures."""

import pytest


@pytest.fixture
def sample_request():
    """Sample upscale request data."""
    return {
        "video_url": "https://example.com/input.mp4",
        "upload_url": "https://example.com/upload?signature=abc",
        "target_resolution": "1080p",
        "high_quality": False,
        "deflicker": False,
        "model": "realesr-animevideov3",
        "preserve_audio": True,
    }


@pytest.fixture
def sample_4k_request():
    """Sample 4K upscale request data."""
    return {
        "video_url": "https://example.com/input.mp4",
        "upload_url": "https://example.com/upload?signature=abc",
        "target_resolution": "4k",
        "high_quality": True,
        "deflicker": True,
        "model": "realesr-animevideov3",
        "preserve_audio": True,
    }
