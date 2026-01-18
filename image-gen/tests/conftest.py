"""
Pytest configuration and fixtures for Image-Gen tests.
"""

import pytest


@pytest.fixture
def sample_generate_request():
    """Sample generate request data."""
    return {
        "prompt": "A test image prompt",
        "upload_url": "https://example.com/upload",
        "width": 512,
        "height": 512,
        "num_steps": 4,
        "guidance_scale": 0.0,
        "seed": 42,
        "output_format": "png",
    }


@pytest.fixture
def sample_edit_request():
    """Sample edit request data."""
    return {
        "prompt": "Edit this image",
        "reference_images": [
            {"url": "https://example.com/ref1.png", "weight": 1.0},
        ],
        "upload_url": "https://example.com/upload",
        "width": 512,
        "height": 512,
        "num_steps": 4,
        "guidance_scale": 0.0,
        "output_format": "png",
    }
