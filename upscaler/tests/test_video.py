"""Tests for video utilities."""

import pytest

from upscaler.video import (
    calculate_output_resolution,
    calculate_scale_factor,
)


class TestCalculateOutputResolution:
    """Tests for calculate_output_resolution function."""

    def test_480p_to_1080p(self):
        """Test 480p to 1080p upscale."""
        width, height = calculate_output_resolution(854, 480, "1080p")
        assert height == 1080
        # Aspect ratio preserved
        assert abs(width / height - 854 / 480) < 0.01

    def test_480p_to_4k(self):
        """Test 480p to 4K upscale."""
        width, height = calculate_output_resolution(854, 480, "4k")
        assert height == 2160
        # Aspect ratio preserved
        assert abs(width / height - 854 / 480) < 0.01

    def test_720p_to_1080p(self):
        """Test 720p to 1080p upscale."""
        width, height = calculate_output_resolution(1280, 720, "1080p")
        assert height == 1080
        assert width == 1920

    def test_even_dimensions(self):
        """Test that output dimensions are always even."""
        # Odd input dimensions
        width, height = calculate_output_resolution(853, 479, "1080p")
        assert width % 2 == 0
        assert height % 2 == 0


class TestCalculateScaleFactor:
    """Tests for calculate_scale_factor function."""

    def test_2x_scale(self):
        """Test 2x scale factor."""
        scale = calculate_scale_factor(960, 540, 1920, 1080)
        assert scale == 2.0

    def test_4x_scale(self):
        """Test 4x scale factor."""
        scale = calculate_scale_factor(480, 270, 1920, 1080)
        assert scale == 4.0

    def test_480p_to_4k_scale(self):
        """Test 480p to 4K scale factor."""
        scale = calculate_scale_factor(854, 480, 3840, 2160)
        assert 4.4 < scale < 4.6  # ~4.5x
