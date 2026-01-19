"""Tests for Pydantic schemas."""

import pytest
from pydantic import ValidationError

from upscaler.server.schemas import (
    UpscaleRequest,
    UpscaleResponse,
    HealthResponse,
)


class TestUpscaleRequest:
    """Tests for UpscaleRequest schema."""

    def test_valid_request(self, sample_request):
        """Test valid request parsing."""
        request = UpscaleRequest(**sample_request)
        assert request.video_url == sample_request["video_url"]
        assert request.target_resolution == "1080p"
        assert request.high_quality is False
        assert request.deflicker is False

    def test_defaults(self):
        """Test default values."""
        request = UpscaleRequest(
            video_url="https://example.com/video.mp4",
            upload_url="https://example.com/upload",
        )
        assert request.target_resolution == "1080p"
        assert request.high_quality is False
        assert request.deflicker is False
        assert request.model == "realesr-animevideov3"
        assert request.preserve_audio is True

    def test_4k_request(self, sample_4k_request):
        """Test 4K request with all options."""
        request = UpscaleRequest(**sample_4k_request)
        assert request.target_resolution == "4k"
        assert request.high_quality is True
        assert request.deflicker is True

    def test_invalid_resolution(self, sample_request):
        """Test invalid resolution value."""
        sample_request["target_resolution"] = "8k"
        with pytest.raises(ValidationError):
            UpscaleRequest(**sample_request)

    def test_invalid_model(self, sample_request):
        """Test invalid model value."""
        sample_request["model"] = "invalid-model"
        with pytest.raises(ValidationError):
            UpscaleRequest(**sample_request)

    def test_missing_required_fields(self):
        """Test missing required fields."""
        with pytest.raises(ValidationError):
            UpscaleRequest(video_url="https://example.com/video.mp4")

        with pytest.raises(ValidationError):
            UpscaleRequest(upload_url="https://example.com/upload")


class TestUpscaleResponse:
    """Tests for UpscaleResponse schema."""

    def test_success_response(self):
        """Test successful response."""
        response = UpscaleResponse(
            success=True,
            output_url="https://example.com/output.mp4",
            input_resolution="854x480",
            output_resolution="1920x1080",
            input_duration_ms=60000,
            output_duration_ms=60000,
            frames_processed=1440,
            output_size_bytes=50_000_000,
            model_used="realesr-animevideov3",
            upscale_passes=1,
            deflicker_applied=False,
            timing_total_ms=180000,
        )
        assert response.success is True
        assert response.output_resolution == "1920x1080"

    def test_error_response(self):
        """Test error response."""
        response = UpscaleResponse(
            success=False,
            error_message="Failed to download video",
        )
        assert response.success is False
        assert response.error_message == "Failed to download video"
        assert response.output_url is None


class TestHealthResponse:
    """Tests for HealthResponse schema."""

    def test_healthy_response(self):
        """Test healthy status."""
        response = HealthResponse(
            status="healthy",
            model_loaded=True,
            deflicker_available=True,
            gpu_available=True,
            gpu_name="NVIDIA RTX 4090",
            gpu_memory_gb=24.0,
            version="0.1.0",
        )
        assert response.status == "healthy"
        assert response.model_loaded is True

    def test_unhealthy_response(self):
        """Test unhealthy status."""
        response = HealthResponse(
            status="unhealthy",
            model_loaded=False,
            deflicker_available=False,
            gpu_available=False,
            version="0.1.0",
        )
        assert response.status == "unhealthy"
        assert response.gpu_name is None
