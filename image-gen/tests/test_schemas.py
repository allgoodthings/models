"""
Tests for Pydantic schemas.

These tests verify schema validation without requiring GPU or model loading.
"""

import pytest
from pydantic import ValidationError

from imagegen.server.schemas import (
    EditRequest,
    EditResponse,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
    ReferenceImage,
)


class TestHealthResponse:
    """Tests for HealthResponse schema."""

    def test_healthy_status(self):
        """Test healthy response with all fields."""
        response = HealthResponse(
            status="healthy",
            flux_loaded=True,
            gpu_available=True,
            gpu_name="NVIDIA RTX 4090",
            gpu_memory_gb=24.0,
            gpu_memory_used_gb=15.5,
        )
        assert response.status == "healthy"
        assert response.flux_loaded is True

    def test_loading_status(self):
        """Test loading status."""
        response = HealthResponse(
            status="loading",
            flux_loaded=False,
            gpu_available=True,
        )
        assert response.status == "loading"

    def test_invalid_status(self):
        """Test invalid status is rejected."""
        with pytest.raises(ValidationError):
            HealthResponse(
                status="invalid",
                flux_loaded=True,
                gpu_available=True,
            )


class TestGenerateRequest:
    """Tests for GenerateRequest schema."""

    def test_minimal_request(self):
        """Test minimal valid request."""
        request = GenerateRequest(
            prompt="A test prompt",
            upload_url="https://example.com/upload",
        )
        assert request.prompt == "A test prompt"
        assert request.width == 1024  # default
        assert request.height == 1024  # default
        assert request.num_steps == 4  # default

    def test_full_request(self, sample_generate_request):
        """Test request with all fields."""
        request = GenerateRequest(**sample_generate_request)
        assert request.prompt == "A test image prompt"
        assert request.width == 512
        assert request.seed == 42

    def test_empty_prompt_rejected(self):
        """Test empty prompt is rejected."""
        with pytest.raises(ValidationError):
            GenerateRequest(
                prompt="",
                upload_url="https://example.com/upload",
            )

    def test_invalid_dimensions(self):
        """Test invalid dimensions are rejected."""
        with pytest.raises(ValidationError):
            GenerateRequest(
                prompt="Test",
                upload_url="https://example.com/upload",
                width=100,  # Below minimum
            )

        with pytest.raises(ValidationError):
            GenerateRequest(
                prompt="Test",
                upload_url="https://example.com/upload",
                height=5000,  # Above maximum
            )

    def test_invalid_output_format(self):
        """Test invalid output format is rejected."""
        with pytest.raises(ValidationError):
            GenerateRequest(
                prompt="Test",
                upload_url="https://example.com/upload",
                output_format="gif",  # Not supported
            )


class TestGenerateResponse:
    """Tests for GenerateResponse schema."""

    def test_success_response(self):
        """Test successful response."""
        response = GenerateResponse(
            success=True,
            output_url="https://example.com/output.png",
            width=1024,
            height=1024,
            seed=42,
            timing_inference_ms=2000,
            timing_upload_ms=500,
            timing_total_ms=2500,
        )
        assert response.success is True
        assert response.seed == 42

    def test_failure_response(self):
        """Test failure response."""
        response = GenerateResponse(
            success=False,
            error_message="Out of memory",
        )
        assert response.success is False
        assert response.error_message == "Out of memory"


class TestReferenceImage:
    """Tests for ReferenceImage schema."""

    def test_default_weight(self):
        """Test default weight is 1.0."""
        ref = ReferenceImage(url="https://example.com/image.png")
        assert ref.weight == 1.0

    def test_custom_weight(self):
        """Test custom weight."""
        ref = ReferenceImage(url="https://example.com/image.png", weight=0.5)
        assert ref.weight == 0.5

    def test_invalid_weight(self):
        """Test invalid weight is rejected."""
        with pytest.raises(ValidationError):
            ReferenceImage(url="https://example.com/image.png", weight=3.0)

        with pytest.raises(ValidationError):
            ReferenceImage(url="https://example.com/image.png", weight=-1.0)


class TestEditRequest:
    """Tests for EditRequest schema."""

    def test_minimal_request(self):
        """Test minimal valid request."""
        request = EditRequest(
            prompt="Edit prompt",
            reference_images=[{"url": "https://example.com/ref.png"}],
            upload_url="https://example.com/upload",
        )
        assert len(request.reference_images) == 1

    def test_multiple_references(self):
        """Test multiple reference images."""
        request = EditRequest(
            prompt="Edit prompt",
            reference_images=[
                {"url": "https://example.com/ref1.png", "weight": 1.0},
                {"url": "https://example.com/ref2.png", "weight": 0.5},
                {"url": "https://example.com/ref3.png", "weight": 0.8},
            ],
            upload_url="https://example.com/upload",
        )
        assert len(request.reference_images) == 3

    def test_empty_references_rejected(self):
        """Test empty references are rejected."""
        with pytest.raises(ValidationError):
            EditRequest(
                prompt="Edit prompt",
                reference_images=[],
                upload_url="https://example.com/upload",
            )

    def test_too_many_references_rejected(self):
        """Test more than 4 references are rejected."""
        with pytest.raises(ValidationError):
            EditRequest(
                prompt="Edit prompt",
                reference_images=[
                    {"url": f"https://example.com/ref{i}.png"}
                    for i in range(5)
                ],
                upload_url="https://example.com/upload",
            )


class TestEditResponse:
    """Tests for EditResponse schema."""

    def test_success_response(self):
        """Test successful edit response."""
        response = EditResponse(
            success=True,
            output_url="https://example.com/output.png",
            references_loaded=2,
            width=1024,
            height=1024,
            seed=42,
            timing_download_ms=500,
            timing_inference_ms=2000,
            timing_upload_ms=300,
            timing_total_ms=2800,
        )
        assert response.success is True
        assert response.references_loaded == 2
