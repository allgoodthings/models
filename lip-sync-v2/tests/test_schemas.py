"""
Unit tests for Pydantic schemas.

These tests validate schema structure and validation without any models.
Run with: pytest tests/test_schemas.py -v
"""

import importlib.util
from pathlib import Path

import pytest

# Check for pydantic
try:
    from pydantic import ValidationError
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False

pytestmark = pytest.mark.skipif(not HAS_PYDANTIC, reason="pydantic not installed")

# Load schemas module directly from file (avoid package __init__.py which imports heavy deps)
_module_path = Path(__file__).parent.parent / "lipsync" / "server" / "schemas.py"
_spec = importlib.util.spec_from_file_location("schemas", _module_path)
_schemas = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_schemas)

# Import schemas
CharacterReference = _schemas.CharacterReference
LipSyncRequest = _schemas.LipSyncRequest
LipSyncResponse = _schemas.LipSyncResponse
FaceTrackingRequest = _schemas.FaceTrackingRequest
FaceTrackingResponse = _schemas.FaceTrackingResponse
HealthResponse = _schemas.HealthResponse


class TestCharacterReference:
    """Tests for CharacterReference schema."""

    def test_valid_character(self):
        """Test valid character reference."""
        char = CharacterReference(
            id="char_1",
            name="Alice",
            reference_image_url="https://example.com/alice.jpg",
        )
        assert char.id == "char_1"
        assert char.name == "Alice"
        assert char.reference_image_url == "https://example.com/alice.jpg"

    def test_missing_required_fields(self):
        """Test that required fields are enforced."""
        with pytest.raises(ValidationError):
            CharacterReference(id="char_1")  # Missing name and reference_image_url


class TestLipSyncRequest:
    """Tests for LipSyncRequest schema."""

    def test_valid_request(self):
        """Test valid lip-sync request."""
        request = LipSyncRequest(
            video_url="https://example.com/video.mp4",
            audio_url="https://example.com/audio.wav",
            upload_url="https://example.com/upload",
            characters=[
                CharacterReference(
                    id="char_1",
                    name="Alice",
                    reference_image_url="https://example.com/alice.jpg",
                )
            ],
        )
        assert request.video_url == "https://example.com/video.mp4"
        assert request.enhance_quality is True  # Default
        assert request.temporal_smoothing == 0.3  # Default
        assert request.similarity_threshold == 0.5  # Default

    def test_custom_parameters(self):
        """Test custom parameter values."""
        request = LipSyncRequest(
            video_url="https://example.com/video.mp4",
            audio_url="https://example.com/audio.wav",
            upload_url="https://example.com/upload",
            characters=[
                CharacterReference(
                    id="char_1",
                    name="Alice",
                    reference_image_url="https://example.com/alice.jpg",
                )
            ],
            enhance_quality=False,
            temporal_smoothing=0.5,
            similarity_threshold=0.7,
        )
        assert request.enhance_quality is False
        assert request.temporal_smoothing == 0.5
        assert request.similarity_threshold == 0.7

    def test_similarity_threshold_bounds(self):
        """Test similarity threshold validation bounds."""
        with pytest.raises(ValidationError):
            LipSyncRequest(
                video_url="https://example.com/video.mp4",
                audio_url="https://example.com/audio.wav",
                upload_url="https://example.com/upload",
                characters=[
                    CharacterReference(
                        id="char_1",
                        name="Alice",
                        reference_image_url="https://example.com/alice.jpg",
                    )
                ],
                similarity_threshold=1.5,  # Out of bounds
            )

    def test_empty_characters_rejected(self):
        """Test that empty characters list is rejected."""
        with pytest.raises(ValidationError):
            LipSyncRequest(
                video_url="https://example.com/video.mp4",
                audio_url="https://example.com/audio.wav",
                upload_url="https://example.com/upload",
                characters=[],  # Empty not allowed
            )


class TestLipSyncResponse:
    """Tests for LipSyncResponse schema."""

    def test_success_response(self):
        """Test successful response."""
        response = LipSyncResponse(
            success=True,
            output_url="https://example.com/output.mp4",
            processed_characters=["char_1"],
            output_duration_ms=5000,
            output_width=1920,
            output_height=1080,
            output_fps=30.0,
            output_size_bytes=10000000,
            timing_download_ms=1000,
            timing_tracking_ms=2000,
            timing_lipsync_ms=5000,
            timing_upload_ms=1000,
            timing_total_ms=9000,
        )
        assert response.success is True
        assert response.output_url == "https://example.com/output.mp4"

    def test_failure_response(self):
        """Test failure response."""
        response = LipSyncResponse(
            success=False,
            error_message="No characters could be processed",
        )
        assert response.success is False
        assert response.error_message == "No characters could be processed"


class TestFaceTrackingRequest:
    """Tests for FaceTrackingRequest schema."""

    def test_valid_request(self):
        """Test valid face tracking request."""
        request = FaceTrackingRequest(
            video_url="https://example.com/video.mp4",
            upload_url="https://example.com/upload",
            characters=[
                CharacterReference(
                    id="char_1",
                    name="Alice",
                    reference_image_url="https://example.com/alice.jpg",
                )
            ],
        )
        assert request.video_url == "https://example.com/video.mp4"
        assert request.temporal_smoothing == 0.3  # Default


class TestFaceTrackingResponse:
    """Tests for FaceTrackingResponse schema."""

    def test_valid_response(self):
        """Test valid face tracking response."""
        response = FaceTrackingResponse(
            output_url="https://example.com/tracking.mp4",
            character_colors={"char_1": "#FF0000"},
            frame_count=300,
            fps=30.0,
            total_ms=5000,
        )
        assert response.output_url == "https://example.com/tracking.mp4"
        assert response.character_colors == {"char_1": "#FF0000"}


class TestHealthResponse:
    """Tests for HealthResponse schema."""

    def test_healthy_response(self):
        """Test healthy server response."""
        response = HealthResponse(
            status="healthy",
            insightface_loaded=True,
            wav2lip_loaded=True,
            gpu_available=True,
            gpu_name="NVIDIA RTX 4090",
            gpu_memory_gb=24.0,
        )
        assert response.status == "healthy"
        assert response.insightface_loaded is True

    def test_unhealthy_response(self):
        """Test unhealthy server response."""
        response = HealthResponse(
            status="unhealthy",
            insightface_loaded=False,
            wav2lip_loaded=False,
            gpu_available=False,
        )
        assert response.status == "unhealthy"
        assert response.gpu_name is None
