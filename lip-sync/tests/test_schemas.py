"""
Unit tests for Pydantic schemas.

These tests validate schema structure and validation without any models.
Run with: pytest tests/test_schemas.py -v
"""

import importlib.util
import sys
from pathlib import Path

import pytest

# Check for pydantic
try:
    import pydantic
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False

pytestmark = pytest.mark.skipif(not HAS_PYDANTIC, reason="pydantic not installed")

# Load schemas module directly from file (avoid package __init__.py)
_module_path = Path(__file__).parent.parent / "lipsync" / "server" / "schemas.py"
_spec = importlib.util.spec_from_file_location("schemas", _module_path)
_schemas = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_schemas)

# Import schemas
CharacterReference = _schemas.CharacterReference
DetectedFaceWithMetadata = _schemas.DetectedFaceWithMetadata
DetectFacesRequest = _schemas.DetectFacesRequest
DetectFacesResponse = _schemas.DetectFacesResponse
FaceJobRequest = _schemas.FaceJobRequest
LipSyncRequest = _schemas.LipSyncRequest
LipSyncResponse = _schemas.LipSyncResponse
FaceSegment = _schemas.FaceSegment
FaceSummary = _schemas.FaceSummary
FaceResult = _schemas.FaceResult
FacesResult = _schemas.FacesResult
OutputMetadata = _schemas.OutputMetadata
TimingBreakdown = _schemas.TimingBreakdown


class TestCharacterReference:
    """Test CharacterReference schema."""

    def test_valid_character(self):
        char = CharacterReference(
            id="alice",
            name="Alice",
            reference_image_url="https://example.com/alice.jpg",
        )
        assert char.id == "alice"
        assert char.name == "Alice"

    def test_missing_required_field(self):
        with pytest.raises(pydantic.ValidationError):
            CharacterReference(id="alice", name="Alice")  # Missing reference_image_url


class TestDetectFacesRequest:
    """Test DetectFacesRequest schema."""

    def test_valid_request(self):
        req = DetectFacesRequest(
            video_url="https://example.com/video.mp4",
            characters=[
                CharacterReference(
                    id="alice",
                    name="Alice",
                    reference_image_url="https://example.com/alice.jpg",
                )
            ],
        )
        assert req.sample_fps == 3  # Default value
        assert req.similarity_threshold == 0.5  # Default value

    def test_sample_fps_bounds(self):
        # Valid: 1-30
        req = DetectFacesRequest(
            video_url="https://example.com/video.mp4",
            sample_fps=30,
            characters=[
                CharacterReference(id="a", name="A", reference_image_url="http://x.com/a.jpg")
            ],
        )
        assert req.sample_fps == 30

        # Invalid: > 30
        with pytest.raises(pydantic.ValidationError):
            DetectFacesRequest(
                video_url="https://example.com/video.mp4",
                sample_fps=31,
                characters=[
                    CharacterReference(id="a", name="A", reference_image_url="http://x.com/a.jpg")
                ],
            )

        # Invalid: < 1
        with pytest.raises(pydantic.ValidationError):
            DetectFacesRequest(
                video_url="https://example.com/video.mp4",
                sample_fps=0,
                characters=[
                    CharacterReference(id="a", name="A", reference_image_url="http://x.com/a.jpg")
                ],
            )

    def test_empty_characters_invalid(self):
        with pytest.raises(pydantic.ValidationError):
            DetectFacesRequest(
                video_url="https://example.com/video.mp4",
                characters=[],
            )


class TestLipSyncRequest:
    """Test LipSyncRequest schema."""

    def test_valid_request(self):
        req = LipSyncRequest(
            video_url="https://example.com/video.mp4",
            audio_url="https://example.com/audio.mp3",
            upload_url="https://storage.example.com/presigned",
            faces=[
                FaceJobRequest(
                    character_id="alice",
                    bbox=(100, 50, 200, 250),
                    end_time_ms=5000,
                )
            ],
        )
        assert req.enhance_quality is True  # Default
        assert req.fidelity_weight == 0.7  # Default

    def test_missing_upload_url(self):
        with pytest.raises(pydantic.ValidationError):
            LipSyncRequest(
                video_url="https://example.com/video.mp4",
                audio_url="https://example.com/audio.mp3",
                # Missing upload_url
                faces=[
                    FaceJobRequest(
                        character_id="alice",
                        bbox=(100, 50, 200, 250),
                        end_time_ms=5000,
                    )
                ],
            )

    def test_empty_faces_invalid(self):
        with pytest.raises(pydantic.ValidationError):
            LipSyncRequest(
                video_url="https://example.com/video.mp4",
                audio_url="https://example.com/audio.mp3",
                upload_url="https://storage.example.com/presigned",
                faces=[],
            )

    def test_fidelity_weight_bounds(self):
        # Valid: 0.0-1.0
        req = LipSyncRequest(
            video_url="https://example.com/video.mp4",
            audio_url="https://example.com/audio.mp3",
            upload_url="https://storage.example.com/presigned",
            faces=[
                FaceJobRequest(character_id="a", bbox=(0, 0, 100, 100), end_time_ms=1000)
            ],
            fidelity_weight=0.0,
        )
        assert req.fidelity_weight == 0.0

        # Invalid: > 1.0
        with pytest.raises(pydantic.ValidationError):
            LipSyncRequest(
                video_url="https://example.com/video.mp4",
                audio_url="https://example.com/audio.mp3",
                upload_url="https://storage.example.com/presigned",
                faces=[
                    FaceJobRequest(character_id="a", bbox=(0, 0, 100, 100), end_time_ms=1000)
                ],
                fidelity_weight=1.5,
            )


class TestFaceSegment:
    """Test FaceSegment schema."""

    def test_synced_segment(self):
        seg = FaceSegment(
            start_ms=0,
            end_ms=1000,
            synced=True,
            skip_reason=None,
            avg_quality=0.9,
            avg_bbox=(100, 50, 200, 250),
            avg_head_pose=(5.0, -10.0, 2.0),
        )
        assert seg.synced is True
        assert seg.avg_quality == 0.9

    def test_skipped_segment(self):
        seg = FaceSegment(
            start_ms=1000,
            end_ms=2000,
            synced=False,
            skip_reason="profile_view",
            avg_quality=None,
            avg_bbox=(100, 50, 200, 250),
            avg_head_pose=(5.0, 52.0, 2.0),
        )
        assert seg.synced is False
        assert seg.skip_reason == "profile_view"


class TestLipSyncResponse:
    """Test LipSyncResponse schema."""

    def test_successful_response(self):
        resp = LipSyncResponse(
            success=True,
            faces=FacesResult(
                total_detected=2,
                processed=1,
                unknown=1,
                results=[
                    FaceResult(
                        character_id="alice",
                        success=True,
                        segments=[
                            FaceSegment(
                                start_ms=0,
                                end_ms=5000,
                                synced=True,
                                skip_reason=None,
                                avg_quality=0.9,
                                avg_bbox=(100, 50, 200, 250),
                                avg_head_pose=(5.0, -10.0, 2.0),
                            )
                        ],
                        summary=FaceSummary(total_ms=5000, synced_ms=5000, skipped_ms=0),
                    )
                ],
            ),
            output=OutputMetadata(
                duration_ms=5000,
                width=1920,
                height=1080,
                fps=30.0,
                file_size_bytes=2500000,
            ),
            timing=TimingBreakdown(
                total_ms=45000,
                download_ms=3000,
                detection_ms=5000,
                lipsync_ms=30000,
                enhancement_ms=4000,
                encoding_ms=2000,
                upload_ms=1000,
            ),
        )

        assert resp.success is True
        assert resp.faces.total_detected == 2
        assert resp.output.duration_ms == 5000
        assert resp.timing.total_ms == 45000

    def test_failed_response(self):
        resp = LipSyncResponse(
            success=False,
            error_message="Failed to download video",
        )
        assert resp.success is False
        assert resp.error_message == "Failed to download video"
        assert resp.faces is None
        assert resp.output is None


class TestDetectedFaceWithMetadata:
    """Test DetectedFaceWithMetadata schema."""

    def test_syncable_face(self):
        face = DetectedFaceWithMetadata(
            character_id="alice",
            bbox=(100, 50, 200, 250),
            confidence=0.95,
            head_pose=(5.0, -10.0, 2.0),
            syncable=True,
            sync_quality=0.9,
        )
        assert face.syncable is True
        assert face.skip_reason is None

    def test_unsyncable_face(self):
        face = DetectedFaceWithMetadata(
            character_id="bob",
            bbox=(100, 50, 200, 250),
            confidence=0.90,
            head_pose=(5.0, 52.0, 2.0),
            syncable=False,
            sync_quality=0.0,
            skip_reason="profile_view",
        )
        assert face.syncable is False
        assert face.skip_reason == "profile_view"
