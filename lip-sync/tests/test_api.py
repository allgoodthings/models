"""
API integration tests with mocked models.

These tests validate the full API flow without requiring GPU or actual inference.
Run with: pytest tests/test_api.py -v

Note: These tests require fastapi, httpx, numpy, opencv-python-headless.
Tests will be skipped if dependencies are not available.
"""

import json
import os
import sys
import tempfile
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Check for required dependencies
try:
    from fastapi.testclient import TestClient
    import numpy as np
    HAS_DEPS = True
except ImportError as e:
    HAS_DEPS = False
    MISSING_DEP = str(e)

# Skip all tests in this module if deps missing
pytestmark = pytest.mark.skipif(
    not HAS_DEPS,
    reason=f"Missing dependencies for integration tests: {MISSING_DEP if not HAS_DEPS else ''}"
)

from conftest import MockFaceAnalysisResult, MockInsightFaceDetector, MockLipSyncPipeline


class TestHealthEndpoint:
    """Test /health endpoint."""

    def test_health_returns_status(self, app_with_mocks):
        client, mock_detector, mock_pipeline = app_with_mocks

        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["insightface_loaded"] is True
        assert data["musetalk_loaded"] is True
        assert data["liveportrait_loaded"] is True
        assert data["codeformer_loaded"] is True


class TestDetectFacesEndpoint:
    """Test /detect-faces endpoint."""

    @pytest.mark.asyncio
    async def test_detect_faces_basic(self, app_with_mocks, sample_video_path, sample_image_path):
        client, mock_detector, mock_pipeline = app_with_mocks

        # Configure mock to return specific faces
        mock_detector.set_mock_faces([
            MockFaceAnalysisResult(
                character_id="alice",
                bbox=(100, 50, 200, 250),
                confidence=0.95,
                head_pose=(5.0, -10.0, 2.0),
                syncable=True,
                sync_quality=0.9,
            ),
        ])

        # Mock file downloads
        with patch("lipsync.server.main.download_file") as mock_download, \
             patch("lipsync.server.main.download_image") as mock_download_img, \
             patch("lipsync.server.main.get_video_info") as mock_video_info, \
             patch("lipsync.server.main.extract_frames_at_fps") as mock_extract:

            # Setup mocks
            async def fake_download(url, path):
                # Copy sample video to destination
                import shutil
                shutil.copy(sample_video_path, path)
                return 100.0

            mock_download.side_effect = fake_download

            async def fake_download_img(url):
                import cv2
                return cv2.imread(sample_image_path)

            mock_download_img.side_effect = fake_download_img

            mock_video_info.return_value = {
                "duration_ms": 2000,
                "width": 320,
                "height": 240,
                "fps": 30.0,
            }

            # Return fake frame paths
            mock_extract.return_value = [
                (0, sample_image_path),
                (333, sample_image_path),
                (666, sample_image_path),
            ]

            # Make request
            response = client.post("/detect-faces", json={
                "video_url": "https://example.com/video.mp4",
                "sample_fps": 3,
                "characters": [
                    {
                        "id": "alice",
                        "name": "Alice",
                        "reference_image_url": "https://example.com/alice.jpg",
                    }
                ],
                "similarity_threshold": 0.5,
            })

        assert response.status_code == 200
        data = response.json()

        assert "frames" in data
        assert data["frame_width"] == 320
        assert data["frame_height"] == 240
        assert data["video_duration_ms"] == 2000
        assert data["sample_fps"] == 3


class TestLipsyncEndpoint:
    """Test /lipsync endpoint."""

    @pytest.mark.asyncio
    async def test_lipsync_basic(
        self, app_with_mocks, sample_video_path, sample_audio_path, sample_image_path
    ):
        client, mock_detector, mock_pipeline = app_with_mocks

        # Configure mock faces
        mock_detector.set_mock_faces([
            MockFaceAnalysisResult(
                character_id="alice",
                bbox=(100, 50, 200, 250),
                confidence=0.95,
                head_pose=(5.0, -10.0, 2.0),
                syncable=True,
                sync_quality=0.9,
            ),
        ])

        with patch("lipsync.server.main.download_file") as mock_download, \
             patch("lipsync.server.main.upload_file") as mock_upload, \
             patch("lipsync.server.main.get_video_info") as mock_video_info, \
             patch("lipsync.server.main.extract_frames_at_fps") as mock_extract, \
             patch("cv2.imread") as mock_imread:

            # Setup download mock
            async def fake_download(url, path):
                import shutil
                if "video" in url:
                    shutil.copy(sample_video_path, path)
                else:
                    shutil.copy(sample_audio_path, path)
                return 100.0

            mock_download.side_effect = fake_download

            # Setup upload mock
            async def fake_upload(url, path):
                return 50.0  # Upload time in ms

            mock_upload.side_effect = fake_upload

            mock_video_info.return_value = {
                "duration_ms": 2000,
                "width": 320,
                "height": 240,
                "fps": 30.0,
            }

            # Return fake frames for detection
            mock_extract.return_value = [
                (0, "/fake/frame_0.jpg"),
                (200, "/fake/frame_1.jpg"),
                (400, "/fake/frame_2.jpg"),
            ]

            # Mock cv2.imread to return a fake image
            import numpy as np
            mock_imread.return_value = np.zeros((240, 320, 3), dtype=np.uint8)

            # Make request
            response = client.post("/lipsync", json={
                "video_url": "https://example.com/video.mp4",
                "audio_url": "https://example.com/audio.mp3",
                "upload_url": "https://example.com/upload",
                "faces": [
                    {
                        "character_id": "alice",
                        "bbox": [100, 50, 200, 250],
                        "start_time_ms": 0,
                        "end_time_ms": 2000,
                    }
                ],
                "enhance_quality": True,
                "fidelity_weight": 0.7,
            })

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert "faces" in data
        assert "output" in data
        assert "timing" in data

        # Check faces result
        faces = data["faces"]
        assert faces["processed"] == 1
        assert len(faces["results"]) == 1
        assert faces["results"][0]["character_id"] == "alice"
        assert faces["results"][0]["success"] is True

        # Check timing has all fields
        timing = data["timing"]
        assert "total_ms" in timing
        assert "download_ms" in timing
        assert "detection_ms" in timing
        assert "lipsync_ms" in timing
        assert "upload_ms" in timing


class TestSchemaValidation:
    """Test request/response schema validation."""

    def test_detect_faces_requires_characters(self, app_with_mocks):
        client, _, _ = app_with_mocks

        response = client.post("/detect-faces", json={
            "video_url": "https://example.com/video.mp4",
            "sample_fps": 3,
            # Missing 'characters' field
        })

        assert response.status_code == 422  # Validation error

    def test_detect_faces_requires_video_url(self, app_with_mocks):
        client, _, _ = app_with_mocks

        response = client.post("/detect-faces", json={
            "sample_fps": 3,
            "characters": [{"id": "alice", "name": "Alice", "reference_image_url": "http://x.com/a.jpg"}],
        })

        assert response.status_code == 422

    def test_lipsync_requires_upload_url(self, app_with_mocks):
        client, _, _ = app_with_mocks

        response = client.post("/lipsync", json={
            "video_url": "https://example.com/video.mp4",
            "audio_url": "https://example.com/audio.mp3",
            # Missing 'upload_url'
            "faces": [
                {"character_id": "alice", "bbox": [100, 50, 200, 250], "start_time_ms": 0, "end_time_ms": 2000}
            ],
        })

        assert response.status_code == 422

    def test_lipsync_requires_faces(self, app_with_mocks):
        client, _, _ = app_with_mocks

        response = client.post("/lipsync", json={
            "video_url": "https://example.com/video.mp4",
            "audio_url": "https://example.com/audio.mp3",
            "upload_url": "https://example.com/upload",
            # Missing 'faces'
        })

        assert response.status_code == 422

    def test_sample_fps_bounds(self, app_with_mocks):
        client, _, _ = app_with_mocks

        # FPS too high
        response = client.post("/detect-faces", json={
            "video_url": "https://example.com/video.mp4",
            "sample_fps": 100,  # Max is 30
            "characters": [{"id": "alice", "name": "Alice", "reference_image_url": "http://x.com/a.jpg"}],
        })

        assert response.status_code == 422

        # FPS too low
        response = client.post("/detect-faces", json={
            "video_url": "https://example.com/video.mp4",
            "sample_fps": 0,  # Min is 1
            "characters": [{"id": "alice", "name": "Alice", "reference_image_url": "http://x.com/a.jpg"}],
        })

        assert response.status_code == 422


class TestResponseFormats:
    """Test response format correctness."""

    def test_face_segment_has_required_fields(self, app_with_mocks, sample_video_path, sample_audio_path):
        """Verify FaceSegment has all required fields."""
        client, mock_detector, mock_pipeline = app_with_mocks

        # Set up mocks similar to lipsync test
        mock_detector.set_mock_faces([
            MockFaceAnalysisResult(
                character_id="alice",
                bbox=(100, 50, 200, 250),
                confidence=0.95,
                head_pose=(5.0, -10.0, 2.0),
                syncable=True,
                sync_quality=0.9,
            ),
        ])

        with patch("lipsync.server.main.download_file") as mock_download, \
             patch("lipsync.server.main.upload_file") as mock_upload, \
             patch("lipsync.server.main.get_video_info") as mock_video_info, \
             patch("lipsync.server.main.extract_frames_at_fps") as mock_extract, \
             patch("cv2.imread") as mock_imread:

            async def fake_download(url, path):
                import shutil
                if "video" in url:
                    shutil.copy(sample_video_path, path)
                else:
                    shutil.copy(sample_audio_path, path)
                return 100.0

            mock_download.side_effect = fake_download

            async def fake_upload(url, path):
                return 50.0

            mock_upload.side_effect = fake_upload

            mock_video_info.return_value = {
                "duration_ms": 2000,
                "width": 320,
                "height": 240,
                "fps": 30.0,
            }

            mock_extract.return_value = [
                (0, "/fake/frame_0.jpg"),
                (200, "/fake/frame_1.jpg"),
            ]

            import numpy as np
            mock_imread.return_value = np.zeros((240, 320, 3), dtype=np.uint8)

            response = client.post("/lipsync", json={
                "video_url": "https://example.com/video.mp4",
                "audio_url": "https://example.com/audio.mp3",
                "upload_url": "https://example.com/upload",
                "faces": [
                    {"character_id": "alice", "bbox": [100, 50, 200, 250], "start_time_ms": 0, "end_time_ms": 2000}
                ],
            })

        assert response.status_code == 200
        data = response.json()

        # Check segment structure
        if data["faces"]["results"][0]["segments"]:
            segment = data["faces"]["results"][0]["segments"][0]
            assert "start_ms" in segment
            assert "end_ms" in segment
            assert "synced" in segment
            assert "skip_reason" in segment
            assert "avg_quality" in segment
            assert "avg_bbox" in segment
            assert "avg_head_pose" in segment

            # avg_bbox should be a list of 4 ints
            assert len(segment["avg_bbox"]) == 4

            # avg_head_pose should be a list of 3 floats
            assert len(segment["avg_head_pose"]) == 3
