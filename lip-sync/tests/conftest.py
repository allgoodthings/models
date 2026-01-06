"""
Pytest fixtures and mocks for lip-sync tests.

This module provides:
- Mock models that skip actual inference
- Test data (short video/audio URLs or local files)
- API client fixtures
- Temporary file management

Note: Heavy imports (numpy, fastapi, etc.) are done lazily within fixtures
to allow running lightweight tests (like segment_builder) without all dependencies.
"""

import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =============================================================================
# Mock Model Classes (no heavy deps needed)
# =============================================================================


@dataclass
class MockFaceAnalysisResult:
    """Mock result from InsightFace face analysis."""
    character_id: str
    bbox: Tuple[int, int, int, int]
    confidence: float
    head_pose: Tuple[float, float, float]
    syncable: bool
    sync_quality: float
    skip_reason: Optional[str] = None


class MockInsightFaceDetector:
    """Mock InsightFace detector that returns predefined face data."""

    def __init__(self, *args, **kwargs):
        self.is_loaded = False
        self._mock_faces: List[MockFaceAnalysisResult] = []
        self._references: Dict[str, Any] = {}

    def load(self):
        self.is_loaded = True

    def unload(self):
        self.is_loaded = False

    def load_reference(self, character_id: str, image: Any) -> bool:
        self._references[character_id] = "mock_embedding"
        return True

    def clear_references(self):
        self._references.clear()

    def detect_faces(
        self, frame: Any, similarity_threshold: float = 0.5
    ) -> List[MockFaceAnalysisResult]:
        """Return mock faces. Can be configured via set_mock_faces()."""
        if not self._mock_faces:
            # Default: return one syncable face
            return [
                MockFaceAnalysisResult(
                    character_id="face_1",
                    bbox=(100, 50, 200, 250),
                    confidence=0.95,
                    head_pose=(5.0, -10.0, 2.0),
                    syncable=True,
                    sync_quality=0.9,
                )
            ]
        return self._mock_faces

    def set_mock_faces(self, faces: List[MockFaceAnalysisResult]):
        """Configure faces to return from detect_faces()."""
        self._mock_faces = faces


class MockLipSyncPipeline:
    """Mock pipeline that creates a simple output video without inference."""

    def __init__(self, config=None):
        self.config = config or MagicMock()
        self._models_loaded = False

    def load_models(self):
        self._models_loaded = True

    def unload_models(self):
        self._models_loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._models_loaded

    def process_single_face(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
        target_bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> str:
        """Create a minimal output video (copy input with slight modification)."""
        import shutil
        shutil.copy(video_path, output_path)
        return output_path

    def process_multi_face(
        self,
        video_path: str,
        face_jobs: list,
        output_path: str,
    ) -> str:
        """Create a minimal output video for multi-face."""
        import shutil
        shutil.copy(video_path, output_path)
        return output_path


# =============================================================================
# Basic Fixtures (no heavy deps)
# =============================================================================


@pytest.fixture
def mock_detector():
    """Provide a mock InsightFace detector."""
    return MockInsightFaceDetector()


@pytest.fixture
def mock_pipeline():
    """Provide a mock lip-sync pipeline."""
    return MockLipSyncPipeline()


@pytest.fixture
def temp_dir():
    """Provide a temporary directory that's cleaned up after tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


# =============================================================================
# Media File Fixtures (require ffmpeg)
# =============================================================================


@pytest.fixture
def sample_video_path(temp_dir):
    """Create a minimal test video file."""
    video_path = os.path.join(temp_dir, "test_video.mp4")

    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", "color=c=blue:s=320x240:d=2",
        "-f", "lavfi",
        "-i", "sine=frequency=440:duration=2",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-c:a", "aac",
        video_path,
    ]

    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        pytest.skip("ffmpeg not available for test video generation")

    return video_path


@pytest.fixture
def sample_audio_path(temp_dir):
    """Create a minimal test audio file."""
    audio_path = os.path.join(temp_dir, "test_audio.wav")

    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", "sine=frequency=440:duration=2",
        audio_path,
    ]

    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        pytest.skip("ffmpeg not available for test audio generation")

    return audio_path


@pytest.fixture
def sample_image_path(temp_dir):
    """Create a minimal test image file."""
    image_path = os.path.join(temp_dir, "test_image.jpg")

    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", "color=c=red:s=200x200:d=1",
        "-frames:v", "1",
        image_path,
    ]

    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        pytest.skip("ffmpeg not available for test image generation")

    return image_path


@pytest.fixture
def mock_presigned_upload_url():
    """Provide a mock presigned URL."""
    return "http://localhost:9999/mock-upload"


# =============================================================================
# App Fixtures (require fastapi, numpy, etc.)
# =============================================================================


@pytest.fixture
def app_with_mocks():
    """
    Create FastAPI app with mocked models for integration testing.

    This fixture requires heavy dependencies (fastapi, numpy, etc).
    Tests using this fixture should be marked appropriately.
    """
    try:
        from fastapi.testclient import TestClient
    except ImportError:
        pytest.skip("fastapi not installed - skipping integration test")

    try:
        from lipsync.server import main as server_main
    except ImportError as e:
        pytest.skip(f"Could not import server module: {e}")

    # Store original values
    original_pipeline = server_main.pipeline
    original_detector = server_main.face_detector

    # Replace with mocks
    mock_det = MockInsightFaceDetector()
    mock_det.load()
    mock_pipe = MockLipSyncPipeline()
    mock_pipe.load_models()

    server_main.face_detector = mock_det
    server_main.pipeline = mock_pipe

    # Create test client
    client = TestClient(server_main.app)

    yield client, mock_det, mock_pipe

    # Restore originals
    server_main.pipeline = original_pipeline
    server_main.face_detector = original_detector


# =============================================================================
# Test Data Helpers
# =============================================================================


def create_mock_frame_data(
    character_id: str,
    num_frames: int = 10,
    frame_interval_ms: int = 200,
    syncable_pattern: Optional[List[bool]] = None,
) -> List[dict]:
    """
    Create mock per-frame face data for testing.

    Args:
        character_id: Character ID
        num_frames: Number of frames to generate
        frame_interval_ms: Time between frames
        syncable_pattern: Pattern of syncable states (repeats if shorter than num_frames)

    Returns:
        List of frame data dicts
    """
    if syncable_pattern is None:
        syncable_pattern = [True]  # All syncable by default

    frames = []
    for i in range(num_frames):
        syncable = syncable_pattern[i % len(syncable_pattern)]
        frames.append({
            "timestamp_ms": i * frame_interval_ms,
            "character_id": character_id,
            "bbox": (100 + i, 50 + i, 200, 250),
            "head_pose": (5.0 + i * 0.1, -10.0 + i * 0.2, 2.0),
            "confidence": 0.95 - i * 0.01,
            "syncable": syncable,
            "sync_quality": 0.9 if syncable else 0.0,
            "skip_reason": None if syncable else "profile_view",
        })

    return frames
