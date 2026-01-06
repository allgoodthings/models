"""
FastAPI server for multi-face lip-sync.
"""

from .insightface_detector import InsightFaceDetector
from .main import app
from .schemas import (
    CharacterReference,
    DetectedFaceWithMetadata,
    DetectFacesRequest,
    DetectFacesResponse,
    FaceJobRequest,
    FrameAnalysis,
    HealthResponse,
    LipSyncRequest,
    LipSyncResponse,
)

__all__ = [
    "app",
    "InsightFaceDetector",
    "CharacterReference",
    "DetectedFaceWithMetadata",
    "DetectFacesRequest",
    "DetectFacesResponse",
    "FaceJobRequest",
    "FrameAnalysis",
    "HealthResponse",
    "LipSyncRequest",
    "LipSyncResponse",
]
