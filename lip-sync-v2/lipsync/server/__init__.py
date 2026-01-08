"""
FastAPI server for multi-face lip-sync.

Note: Import app directly from main.py to avoid circular imports:
    from lipsync.server.main import app
"""

# Only export items that don't cause circular imports
from .insightface_detector import InsightFaceDetector
from .schemas import (
    CharacterReference,
    FaceTrackingRequest,
    FaceTrackingResponse,
    HealthResponse,
    LipSyncRequest,
    LipSyncResponse,
)

__all__ = [
    "InsightFaceDetector",
    "CharacterReference",
    "FaceTrackingRequest",
    "FaceTrackingResponse",
    "HealthResponse",
    "LipSyncRequest",
    "LipSyncResponse",
]
