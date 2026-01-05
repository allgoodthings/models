"""
Pydantic schemas for lip-sync API.
"""

from typing import List, Optional, Tuple
from pydantic import BaseModel, Field


class FaceJobRequest(BaseModel):
    """Single face lip-sync job definition."""

    character_id: str = Field(
        ...,
        description="Unique identifier for the character/face",
    )
    bbox: Tuple[int, int, int, int] = Field(
        ...,
        description="Bounding box (x, y, width, height) for the face in the video",
    )
    audio_url: Optional[str] = Field(
        None,
        description="URL to audio file for this face (if separate from main audio)",
    )
    start_time_ms: int = Field(
        0,
        description="Start time in milliseconds for this face's lip-sync",
    )
    end_time_ms: int = Field(
        ...,
        description="End time in milliseconds for this face's lip-sync",
    )


class LipSyncRequest(BaseModel):
    """Multi-face lip-sync request."""

    faces: List[FaceJobRequest] = Field(
        ...,
        description="List of face jobs to process",
        min_length=1,
    )
    enhance_quality: bool = Field(
        True,
        description="Apply CodeFormer enhancement after lip-sync",
    )
    fidelity_weight: float = Field(
        0.7,
        description="CodeFormer fidelity weight (0=quality, 1=fidelity)",
        ge=0.0,
        le=1.0,
    )


class FaceResultInfo(BaseModel):
    """Result info for a processed face."""

    character_id: str
    success: bool
    error_message: Optional[str] = None


class LipSyncResponse(BaseModel):
    """Lip-sync processing response."""

    success: bool
    faces_processed: int
    face_results: List[FaceResultInfo]
    processing_time_ms: int
    output_url: Optional[str] = None
    error_message: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    models_downloaded: bool
    models_loaded: bool
    gpu_available: bool
    gpu_name: Optional[str] = None
    gpu_memory_gb: Optional[float] = None


class DetectFacesRequest(BaseModel):
    """Request to detect faces using Qwen-VL."""

    characters: List[dict] = Field(
        ...,
        description="List of character definitions [{id, name, description}]",
    )


class DetectedFace(BaseModel):
    """Detected face result."""

    character_id: str
    bbox: Tuple[int, int, int, int]
    confidence: float


class DetectFacesResponse(BaseModel):
    """Face detection response."""

    faces: List[DetectedFace]
    frame_width: int
    frame_height: int
