"""
Pydantic schemas for lip-sync API.
"""

from typing import List, Optional, Tuple
from pydantic import BaseModel, Field, HttpUrl


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


class DetectedFaceAtTime(BaseModel):
    """Detected face with timestamp."""

    character_id: str
    bbox: Tuple[int, int, int, int]
    confidence: float
    timestamp_ms: int


class FrameDetection(BaseModel):
    """Face detections for a single frame."""

    timestamp_ms: int
    faces: List[DetectedFace]


class DetectFacesResponse(BaseModel):
    """Face detection response (single frame, legacy)."""

    faces: List[DetectedFace]
    frame_width: int
    frame_height: int


class DetectFacesMultiFrameResponse(BaseModel):
    """Face detection response with multi-frame tracking."""

    frames: List[FrameDetection]
    frame_width: int
    frame_height: int
    sample_fps: float
    video_duration_ms: int


# URL-based request schemas (JSON API)

class CharacterDefinition(BaseModel):
    """Character definition for face detection."""

    id: str = Field(..., description="Unique identifier for the character")
    name: str = Field(..., description="Character name")
    description: Optional[str] = Field(None, description="Visual description of the character")


class DetectFacesUrlRequest(BaseModel):
    """Face detection request with video URL."""

    video_url: str = Field(
        ...,
        description="URL to video file",
    )
    sample_fps: float = Field(
        3.0,
        description="Frames per second to sample for face detection (e.g., 3 = 3 frames/second)",
        gt=0.0,
        le=30.0,
    )
    start_time_ms: Optional[int] = Field(
        None,
        description="Start time in milliseconds (default: 0)",
    )
    end_time_ms: Optional[int] = Field(
        None,
        description="End time in milliseconds (default: video duration)",
    )
    characters: List[CharacterDefinition] = Field(
        ...,
        description="List of characters to detect",
    )


class LipSyncUrlRequest(BaseModel):
    """Lip-sync request with URLs."""

    video_url: str = Field(
        ...,
        description="URL to input video file",
    )
    audio_url: str = Field(
        ...,
        description="URL to audio file for lip-sync",
    )
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
