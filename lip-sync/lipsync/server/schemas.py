"""
Pydantic schemas for lip-sync API.
"""

from typing import List, Optional, Tuple

from pydantic import BaseModel, Field


# =============================================================================
# Character Reference
# =============================================================================


class CharacterReference(BaseModel):
    """Character with reference image for identity matching."""

    id: str = Field(..., description="Unique identifier for the character")
    name: str = Field(..., description="Character name")
    reference_image_url: str = Field(
        ..., description="URL to headshot/reference image for face matching"
    )


# =============================================================================
# Face Detection
# =============================================================================


class DetectedFaceWithMetadata(BaseModel):
    """Face detection with full analysis metadata."""

    character_id: str = Field(
        ...,
        description="Matched character ID or auto-assigned (face_1, face_2, etc.)",
    )
    bbox: Tuple[int, int, int, int] = Field(
        ..., description="Bounding box (x, y, width, height)"
    )
    confidence: float = Field(..., description="Detection confidence 0-1")
    head_pose: Tuple[float, float, float] = Field(
        ..., description="Head pose (pitch, yaw, roll) in degrees"
    )
    syncable: bool = Field(
        ..., description="Whether face is suitable for lip-sync"
    )
    sync_quality: float = Field(
        ..., description="Recommended sync blend strength 0-1"
    )
    skip_reason: Optional[str] = Field(
        None,
        description="Reason if not syncable (profile_view, face_too_small, low_detection_quality)",
    )


class FrameAnalysis(BaseModel):
    """Complete analysis for one frame."""

    timestamp_ms: int = Field(..., description="Frame timestamp in milliseconds")
    faces: List[DetectedFaceWithMetadata] = Field(
        default_factory=list, description="Detected faces in this frame"
    )


class DetectFacesRequest(BaseModel):
    """Face detection request with character references."""

    video_url: str = Field(..., description="URL to video file")
    sample_fps: int = Field(
        3,
        description="Frames per second to sample (e.g., 3 = sample every ~333ms)",
        gt=0,
        le=30,
    )
    start_time_ms: Optional[int] = Field(
        None, description="Start time in milliseconds (default: 0)"
    )
    end_time_ms: Optional[int] = Field(
        None, description="End time in milliseconds (default: video duration)"
    )
    characters: List[CharacterReference] = Field(
        ...,
        description="Characters to detect with reference images",
        min_length=1,
    )
    similarity_threshold: float = Field(
        0.5,
        description="Cosine similarity threshold for face matching (0-1)",
        ge=0.0,
        le=1.0,
    )


class DetectFacesResponse(BaseModel):
    """Face detection response with tracking and metadata."""

    frames: List[FrameAnalysis] = Field(
        ..., description="Per-frame face analysis results"
    )
    frame_width: int = Field(..., description="Video frame width in pixels")
    frame_height: int = Field(..., description="Video frame height in pixels")
    sample_fps: int = Field(..., description="Frames per second sampled")
    video_duration_ms: int = Field(..., description="Total video duration in milliseconds")
    characters_detected: List[str] = Field(
        default_factory=list,
        description="IDs of characters that were matched at least once",
    )


# =============================================================================
# Lip-Sync Processing
# =============================================================================


class FaceJobRequest(BaseModel):
    """Single face lip-sync job definition."""

    character_id: str = Field(
        ..., description="Unique identifier for the character/face"
    )
    bbox: Tuple[int, int, int, int] = Field(
        ..., description="Bounding box (x, y, width, height) for the face"
    )
    audio_url: Optional[str] = Field(
        None,
        description="URL to audio file for this face (if separate from main audio)",
    )
    start_time_ms: int = Field(
        0, description="Start time in milliseconds for this face's lip-sync"
    )
    end_time_ms: int = Field(
        ..., description="End time in milliseconds for this face's lip-sync"
    )


class LipSyncRequest(BaseModel):
    """Lip-sync request with URLs."""

    video_url: str = Field(..., description="URL to input video file")
    audio_url: str = Field(..., description="URL to audio file for lip-sync")
    faces: List[FaceJobRequest] = Field(
        ..., description="List of face jobs to process", min_length=1
    )
    enhance_quality: bool = Field(
        True, description="Apply CodeFormer enhancement after lip-sync"
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


# =============================================================================
# Health Check
# =============================================================================


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Server status (healthy, unhealthy, loading)")
    insightface_loaded: bool = Field(
        ..., description="Whether InsightFace model is loaded"
    )
    musetalk_loaded: bool = Field(
        ..., description="Whether MuseTalk model is loaded"
    )
    liveportrait_loaded: bool = Field(
        ..., description="Whether LivePortrait model is loaded"
    )
    codeformer_loaded: bool = Field(
        ..., description="Whether CodeFormer model is loaded"
    )
    gpu_available: bool = Field(..., description="Whether GPU is available")
    gpu_name: Optional[str] = Field(None, description="GPU device name")
    gpu_memory_gb: Optional[float] = Field(None, description="GPU memory in GB")
