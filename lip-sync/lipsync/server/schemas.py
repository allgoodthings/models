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


class DetectionTimingBreakdown(BaseModel):
    """Timing breakdown for face detection."""

    total_ms: int = Field(..., description="Total processing time")
    download_video_ms: int = Field(..., description="Time to download video")
    download_refs_ms: int = Field(..., description="Time to download reference images")
    frame_extraction_ms: int = Field(..., description="Time to extract frames with ffmpeg")
    detection_ms: int = Field(..., description="Time for InsightFace detection on all frames")


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
    timing: Optional[DetectionTimingBreakdown] = Field(
        None, description="Processing time breakdown"
    )


# =============================================================================
# Lip-Sync Request
# =============================================================================


class LipSyncRequest(BaseModel):
    """Lip-sync request - detection happens automatically."""

    video_url: str = Field(..., description="URL to input video file")
    audio_url: str = Field(..., description="URL to audio file for lip-sync")
    upload_url: str = Field(
        ...,
        description="Presigned URL for uploading the output video (PUT request)",
    )
    characters: List[CharacterReference] = Field(
        ...,
        description="Characters to lip-sync with reference images for face matching",
        min_length=1,
    )
    start_time_ms: Optional[int] = Field(
        None, description="Start time in milliseconds (default: 0)"
    )
    end_time_ms: Optional[int] = Field(
        None, description="End time in milliseconds (default: video duration)"
    )
    similarity_threshold: float = Field(
        0.5,
        description="Cosine similarity threshold for face matching (0-1)",
        ge=0.0,
        le=1.0,
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


# =============================================================================
# Lip-Sync Response - Segment-based
# =============================================================================


class FaceSegment(BaseModel):
    """A contiguous time range where face had same sync state."""

    start_ms: int = Field(..., description="Segment start time in milliseconds")
    end_ms: int = Field(..., description="Segment end time in milliseconds")
    synced: bool = Field(..., description="Whether lip-sync was applied in this segment")
    skip_reason: Optional[str] = Field(
        None,
        description="Why sync was skipped (profile_view, face_too_small, etc.)",
    )
    avg_quality: Optional[float] = Field(
        None, description="Average sync quality in this segment (if synced)"
    )
    avg_bbox: Tuple[int, int, int, int] = Field(
        ..., description="Average bounding box in this segment"
    )
    avg_head_pose: Tuple[float, float, float] = Field(
        ..., description="Average head pose (pitch, yaw, roll) in this segment"
    )


class FaceSummary(BaseModel):
    """Summary of sync coverage for a face."""

    total_ms: int = Field(..., description="Total time range for this face")
    synced_ms: int = Field(..., description="Milliseconds where sync was applied")
    skipped_ms: int = Field(..., description="Milliseconds where sync was skipped")


class FaceResult(BaseModel):
    """Complete result for a processed face."""

    character_id: str = Field(..., description="Character identifier")
    success: bool = Field(..., description="Whether processing succeeded")
    error_message: Optional[str] = Field(None, description="Error if failed")
    segments: List[FaceSegment] = Field(
        default_factory=list, description="Time segments with sync state"
    )
    summary: Optional[FaceSummary] = Field(
        None, description="Summary of sync coverage"
    )


class UnknownFace(BaseModel):
    """A face detected but not in the request."""

    character_id: str = Field(
        ..., description="Auto-assigned ID (face_1, face_2, etc.)"
    )
    segments: List[FaceSegment] = Field(
        ..., description="Time segments where this face was visible"
    )


class FacesResult(BaseModel):
    """Container for all face-related results."""

    total_detected: int = Field(..., description="Total faces detected in video")
    processed: int = Field(..., description="Faces that were processed (from request)")
    unknown: int = Field(..., description="Faces detected but not in request")
    results: List[FaceResult] = Field(
        ..., description="Results for requested faces"
    )
    unknown_faces: List[UnknownFace] = Field(
        default_factory=list, description="Faces detected but not in request"
    )


class OutputMetadata(BaseModel):
    """Metadata about the output video."""

    duration_ms: int = Field(..., description="Output video duration in milliseconds")
    width: int = Field(..., description="Video width in pixels")
    height: int = Field(..., description="Video height in pixels")
    fps: float = Field(..., description="Video frame rate")
    file_size_bytes: int = Field(..., description="Output file size in bytes")


class TimingBreakdown(BaseModel):
    """Detailed timing breakdown of processing stages."""

    total_ms: int = Field(..., description="Total processing time")
    download_ms: int = Field(..., description="Time to download video and audio")
    detection_ms: int = Field(..., description="Time for face detection/tracking")
    lipsync_ms: int = Field(..., description="Time for lip-sync processing")
    enhancement_ms: int = Field(..., description="Time for CodeFormer enhancement")
    encoding_ms: int = Field(..., description="Time for final video encoding")
    upload_ms: int = Field(..., description="Time to upload to presigned URL")


class LipSyncResponse(BaseModel):
    """Lip-sync processing response with segment-based face data."""

    success: bool = Field(..., description="Whether processing succeeded")
    error_message: Optional[str] = Field(None, description="Error if failed")
    faces: Optional[FacesResult] = Field(
        None, description="Face detection and processing results"
    )
    output: Optional[OutputMetadata] = Field(
        None, description="Output video metadata"
    )
    timing: Optional[TimingBreakdown] = Field(
        None, description="Processing time breakdown"
    )


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
