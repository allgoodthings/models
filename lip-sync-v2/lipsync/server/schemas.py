"""
Pydantic schemas for lip-sync API.
"""

from typing import List, Literal, Optional, Tuple

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
# Lip-Sync Request
# =============================================================================


class LipSyncRequest(BaseModel):
    """Lip-sync request with Wav2Lip-HD."""

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
    similarity_threshold: float = Field(
        0.5,
        description="Cosine similarity threshold for face matching (0-1)",
        ge=0.0,
        le=1.0,
    )
    enhance_quality: bool = Field(
        True, description="Apply GFPGAN enhancement after lip-sync"
    )
    loop_mode: Literal["none", "repeat", "pingpong", "crossfade", "smart"] = Field(
        "smart",
        description=(
            "How to handle audio longer than video: "
            "'none' = trim to shortest, "
            "'repeat' = loop video with hard cut, "
            "'pingpong' = forward-backward loop (good for portraits), "
            "'crossfade' = loop with smooth blend at boundaries, "
            "'smart' = find best loop point + RIFE interpolation (default, best quality)"
        ),
    )
    crossfade_frames: int = Field(
        10,
        description="Number of frames to crossfade at loop boundary (only for crossfade mode)",
        ge=1,
        le=30,
    )
    temporal_smoothing: float = Field(
        0.3,
        description="EMA smoothing factor for bbox tracking (0=no smoothing, 1=no history)",
        ge=0.0,
        le=1.0,
    )


# =============================================================================
# Face Tracking (Visual Validation)
# =============================================================================


class FaceTrackingRequest(BaseModel):
    """Request to generate face tracking visualization video."""

    video_url: str = Field(..., description="URL to video file")
    upload_url: str = Field(
        ...,
        description="Presigned URL for uploading the visualization video",
    )
    characters: List[CharacterReference] = Field(
        ...,
        description="Characters to track with reference images",
        min_length=1,
    )
    similarity_threshold: float = Field(
        0.5,
        description="Cosine similarity threshold for face matching (0-1)",
        ge=0.0,
        le=1.0,
    )
    temporal_smoothing: float = Field(
        0.3,
        description="EMA smoothing factor for bbox tracking",
        ge=0.0,
        le=1.0,
    )


class FaceTrackingResponse(BaseModel):
    """Response from face tracking visualization."""

    output_url: str = Field(..., description="URL to uploaded visualization video")
    character_colors: dict = Field(
        ..., description="Character ID to hex color mapping"
    )
    frame_count: int = Field(..., description="Total frames in video")
    fps: float = Field(..., description="Video frame rate")
    total_ms: int = Field(..., description="Total processing time")


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
    """Lip-sync V2 processing response."""

    success: bool = Field(..., description="Whether processing succeeded")
    error_message: Optional[str] = Field(None, description="Error if failed")
    output_url: Optional[str] = Field(None, description="URL to uploaded output video")
    processed_characters: List[str] = Field(
        default_factory=list, description="IDs of characters that were processed"
    )
    output_duration_ms: Optional[int] = Field(None, description="Output video duration")
    output_width: Optional[int] = Field(None, description="Output video width")
    output_height: Optional[int] = Field(None, description="Output video height")
    output_fps: Optional[float] = Field(None, description="Output video FPS")
    output_size_bytes: Optional[int] = Field(None, description="Output file size")
    timing_download_ms: Optional[int] = Field(None, description="Download time")
    timing_tracking_ms: Optional[int] = Field(None, description="Face tracking time")
    timing_lipsync_ms: Optional[int] = Field(None, description="Wav2Lip processing time")
    timing_upload_ms: Optional[int] = Field(None, description="Upload time")
    timing_total_ms: Optional[int] = Field(None, description="Total processing time")


# =============================================================================
# Health Check
# =============================================================================


class HealthResponse(BaseModel):
    """Health check response for V2."""

    status: str = Field(..., description="Server status (healthy, unhealthy, loading)")
    insightface_loaded: bool = Field(
        ..., description="Whether InsightFace model is loaded"
    )
    wav2lip_loaded: bool = Field(
        ..., description="Whether Wav2Lip-HD is initialized"
    )
    gpu_available: bool = Field(..., description="Whether GPU is available")
    gpu_name: Optional[str] = Field(None, description="GPU device name")
    gpu_memory_gb: Optional[float] = Field(None, description="GPU memory in GB")
