"""
Pydantic schemas for Video Upscaler API.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field


# =============================================================================
# Enums and Types
# =============================================================================

TargetResolution = Literal["1080p", "4k"]
ModelName = Literal["realesr-animevideov3", "realesrgan-x4plus"]


# =============================================================================
# Upscale Request
# =============================================================================


class UpscaleRequest(BaseModel):
    """Video upscaling request."""

    video_url: str = Field(
        ...,
        description="URL to input video file (typically 480p)",
    )
    upload_url: str = Field(
        ...,
        description="Presigned URL for uploading the output video (PUT request)",
    )
    target_resolution: TargetResolution = Field(
        "1080p",
        description="Target output resolution: '1080p' (1920x1080) or '4k' (3840x2160)",
    )
    high_quality: bool = Field(
        False,
        description=(
            "Use two-pass upscaling for better quality. "
            "Pass 1: 4x upscale, Pass 2: 2x upscale. "
            "Slower but produces sharper results, especially for 4K output."
        ),
    )
    deflicker: bool = Field(
        False,
        description=(
            "Apply temporal deflicker post-processing to reduce frame-to-frame flickering. "
            "Adds ~50% processing time but improves temporal consistency."
        ),
    )
    model: ModelName = Field(
        "realesr-animevideov3",
        description=(
            "Upscaling model to use: "
            "'realesr-animevideov3' (fast, optimized for anime/AI video) or "
            "'realesrgan-x4plus' (slower, better for photorealistic content)"
        ),
    )
    preserve_audio: bool = Field(
        True,
        description="Preserve original audio track in output video",
    )


# =============================================================================
# Upscale Response
# =============================================================================


class UpscaleResponse(BaseModel):
    """Video upscaling response."""

    success: bool = Field(..., description="Whether upscaling succeeded")
    error_message: Optional[str] = Field(None, description="Error message if failed")

    # Output info
    output_url: Optional[str] = Field(None, description="URL to uploaded output video")
    input_resolution: Optional[str] = Field(
        None, description="Input video resolution (e.g., '854x480')"
    )
    output_resolution: Optional[str] = Field(
        None, description="Output video resolution (e.g., '3840x2160')"
    )
    input_duration_ms: Optional[int] = Field(None, description="Input video duration in ms")
    output_duration_ms: Optional[int] = Field(None, description="Output video duration in ms")
    frames_processed: Optional[int] = Field(None, description="Total frames processed")
    output_size_bytes: Optional[int] = Field(None, description="Output file size in bytes")

    # Processing details
    model_used: Optional[str] = Field(None, description="Model used for upscaling")
    upscale_passes: Optional[int] = Field(None, description="Number of upscale passes (1 or 2)")
    deflicker_applied: Optional[bool] = Field(None, description="Whether deflicker was applied")

    # Timing breakdown
    timing_download_ms: Optional[int] = Field(None, description="Time to download input video")
    timing_extract_ms: Optional[int] = Field(None, description="Time to extract frames")
    timing_upscale_ms: Optional[int] = Field(None, description="Time for upscaling passes")
    timing_deflicker_ms: Optional[int] = Field(None, description="Time for deflicker (if applied)")
    timing_encode_ms: Optional[int] = Field(None, description="Time to encode output video")
    timing_upload_ms: Optional[int] = Field(None, description="Time to upload output")
    timing_total_ms: Optional[int] = Field(None, description="Total processing time")


# =============================================================================
# Health Check
# =============================================================================


class HealthResponse(BaseModel):
    """Health check response."""

    status: Literal["healthy", "unhealthy", "loading"] = Field(
        ..., description="Server health status"
    )
    model_loaded: bool = Field(..., description="Whether upscaler model is loaded")
    deflicker_available: bool = Field(
        ..., description="Whether deflicker post-processing is available"
    )
    gpu_available: bool = Field(..., description="Whether GPU is available")
    gpu_name: Optional[str] = Field(None, description="GPU device name")
    gpu_memory_gb: Optional[float] = Field(None, description="GPU memory in GB")
    version: str = Field(..., description="API version")
