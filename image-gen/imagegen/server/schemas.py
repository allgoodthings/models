"""Pydantic schemas for Image-Gen API."""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field, model_validator


# =============================================================================
# Health Check
# =============================================================================


class HealthResponse(BaseModel):
    """Health check response."""

    status: Literal["healthy", "loading", "unhealthy"]
    flux_loaded: bool
    gpu_available: bool
    gpu_name: Optional[str] = None
    gpu_memory_gb: Optional[float] = None
    gpu_memory_used_gb: Optional[float] = None


# =============================================================================
# Batch Image Generation
# =============================================================================


class ImageSpec(BaseModel):
    """Specification for a single image to generate."""

    prompt: str
    referenceImageUrls: List[str] = Field(default_factory=list)


class GenerationConfig(BaseModel):
    """Configuration for image generation."""

    width: int = Field(1024, ge=256, le=2048)
    height: int = Field(1024, ge=256, le=2048)
    seed: Optional[int] = None
    steps: int = Field(4, ge=1, le=50)
    guidanceScale: float = Field(1.0, ge=0.0, le=20.0)


class BatchImageRequest(BaseModel):
    """Batch image generation request."""

    images: List[ImageSpec] = Field(..., min_length=1, max_length=20)
    uploadUrls: List[str] = Field(..., min_length=1, max_length=20)
    config: GenerationConfig
    sequential: bool = False

    @model_validator(mode="after")
    def validate_lengths(self):
        if len(self.images) != len(self.uploadUrls):
            raise ValueError("images and uploadUrls must have same length")
        return self


class ImageResult(BaseModel):
    """Result for a single generated image."""

    index: int
    success: bool
    error: Optional[str] = None


class BatchImageResponse(BaseModel):
    """Batch image generation response."""

    results: List[ImageResult]


# =============================================================================
# Image Upscaling (Standalone)
# =============================================================================


class UpscaleRequest(BaseModel):
    """Image upscaling request."""

    image_url: str
    upload_url: str
    scale: Literal[2, 4] = 4
    output_format: Literal["png", "jpeg", "webp"] = "png"


class UpscaleResponse(BaseModel):
    """Image upscaling response."""

    success: bool
    error: Optional[str] = None
    output_url: Optional[str] = None
    input_width: Optional[int] = None
    input_height: Optional[int] = None
    output_width: Optional[int] = None
    output_height: Optional[int] = None
    scale: Optional[int] = None
    timing_download_ms: Optional[int] = None
    timing_load_ms: Optional[int] = None
    timing_upscale_ms: Optional[int] = None
    timing_upload_ms: Optional[int] = None
    timing_total_ms: Optional[int] = None
