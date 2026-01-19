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
# Image Generation (Batch)
# =============================================================================


class ReferenceImage(BaseModel):
    """Reference image for guided generation."""

    url: str
    weight: float = Field(1.0, ge=0.0, le=2.0)


class GenerateRequest(BaseModel):
    """Batch image generation request."""

    prompts: List[str] = Field(..., min_length=1, max_length=20)
    upload_urls: List[str] = Field(..., min_length=1, max_length=20)
    images: List[ReferenceImage] = Field(default_factory=list, max_length=16)
    width: int = Field(1024, ge=256, le=2048)
    height: int = Field(1024, ge=256, le=2048)
    num_steps: int = Field(4, ge=1, le=50)
    guidance_scale: float = Field(1.0, ge=0.0, le=20.0)
    seed: Optional[int] = None
    upscale: Optional[Literal[2, 4]] = None
    output_format: Literal["png", "jpeg", "webp"] = "png"

    @model_validator(mode="after")
    def validate_lengths(self):
        if len(self.prompts) != len(self.upload_urls):
            raise ValueError("prompts and upload_urls must have same length")
        return self


class GenerateResult(BaseModel):
    """Result for a single generated image."""

    index: int
    success: bool
    error: Optional[str] = None
    output_url: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    seed: Optional[int] = None
    timing_inference_ms: Optional[int] = None
    timing_upscale_ms: Optional[int] = None
    timing_upload_ms: Optional[int] = None


class GenerateResponse(BaseModel):
    """Batch image generation response."""

    success: bool
    results: List[GenerateResult]
    timing_total_ms: int


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
