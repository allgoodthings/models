"""
Pydantic schemas for Image-Gen API.
"""

from typing import List, Literal, Optional

from pydantic import BaseModel, Field


# =============================================================================
# Health Check
# =============================================================================


class HealthResponse(BaseModel):
    """Health check response."""

    status: Literal["healthy", "loading", "unhealthy"] = Field(
        ..., description="Server status"
    )
    flux_loaded: bool = Field(..., description="Whether FLUX model is loaded")
    gpu_available: bool = Field(..., description="Whether GPU is available")
    gpu_name: Optional[str] = Field(None, description="GPU device name")
    gpu_memory_gb: Optional[float] = Field(None, description="Total GPU memory in GB")
    gpu_memory_used_gb: Optional[float] = Field(None, description="Used GPU memory in GB")


# =============================================================================
# Text-to-Image Generation
# =============================================================================


class GenerateRequest(BaseModel):
    """Text-to-image generation request."""

    prompt: str = Field(
        ...,
        description="Text prompt describing the image to generate",
        min_length=1,
        max_length=2000,
    )
    upload_url: str = Field(
        ...,
        description="Presigned URL for uploading the output image (PUT request)",
    )
    width: int = Field(
        1024,
        description="Output image width in pixels",
        ge=256,
        le=2048,
    )
    height: int = Field(
        1024,
        description="Output image height in pixels",
        ge=256,
        le=2048,
    )
    num_steps: int = Field(
        4,
        description="Number of inference steps (FLUX.2 klein uses 4 steps)",
        ge=1,
        le=50,
    )
    guidance_scale: float = Field(
        1.0,
        description="Classifier-free guidance scale (1.0 = no guidance for distilled models)",
        ge=0.0,
        le=20.0,
    )
    seed: Optional[int] = Field(
        None,
        description="Random seed for reproducibility (None = random)",
    )
    output_format: Literal["png", "jpeg", "webp"] = Field(
        "png",
        description="Output image format",
    )
    upscale: Optional[Literal[2, 4]] = Field(
        None,
        description="Optional upscale factor (2x or 4x) to apply after generation",
    )


class GenerateResponse(BaseModel):
    """Text-to-image generation response."""

    success: bool = Field(..., description="Whether generation succeeded")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    output_url: Optional[str] = Field(
        None, description="URL to uploaded output image (without query params)"
    )
    width: Optional[int] = Field(None, description="Generated image width")
    height: Optional[int] = Field(None, description="Generated image height")
    seed: Optional[int] = Field(None, description="Seed used for generation")
    timing_inference_ms: Optional[int] = Field(
        None, description="Time for image generation"
    )
    timing_upscale_ms: Optional[int] = Field(
        None, description="Time for upscaling (if requested)"
    )
    timing_upload_ms: Optional[int] = Field(None, description="Time to upload image")
    timing_total_ms: Optional[int] = Field(None, description="Total processing time")


# =============================================================================
# Multi-Reference Image Editing
# =============================================================================


class ReferenceImage(BaseModel):
    """Reference image for guided editing."""

    url: str = Field(..., description="URL to reference image")
    weight: float = Field(
        1.0,
        description="Weight/influence of this reference (0.0-2.0)",
        ge=0.0,
        le=2.0,
    )


class EditRequest(BaseModel):
    """Multi-reference image editing request."""

    prompt: str = Field(
        ...,
        description="Text prompt describing the desired edit/output",
        min_length=1,
        max_length=2000,
    )
    reference_images: List[ReferenceImage] = Field(
        ...,
        description="Reference images to guide generation (1-4 images)",
        min_length=1,
        max_length=4,
    )
    upload_url: str = Field(
        ...,
        description="Presigned URL for uploading the output image (PUT request)",
    )
    width: int = Field(
        1024,
        description="Output image width in pixels",
        ge=256,
        le=2048,
    )
    height: int = Field(
        1024,
        description="Output image height in pixels",
        ge=256,
        le=2048,
    )
    num_steps: int = Field(
        4,
        description="Number of inference steps",
        ge=1,
        le=50,
    )
    guidance_scale: float = Field(
        1.0,
        description="Classifier-free guidance scale",
        ge=0.0,
        le=20.0,
    )
    seed: Optional[int] = Field(
        None,
        description="Random seed for reproducibility",
    )
    output_format: Literal["png", "jpeg", "webp"] = Field(
        "png",
        description="Output image format",
    )
    upscale: Optional[Literal[2, 4]] = Field(
        None,
        description="Optional upscale factor (2x or 4x) to apply after generation",
    )


class EditResponse(BaseModel):
    """Multi-reference image editing response."""

    success: bool = Field(..., description="Whether editing succeeded")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    output_url: Optional[str] = Field(
        None, description="URL to uploaded output image"
    )
    references_loaded: Optional[int] = Field(
        None, description="Number of reference images loaded"
    )
    width: Optional[int] = Field(None, description="Generated image width")
    height: Optional[int] = Field(None, description="Generated image height")
    seed: Optional[int] = Field(None, description="Seed used for generation")
    timing_download_ms: Optional[int] = Field(
        None, description="Time to download reference images"
    )
    timing_inference_ms: Optional[int] = Field(
        None, description="Time for image generation"
    )
    timing_upscale_ms: Optional[int] = Field(
        None, description="Time for upscaling (if requested)"
    )
    timing_upload_ms: Optional[int] = Field(None, description="Time to upload image")
    timing_total_ms: Optional[int] = Field(None, description="Total processing time")


# =============================================================================
# Image Upscaling
# =============================================================================


class UpscaleRequest(BaseModel):
    """Image upscaling request."""

    image_url: str = Field(
        ...,
        description="URL of the image to upscale",
    )
    upload_url: str = Field(
        ...,
        description="Presigned URL for uploading the upscaled image (PUT request)",
    )
    scale: Literal[2, 4] = Field(
        4,
        description="Upscale factor (2x or 4x)",
    )
    output_format: Literal["png", "jpeg", "webp"] = Field(
        "png",
        description="Output image format",
    )


class UpscaleResponse(BaseModel):
    """Image upscaling response."""

    success: bool = Field(..., description="Whether upscaling succeeded")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    output_url: Optional[str] = Field(
        None, description="URL to uploaded upscaled image"
    )
    input_width: Optional[int] = Field(None, description="Input image width")
    input_height: Optional[int] = Field(None, description="Input image height")
    output_width: Optional[int] = Field(None, description="Output image width")
    output_height: Optional[int] = Field(None, description="Output image height")
    scale: Optional[int] = Field(None, description="Scale factor used")
    timing_download_ms: Optional[int] = Field(
        None, description="Time to download input image"
    )
    timing_load_ms: Optional[int] = Field(
        None, description="Time to load upscaler model"
    )
    timing_upscale_ms: Optional[int] = Field(
        None, description="Time to upscale image"
    )
    timing_upload_ms: Optional[int] = Field(None, description="Time to upload image")
    timing_total_ms: Optional[int] = Field(None, description="Total processing time")
