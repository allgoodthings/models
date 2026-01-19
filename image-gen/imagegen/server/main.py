"""
Image-Gen Server - FLUX.2 klein image generation.

Usage:
    uvicorn imagegen.server.main:app --host 0.0.0.0 --port 7000
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import httpx
import torch
from fastapi import FastAPI, HTTPException
from PIL import Image

from .flux_pipeline import FluxConfig, FluxPipeline, get_content_type, image_to_bytes
from .schemas import (
    EditRequest,
    EditResponse,
    GenerateRequest,
    GenerateResponse,
    HealthResponse,
)

# Configure logging
log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("imagegen.server")

# Global pipeline instance
pipeline: Optional[FluxPipeline] = None


async def download_image(url: str) -> Image.Image:
    """Download image from URL and return as PIL Image."""
    logger.debug(f"Downloading image from {url}")

    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()

        from io import BytesIO

        image = Image.open(BytesIO(response.content))
        # Convert to RGB if necessary (handles RGBA, palette, etc.)
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")

        return image


async def upload_image(url: str, image_bytes: bytes, content_type: str) -> float:
    """Upload image to presigned URL. Returns upload time in ms."""
    logger.info(f"Uploading image ({len(image_bytes) / 1024:.1f}KB)...")
    start_time = time.time()

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.put(
            url,
            content=image_bytes,
            headers={"Content-Type": content_type},
        )
        response.raise_for_status()

    elapsed_ms = (time.time() - start_time) * 1000
    logger.info(f"  Uploaded in {elapsed_ms:.0f}ms")

    return elapsed_ms


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    global pipeline

    logger.info("=" * 60)
    logger.info("STARTING IMAGE-GEN SERVER")
    logger.info("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f}GB)")

    # Check for HF token
    hf_token = os.environ.get("HUGGING_FACE_TOKEN") or os.environ.get("HF_TOKEN")
    if not hf_token:
        logger.error("HUGGING_FACE_TOKEN environment variable is required!")
        logger.error("Get your token from: https://huggingface.co/settings/tokens")
        raise RuntimeError("HUGGING_FACE_TOKEN required")

    skip_load = os.environ.get("PRELOAD_MODELS", "true").lower() in ("false", "0", "no")

    if skip_load:
        logger.info("PRELOAD_MODELS=false - Skipping model loading")
        yield
        return

    logger.info("-" * 40)
    logger.info("Loading FLUX model...")

    # Quantization mode: "none" (BF16, needs 32GB), "fp8" (24GB), "int8" (20GB)
    quantization = os.environ.get("QUANTIZATION", "none")
    if quantization not in ("none", "fp8", "int8"):
        logger.warning(f"Invalid QUANTIZATION={quantization}, using 'none'")
        quantization = "none"

    logger.info(f"Quantization mode: {quantization}")

    config = FluxConfig(
        hf_token=hf_token,
        quantization=quantization,  # type: ignore
    )
    pipeline = FluxPipeline(config)
    pipeline.load()

    logger.info("-" * 40)
    logger.info("MODEL LOADED - Server ready")
    logger.info("=" * 60)

    yield

    logger.info("Shutting down server...")
    if pipeline is not None:
        pipeline.unload()


app = FastAPI(
    title="Image-Gen API",
    description="FLUX.2 klein image generation service",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    gpu_available = torch.cuda.is_available()
    gpu_name = None
    gpu_memory_gb = None
    gpu_memory_used_gb = None

    if gpu_available:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
        gpu_memory_used_gb = round(torch.cuda.memory_allocated(0) / (1024**3), 2)

    flux_loaded = pipeline is not None and pipeline.is_loaded

    if flux_loaded:
        status = "healthy"
    elif pipeline is not None:
        status = "loading"
    else:
        status = "unhealthy"

    return HealthResponse(
        status=status,
        flux_loaded=flux_loaded,
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        gpu_memory_gb=gpu_memory_gb,
        gpu_memory_used_gb=gpu_memory_used_gb,
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """Generate an image from a text prompt."""
    if pipeline is None or not pipeline.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    total_start = time.time()

    logger.info("=" * 60)
    logger.info("GENERATE IMAGE")
    logger.info("=" * 60)
    logger.info(f"  Prompt: {request.prompt[:100]}...")
    logger.info(f"  Size: {request.width}x{request.height}")
    logger.info(f"  Steps: {request.num_steps}")
    logger.info(f"  Format: {request.output_format}")

    try:
        # Generate image
        inference_start = time.time()
        image, seed = pipeline.generate(
            prompt=request.prompt,
            width=request.width,
            height=request.height,
            num_steps=request.num_steps,
            guidance_scale=request.guidance_scale,
            seed=request.seed,
        )
        inference_ms = int((time.time() - inference_start) * 1000)
        logger.info(f"  Generated in {inference_ms}ms, seed={seed}")

        # Convert to bytes
        image_bytes = image_to_bytes(image, request.output_format)
        content_type = get_content_type(request.output_format)

        # Upload
        upload_ms = int(await upload_image(request.upload_url, image_bytes, content_type))

        total_ms = int((time.time() - total_start) * 1000)
        logger.info(f"  Complete in {total_ms}ms")

        return GenerateResponse(
            success=True,
            output_url=request.upload_url.split("?")[0],
            width=request.width,
            height=request.height,
            seed=seed,
            timing_inference_ms=inference_ms,
            timing_upload_ms=upload_ms,
            timing_total_ms=total_ms,
        )

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return GenerateResponse(
            success=False,
            error_message=str(e),
        )


@app.post("/edit", response_model=EditResponse)
async def edit(request: EditRequest):
    """Edit/generate an image using reference images."""
    if pipeline is None or not pipeline.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    total_start = time.time()

    logger.info("=" * 60)
    logger.info("EDIT IMAGE")
    logger.info("=" * 60)
    logger.info(f"  Prompt: {request.prompt[:100]}...")
    logger.info(f"  References: {len(request.reference_images)}")
    logger.info(f"  Size: {request.width}x{request.height}")

    try:
        # Download reference images
        download_start = time.time()
        reference_data = []

        for ref in request.reference_images:
            logger.info(f"  Downloading reference: {ref.url[:50]}...")
            image = await download_image(ref.url)
            reference_data.append((image, ref.weight))

        download_ms = int((time.time() - download_start) * 1000)
        logger.info(f"  Downloaded {len(reference_data)} references in {download_ms}ms")

        # Generate with references
        inference_start = time.time()
        image, seed = pipeline.edit(
            prompt=request.prompt,
            reference_images=reference_data,
            width=request.width,
            height=request.height,
            num_steps=request.num_steps,
            guidance_scale=request.guidance_scale,
            seed=request.seed,
        )
        inference_ms = int((time.time() - inference_start) * 1000)
        logger.info(f"  Generated in {inference_ms}ms, seed={seed}")

        # Convert to bytes
        image_bytes = image_to_bytes(image, request.output_format)
        content_type = get_content_type(request.output_format)

        # Upload
        upload_ms = int(await upload_image(request.upload_url, image_bytes, content_type))

        total_ms = int((time.time() - total_start) * 1000)
        logger.info(f"  Complete in {total_ms}ms")

        return EditResponse(
            success=True,
            output_url=request.upload_url.split("?")[0],
            references_loaded=len(reference_data),
            width=request.width,
            height=request.height,
            seed=seed,
            timing_download_ms=download_ms,
            timing_inference_ms=inference_ms,
            timing_upload_ms=upload_ms,
            timing_total_ms=total_ms,
        )

    except Exception as e:
        logger.error(f"Edit failed: {e}")
        return EditResponse(
            success=False,
            error_message=str(e),
        )


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Image-Gen API",
        "version": "0.1.0",
        "model": "black-forest-labs/FLUX.2-klein-9B",
        "docs": "/docs",
        "health": "/health",
        "endpoints": [
            "POST /generate - Text-to-image generation",
            "POST /edit - Multi-reference image editing (1-4 images)",
            "GET /health - Health check",
        ],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "imagegen.server.main:app",
        host="0.0.0.0",
        port=7000,
        reload=True,
    )
