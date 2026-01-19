"""
Image-Gen Server - FLUX.2 klein image generation.

Usage:
    uvicorn imagegen.server.main:app --host 0.0.0.0 --port 7000
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from io import BytesIO
from typing import List, Optional, Tuple

import httpx
import torch
from fastapi import FastAPI, HTTPException
from PIL import Image

from .flux_pipeline import FluxConfig, FluxPipeline, get_content_type, image_to_bytes
from .schemas import (
    BatchImageRequest,
    BatchImageResponse,
    HealthResponse,
    ImageResult,
    ImageSpec,
    UpscaleRequest,
    UpscaleResponse,
)
from .upscaler import Upscaler, UpscalerConfig

# Logging setup
log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("imagegen.server")

# Global pipeline
pipeline: Optional[FluxPipeline] = None


# =============================================================================
# Helper Functions
# =============================================================================


async def download_image(url: str) -> Image.Image:
    """Download image from URL."""
    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        return image


async def upload_image(url: str, image_bytes: bytes, content_type: str) -> None:
    """Upload image to presigned URL."""
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.put(url, content=image_bytes, headers={"Content-Type": content_type})
        response.raise_for_status()


async def download_references(urls: List[str]) -> List[Tuple[Image.Image, float]]:
    """Download reference images from URLs."""
    refs = []
    for url in urls:
        img = await download_image(url)
        refs.append((img, 1.0))
    return refs


def get_base_seed(seed: Optional[int]) -> int:
    """Get base seed, generating random if None."""
    if seed is not None:
        return seed
    return torch.randint(0, 2**32 - 1, (1,)).item()


# =============================================================================
# Lifespan
# =============================================================================


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
        props = torch.cuda.get_device_properties(0)
        logger.info(f"GPU: {props.name} ({props.total_memory / 1024**3:.1f}GB)")

    hf_token = os.environ.get("HUGGING_FACE_TOKEN") or os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HUGGING_FACE_TOKEN required")

    if os.environ.get("PRELOAD_MODELS", "true").lower() in ("false", "0", "no"):
        logger.info("PRELOAD_MODELS=false - Skipping model loading")
        yield
        return

    quantization = os.environ.get("QUANTIZATION", "none")
    if quantization not in ("none", "fp8", "int8"):
        quantization = "none"

    logger.info(f"Loading FLUX model (quantization={quantization})...")
    pipeline = FluxPipeline(FluxConfig(hf_token=hf_token, quantization=quantization))
    pipeline.load()
    logger.info("MODEL LOADED - Server ready")

    yield

    if pipeline:
        pipeline.unload()


app = FastAPI(
    title="Image-Gen API",
    description="FLUX.2 klein batch image generation",
    version="0.3.0",
    lifespan=lifespan,
)


# =============================================================================
# Endpoints
# =============================================================================


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check."""
    gpu_available = torch.cuda.is_available()
    gpu_name = gpu_memory_gb = gpu_memory_used_gb = None

    if gpu_available:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory_gb = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
        gpu_memory_used_gb = round(torch.cuda.memory_allocated(0) / 1024**3, 2)

    flux_loaded = pipeline is not None and pipeline.is_loaded
    status = "healthy" if flux_loaded else ("loading" if pipeline else "unhealthy")

    return HealthResponse(
        status=status,
        flux_loaded=flux_loaded,
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        gpu_memory_gb=gpu_memory_gb,
        gpu_memory_used_gb=gpu_memory_used_gb,
    )


@app.post("/generate", response_model=BatchImageResponse)
async def generate(request: BatchImageRequest):
    """Generate batch of images."""
    if pipeline is None or not pipeline.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")

    logger.info("=" * 60)
    logger.info(f"GENERATE: {len(request.images)} images, sequential={request.sequential}")
    logger.info("=" * 60)

    base_seed = get_base_seed(request.config.seed)
    results: List[ImageResult] = []
    prev_output: Optional[Image.Image] = None

    for i, (spec, upload_url) in enumerate(zip(request.images, request.uploadUrls)):
        result = await generate_single(
            index=i,
            spec=spec,
            upload_url=upload_url,
            config=request.config,
            seed=base_seed + i,
            prev_output=prev_output if request.sequential else None,
        )
        results.append(result)

        # Store output for next iteration in sequential mode
        if request.sequential and result.success and hasattr(result, "_output_image"):
            prev_output = result._output_image

    logger.info(f"Complete: {sum(r.success for r in results)}/{len(results)}")
    return BatchImageResponse(results=results)


async def generate_single(
    index: int,
    spec: ImageSpec,
    upload_url: str,
    config,
    seed: int,
    prev_output: Optional[Image.Image],
) -> ImageResult:
    """Generate a single image."""
    logger.info(f"[{index}] {spec.prompt[:60]}...")

    try:
        # Build reference images list
        ref_images: List[Tuple[Image.Image, float]] = []

        # In sequential mode, prepend previous output as first reference
        if prev_output is not None:
            ref_images.append((prev_output, 1.0))

        # Download reference images from URLs
        if spec.referenceImageUrls:
            logger.info(f"[{index}] Downloading {len(spec.referenceImageUrls)} refs...")
            downloaded = await download_references(spec.referenceImageUrls)
            ref_images.extend(downloaded)

        # Generate
        start = time.time()
        if ref_images:
            image, _ = pipeline.edit(
                prompt=spec.prompt,
                reference_images=ref_images,
                width=config.width,
                height=config.height,
                num_steps=config.steps,
                guidance_scale=config.guidanceScale,
                seed=seed,
            )
        else:
            image, _ = pipeline.generate(
                prompt=spec.prompt,
                width=config.width,
                height=config.height,
                num_steps=config.steps,
                guidance_scale=config.guidanceScale,
                seed=seed,
            )
        gen_ms = int((time.time() - start) * 1000)

        # Upload
        image_bytes = image_to_bytes(image, "png")
        content_type = get_content_type("png")
        await upload_image(upload_url, image_bytes, content_type)

        logger.info(f"[{index}] Done: {gen_ms}ms")

        # Create result and attach output image for sequential mode
        result = ImageResult(index=index, success=True)
        result._output_image = image  # Attach for sequential chaining
        return result

    except Exception as e:
        logger.error(f"[{index}] Failed: {e}")
        return ImageResult(index=index, success=False, error=str(e))


@app.post("/upscale", response_model=UpscaleResponse)
async def upscale_endpoint(request: UpscaleRequest):
    """Upscale an image using Real-ESRGAN."""
    total_start = time.time()
    logger.info(f"UPSCALE: {request.scale}x")

    try:
        # Download
        download_start = time.time()
        image = await download_image(request.image_url)
        input_w, input_h = image.size
        download_ms = int((time.time() - download_start) * 1000)

        # Load upscaler
        load_start = time.time()
        upscaler = Upscaler(UpscalerConfig())
        upscaler.load()
        load_ms = int((time.time() - load_start) * 1000)

        # Upscale
        upscale_start = time.time()
        upscaled = upscaler.upscale(image, scale=request.scale)
        output_w, output_h = upscaled.size
        upscale_ms = int((time.time() - upscale_start) * 1000)

        upscaler.unload()

        # Upload
        def extract_base_url(url: str) -> str:
            return url.split("?")[0]

        image_bytes = image_to_bytes(upscaled, request.output_format)
        content_type = get_content_type(request.output_format)
        upload_start = time.time()
        await upload_image(request.upload_url, image_bytes, content_type)
        upload_ms = int((time.time() - upload_start) * 1000)

        total_ms = int((time.time() - total_start) * 1000)

        return UpscaleResponse(
            success=True,
            output_url=extract_base_url(request.upload_url),
            input_width=input_w,
            input_height=input_h,
            output_width=output_w,
            output_height=output_h,
            scale=request.scale,
            timing_download_ms=download_ms,
            timing_load_ms=load_ms,
            timing_upscale_ms=upscale_ms,
            timing_upload_ms=upload_ms,
            timing_total_ms=total_ms,
        )

    except Exception as e:
        logger.error(f"Upscale failed: {e}")
        return UpscaleResponse(success=False, error=str(e))


@app.get("/")
async def root():
    """API info."""
    return {
        "name": "Image-Gen API",
        "version": "0.3.0",
        "endpoints": [
            "POST /generate - Batch image generation",
            "POST /upscale - Image upscaling (2x/4x)",
            "GET /health - Health check",
        ],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("imagegen.server.main:app", host="0.0.0.0", port=7000, reload=True)
