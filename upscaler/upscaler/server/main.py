"""
Video Upscaler Server - Real-ESRGAN based video upscaling service.

Usage:
    uvicorn upscaler.server.main:app --host 0.0.0.0 --port 8000
"""

import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager
from typing import Optional

import httpx
import torch
from fastapi import FastAPI, HTTPException

from .schemas import HealthResponse, UpscaleRequest, UpscaleResponse
from ..realesrgan import VideoUpscaler, UpscalerConfig, resize_frames_to_target
from ..deflicker import Deflicker, SimpleDeflicker, get_deflicker, clear_memory
from ..video import (
    get_video_info,
    extract_frames,
    extract_audio,
    frames_to_video,
    calculate_output_resolution,
    calculate_scale_factor,
)
from .. import __version__

# Configure logging
log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("upscaler.server")

# Global instances
upscaler: Optional[VideoUpscaler] = None
deflicker: Optional[Deflicker] = None

# Default models directory
MODELS_DIR = os.environ.get("MODELS_DIR", "/app/models")


async def download_file(url: str, dest_path: str) -> float:
    """Download a file from URL. Returns download time in ms."""
    logger.info(f"Downloading from {url[:80]}...")
    start_time = time.time()

    async with httpx.AsyncClient(timeout=600.0, follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()

        with open(dest_path, "wb") as f:
            f.write(response.content)

    elapsed_ms = (time.time() - start_time) * 1000
    file_size = os.path.getsize(dest_path) / (1024 * 1024)
    logger.info(f"  Downloaded {file_size:.2f}MB in {elapsed_ms:.0f}ms")

    return elapsed_ms


async def upload_file(url: str, file_path: str) -> float:
    """Upload a file to a presigned URL using PUT. Returns upload time in ms."""
    logger.info(f"Uploading to presigned URL...")
    start_time = time.time()

    file_size = os.path.getsize(file_path)

    async with httpx.AsyncClient(timeout=600.0) as client:
        with open(file_path, "rb") as f:
            response = await client.put(
                url,
                content=f.read(),
                headers={"Content-Type": "video/mp4"},
            )
            response.raise_for_status()

    elapsed_ms = (time.time() - start_time) * 1000
    logger.info(f"  Uploaded {file_size / (1024*1024):.2f}MB in {elapsed_ms:.0f}ms")

    return elapsed_ms


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, cleanup on shutdown."""
    global upscaler, deflicker

    logger.info("=" * 60)
    logger.info("STARTING VIDEO UPSCALER SERVER")
    logger.info("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f}GB)")

    skip_preload = os.environ.get("PRELOAD_MODELS", "true").lower() in ("false", "0", "no")

    if skip_preload:
        logger.info("PRELOAD_MODELS=false - Models will be loaded on first request")
        yield
        return

    logger.info("-" * 40)
    logger.info("[1/2] Loading Real-ESRGAN upscaler...")

    config = UpscalerConfig(
        model_name="realesr-animevideov3",
        cache_dir=MODELS_DIR,
    )
    upscaler = VideoUpscaler(config)
    upscaler.load()

    logger.info("[1/2] Real-ESRGAN loaded successfully")

    logger.info("-" * 40)
    logger.info("[2/2] Checking deflicker availability...")

    deflicker = get_deflicker()
    if deflicker:
        logger.info("[2/2] Deflicker available")
    else:
        logger.info("[2/2] Deflicker not available (optional)")

    logger.info("-" * 40)
    logger.info("SERVER READY")
    logger.info("=" * 60)

    yield

    logger.info("Shutting down server...")
    if upscaler:
        upscaler.unload()


app = FastAPI(
    title="Video Upscaler API",
    description="Real-ESRGAN based video upscaling with optional temporal deflicker",
    version=__version__,
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    gpu_available = torch.cuda.is_available()
    gpu_name = None
    gpu_memory_gb = None

    if gpu_available:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)

    model_loaded = upscaler is not None and upscaler.is_loaded
    deflicker_available = deflicker is not None and deflicker.is_available

    if model_loaded:
        status = "healthy"
    elif upscaler is not None:
        status = "loading"
    else:
        status = "unhealthy"

    return HealthResponse(
        status=status,
        model_loaded=model_loaded,
        deflicker_available=deflicker_available,
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        gpu_memory_gb=gpu_memory_gb,
        version=__version__,
    )


@app.post("/upscale", response_model=UpscaleResponse)
async def upscale(request: UpscaleRequest):
    """
    Upscale a video using Real-ESRGAN.

    Process:
    1. Download input video
    2. Extract frames
    3. Upscale frames (single or two-pass)
    4. Optional: Apply deflicker
    5. Encode output video
    6. Upload to presigned URL
    """
    global upscaler

    # Lazy load if not preloaded
    if upscaler is None or not upscaler.is_loaded:
        logger.info("Loading upscaler on first request...")
        config = UpscalerConfig(
            model_name=request.model,
            cache_dir=MODELS_DIR,
        )
        upscaler = VideoUpscaler(config)
        upscaler.load()

    total_start = time.time()
    timing = {
        "download_ms": 0,
        "extract_ms": 0,
        "upscale_ms": 0,
        "deflicker_ms": 0,
        "encode_ms": 0,
        "upload_ms": 0,
    }

    logger.info("=" * 60)
    logger.info("VIDEO UPSCALE REQUEST")
    logger.info("=" * 60)
    logger.info(f"  Target: {request.target_resolution}")
    logger.info(f"  High Quality: {request.high_quality}")
    logger.info(f"  Deflicker: {request.deflicker}")
    logger.info(f"  Model: {request.model}")

    with tempfile.TemporaryDirectory() as tmpdir:
        input_video = os.path.join(tmpdir, "input.mp4")
        frames_dir = os.path.join(tmpdir, "frames")
        upscaled_dir = os.path.join(tmpdir, "upscaled")
        final_frames_dir = os.path.join(tmpdir, "final_frames")
        audio_file = os.path.join(tmpdir, "audio.aac")
        output_video = os.path.join(tmpdir, "output.mp4")

        # 1. Download input video
        try:
            timing["download_ms"] = int(await download_file(request.video_url, input_video))
        except Exception as e:
            logger.error(f"Download failed: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to download video: {e}")

        # Get video info
        try:
            video_info = get_video_info(input_video)
            logger.info(f"  Input: {video_info.resolution} @ {video_info.fps:.2f}fps, {video_info.frame_count} frames")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid video: {e}")

        # Calculate output resolution
        output_width, output_height = calculate_output_resolution(
            video_info.width,
            video_info.height,
            request.target_resolution,
        )
        scale_factor = calculate_scale_factor(
            video_info.width,
            video_info.height,
            output_width,
            output_height,
        )
        logger.info(f"  Output: {output_width}x{output_height} (scale: {scale_factor:.2f}x)")

        # 2. Extract frames
        extract_start = time.time()
        try:
            frame_count, fps = extract_frames(input_video, frames_dir)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Frame extraction failed: {e}")
        timing["extract_ms"] = int((time.time() - extract_start) * 1000)

        # Extract audio if requested
        has_audio = False
        if request.preserve_audio:
            has_audio = extract_audio(input_video, audio_file)

        # 3. Upscale frames
        upscale_start = time.time()
        try:
            # Upscale slightly larger than target, then resize down for quality
            upscale_factor = scale_factor * 1.05  # 5% overscale
            upscale_factor = min(upscale_factor, 8.0)  # Cap at 8x

            frames_processed = upscaler.upscale_directory(
                input_dir=frames_dir,
                output_dir=upscaled_dir,
                outscale=upscale_factor,
                high_quality=request.high_quality,
            )
            logger.info(f"  Upscaled {frames_processed} frames")
        except Exception as e:
            logger.error(f"Upscaling failed: {e}")
            raise HTTPException(status_code=500, detail=f"Upscaling failed: {e}")
        timing["upscale_ms"] = int((time.time() - upscale_start) * 1000)

        # 4. Resize to exact target resolution
        resize_frames_to_target(upscaled_dir, final_frames_dir, output_width, output_height)
        current_frames_dir = final_frames_dir

        # 5. Optional deflicker
        deflicker_applied = False
        if request.deflicker:
            deflicker_start = time.time()

            if deflicker and deflicker.is_available:
                logger.info("  Applying neural deflicker...")
                # For neural deflicker, we need to encode a temp video first
                temp_video = os.path.join(tmpdir, "temp_for_deflicker.mp4")
                frames_to_video(current_frames_dir, temp_video, fps)

                deflickered_video = os.path.join(tmpdir, "deflickered.mp4")
                if deflicker.process_video(temp_video, deflickered_video):
                    # Re-extract frames from deflickered video
                    deflickered_frames = os.path.join(tmpdir, "deflickered_frames")
                    extract_frames(deflickered_video, deflickered_frames)
                    current_frames_dir = deflickered_frames
                    deflicker_applied = True
                    logger.info("  Neural deflicker applied")
                else:
                    logger.warning("  Neural deflicker failed, using simple fallback")
            else:
                logger.info("  Applying simple temporal smoothing...")
                simple_deflicker = SimpleDeflicker(window_size=5)
                smoothed_dir = os.path.join(tmpdir, "smoothed")
                simple_deflicker.process_frames(current_frames_dir, smoothed_dir)
                current_frames_dir = smoothed_dir
                deflicker_applied = True
                logger.info("  Simple deflicker applied")

            timing["deflicker_ms"] = int((time.time() - deflicker_start) * 1000)
            clear_memory()

        # 6. Encode output video
        encode_start = time.time()
        try:
            frames_to_video(
                frames_dir=current_frames_dir,
                output_path=output_video,
                fps=fps,
                audio_path=audio_file if has_audio else None,
                crf=18,
                preset="medium",
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Video encoding failed: {e}")
        timing["encode_ms"] = int((time.time() - encode_start) * 1000)

        # 7. Upload output
        try:
            timing["upload_ms"] = int(await upload_file(request.upload_url, output_video))
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            raise HTTPException(status_code=500, detail=f"Upload failed: {e}")

        # Get output info
        output_info = get_video_info(output_video)
        output_size = os.path.getsize(output_video)

    total_ms = int((time.time() - total_start) * 1000)

    logger.info("-" * 40)
    logger.info(f"  Total time: {total_ms}ms")
    logger.info(f"  Output size: {output_size / (1024*1024):.1f}MB")
    logger.info("=" * 60)

    return UpscaleResponse(
        success=True,
        output_url=request.upload_url.split("?")[0],
        input_resolution=video_info.resolution,
        output_resolution=f"{output_width}x{output_height}",
        input_duration_ms=video_info.duration_ms,
        output_duration_ms=output_info.duration_ms,
        frames_processed=frame_count,
        output_size_bytes=output_size,
        model_used=request.model,
        upscale_passes=2 if request.high_quality else 1,
        deflicker_applied=deflicker_applied,
        timing_download_ms=timing["download_ms"],
        timing_extract_ms=timing["extract_ms"],
        timing_upscale_ms=timing["upscale_ms"],
        timing_deflicker_ms=timing["deflicker_ms"] if deflicker_applied else None,
        timing_encode_ms=timing["encode_ms"],
        timing_upload_ms=timing["upload_ms"],
        timing_total_ms=total_ms,
    )


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Video Upscaler API",
        "version": __version__,
        "docs": "/docs",
        "health": "/health",
        "endpoints": [
            "POST /upscale - Upscale video with Real-ESRGAN",
            "GET /health - Health check",
        ],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "upscaler.server.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
