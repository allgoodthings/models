"""
Lip-Sync V2 Server - Multi-face lip-sync with Wav2Lip-HD.

Usage:
    uvicorn lipsync.server.main:app --host 0.0.0.0 --port 8000
"""

# =============================================================================
# TORCHVISION COMPATIBILITY FIX (must be at the very top, before any imports)
# =============================================================================
# basicsr 1.4.2 imports from torchvision.transforms.functional_tensor, which was
# removed in torchvision 0.18+. This fix creates a dummy module so basicsr works.
# See: https://github.com/XPixelGroup/BasicSR/pull/677
import sys
import types

try:
    from torchvision.transforms.functional_tensor import rgb_to_grayscale
except ImportError:
    from torchvision.transforms.functional import rgb_to_grayscale
    # Create a fake module for backward compatibility
    _functional_tensor = types.ModuleType("torchvision.transforms.functional_tensor")
    _functional_tensor.rgb_to_grayscale = rgb_to_grayscale
    sys.modules["torchvision.transforms.functional_tensor"] = _functional_tensor
# =============================================================================

import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager
from typing import List, Optional

import cv2
import httpx
import numpy as np
import torch
from fastapi import FastAPI, HTTPException

from .insightface_detector import InsightFaceDetector
from .schemas import (
    CharacterReference,
    HealthResponse,
    LipSyncRequest,
    LipSyncResponse,
    FaceTrackingRequest,
    FaceTrackingResponse,
)
from ..tracker import FaceTracker, generate_tracking_video, bbox_to_hex_color
from ..wav2lip import Wav2LipHD, Wav2LipConfig
from ..loop_handler import LoopHandler, LoopConfig

# Configure logging
log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("lipsync.server")

# Global instances
face_detector: Optional[InsightFaceDetector] = None
wav2lip: Optional[Wav2LipHD] = None

# Default models directory
MODELS_DIR = os.environ.get("MODELS_DIR", "/app/models")

# Model download URLs (HuggingFace mirrors - more reliable than SharePoint)
MODEL_URLS = {
    "wav2lip_gan.pth": "https://huggingface.co/numz/wav2lip_studio/resolve/main/Wav2lip/wav2lip_gan.pth",
    "GFPGANv1.4.pth": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth",
}

# Wav2Lip's S3FD face detection model (different path)
S3FD_MODEL_URL = "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth"
S3FD_MODEL_PATH = "/app/vendors/wav2lip-hd/Wav2Lip-master/face_detection/detection/sfd/s3fd.pth"


async def ensure_models_downloaded():
    """Download model files if they don't exist."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    # Download main models (wav2lip, gfpgan)
    for filename, url in MODEL_URLS.items():
        model_path = os.path.join(MODELS_DIR, filename)

        if os.path.exists(model_path):
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            logger.info(f"  Model exists: {filename} ({size_mb:.1f}MB)")
            continue

        logger.info(f"  Downloading {filename} from {url}...")
        try:
            await download_file(url, model_path)
            size_mb = os.path.getsize(model_path) / (1024 * 1024)
            logger.info(f"  Downloaded {filename} ({size_mb:.1f}MB)")
        except Exception as e:
            logger.error(f"  Failed to download {filename}: {e}")
            raise RuntimeError(f"Failed to download required model {filename}: {e}")

    # Download S3FD face detection model for Wav2Lip
    if not os.path.exists(S3FD_MODEL_PATH):
        logger.info(f"  Downloading S3FD face detection model...")
        os.makedirs(os.path.dirname(S3FD_MODEL_PATH), exist_ok=True)
        try:
            await download_file(S3FD_MODEL_URL, S3FD_MODEL_PATH)
            size_mb = os.path.getsize(S3FD_MODEL_PATH) / (1024 * 1024)
            logger.info(f"  Downloaded s3fd.pth ({size_mb:.1f}MB)")
        except Exception as e:
            logger.error(f"  Failed to download S3FD model: {e}")
            raise RuntimeError(f"Failed to download S3FD face detection model: {e}")
    else:
        size_mb = os.path.getsize(S3FD_MODEL_PATH) / (1024 * 1024)
        logger.info(f"  Model exists: s3fd.pth ({size_mb:.1f}MB)")


async def download_file(url: str, dest_path: str) -> float:
    """Download a file from URL. Returns download time in ms."""
    logger.info(f"Downloading file from {url}")
    start_time = time.time()

    async with httpx.AsyncClient(timeout=300.0, follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()

        with open(dest_path, "wb") as f:
            f.write(response.content)

    elapsed_ms = (time.time() - start_time) * 1000
    file_size = os.path.getsize(dest_path) / (1024 * 1024)
    logger.info(f"  Downloaded {file_size:.2f}MB in {elapsed_ms:.0f}ms")

    return elapsed_ms


async def download_image(url: str) -> np.ndarray:
    """Download image from URL and return as numpy array (BGR)."""
    logger.debug(f"Downloading image from {url}")

    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()

        img_array = np.frombuffer(response.content, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError(f"Failed to decode image from {url}")

        return img


async def upload_file(url: str, file_path: str) -> float:
    """Upload a file to a presigned URL using PUT. Returns upload time in ms."""
    logger.info(f"Uploading file to presigned URL...")
    start_time = time.time()

    file_size = os.path.getsize(file_path)

    async with httpx.AsyncClient(timeout=300.0) as client:
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


def get_video_info(video_path: str) -> dict:
    """Get video duration, dimensions, and fps using OpenCV."""
    cap = cv2.VideoCapture(video_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_ms = int((frame_count / fps) * 1000) if fps > 0 else 0

    cap.release()

    return {
        "duration_ms": duration_ms,
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
    }


def get_audio_duration(audio_path: str) -> float:
    """Get audio duration in seconds using ffprobe."""
    import subprocess

    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        audio_path
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except Exception as e:
        logger.warning(f"Failed to get audio duration: {e}")
        # Fallback: try to read with cv2 (works for some formats)
        cap = cv2.VideoCapture(audio_path)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()
            if fps > 0:
                return frame_count / fps
        return 10.0  # Default fallback


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, cleanup on shutdown."""
    global face_detector, wav2lip

    logger.info("=" * 60)
    logger.info("STARTING LIP-SYNC V2 SERVER")
    logger.info("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f}GB)")

    skip_load = os.environ.get("PRELOAD_MODELS", "true").lower() in ("false", "0", "no")

    if skip_load:
        logger.info("PRELOAD_MODELS=false - Skipping model loading")
        yield
        return

    logger.info("-" * 40)
    logger.info("[0/2] Ensuring models are downloaded...")
    await ensure_models_downloaded()

    logger.info("-" * 40)
    logger.info("[1/2] Loading InsightFace...")
    face_detector = InsightFaceDetector(model_name="buffalo_l", device=device)
    face_detector.load()
    logger.info("[1/2] InsightFace loaded successfully")

    logger.info("-" * 40)
    logger.info("[2/2] Initializing Wav2Lip-HD...")
    wav2lip_config = Wav2LipConfig(
        checkpoint_path=os.path.join(MODELS_DIR, "wav2lip_gan.pth"),
        gfpgan_checkpoint=os.path.join(MODELS_DIR, "GFPGANv1.4.pth"),
    )
    wav2lip = Wav2LipHD(wav2lip_config)
    logger.info("[2/2] Wav2Lip-HD initialized")

    logger.info("-" * 40)
    logger.info("ALL MODELS LOADED - Server ready")
    logger.info("=" * 60)

    yield

    logger.info("Shutting down server...")


app = FastAPI(
    title="Lip-Sync V2 API",
    description="Multi-face lip-sync using Wav2Lip-HD with InsightFace tracking",
    version="2.0.0",
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

    insightface_loaded = face_detector is not None and face_detector.is_loaded
    wav2lip_loaded = wav2lip is not None

    if insightface_loaded and wav2lip_loaded:
        status = "healthy"
    elif insightface_loaded or wav2lip_loaded:
        status = "loading"
    else:
        status = "unhealthy"

    return HealthResponse(
        status=status,
        insightface_loaded=insightface_loaded,
        wav2lip_loaded=wav2lip_loaded,
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        gpu_memory_gb=gpu_memory_gb,
    )


@app.post("/face-tracking", response_model=FaceTrackingResponse)
async def face_tracking(request: FaceTrackingRequest):
    """Generate visualization video with colored bboxes and character labels."""
    if face_detector is None or not face_detector.is_loaded:
        raise HTTPException(status_code=503, detail="Face detector not loaded")

    total_start = time.time()

    logger.info("=" * 60)
    logger.info("FACE TRACKING VISUALIZATION")
    logger.info("=" * 60)
    logger.info(f"  Video URL: {request.video_url}")
    logger.info(f"  Characters: {len(request.characters)}")

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, "video.mp4")
        output_path = os.path.join(tmpdir, "tracking.mp4")

        # Download video
        try:
            await download_file(request.video_url, video_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to download video: {e}")

        # Load reference images
        face_detector.clear_references()
        for char in request.characters:
            try:
                ref_image = await download_image(char.reference_image_url)
                face_detector.load_reference(char.id, ref_image)
            except Exception as e:
                logger.warning(f"Failed to load reference for {char.id}: {e}")

        # Track faces with sampling + interpolation
        tracker = FaceTracker(
            detector=face_detector,
            sample_interval=5,  # Detect every 5th frame
            smoothing_factor=request.temporal_smoothing,
            similarity_threshold=request.similarity_threshold,
        )

        characters = [
            {"id": c.id, "name": c.name, "reference_image_url": c.reference_image_url}
            for c in request.characters
        ]

        tracking_result = tracker.track_video(video_path, characters)

        # Generate visualization
        generate_tracking_video(video_path, tracking_result, output_path)

        # Upload
        try:
            await upload_file(request.upload_url, output_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to upload: {e}")

    total_ms = int((time.time() - total_start) * 1000)
    logger.info(f"  Complete in {total_ms}ms")

    # Build color map
    character_colors = {
        c.character_id: bbox_to_hex_color(c.color)
        for c in tracking_result.characters
    }

    return FaceTrackingResponse(
        output_url=request.upload_url.split("?")[0],
        character_colors=character_colors,
        frame_count=tracking_result.frame_count,
        fps=tracking_result.fps,
        total_ms=total_ms,
    )


@app.post("/lipsync", response_model=LipSyncResponse)
async def lipsync(request: LipSyncRequest):
    """Process lip-sync with Wav2Lip-HD and per-frame face tracking."""
    if face_detector is None or not face_detector.is_loaded:
        raise HTTPException(status_code=503, detail="Face detector not loaded")
    if wav2lip is None:
        raise HTTPException(status_code=503, detail="Wav2Lip not initialized")

    total_start = time.time()
    timing = {
        "download_ms": 0,
        "tracking_ms": 0,
        "lipsync_ms": 0,
        "upload_ms": 0,
    }

    logger.info("=" * 60)
    logger.info("LIP-SYNC V2")
    logger.info("=" * 60)
    logger.info(f"  Video URL: {request.video_url}")
    logger.info(f"  Audio URL: {request.audio_url}")
    logger.info(f"  Characters: {len(request.characters)}")
    logger.info(f"  Enhance: {request.enhance_quality}")
    logger.info(f"  Smoothing: {request.temporal_smoothing}")

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, "video.mp4")
        audio_path = os.path.join(tmpdir, "audio.wav")
        output_path = os.path.join(tmpdir, "output.mp4")

        # Download video, audio, and reference images
        download_start = time.time()
        try:
            await download_file(request.video_url, video_path)
            await download_file(request.audio_url, audio_path)

            face_detector.clear_references()
            for char in request.characters:
                ref_img = await download_image(char.reference_image_url)
                if face_detector.load_reference(char.id, ref_img):
                    logger.info(f"    Registered reference for {char.id}")
                else:
                    logger.warning(f"    Failed to register reference for {char.id}")

            timing["download_ms"] = int((time.time() - download_start) * 1000)
        except Exception as e:
            logger.error(f"Download failed: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to download: {e}")

        # Track faces with sampling + interpolation
        tracking_start = time.time()
        logger.info("  Tracking faces...")

        tracker = FaceTracker(
            detector=face_detector,
            sample_interval=5,  # Detect every 5th frame
            smoothing_factor=request.temporal_smoothing,
            similarity_threshold=request.similarity_threshold,
        )

        characters = [
            {"id": c.id, "name": c.name, "reference_image_url": c.reference_image_url}
            for c in request.characters
        ]

        # Sampling + interpolation + smoothing all handled internally
        tracking_result = tracker.track_video(video_path, characters)

        timing["tracking_ms"] = int((time.time() - tracking_start) * 1000)

        # Process each character sequentially
        lipsync_start = time.time()
        processed_characters: List[str] = []
        current_video = video_path

        # For "smart" loop mode, we don't loop in wav2lip - we do it after
        wav2lip_loop_mode = "none" if request.loop_mode == "smart" else request.loop_mode

        for char in request.characters:
            char_bboxes = tracking_result.tracks.get(char.id, [])
            has_detections = any(b is not None for b in char_bboxes)

            if not has_detections:
                logger.warning(f"  No detections for {char.id}, skipping")
                continue

            logger.info(f"  Processing {char.id}...")

            # Output for this character
            char_output = os.path.join(tmpdir, f"{char.id}_output.mp4")

            try:
                wav2lip.process(
                    video_path=current_video,
                    audio_path=audio_path,
                    output_path=char_output,
                    bboxes=char_bboxes,
                    enhance=request.enhance_quality,
                    loop_mode=wav2lip_loop_mode,
                    crossfade_frames=request.crossfade_frames,
                )

                # Use this output as input for next character
                current_video = char_output
                processed_characters.append(char.id)

            except Exception as e:
                logger.error(f"  Failed to process {char.id}: {e}")
                continue

        timing["lipsync_ms"] = int((time.time() - lipsync_start) * 1000)

        # Apply smart loop if requested
        if request.loop_mode == "smart" and processed_characters:
            logger.info("  Applying smart loop with RIFE...")
            loop_start = time.time()

            try:
                # Get audio duration
                audio_duration = get_audio_duration(audio_path)
                logger.info(f"    Audio duration: {audio_duration:.2f}s")

                # Get the bboxes from the first processed character for loop detection
                first_char_bboxes = tracking_result.tracks.get(processed_characters[0], [])

                # Create seamless loop
                loop_config = LoopConfig(
                    min_loop_frames=30,  # 1s minimum at 30fps
                    transition_frames=request.crossfade_frames,
                    similarity_threshold=0.7,
                )
                loop_handler = LoopHandler(loop_config)

                loop_output = os.path.join(tmpdir, "looped_output.mp4")
                loop_handler.create_seamless_loop(
                    video_path=current_video,
                    output_path=loop_output,
                    target_duration=audio_duration,
                    bboxes=first_char_bboxes,
                    fps=tracking_result.fps,
                )
                loop_handler.unload()

                current_video = loop_output
                logger.info(f"    Smart loop completed in {time.time() - loop_start:.2f}s")

            except Exception as e:
                logger.warning(f"    Smart loop failed, using original: {e}")
                # Fall back to original output without looping

        if not processed_characters:
            return LipSyncResponse(
                success=False,
                error_message="No characters could be processed",
            )

        # Prepare final output
        import shutil
        import subprocess

        if request.include_audio:
            # Keep audio in output
            shutil.copy(current_video, output_path)
        else:
            # Strip audio for video-only output (useful for multi-language dubbing)
            logger.info("  Stripping audio from output...")
            strip_cmd = [
                "ffmpeg", "-y", "-i", current_video,
                "-c:v", "copy", "-an",  # Copy video, no audio
                output_path
            ]
            subprocess.run(strip_cmd, capture_output=True, check=True)
            logger.info("  Audio stripped")

        # Upload
        try:
            timing["upload_ms"] = int(await upload_file(request.upload_url, output_path))
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to upload: {e}")

        # Get output info
        output_info = get_video_info(output_path)
        output_size = os.path.getsize(output_path)

    total_ms = int((time.time() - total_start) * 1000)
    logger.info(f"  Complete in {total_ms}ms")
    logger.info(f"  Processed: {processed_characters}")

    return LipSyncResponse(
        success=True,
        output_url=request.upload_url.split("?")[0],
        processed_characters=processed_characters,
        output_duration_ms=output_info["duration_ms"],
        output_width=output_info["width"],
        output_height=output_info["height"],
        output_fps=output_info["fps"],
        output_size_bytes=output_size,
        timing_download_ms=timing["download_ms"],
        timing_tracking_ms=timing["tracking_ms"],
        timing_lipsync_ms=timing["lipsync_ms"],
        timing_upload_ms=timing["upload_ms"],
        timing_total_ms=total_ms,
    )


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Lip-Sync V2 API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": [
            "POST /face-tracking - Generate visualization video with tracked bboxes",
            "POST /lipsync - Process lip-sync with Wav2Lip-HD",
            "GET /health - Health check",
        ],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "lipsync.server.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
