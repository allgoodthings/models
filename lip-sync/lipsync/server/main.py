"""
FastAPI server for multi-face lip-sync processing.

Usage:
    uvicorn lipsync.server.main:app --host 0.0.0.0 --port 8000
"""

import base64
import logging
import os
import subprocess
import sys
import tempfile
import time
from contextlib import asynccontextmanager
from typing import Optional

import cv2
import httpx
import numpy as np
import torch
from fastapi import FastAPI, HTTPException

from ..pipeline import FaceJob, LipSyncPipeline, PipelineConfig
from .insightface_detector import InsightFaceDetector
from .schemas import (
    DetectedFaceWithMetadata,
    DetectFacesRequest,
    DetectFacesResponse,
    FaceResultInfo,
    FrameAnalysis,
    HealthResponse,
    LipSyncRequest,
    LipSyncResponse,
)

# Configure logging
log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("lipsync.server")

# Global instances
pipeline: Optional[LipSyncPipeline] = None
face_detector: Optional[InsightFaceDetector] = None

# Default models directory
MODELS_DIR = os.environ.get("MODELS_DIR", "/app/models")


def check_models_exist() -> bool:
    """Check if model weights are present."""
    logger.debug(f"Checking for models in {MODELS_DIR}")
    required_dirs = ["musetalk", "liveportrait", "codeformer"]
    for model_dir in required_dirs:
        path = os.path.join(MODELS_DIR, model_dir)
        if not os.path.exists(path):
            logger.debug(f"  {model_dir}: NOT FOUND")
            return False
        # Check if directory has files (not just empty)
        if not any(os.scandir(path)):
            logger.debug(f"  {model_dir}: EMPTY")
            return False
        logger.debug(f"  {model_dir}: OK")
    return True


def download_models() -> None:
    """Download model weights if not present."""
    logger.info(f"Downloading models to {MODELS_DIR}...")
    logger.info("This may take 10-15 minutes on first run...")

    # Try to import and run the download script
    try:
        # Add scripts directory to path
        scripts_dir = os.path.join(os.path.dirname(__file__), "..", "..", "scripts")
        sys.path.insert(0, scripts_dir)

        from download_models import (
            download_codeformer,
            download_liveportrait,
            download_musetalk,
        )

        os.makedirs(MODELS_DIR, exist_ok=True)

        logger.info("Downloading MuseTalk...")
        download_musetalk(MODELS_DIR)

        logger.info("Downloading LivePortrait...")
        download_liveportrait(MODELS_DIR)

        logger.info("Downloading CodeFormer...")
        download_codeformer(MODELS_DIR)

        logger.info("All models downloaded successfully!")

    except ImportError as e:
        logger.warning(f"Could not import download functions: {e}")
        # Fallback: run the script directly
        script_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "scripts", "download_models.py"
        )
        if os.path.exists(script_path):
            logger.info(f"Running download script: {script_path}")
            subprocess.run(
                [sys.executable, script_path, "--models-dir", MODELS_DIR],
                check=True,
            )
        else:
            raise RuntimeError(
                f"Cannot find download script and models not present at {MODELS_DIR}"
            )


async def download_file(url: str, dest_path: str) -> str:
    """
    Download a file from URL.

    Args:
        url: URL to download from
        dest_path: Local path to save file

    Returns:
        Path to downloaded file
    """
    logger.info(f"Downloading file from {url}")
    start_time = time.time()

    async with httpx.AsyncClient(timeout=300.0, follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()

        with open(dest_path, "wb") as f:
            f.write(response.content)

    elapsed = time.time() - start_time
    file_size = os.path.getsize(dest_path) / (1024 * 1024)
    logger.info(f"  Downloaded {file_size:.2f}MB in {elapsed:.2f}s")

    return dest_path


async def download_image(url: str) -> np.ndarray:
    """
    Download image from URL and return as numpy array (BGR).

    Args:
        url: URL to image

    Returns:
        BGR numpy array
    """
    logger.debug(f"Downloading image from {url}")

    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        response = await client.get(url)
        response.raise_for_status()

        img_array = np.frombuffer(response.content, np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        if img is None:
            raise ValueError(f"Failed to decode image from {url}")

        return img


def get_video_info(video_path: str) -> dict:
    """
    Get video duration and dimensions using ffprobe.

    Returns:
        Dict with duration_ms, width, height
    """
    import json

    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        video_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")

    data = json.loads(result.stdout)

    duration = float(data.get("format", {}).get("duration", 0))
    duration_ms = int(duration * 1000)

    # Find video stream
    width, height = 0, 0
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            width = stream.get("width", 0)
            height = stream.get("height", 0)
            break

    return {
        "duration_ms": duration_ms,
        "width": width,
        "height": height,
    }


def extract_frames_at_fps(
    video_path: str,
    output_dir: str,
    fps: int,
    start_time_ms: int = 0,
    end_time_ms: Optional[int] = None,
) -> list:
    """
    Extract frames from video at specified FPS.

    Args:
        video_path: Path to video file
        output_dir: Directory to save frames
        fps: Frames per second to extract (integer)
        start_time_ms: Start time in milliseconds
        end_time_ms: End time in milliseconds (None = video end)

    Returns:
        List of (timestamp_ms, frame_path) tuples
    """
    video_info = get_video_info(video_path)
    video_duration_ms = video_info["duration_ms"]

    if end_time_ms is None:
        end_time_ms = video_duration_ms
    else:
        end_time_ms = min(end_time_ms, video_duration_ms)

    interval_ms = 1000 // fps
    frames = []

    logger.info(f"Extracting frames at {fps} FPS from {start_time_ms}ms to {end_time_ms}ms")

    timestamp_ms = start_time_ms
    frame_idx = 0

    while timestamp_ms < end_time_ms:
        time_seconds = timestamp_ms / 1000.0
        frame_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.jpg")

        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            str(time_seconds),
            "-i",
            video_path,
            "-vframes",
            "1",
            "-f",
            "image2",
            frame_path,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0 and os.path.exists(frame_path):
            frames.append((timestamp_ms, frame_path))

        timestamp_ms += interval_ms
        frame_idx += 1

    logger.info(f"  Extracted {len(frames)} frames")
    return frames


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, cleanup on shutdown."""
    global pipeline, face_detector

    logger.info("=" * 60)
    logger.info("STARTING LIP-SYNC SERVER")
    logger.info("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    if device == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"GPU: {gpu_name} ({gpu_memory:.1f}GB)")

    # Check if models exist, download if not
    logger.info("-" * 40)
    logger.info("Checking model weights...")
    if not check_models_exist():
        logger.info(f"Models not found at {MODELS_DIR}")
        download_models()
    else:
        logger.info(f"Models found at {MODELS_DIR}")

    # Initialize face detector
    logger.info("-" * 40)
    logger.info("[1/4] Loading InsightFace...")
    face_detector = InsightFaceDetector(model_name="buffalo_l", device=device)
    face_detector.load()
    logger.info("[1/4] InsightFace loaded successfully")

    # Initialize lip-sync pipeline
    logger.info("-" * 40)
    logger.info("[2/4] Initializing LipSyncPipeline...")
    config = PipelineConfig(
        musetalk_path=os.path.join(MODELS_DIR, "musetalk"),
        liveportrait_path=os.path.join(MODELS_DIR, "liveportrait"),
        codeformer_path=os.path.join(MODELS_DIR, "codeformer"),
        device=device,
        fp16=device == "cuda",
        enhance_quality=True,
    )
    pipeline = LipSyncPipeline(config)

    # Load pipeline models in sequence
    logger.info("[3/4] Loading MuseTalk + LivePortrait...")
    pipeline.load_models()
    logger.info("[3/4] MuseTalk + LivePortrait loaded successfully")

    logger.info("[4/4] Loading CodeFormer...")
    # CodeFormer is loaded lazily by pipeline, force load it
    if hasattr(pipeline, "_load_codeformer"):
        pipeline._load_codeformer()
    logger.info("[4/4] CodeFormer loaded successfully")

    logger.info("-" * 40)
    logger.info("ALL MODELS LOADED - Server ready")
    logger.info("=" * 60)

    yield

    # Cleanup
    logger.info("Shutting down server...")
    if pipeline is not None:
        logger.info("Unloading pipeline models...")
        pipeline.unload_models()
    logger.info("Shutdown complete")


app = FastAPI(
    title="Multi-Face Lip-Sync API",
    description="Process lip-sync for multiple faces using MuseTalk + LivePortrait + CodeFormer with InsightFace detection",
    version="0.3.0",
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

    # Check model status
    insightface_loaded = face_detector is not None and face_detector.is_loaded
    pipeline_loaded = pipeline is not None and pipeline.is_loaded

    # Determine status
    if insightface_loaded and pipeline_loaded:
        status = "healthy"
    elif insightface_loaded or pipeline_loaded:
        status = "loading"
    else:
        status = "unhealthy"

    return HealthResponse(
        status=status,
        insightface_loaded=insightface_loaded,
        musetalk_loaded=pipeline_loaded,
        liveportrait_loaded=pipeline_loaded,
        codeformer_loaded=pipeline_loaded,
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        gpu_memory_gb=gpu_memory_gb,
    )


@app.post("/detect-faces", response_model=DetectFacesResponse)
async def detect_faces(request: DetectFacesRequest):
    """
    Detect and track faces across video frames using InsightFace.

    Downloads video and reference images from URLs, extracts face embeddings
    from reference images, samples video frames at specified FPS, and matches
    detected faces to characters using embedding similarity.

    Returns per-frame bounding boxes with syncability metadata.
    """
    if face_detector is None or not face_detector.is_loaded:
        raise HTTPException(status_code=503, detail="Face detector not loaded")

    logger.info("=" * 60)
    logger.info("DETECT FACES")
    logger.info("=" * 60)
    logger.info(f"  Video URL: {request.video_url}")
    logger.info(f"  Sample FPS: {request.sample_fps}")
    logger.info(f"  Time range: {request.start_time_ms}ms - {request.end_time_ms}ms")
    logger.info(f"  Similarity threshold: {request.similarity_threshold}")
    logger.info(f"  Characters: {len(request.characters)}")
    for char in request.characters:
        logger.info(f"    - {char.id}: {char.name}")

    start_time = time.time()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Download video
        video_path = os.path.join(tmpdir, "video.mp4")
        try:
            await download_file(request.video_url, video_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to download video: {e}")

        # Get video info
        try:
            video_info = get_video_info(video_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to get video info: {e}")

        width = video_info["width"]
        height = video_info["height"]
        video_duration_ms = video_info["duration_ms"]

        logger.info(f"  Video: {width}x{height}, {video_duration_ms}ms duration")

        # Clear existing references and load new ones
        face_detector.clear_references()

        logger.info("  Loading reference images...")
        for char in request.characters:
            try:
                ref_image = await download_image(char.reference_image_url)
                success = face_detector.load_reference(char.id, ref_image)
                if success:
                    logger.info(f"    - {char.id}: Reference loaded")
                else:
                    logger.warning(f"    - {char.id}: No face found in reference image")
            except Exception as e:
                logger.warning(f"    - {char.id}: Failed to load reference: {e}")

        # Extract frames at FPS
        frames_dir = os.path.join(tmpdir, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        try:
            frame_paths = extract_frames_at_fps(
                video_path,
                frames_dir,
                request.sample_fps,
                request.start_time_ms or 0,
                request.end_time_ms,
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to extract frames: {e}")

        if not frame_paths:
            raise HTTPException(status_code=500, detail="No frames extracted from video")

        logger.info(f"  Running face detection on {len(frame_paths)} frames...")

        # Process each frame
        frame_analyses = []
        characters_detected = set()
        total_faces = 0

        for timestamp_ms, frame_path in frame_paths:
            # Read frame
            frame = cv2.imread(frame_path)
            if frame is None:
                logger.warning(f"Failed to read frame at {timestamp_ms}ms")
                frame_analyses.append(FrameAnalysis(timestamp_ms=timestamp_ms, faces=[]))
                continue

            # Detect faces
            faces = face_detector.detect_faces(frame, request.similarity_threshold)

            # Convert to response format
            face_results = []
            for face in faces:
                face_results.append(
                    DetectedFaceWithMetadata(
                        character_id=face.character_id,
                        bbox=face.bbox,
                        confidence=face.confidence,
                        head_pose=face.head_pose,
                        syncable=face.syncable,
                        sync_quality=face.sync_quality,
                        skip_reason=face.skip_reason,
                    )
                )

                # Track which characters were detected
                if not face.character_id.startswith("face_"):
                    characters_detected.add(face.character_id)

            frame_analyses.append(
                FrameAnalysis(timestamp_ms=timestamp_ms, faces=face_results)
            )
            total_faces += len(face_results)

        elapsed = time.time() - start_time
        logger.info(f"  Detected {total_faces} faces across {len(frame_analyses)} frames")
        logger.info(f"  Characters found: {list(characters_detected)}")
        logger.info(f"  Processing time: {elapsed:.2f}s")

    return DetectFacesResponse(
        frames=frame_analyses,
        frame_width=width,
        frame_height=height,
        sample_fps=request.sample_fps,
        video_duration_ms=video_duration_ms,
        characters_detected=list(characters_detected),
    )


@app.post("/lipsync", response_model=LipSyncResponse)
async def lipsync(request: LipSyncRequest):
    """
    Process multi-face lip-sync from URLs.

    Downloads video and audio from URLs, processes lip-sync for each face,
    returns result with base64-encoded output video.

    Example request:
    ```json
    {
        "video_url": "https://example.com/video.mp4",
        "audio_url": "https://example.com/audio.mp3",
        "faces": [
            {
                "character_id": "alice",
                "bbox": [100, 50, 200, 250],
                "start_time_ms": 0,
                "end_time_ms": 5000
            }
        ],
        "enhance_quality": true
    }
    ```
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    logger.info("=" * 60)
    logger.info("LIP-SYNC")
    logger.info("=" * 60)
    logger.info(f"  Video URL: {request.video_url}")
    logger.info(f"  Audio URL: {request.audio_url}")
    logger.info(f"  Faces: {len(request.faces)}")
    for face in request.faces:
        logger.info(
            f"    - {face.character_id}: bbox={face.bbox}, "
            f"{face.start_time_ms}ms-{face.end_time_ms}ms"
        )
    logger.info(f"  Enhance: {request.enhance_quality}")
    logger.info(f"  Fidelity: {request.fidelity_weight}")

    start_time = time.time()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Download video and audio
        video_path = os.path.join(tmpdir, "video.mp4")
        audio_path = os.path.join(tmpdir, "audio.wav")
        output_path = os.path.join(tmpdir, "output.mp4")

        try:
            await download_file(request.video_url, video_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to download video: {e}")

        try:
            await download_file(request.audio_url, audio_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to download audio: {e}")

        # Convert request faces to FaceJob objects
        face_jobs = [
            FaceJob(
                character_id=face.character_id,
                bbox=face.bbox,
                audio_path=audio_path,
                start_time_ms=face.start_time_ms,
                end_time_ms=face.end_time_ms,
            )
            for face in request.faces
        ]

        face_results = []

        try:
            # Update pipeline config
            pipeline.config.enhance_quality = request.enhance_quality
            pipeline.config.fidelity_weight = request.fidelity_weight

            # Process lip-sync
            if len(face_jobs) == 1:
                job = face_jobs[0]
                pipeline.process_single_face(
                    video_path,
                    audio_path,
                    output_path,
                    target_bbox=job.bbox,
                )
            else:
                pipeline.process_multi_face(
                    video_path,
                    face_jobs,
                    output_path,
                )

            # Mark all faces as successful
            for face in request.faces:
                face_results.append(
                    FaceResultInfo(
                        character_id=face.character_id,
                        success=True,
                    )
                )

        except Exception as e:
            logger.error(f"Lip-sync processing failed: {e}")
            # Mark all faces as failed
            for face in request.faces:
                face_results.append(
                    FaceResultInfo(
                        character_id=face.character_id,
                        success=False,
                        error_message=str(e),
                    )
                )

            processing_time = int((time.time() - start_time) * 1000)
            return LipSyncResponse(
                success=False,
                faces_processed=0,
                face_results=face_results,
                processing_time_ms=processing_time,
                error_message=str(e),
            )

        # Read output file and encode as base64
        if not os.path.exists(output_path):
            raise HTTPException(status_code=500, detail="Output file not created")

        with open(output_path, "rb") as f:
            output_bytes = f.read()

        video_base64 = base64.b64encode(output_bytes).decode("utf-8")
        output_url = f"data:video/mp4;base64,{video_base64}"

    processing_time = int((time.time() - start_time) * 1000)

    logger.info(f"  Processing complete in {processing_time}ms")
    logger.info(f"  Output size: {len(output_bytes) / (1024*1024):.2f}MB")

    return LipSyncResponse(
        success=True,
        faces_processed=len(face_results),
        face_results=face_results,
        processing_time_ms=processing_time,
        output_url=output_url,
    )


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Multi-Face Lip-Sync API",
        "version": "0.3.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": [
            "POST /detect-faces - Detect faces with InsightFace (JSON)",
            "POST /lipsync - Process lip-sync from URLs (JSON)",
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
