"""
FastAPI server for multi-face lip-sync processing.

Usage:
    uvicorn lipsync.server.main:app --host 0.0.0.0 --port 8000
"""

import base64
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import httpx
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response

from ..pipeline import LipSyncPipeline, PipelineConfig, FaceJob
from .schemas import (
    LipSyncRequest,
    LipSyncResponse,
    FaceResultInfo,
    HealthResponse,
    DetectFacesRequest,
    DetectFacesResponse,
    DetectFacesMultiFrameResponse,
    DetectFacesUrlRequest,
    LipSyncUrlRequest,
    DetectedFace,
    FrameDetection,
)

# Configure logging
log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('lipsync.server')

# Global pipeline instance
pipeline: Optional[LipSyncPipeline] = None

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

        from download_models import download_musetalk, download_liveportrait, download_codeformer

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


def extract_frame(video_path: str, output_path: str, time_seconds: float = 0.5) -> str:
    """
    Extract a frame from video using ffmpeg.

    Args:
        video_path: Path to video file
        output_path: Path to save frame image
        time_seconds: Time in seconds to extract frame

    Returns:
        Path to extracted frame
    """
    logger.info(f"Extracting frame at {time_seconds}s from {video_path}")

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(time_seconds),
        "-i", video_path,
        "-vframes", "1",
        "-f", "image2",
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"ffmpeg error: {result.stderr}")
        raise RuntimeError(f"Failed to extract frame: {result.stderr}")

    logger.info(f"  Frame saved to {output_path}")
    return output_path


def get_video_info(video_path: str) -> dict:
    """
    Get video duration and dimensions using ffprobe.

    Returns:
        Dict with duration_ms, width, height
    """
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_format", "-show_streams",
        video_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed: {result.stderr}")

    import json
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
    fps: float,
    start_time_ms: int = 0,
    end_time_ms: Optional[int] = None,
) -> list:
    """
    Extract frames from video at specified FPS.

    Args:
        video_path: Path to video file
        output_dir: Directory to save frames
        fps: Frames per second to extract
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

    interval_ms = int(1000 / fps)
    frames = []

    logger.info(f"Extracting frames at {fps} FPS from {start_time_ms}ms to {end_time_ms}ms")

    timestamp_ms = start_time_ms
    frame_idx = 0

    while timestamp_ms < end_time_ms:
        time_seconds = timestamp_ms / 1000.0
        frame_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.jpg")

        cmd = [
            "ffmpeg", "-y",
            "-ss", str(time_seconds),
            "-i", video_path,
            "-vframes", "1",
            "-f", "image2",
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
    global pipeline

    print("Initializing lip-sync pipeline...")

    # Check if models exist, download if not
    if not check_models_exist():
        print(f"Models not found at {MODELS_DIR}")
        download_models()
    else:
        print(f"Models found at {MODELS_DIR}")

    config = PipelineConfig(
        musetalk_path=os.path.join(MODELS_DIR, "musetalk"),
        liveportrait_path=os.path.join(MODELS_DIR, "liveportrait"),
        codeformer_path=os.path.join(MODELS_DIR, "codeformer"),
        device="cuda" if torch.cuda.is_available() else "cpu",
        fp16=torch.cuda.is_available(),
        enhance_quality=True,
    )

    pipeline = LipSyncPipeline(config)

    # Optionally pre-load models (can be slow)
    if os.environ.get("PRELOAD_MODELS", "false").lower() == "true":
        print("Pre-loading models...")
        pipeline.load_models()
        print("Models pre-loaded")
    else:
        print("Models will be loaded on first request")

    yield

    # Cleanup
    if pipeline is not None:
        print("Unloading models...")
        pipeline.unload_models()


app = FastAPI(
    title="Multi-Face Lip-Sync API",
    description="Process lip-sync for multiple faces in a video using MuseTalk + LivePortrait + CodeFormer",
    version="0.1.0",
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

    return HealthResponse(
        status="healthy",
        models_downloaded=check_models_exist(),
        models_loaded=pipeline.is_loaded if pipeline else False,
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        gpu_memory_gb=gpu_memory_gb,
    )


@app.post("/lipsync", response_model=LipSyncResponse)
async def lipsync(
    video: UploadFile = File(..., description="Input video file"),
    audio: UploadFile = File(..., description="Audio file for lip-sync"),
    request: str = Form(..., description="JSON string of LipSyncRequest"),
):
    """
    Process multi-face lip-sync.

    - **video**: Input video file (mp4, mov, etc.)
    - **audio**: Audio file for lip-sync (wav, mp3, etc.)
    - **request**: JSON configuration with face jobs

    Example request JSON:
    ```json
    {
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

    start_time = time.time()

    try:
        req = LipSyncRequest(**json.loads(request))
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON in request: {e}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid request format: {e}")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save uploaded files
        video_path = os.path.join(tmpdir, "input_video.mp4")
        audio_path = os.path.join(tmpdir, "input_audio.wav")
        output_path = os.path.join(tmpdir, "output.mp4")

        video_content = await video.read()
        with open(video_path, "wb") as f:
            f.write(video_content)

        audio_content = await audio.read()
        with open(audio_path, "wb") as f:
            f.write(audio_content)

        # Convert request faces to FaceJob objects
        face_jobs = [
            FaceJob(
                character_id=face.character_id,
                bbox=face.bbox,
                audio_path=audio_path,  # Use main audio for all faces for now
                start_time_ms=face.start_time_ms,
                end_time_ms=face.end_time_ms,
            )
            for face in req.faces
        ]

        face_results = []

        try:
            # Update pipeline config
            pipeline.config.enhance_quality = req.enhance_quality
            pipeline.config.fidelity_weight = req.fidelity_weight

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
            for face in req.faces:
                face_results.append(FaceResultInfo(
                    character_id=face.character_id,
                    success=True,
                ))

        except Exception as e:
            # Mark all faces as failed
            for face in req.faces:
                face_results.append(FaceResultInfo(
                    character_id=face.character_id,
                    success=False,
                    error_message=str(e),
                ))

            processing_time = int((time.time() - start_time) * 1000)
            return LipSyncResponse(
                success=False,
                faces_processed=0,
                face_results=face_results,
                processing_time_ms=processing_time,
                error_message=str(e),
            )

        # Read output file
        if not os.path.exists(output_path):
            raise HTTPException(status_code=500, detail="Output file not created")

        with open(output_path, "rb") as f:
            output_bytes = f.read()

    processing_time = int((time.time() - start_time) * 1000)

    # Return video directly
    return Response(
        content=output_bytes,
        media_type="video/mp4",
        headers={
            "X-Faces-Processed": str(len(face_results)),
            "X-Processing-Time-Ms": str(processing_time),
        },
    )


@app.post("/lipsync/json", response_model=LipSyncResponse)
async def lipsync_json(
    video: UploadFile = File(...),
    audio: UploadFile = File(...),
    request: str = Form(...),
):
    """
    Process lip-sync and return JSON response with base64 video.

    Same as /lipsync but returns JSON with metadata instead of raw video.
    """
    import base64

    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")

    start_time = time.time()

    try:
        req = LipSyncRequest(**json.loads(request))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid request: {e}")

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, "input_video.mp4")
        audio_path = os.path.join(tmpdir, "input_audio.wav")
        output_path = os.path.join(tmpdir, "output.mp4")

        video_content = await video.read()
        with open(video_path, "wb") as f:
            f.write(video_content)

        audio_content = await audio.read()
        with open(audio_path, "wb") as f:
            f.write(audio_content)

        face_jobs = [
            FaceJob(
                character_id=face.character_id,
                bbox=face.bbox,
                audio_path=audio_path,
                start_time_ms=face.start_time_ms,
                end_time_ms=face.end_time_ms,
            )
            for face in req.faces
        ]

        face_results = []

        try:
            pipeline.config.enhance_quality = req.enhance_quality
            pipeline.config.fidelity_weight = req.fidelity_weight

            if len(face_jobs) == 1:
                job = face_jobs[0]
                pipeline.process_single_face(
                    video_path,
                    audio_path,
                    output_path,
                    target_bbox=job.bbox,
                )
            else:
                pipeline.process_multi_face(video_path, face_jobs, output_path)

            for face in req.faces:
                face_results.append(FaceResultInfo(
                    character_id=face.character_id,
                    success=True,
                ))

            with open(output_path, "rb") as f:
                output_bytes = f.read()

            # Encode as base64 data URL
            video_base64 = base64.b64encode(output_bytes).decode("utf-8")
            output_url = f"data:video/mp4;base64,{video_base64}"

        except Exception as e:
            for face in req.faces:
                face_results.append(FaceResultInfo(
                    character_id=face.character_id,
                    success=False,
                    error_message=str(e),
                ))

            processing_time = int((time.time() - start_time) * 1000)
            return LipSyncResponse(
                success=False,
                faces_processed=0,
                face_results=face_results,
                processing_time_ms=processing_time,
                error_message=str(e),
            )

    processing_time = int((time.time() - start_time) * 1000)

    return LipSyncResponse(
        success=True,
        faces_processed=len(face_results),
        face_results=face_results,
        processing_time_ms=processing_time,
        output_url=output_url,
    )


@app.post("/detect-faces", response_model=DetectFacesResponse)
async def detect_faces_multipart(
    frame: UploadFile = File(..., description="Video frame image"),
    request: str = Form(..., description="JSON with character definitions"),
):
    """
    Detect character faces in a frame using Qwen-VL (multipart upload).

    Requires OPENROUTER_API_KEY environment variable.

    DEPRECATED: Use POST /detect-faces/url with JSON body instead.
    """
    try:
        req = DetectFacesRequest(**json.loads(request))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid request: {e}")

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="OPENROUTER_API_KEY not configured",
        )

    # Import Qwen client here to avoid circular import
    from .qwen_client import QwenVLClient

    client = QwenVLClient(api_key)

    # Read frame
    frame_bytes = await frame.read()

    # Get image dimensions
    import cv2
    import numpy as np

    nparr = np.frombuffer(frame_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Invalid image file")

    height, width = img.shape[:2]

    # Detect faces
    frame_base64 = base64.b64encode(frame_bytes).decode("utf-8")

    detected = await client.detect_characters(frame_base64, req.characters)

    return DetectFacesResponse(
        faces=detected,
        frame_width=width,
        frame_height=height,
    )


@app.post("/detect-faces/url", response_model=DetectFacesMultiFrameResponse)
async def detect_faces_url(request: DetectFacesUrlRequest):
    """
    Detect character faces across video frames using Qwen-VL.

    Downloads video from URL, samples frames at specified FPS, and detects faces
    in each frame. Returns per-frame bounding boxes for face tracking.

    Requires OPENROUTER_API_KEY environment variable.
    """
    logger.info("=" * 60)
    logger.info("DETECT FACES (URL) - Multi-frame")
    logger.info("=" * 60)
    logger.info(f"  Video URL: {request.video_url}")
    logger.info(f"  Sample FPS: {request.sample_fps}")
    logger.info(f"  Time range: {request.start_time_ms}ms - {request.end_time_ms}ms")
    logger.info(f"  Characters: {[c.id for c in request.characters]}")

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="OPENROUTER_API_KEY not configured",
        )

    from .qwen_client import QwenVLClient

    client = QwenVLClient(api_key)

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

        logger.info(f"  Extracted {len(frame_paths)} frames")

        # Convert frames to base64
        frames_base64 = []
        for timestamp_ms, frame_path in frame_paths:
            with open(frame_path, "rb") as f:
                frame_bytes = f.read()
            frame_base64 = base64.b64encode(frame_bytes).decode("utf-8")
            frames_base64.append((timestamp_ms, frame_base64))

        # Build character dict
        characters_dict = [
            {"id": c.id, "name": c.name, "description": c.description}
            for c in request.characters
        ]

        # Detect faces in all frames
        logger.info(f"  Running face detection on {len(frames_base64)} frames...")
        try:
            results = await client.detect_characters_multi_frame(
                frames_base64,
                characters_dict,
                max_concurrent=5,
            )
        except Exception as e:
            logger.error(f"Face detection failed: {e}")
            raise HTTPException(status_code=500, detail=f"Face detection failed: {e}")

        # Build response
        frame_detections = []
        total_faces = 0
        for timestamp_ms, faces in results:
            frame_detections.append(FrameDetection(
                timestamp_ms=timestamp_ms,
                faces=faces,
            ))
            total_faces += len(faces)

        logger.info(f"  Total detections: {total_faces} faces across {len(frame_detections)} frames")

    return DetectFacesMultiFrameResponse(
        frames=frame_detections,
        frame_width=width,
        frame_height=height,
        sample_fps=request.sample_fps,
        video_duration_ms=video_duration_ms,
    )


@app.post("/lipsync/url", response_model=LipSyncResponse)
async def lipsync_url(request: LipSyncUrlRequest):
    """
    Process multi-face lip-sync from URLs.

    Downloads video and audio from URLs, processes lip-sync, returns result.

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
    logger.info("LIP-SYNC (URL)")
    logger.info("=" * 60)
    logger.info(f"  Video URL: {request.video_url}")
    logger.info(f"  Audio URL: {request.audio_url}")
    logger.info(f"  Faces: {len(request.faces)}")
    for face in request.faces:
        logger.info(f"    - {face.character_id}: bbox={face.bbox}, {face.start_time_ms}ms-{face.end_time_ms}ms")
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
                face_results.append(FaceResultInfo(
                    character_id=face.character_id,
                    success=True,
                ))

        except Exception as e:
            logger.error(f"Lip-sync processing failed: {e}")
            # Mark all faces as failed
            for face in request.faces:
                face_results.append(FaceResultInfo(
                    character_id=face.character_id,
                    success=False,
                    error_message=str(e),
                ))

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
        "version": "0.2.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": [
            "POST /detect-faces/url - Detect faces from video URL (JSON)",
            "POST /lipsync/url - Process lip-sync from URLs (JSON)",
            "GET /health - Health check",
            "--- Legacy (multipart) ---",
            "POST /lipsync - Process lip-sync (multipart, returns video)",
            "POST /lipsync/json - Process lip-sync (multipart, returns JSON)",
            "POST /detect-faces - Detect faces (multipart)",
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
