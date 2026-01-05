"""
FastAPI server for multi-face lip-sync processing.

Usage:
    uvicorn lipsync.server.main:app --host 0.0.0.0 --port 8000
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

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
)

# Global pipeline instance
pipeline: Optional[LipSyncPipeline] = None

# Default models directory
MODELS_DIR = os.environ.get("MODELS_DIR", "/app/models")


def check_models_exist() -> bool:
    """Check if model weights are present."""
    required_dirs = ["musetalk", "liveportrait", "codeformer"]
    for model_dir in required_dirs:
        path = os.path.join(MODELS_DIR, model_dir)
        if not os.path.exists(path):
            return False
        # Check if directory has files (not just empty)
        if not any(os.scandir(path)):
            return False
    return True


def download_models() -> None:
    """Download model weights if not present."""
    print(f"Downloading models to {MODELS_DIR}...")
    print("This may take 10-15 minutes on first run...")

    # Try to import and run the download script
    try:
        # Add scripts directory to path
        scripts_dir = os.path.join(os.path.dirname(__file__), "..", "..", "scripts")
        sys.path.insert(0, scripts_dir)

        from download_models import download_musetalk, download_liveportrait, download_codeformer

        os.makedirs(MODELS_DIR, exist_ok=True)

        print("Downloading MuseTalk...")
        download_musetalk(MODELS_DIR)

        print("Downloading LivePortrait...")
        download_liveportrait(MODELS_DIR)

        print("Downloading CodeFormer...")
        download_codeformer(MODELS_DIR)

        print("All models downloaded successfully!")

    except ImportError:
        # Fallback: run the script directly
        script_path = os.path.join(
            os.path.dirname(__file__), "..", "..", "scripts", "download_models.py"
        )
        if os.path.exists(script_path):
            subprocess.run(
                [sys.executable, script_path, "--models-dir", MODELS_DIR],
                check=True,
            )
        else:
            raise RuntimeError(
                f"Cannot find download script and models not present at {MODELS_DIR}"
            )


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
async def detect_faces(
    frame: UploadFile = File(..., description="Video frame image"),
    request: str = Form(..., description="JSON with character definitions"),
):
    """
    Detect character faces in a frame using Qwen-VL.

    Requires OPENROUTER_API_KEY environment variable.
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
    import base64
    frame_base64 = base64.b64encode(frame_bytes).decode("utf-8")

    detected = await client.detect_characters(frame_base64, req.characters)

    return DetectFacesResponse(
        faces=detected,
        frame_width=width,
        frame_height=height,
    )


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Multi-Face Lip-Sync API",
        "version": "0.1.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": [
            "POST /lipsync - Process lip-sync (returns video)",
            "POST /lipsync/json - Process lip-sync (returns JSON with base64)",
            "POST /detect-faces - Detect character faces using Qwen-VL",
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
