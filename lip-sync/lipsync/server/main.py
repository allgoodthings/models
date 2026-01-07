"""
FastAPI server for multi-face lip-sync processing.

Usage:
    uvicorn lipsync.server.main:app --host 0.0.0.0 --port 8000
"""

import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Set

import cv2
import httpx
import numpy as np
import torch
from fastapi import FastAPI, HTTPException

from ..pipeline import LipSyncPipeline, PipelineConfig
from .insightface_detector import InsightFaceDetector
from .schemas import (
    CharacterReference,
    DetectedFaceWithMetadata,
    DetectFacesRequest,
    DetectFacesResponse,
    DetectionTimingBreakdown,
    FaceResult,
    FaceSegment,
    FacesResult,
    FaceSummary,
    FrameAnalysis,
    HealthResponse,
    LipSyncRequest,
    LipSyncResponse,
    OutputMetadata,
    TimingBreakdown,
    UnknownFace,
)
from .segment_builder import (
    FrameFaceData,
    build_segments_for_face,
    compute_summary,
    extend_segment_end_times,
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

# Detection FPS for tracking during lipsync
LIPSYNC_DETECTION_FPS = 5


def check_models_exist() -> bool:
    """Check if model weights are present."""
    logger.debug(f"Checking for models in {MODELS_DIR}")
    required_dirs = ["musetalk", "liveportrait", "codeformer"]
    for model_dir in required_dirs:
        path = os.path.join(MODELS_DIR, model_dir)
        if not os.path.exists(path):
            logger.debug(f"  {model_dir}: NOT FOUND")
            return False
        if not any(os.scandir(path)):
            logger.debug(f"  {model_dir}: EMPTY")
            return False
        logger.debug(f"  {model_dir}: OK")
    return True


def download_models() -> None:
    """Download model weights if not present."""
    logger.info(f"Downloading models to {MODELS_DIR}...")
    logger.info("This may take 10-15 minutes on first run...")

    try:
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


async def download_file(url: str, dest_path: str) -> float:
    """
    Download a file from URL.

    Returns:
        Download time in milliseconds
    """
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
    """
    Upload a file to a presigned URL using PUT.

    Returns:
        Upload time in milliseconds
    """
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
    """Get video duration, dimensions, and fps using ffprobe."""
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

    width, height, fps = 0, 0, 30.0
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video":
            width = stream.get("width", 0)
            height = stream.get("height", 0)
            # Parse frame rate (e.g., "30/1" or "30000/1001")
            fps_str = stream.get("r_frame_rate", "30/1")
            if "/" in fps_str:
                num, den = fps_str.split("/")
                fps = float(num) / float(den) if float(den) > 0 else 30.0
            else:
                fps = float(fps_str)
            break

    return {
        "duration_ms": duration_ms,
        "width": width,
        "height": height,
        "fps": fps,
    }


def extract_frames_at_fps(
    video_path: str,
    output_dir: str,
    fps: int,
    start_time_ms: int = 0,
    end_time_ms: Optional[int] = None,
) -> list:
    """Extract frames from video at specified FPS using a single ffmpeg command."""
    video_info = get_video_info(video_path)
    video_duration_ms = video_info["duration_ms"]

    if end_time_ms is None:
        end_time_ms = video_duration_ms
    else:
        end_time_ms = min(end_time_ms, video_duration_ms)

    start_seconds = start_time_ms / 1000.0
    duration_seconds = (end_time_ms - start_time_ms) / 1000.0
    interval_ms = 1000 // fps

    logger.info(f"Extracting frames at {fps} FPS from {start_time_ms}ms to {end_time_ms}ms")

    # Extract all frames in a single ffmpeg command
    output_pattern = os.path.join(output_dir, "frame_%04d.jpg")
    cmd = [
        "ffmpeg",
        "-y",
        "-ss", str(start_seconds),
        "-i", video_path,
        "-t", str(duration_seconds),
        "-vf", f"fps={fps}",
        "-q:v", "2",  # High quality JPEG
        output_pattern,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning(f"ffmpeg frame extraction failed: {result.stderr}")
        return []

    # Collect extracted frames with timestamps
    frames = []
    frame_idx = 0
    timestamp_ms = start_time_ms

    while timestamp_ms < end_time_ms:
        # ffmpeg uses 1-based numbering for output pattern
        frame_path = os.path.join(output_dir, f"frame_{frame_idx + 1:04d}.jpg")
        if os.path.exists(frame_path):
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

    skip_download = os.environ.get("SKIP_MODEL_DOWNLOAD", "").lower() in ("true", "1", "yes")
    skip_load = os.environ.get("PRELOAD_MODELS", "true").lower() in ("false", "0", "no")

    if skip_load:
        logger.info("PRELOAD_MODELS=false - Skipping model loading (health check only mode)")
        yield
        return

    logger.info("-" * 40)
    logger.info("Checking model weights...")
    if not check_models_exist():
        if skip_download:
            logger.warning(f"Models not found at {MODELS_DIR} but SKIP_MODEL_DOWNLOAD=true")
        else:
            logger.info(f"Models not found at {MODELS_DIR}")
            download_models()
    else:
        logger.info(f"Models found at {MODELS_DIR}")

    logger.info("-" * 40)
    logger.info("[1/4] Loading InsightFace...")
    face_detector = InsightFaceDetector(model_name="buffalo_l", device=device)
    face_detector.load()
    logger.info("[1/4] InsightFace loaded successfully")

    logger.info("-" * 40)
    logger.info("[2/4] Initializing LipSyncPipeline...")
    config = PipelineConfig(
        musetalk_path=os.path.join(MODELS_DIR, "musetalk"),
        liveportrait_path=os.path.join(MODELS_DIR, "liveportrait"),
        codeformer_path=os.path.join(MODELS_DIR, "codeformer"),
        device=device,
        fp16=device == "cuda",
        use_enhancement=True,
    )
    pipeline = LipSyncPipeline(config)

    logger.info("[3/4] Loading MuseTalk + LivePortrait...")
    pipeline.load_models()
    logger.info("[3/4] MuseTalk + LivePortrait loaded successfully")

    logger.info("[4/4] Loading CodeFormer...")
    if hasattr(pipeline, "_load_codeformer"):
        pipeline._load_codeformer()
    logger.info("[4/4] CodeFormer loaded successfully")

    logger.info("-" * 40)
    logger.info("ALL MODELS LOADED - Server ready")
    logger.info("=" * 60)

    yield

    logger.info("Shutting down server...")
    if pipeline is not None:
        logger.info("Unloading pipeline models...")
        pipeline.unload_models()
    logger.info("Shutdown complete")


app = FastAPI(
    title="Multi-Face Lip-Sync API",
    description="Process lip-sync for multiple faces using MuseTalk + LivePortrait + CodeFormer with InsightFace detection",
    version="0.4.0",
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
    pipeline_loaded = pipeline is not None and pipeline.is_loaded

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
    """
    if face_detector is None or not face_detector.is_loaded:
        raise HTTPException(status_code=503, detail="Face detector not loaded")

    total_start = time.time()
    timing = {
        "download_video_ms": 0,
        "download_refs_ms": 0,
        "frame_extraction_ms": 0,
        "detection_ms": 0,
    }

    logger.info("=" * 60)
    logger.info("DETECT FACES")
    logger.info("=" * 60)
    logger.info(f"  Video URL: {request.video_url}")
    logger.info(f"  Sample FPS: {request.sample_fps}")
    logger.info(f"  Characters: {len(request.characters)}")

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, "video.mp4")

        # Download video
        download_start = time.time()
        try:
            await download_file(request.video_url, video_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to download video: {e}")
        timing["download_video_ms"] = int((time.time() - download_start) * 1000)

        try:
            video_info = get_video_info(video_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to get video info: {e}")

        width = video_info["width"]
        height = video_info["height"]
        video_duration_ms = video_info["duration_ms"]

        # Load reference images
        refs_start = time.time()
        face_detector.clear_references()

        for char in request.characters:
            try:
                ref_image = await download_image(char.reference_image_url)
                face_detector.load_reference(char.id, ref_image)
            except Exception as e:
                logger.warning(f"Failed to load reference for {char.id}: {e}")
        timing["download_refs_ms"] = int((time.time() - refs_start) * 1000)

        # Extract frames
        extraction_start = time.time()
        frames_dir = os.path.join(tmpdir, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        frame_paths = extract_frames_at_fps(
            video_path,
            frames_dir,
            request.sample_fps,
            request.start_time_ms or 0,
            request.end_time_ms,
        )
        timing["frame_extraction_ms"] = int((time.time() - extraction_start) * 1000)

        # Run face detection on all frames
        detection_start = time.time()
        frame_analyses = []
        characters_detected = set()

        for timestamp_ms, frame_path in frame_paths:
            frame = cv2.imread(frame_path)
            if frame is None:
                frame_analyses.append(FrameAnalysis(timestamp_ms=timestamp_ms, faces=[]))
                continue

            faces = face_detector.detect_faces(frame, request.similarity_threshold)

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
                if not face.character_id.startswith("face_"):
                    characters_detected.add(face.character_id)

            frame_analyses.append(
                FrameAnalysis(timestamp_ms=timestamp_ms, faces=face_results)
            )
        timing["detection_ms"] = int((time.time() - detection_start) * 1000)

    total_ms = int((time.time() - total_start) * 1000)
    logger.info(f"  Detection complete in {total_ms}ms")
    logger.info(f"    Download video: {timing['download_video_ms']}ms")
    logger.info(f"    Download refs: {timing['download_refs_ms']}ms")
    logger.info(f"    Frame extraction: {timing['frame_extraction_ms']}ms")
    logger.info(f"    InsightFace detection: {timing['detection_ms']}ms")

    return DetectFacesResponse(
        frames=frame_analyses,
        frame_width=width,
        frame_height=height,
        sample_fps=request.sample_fps,
        video_duration_ms=video_duration_ms,
        characters_detected=list(characters_detected),
        timing=DetectionTimingBreakdown(
            total_ms=total_ms,
            download_video_ms=timing["download_video_ms"],
            download_refs_ms=timing["download_refs_ms"],
            frame_extraction_ms=timing["frame_extraction_ms"],
            detection_ms=timing["detection_ms"],
        ),
    )


@app.post("/lipsync", response_model=LipSyncResponse)
async def lipsync(request: LipSyncRequest):
    """
    Process lip-sync with automatic face detection.

    Downloads video, audio, and reference images. Detects faces in video,
    matches to reference images, processes lip-sync, and uploads result.
    Returns detailed metadata about processed faces and timing.
    """
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    if face_detector is None or not face_detector.is_loaded:
        raise HTTPException(status_code=503, detail="Face detector not loaded")

    logger.info("=" * 60)
    logger.info("LIP-SYNC")
    logger.info("=" * 60)
    logger.info(f"  Video URL: {request.video_url}")
    logger.info(f"  Audio URL: {request.audio_url}")
    logger.info(f"  Upload URL: {request.upload_url[:50]}...")
    logger.info(f"  Characters: {len(request.characters)}")
    for char in request.characters:
        logger.info(f"    - {char.id}: {char.name}")

    total_start = time.time()
    timing = {
        "download_ms": 0,
        "detection_ms": 0,
        "lipsync_ms": 0,
        "enhancement_ms": 0,
        "encoding_ms": 0,
        "upload_ms": 0,
    }

    # Get requested character IDs
    requested_ids: Set[str] = {c.id for c in request.characters}

    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, "video.mp4")
        audio_path = os.path.join(tmpdir, "audio.wav")
        output_path = os.path.join(tmpdir, "output.mp4")

        # Download video, audio, and reference images
        try:
            download_start = time.time()
            await download_file(request.video_url, video_path)
            await download_file(request.audio_url, audio_path)

            # Download and register reference images for face matching
            for char in request.characters:
                ref_img = await download_image(char.reference_image_url)
                face_detector.register_reference(char.id, ref_img)
                logger.info(f"    Registered reference for {char.id}")

            timing["download_ms"] = int((time.time() - download_start) * 1000)
        except Exception as e:
            logger.error(f"Download failed: {e}")
            raise HTTPException(status_code=400, detail=f"Failed to download: {e}")

        # Get video info
        try:
            video_info = get_video_info(video_path)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to get video info: {e}")

        # Determine time range
        start_ms = request.start_time_ms or 0
        end_ms = request.end_time_ms or video_info["duration_ms"]

        # Run face detection for tracking
        detection_start = time.time()
        logger.info("  Running face detection...")

        frames_dir = os.path.join(tmpdir, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        frame_paths = extract_frames_at_fps(
            video_path, frames_dir, LIPSYNC_DETECTION_FPS, start_ms, end_ms
        )

        # Collect per-frame data for each character
        frame_data: Dict[str, List[FrameFaceData]] = {c.id: [] for c in request.characters}
        unknown_frame_data: Dict[str, List[FrameFaceData]] = {}
        all_detected_ids: Set[str] = set()

        # Track best bbox per character (highest confidence detection)
        best_bbox_per_char: Dict[str, tuple] = {}
        best_confidence_per_char: Dict[str, float] = {}

        for timestamp_ms, frame_path in frame_paths:
            frame = cv2.imread(frame_path)
            if frame is None:
                continue

            faces = face_detector.detect_faces(
                frame, similarity_threshold=request.similarity_threshold
            )

            for face in faces:
                all_detected_ids.add(face.character_id)

                face_data_entry = FrameFaceData(
                    timestamp_ms=timestamp_ms,
                    character_id=face.character_id,
                    bbox=face.bbox,
                    head_pose=face.head_pose,
                    confidence=face.confidence,
                    syncable=face.syncable,
                    sync_quality=face.sync_quality,
                    skip_reason=face.skip_reason,
                )

                if face.character_id in requested_ids:
                    frame_data[face.character_id].append(face_data_entry)
                    # Track best bbox (highest confidence)
                    if face.confidence > best_confidence_per_char.get(face.character_id, 0):
                        best_confidence_per_char[face.character_id] = face.confidence
                        best_bbox_per_char[face.character_id] = face.bbox
                else:
                    # Unknown face
                    if face.character_id not in unknown_frame_data:
                        unknown_frame_data[face.character_id] = []
                    unknown_frame_data[face.character_id].append(face_data_entry)

        timing["detection_ms"] = int((time.time() - detection_start) * 1000)
        logger.info(f"  Detection complete: {len(all_detected_ids)} unique faces")

        # Check if any requested characters were found
        found_chars = [c for c in request.characters if c.id in best_bbox_per_char]
        if not found_chars:
            logger.warning("  No requested characters found in video")
            return LipSyncResponse(
                success=False,
                error_message="No requested characters found in video",
            )

        logger.info(f"  Found {len(found_chars)} of {len(request.characters)} characters")
        for char in found_chars:
            bbox = best_bbox_per_char[char.id]
            logger.info(f"    - {char.id}: bbox={bbox}")

        # Process lip-sync
        lipsync_start = time.time()
        face_results: List[FaceResult] = []

        try:
            pipeline.config.use_enhancement = request.enhance_quality
            pipeline.config.fidelity_weight = request.fidelity_weight

            # For single face, use process_single_face with detected bbox
            if len(found_chars) == 1:
                char = found_chars[0]
                bbox = best_bbox_per_char[char.id]
                pipeline.process_single_face(
                    video_path,
                    audio_path,
                    output_path,
                    target_bbox=bbox,
                )
            else:
                # Multi-face: process each face separately (TODO: implement proper multi-face)
                # For now, just process the first character
                char = found_chars[0]
                bbox = best_bbox_per_char[char.id]
                logger.warning(f"  Multi-face not fully implemented, processing only {char.id}")
                pipeline.process_single_face(
                    video_path,
                    audio_path,
                    output_path,
                    target_bbox=bbox,
                )

            timing["lipsync_ms"] = int((time.time() - lipsync_start) * 1000)

            # Build segments for each requested face
            frame_interval_ms = 1000 // LIPSYNC_DETECTION_FPS

            for char in request.characters:
                char_id = char.id
                frames = frame_data.get(char_id, [])

                if frames:
                    segments = build_segments_for_face(frames)
                    extend_segment_end_times(segments, frame_interval_ms)

                    total_ms, synced_ms, skipped_ms = compute_summary(segments)

                    face_segments = [
                        FaceSegment(
                            start_ms=seg.start_ms,
                            end_ms=seg.end_ms,
                            synced=seg.synced,
                            skip_reason=seg.skip_reason,
                            avg_quality=seg.avg_quality,
                            avg_bbox=seg.avg_bbox,
                            avg_head_pose=seg.avg_head_pose,
                        )
                        for seg in segments
                    ]

                    face_results.append(
                        FaceResult(
                            character_id=char_id,
                            success=True,
                            segments=face_segments,
                            summary=FaceSummary(
                                total_ms=total_ms,
                                synced_ms=synced_ms,
                                skipped_ms=skipped_ms,
                            ),
                        )
                    )
                else:
                    # Character not found in video
                    face_results.append(
                        FaceResult(
                            character_id=char_id,
                            success=False,
                            error_message="Character not found in video",
                            segments=[],
                            summary=None,
                        )
                    )

        except Exception as e:
            logger.error(f"Lip-sync processing failed: {e}")
            import traceback
            traceback.print_exc()
            return LipSyncResponse(
                success=False,
                error_message=str(e),
            )

        # Build unknown faces data
        unknown_faces: List[UnknownFace] = []
        for char_id, frames in unknown_frame_data.items():
            segments = build_segments_for_face(frames)
            extend_segment_end_times(segments, frame_interval_ms)

            face_segments = [
                FaceSegment(
                    start_ms=seg.start_ms,
                    end_ms=seg.end_ms,
                    synced=seg.synced,
                    skip_reason=seg.skip_reason,
                    avg_quality=seg.avg_quality,
                    avg_bbox=seg.avg_bbox,
                    avg_head_pose=seg.avg_head_pose,
                )
                for seg in segments
            ]

            unknown_faces.append(
                UnknownFace(character_id=char_id, segments=face_segments)
            )

        # Check output exists
        if not os.path.exists(output_path):
            raise HTTPException(status_code=500, detail="Output file not created")

        # Get output file info
        output_size = os.path.getsize(output_path)
        output_info = get_video_info(output_path)

        # Upload to presigned URL
        try:
            timing["upload_ms"] = int(await upload_file(request.upload_url, output_path))
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to upload: {e}")

    total_ms = int((time.time() - total_start) * 1000)

    logger.info(f"  Processing complete in {total_ms}ms")
    logger.info(f"  Output size: {output_size / (1024*1024):.2f}MB")

    return LipSyncResponse(
        success=True,
        faces=FacesResult(
            total_detected=len(all_detected_ids),
            processed=len([r for r in face_results if r.success]),
            unknown=len(unknown_faces),
            results=face_results,
            unknown_faces=unknown_faces,
        ),
        output=OutputMetadata(
            duration_ms=output_info["duration_ms"],
            width=output_info["width"],
            height=output_info["height"],
            fps=output_info["fps"],
            file_size_bytes=output_size,
        ),
        timing=TimingBreakdown(
            total_ms=total_ms,
            download_ms=timing["download_ms"],
            detection_ms=timing["detection_ms"],
            lipsync_ms=timing["lipsync_ms"],
            enhancement_ms=timing["enhancement_ms"],
            encoding_ms=timing["encoding_ms"],
            upload_ms=timing["upload_ms"],
        ),
    )


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Multi-Face Lip-Sync API",
        "version": "0.4.0",
        "docs": "/docs",
        "health": "/health",
        "endpoints": [
            "POST /detect-faces - Detect faces with InsightFace (JSON)",
            "POST /lipsync - Process lip-sync, upload to presigned URL (JSON)",
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
