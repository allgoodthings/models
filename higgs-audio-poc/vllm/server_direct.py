"""
Higgs Audio v2 Direct Server

Uses HiggsAudioServeEngine directly (no vLLM Docker required).
Note: max_batch_size=1, so only one request at a time.

Usage:
    python server_direct.py
"""

import io
import os
import time
import base64
import asyncio
import functools
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

import torch
import numpy as np
import scipy.io.wavfile as wavfile
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
import uvicorn

from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine
from boson_multimodal.data_types import ChatMLSample, Message, TextContent, AudioContent

app = FastAPI(
    title="Higgs Audio v2 Direct Server",
    description="TTS with HiggsAudioServeEngine (single request mode)",
    version="0.1.0",
)

# Configuration
PORT = int(os.getenv("PORT", "8888"))
MODEL_NAME = "higgs-audio-v2"
MODEL_PATH = "bosonai/higgs-audio-v2-generation-3B-base"
TOKENIZER_PATH = "bosonai/higgs-audio-v2-tokenizer"

# Global engine (initialized on startup)
engine: Optional[HiggsAudioServeEngine] = None
executor = ThreadPoolExecutor(max_workers=1)  # Single worker for sequential processing

# Request tracking
request_stats = {
    "total_requests": 0,
    "successful_requests": 0,
    "failed_requests": 0,
    "total_generation_time": 0.0,
    "concurrent_requests": 0,
    "peak_concurrent": 0,
}

DEFAULT_SYSTEM_PROMPT = (
    "Generate audio following instruction.\n\n"
    "<|scene_desc_start|>\n"
    "Audio is recorded from a quiet room.\n"
    "<|scene_desc_end|>"
)


class GenerateRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize")
    system_prompt: Optional[str] = Field(None, description="Custom system prompt")
    reference_audio_base64: Optional[str] = Field(None, description="Base64-encoded WAV for voice cloning")
    reference_text: Optional[str] = Field(None, description="Transcript of reference audio")
    temperature: float = Field(0.3, ge=0.1, le=1.5)
    top_p: float = Field(0.95, ge=0.0, le=1.0)
    top_k: int = Field(50, ge=1, le=100)
    max_tokens: int = Field(2048, ge=256, le=8192)


def log_vram(label: str = ""):
    """Log current VRAM usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        print(f"[VRAM{' ' + label if label else ''}] Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")


def log_stats(label: str = ""):
    """Log current request stats."""
    print(f"[STATS{' ' + label if label else ''}] "
          f"Total: {request_stats['total_requests']}, "
          f"Success: {request_stats['successful_requests']}, "
          f"Failed: {request_stats['failed_requests']}, "
          f"Concurrent: {request_stats['concurrent_requests']}, "
          f"Peak: {request_stats['peak_concurrent']}")


def generate_sync(
    text: str,
    system_prompt: str,
    reference_audio_base64: Optional[str],
    reference_text: Optional[str],
    temperature: float,
    top_p: float,
    top_k: int,
    max_tokens: int,
) -> bytes:
    """Synchronous generation using HiggsAudioServeEngine."""
    global engine

    messages = [
        Message(role="system", content=[TextContent(text=system_prompt)])
    ]

    # Handle voice cloning
    if reference_audio_base64 and reference_text:
        messages.append(Message(role="user", content=[TextContent(text=reference_text)]))
        # Use raw_audio for base64 encoded audio, audio_url can be empty
        messages.append(Message(role="assistant", content=[AudioContent(audio_url="", raw_audio=reference_audio_base64)]))

    messages.append(Message(role="user", content=[TextContent(text=text)]))

    sample = ChatMLSample(messages=messages)

    result = engine.generate(
        sample,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_new_tokens=max_tokens,
    )

    # Convert to WAV bytes
    audio = result.audio
    if isinstance(audio, torch.Tensor):
        audio = audio.cpu().numpy()

    # Normalize and convert to int16
    audio = np.clip(audio, -1, 1)
    audio_int16 = (audio * 32767).astype(np.int16)

    # Write to WAV buffer
    buf = io.BytesIO()
    wavfile.write(buf, 24000, audio_int16)
    return buf.getvalue()


@app.on_event("startup")
async def startup():
    global engine
    print(f"Loading Higgs Audio model...")
    print(f"Model: {MODEL_PATH}")
    print(f"Tokenizer: {TOKENIZER_PATH}")
    log_vram("before load")

    engine = HiggsAudioServeEngine(
        model_name_or_path=MODEL_PATH,
        audio_tokenizer_name_or_path=TOKENIZER_PATH,
    )

    log_vram("after load")
    print(f"Model loaded successfully")
    print(f"Server ready on port {PORT}")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok" if engine is not None else "not_ready",
        "mode": "direct",
        "model": MODEL_NAME,
        "stats": request_stats,
    }


@app.get("/stats")
async def stats():
    """Get detailed request statistics."""
    avg_time = (request_stats["total_generation_time"] / request_stats["successful_requests"]
                if request_stats["successful_requests"] > 0 else 0)
    return {
        **request_stats,
        "avg_generation_time": round(avg_time, 2),
    }


@app.post("/generate")
async def generate(request: GenerateRequest):
    """
    Generate speech from text.

    Note: Direct mode processes one request at a time.
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    # Track concurrent requests
    request_stats["total_requests"] += 1
    request_stats["concurrent_requests"] += 1
    request_stats["peak_concurrent"] = max(
        request_stats["peak_concurrent"],
        request_stats["concurrent_requests"]
    )

    effective_system_prompt = request.system_prompt or DEFAULT_SYSTEM_PROMPT

    try:
        start = time.time()

        # Run generation in thread pool to not block event loop
        loop = asyncio.get_event_loop()
        audio_bytes = await loop.run_in_executor(
            executor,
            functools.partial(
                generate_sync,
                text=request.text,
                system_prompt=effective_system_prompt,
                reference_audio_base64=request.reference_audio_base64,
                reference_text=request.reference_text,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                max_tokens=request.max_tokens,
            )
        )

        generation_time = time.time() - start

        # Update stats
        request_stats["successful_requests"] += 1
        request_stats["total_generation_time"] += generation_time

        print(f"[GEN] {request.text[:50]}... | {generation_time:.2f}s")
        log_vram()

        return Response(
            content=audio_bytes,
            media_type="audio/wav",
            headers={
                "X-Generation-Time": str(round(generation_time, 2)),
                "X-Concurrent-Requests": str(request_stats["concurrent_requests"]),
                "X-Mode": "direct",
            }
        )

    except Exception as e:
        request_stats["failed_requests"] += 1
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
    finally:
        request_stats["concurrent_requests"] -= 1


@app.post("/generate/batch")
async def generate_batch(requests: list[GenerateRequest]):
    """
    Generate multiple audio files.

    Note: In direct mode, these are processed sequentially.
    """
    results = []
    for i, req in enumerate(requests):
        try:
            response = await generate(req)
            results.append({
                "index": i,
                "success": True,
                "audio_base64": base64.b64encode(response.body).decode(),
                "generation_time": float(response.headers.get("X-Generation-Time", 0)),
            })
        except Exception as e:
            results.append({
                "index": i,
                "success": False,
                "error": str(e),
            })

    return {
        "results": results,
        "total": len(requests),
        "successful": sum(1 for r in results if r["success"]),
        "failed": sum(1 for r in results if not r["success"]),
        "mode": "direct",
    }


if __name__ == "__main__":
    print("Starting Higgs Audio v2 Direct Server...")
    print(f"Mode: HiggsAudioServeEngine (single request)")
    print(f"Model: {MODEL_NAME}")
    print("")
    print("Endpoints:")
    print("  GET  /health        - Health check")
    print("  GET  /stats         - Request statistics")
    print("  POST /generate      - Generate speech (single)")
    print("  POST /generate/batch - Generate speech (batch, sequential)")
    print("  GET  /docs          - API docs")
    print("")

    uvicorn.run(app, host="0.0.0.0", port=PORT)
