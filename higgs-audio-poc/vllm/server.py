"""
Higgs Audio v2 vLLM Proxy Server

Provides the same API as our POC server but proxies to the vLLM backend.
Adds VRAM monitoring and request tracking.

Usage:
    python server.py

The vLLM backend must be running on port 8000 (start with ./start-vllm.sh)
"""

import io
import os
import time
import base64
import asyncio
from typing import Optional
from datetime import datetime

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
import uvicorn

app = FastAPI(
    title="Higgs Audio v2 vLLM Proxy",
    description="High-throughput TTS with vLLM backend",
    version="0.1.0",
)

# Configuration
VLLM_URL = os.getenv("VLLM_URL", "http://localhost:8000")
MODEL_NAME = "higgs-audio-v2"

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
    temperature: float = Field(1.0, ge=0.1, le=1.5)
    top_p: float = Field(0.95, ge=0.0, le=1.0)
    top_k: int = Field(50, ge=1, le=100)
    max_tokens: int = Field(2048, ge=256, le=8192)


def log_stats(label: str = ""):
    """Log current request stats."""
    print(f"[STATS{' ' + label if label else ''}] "
          f"Total: {request_stats['total_requests']}, "
          f"Success: {request_stats['successful_requests']}, "
          f"Failed: {request_stats['failed_requests']}, "
          f"Concurrent: {request_stats['concurrent_requests']}, "
          f"Peak: {request_stats['peak_concurrent']}")


@app.get("/health")
async def health():
    """Health check endpoint."""
    # Check vLLM backend
    vllm_healthy = False
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{VLLM_URL}/health", timeout=5.0)
            vllm_healthy = resp.status_code == 200
    except:
        pass

    return {
        "status": "ok" if vllm_healthy else "degraded",
        "vllm_backend": VLLM_URL,
        "vllm_healthy": vllm_healthy,
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
    Generate speech from text via vLLM backend.

    Supports concurrent requests for high throughput.
    """
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

        # Build messages for vLLM
        messages = [
            {"role": "system", "content": effective_system_prompt},
        ]

        # Handle voice cloning
        if request.reference_audio_base64:
            ref_text = request.reference_text.strip() if request.reference_text else "Please speak naturally."
            messages.append({"role": "user", "content": ref_text})
            messages.append({
                "role": "assistant",
                "content": [{
                    "type": "input_audio",
                    "input_audio": {
                        "data": request.reference_audio_base64,
                        "format": "wav",
                    }
                }]
            })

        messages.append({"role": "user", "content": request.text})

        # Call vLLM backend
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{VLLM_URL}/v1/chat/completions",
                json={
                    "model": MODEL_NAME,
                    "messages": messages,
                    "modalities": ["text", "audio"],
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                    "extra_body": {"top_k": request.top_k},
                    "max_tokens": request.max_tokens,
                    "stop": ["<|eot_id|>", "<|end_of_text|>", "<|audio_eos|>"],
                }
            )

        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=f"vLLM error: {resp.text}")

        result = resp.json()
        generation_time = time.time() - start

        # Extract audio
        choice = result["choices"][0]
        audio_data = choice["message"].get("audio", {}).get("data")

        if not audio_data:
            raise HTTPException(status_code=500, detail="No audio in response")

        # Decode base64 audio
        audio_bytes = base64.b64decode(audio_data)

        # Update stats
        request_stats["successful_requests"] += 1
        request_stats["total_generation_time"] += generation_time

        print(f"[GEN] {request.text[:50]}... | {generation_time:.2f}s | concurrent: {request_stats['concurrent_requests']}")

        return Response(
            content=audio_bytes,
            media_type="audio/wav",
            headers={
                "X-Generation-Time": str(round(generation_time, 2)),
                "X-Concurrent-Requests": str(request_stats["concurrent_requests"]),
            }
        )

    except HTTPException:
        request_stats["failed_requests"] += 1
        raise
    except Exception as e:
        request_stats["failed_requests"] += 1
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
    finally:
        request_stats["concurrent_requests"] -= 1


@app.post("/generate/batch")
async def generate_batch(requests: list[GenerateRequest]):
    """
    Generate multiple audio files concurrently.

    Returns a list of base64-encoded audio or error messages.
    """
    async def process_one(req: GenerateRequest, idx: int):
        try:
            response = await generate(req)
            return {
                "index": idx,
                "success": True,
                "audio_base64": base64.b64encode(response.body).decode(),
                "generation_time": float(response.headers.get("X-Generation-Time", 0)),
            }
        except Exception as e:
            return {
                "index": idx,
                "success": False,
                "error": str(e),
            }

    # Process all requests concurrently
    tasks = [process_one(req, i) for i, req in enumerate(requests)]
    results = await asyncio.gather(*tasks)

    return {
        "results": sorted(results, key=lambda x: x["index"]),
        "total": len(requests),
        "successful": sum(1 for r in results if r["success"]),
        "failed": sum(1 for r in results if not r["success"]),
    }


if __name__ == "__main__":
    print("Starting Higgs Audio v2 vLLM Proxy Server...")
    print(f"vLLM Backend: {VLLM_URL}")
    print(f"Model: {MODEL_NAME}")
    print("")
    print("Endpoints:")
    print("  GET  /health        - Health check")
    print("  GET  /stats         - Request statistics")
    print("  POST /generate      - Generate speech (single)")
    print("  POST /generate/batch - Generate speech (batch)")
    print("  GET  /docs          - API docs")
    print("")

    uvicorn.run(app, host="0.0.0.0", port=8888)
