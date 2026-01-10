"""
Higgs Audio v2 Proxy Server for RunPod

Provides a unified /generate API that proxies to the vLLM backend.
This gives you the same API as other POCs while using vLLM's concurrent inference.

Usage:
    # Start vLLM first
    ./start-vllm.sh --background

    # Then start this proxy
    python server.py
"""

import os
import io
import time
import base64
import asyncio
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
import uvicorn

app = FastAPI(
    title="Higgs Audio v2 Proxy Server",
    description="Unified API proxying to vLLM backend",
    version="0.1.0",
)

# Configuration
PROXY_PORT = int(os.getenv("PROXY_PORT", "8888"))
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
    temperature: float = Field(0.5, ge=0.1, le=1.5)
    top_p: float = Field(0.95, ge=0.0, le=1.0)
    top_k: int = Field(50, ge=1, le=100)
    max_tokens: int = Field(2048, ge=256, le=8192)


@app.get("/health")
async def health():
    """Health check endpoint."""
    # Check vLLM backend
    vllm_healthy = False
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{VLLM_URL}/health")
            vllm_healthy = resp.status_code == 200
    except Exception:
        pass

    return {
        "status": "ok" if vllm_healthy else "degraded",
        "vllm_backend": VLLM_URL,
        "vllm_healthy": vllm_healthy,
        "model": MODEL_NAME,
        "mode": "vllm",
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
    Generate speech from text using vLLM backend.

    Supports concurrent requests via vLLM's continuous batching.
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

    try:
        start = time.time()

        # Build messages for chat completion
        effective_system_prompt = request.system_prompt or DEFAULT_SYSTEM_PROMPT
        messages = [
            {"role": "system", "content": effective_system_prompt}
        ]

        # Add voice cloning context if provided
        if request.reference_audio_base64 and request.reference_text:
            messages.append({"role": "user", "content": request.reference_text})
            messages.append({
                "role": "assistant",
                "content": [{"type": "audio", "audio_url": "", "raw_audio": request.reference_audio_base64}]
            })

        messages.append({"role": "user", "content": request.text})

        # Call vLLM backend - use the OpenAI-compatible speech endpoint
        async with httpx.AsyncClient(timeout=120.0) as client:
            # Try the speech endpoint first (simpler)
            resp = await client.post(
                f"{VLLM_URL}/v1/audio/speech",
                json={
                    "model": MODEL_NAME,
                    "input": request.text,
                    "voice": "alloy",  # Higgs ignores this but it's required
                    "response_format": "wav",
                },
            )

            if resp.status_code != 200:
                raise HTTPException(
                    status_code=resp.status_code,
                    detail=f"vLLM error: {resp.text}"
                )

            audio_bytes = resp.content

        generation_time = time.time() - start

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
                "X-Mode": "vllm",
            }
        )

    except httpx.RequestError as e:
        request_stats["failed_requests"] += 1
        raise HTTPException(status_code=503, detail=f"vLLM backend unavailable: {str(e)}")
    except Exception as e:
        request_stats["failed_requests"] += 1
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
    finally:
        request_stats["concurrent_requests"] -= 1


@app.post("/generate/batch")
async def generate_batch(requests: list[GenerateRequest]):
    """
    Generate multiple audio files concurrently.

    With vLLM, these run in parallel via continuous batching.
    """
    # Run all requests concurrently
    tasks = [generate(req) for req in requests]
    results = []

    responses = await asyncio.gather(*tasks, return_exceptions=True)

    for i, resp in enumerate(responses):
        if isinstance(resp, Exception):
            results.append({
                "index": i,
                "success": False,
                "error": str(resp),
            })
        else:
            results.append({
                "index": i,
                "success": True,
                "audio_base64": base64.b64encode(resp.body).decode(),
                "generation_time": float(resp.headers.get("X-Generation-Time", 0)),
            })

    return {
        "results": results,
        "total": len(requests),
        "successful": sum(1 for r in results if r["success"]),
        "failed": sum(1 for r in results if not r["success"]),
        "mode": "vllm",
    }


if __name__ == "__main__":
    print("Starting Higgs Audio v2 Proxy Server...")
    print(f"vLLM Backend: {VLLM_URL}")
    print(f"Proxy Port: {PROXY_PORT}")
    print("")
    print("Endpoints:")
    print("  GET  /health         - Health check")
    print("  GET  /stats          - Request statistics")
    print("  POST /generate       - Generate speech (concurrent)")
    print("  POST /generate/batch - Generate multiple (parallel)")
    print("  GET  /docs           - API docs")
    print("")

    uvicorn.run(app, host="0.0.0.0", port=PROXY_PORT)
