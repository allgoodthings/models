"""
Higgs Audio v2 POC Server

FastAPI server for testing Higgs Audio v2 TTS capabilities.
Run on a GPU instance with 24GB+ VRAM.
"""

import io
import os
import re
import time
import base64
import tempfile
from typing import Optional, List

import numpy as np
import scipy.io.wavfile
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
import uvicorn

# Higgs Audio imports
try:
    from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine, HiggsAudioResponse
    from boson_multimodal.data_types import ChatMLSample, Message, AudioContent
    from boson_multimodal.audio_processing.higgs_audio_tokenizer import load_higgs_audio_tokenizer
    HIGGS_AVAILABLE = True
except ImportError as e:
    HIGGS_AVAILABLE = False
    print(f"WARNING: Higgs Audio not installed. Error: {e}")

app = FastAPI(
    title="Higgs Audio v2 POC",
    description="POC server for testing Higgs Audio v2 TTS",
    version="0.1.0",
)

# Global model instances
model: Optional[HiggsAudioServeEngine] = None
audio_tokenizer = None
MODEL_ID = "bosonai/higgs-audio-v2-generation-3B-base"
TOKENIZER_ID = "bosonai/higgs-audio-v2-tokenizer"

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


def get_model() -> HiggsAudioServeEngine:
    """Get or load the model."""
    global model, audio_tokenizer
    if model is None:
        if not HIGGS_AVAILABLE:
            raise HTTPException(status_code=503, detail="Higgs Audio not installed.")
        print(f"Loading model: {MODEL_ID}")
        start = time.time()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = HiggsAudioServeEngine(MODEL_ID, TOKENIZER_ID, device=device)
        audio_tokenizer = load_higgs_audio_tokenizer(TOKENIZER_ID, device=device)
        print(f"Model loaded in {time.time() - start:.2f}s")
    return model


def audio_to_wav_bytes(audio_array: np.ndarray, sample_rate: int) -> bytes:
    """Convert numpy audio array to WAV bytes."""
    if audio_array.dtype in (np.float32, np.float64):
        audio_array = np.clip(audio_array, -1.0, 1.0)
        audio_array = (audio_array * 32767).astype(np.int16)
    buffer = io.BytesIO()
    scipy.io.wavfile.write(buffer, sample_rate, audio_array)
    return buffer.getvalue()


def log_vram(label: str = ""):
    """Log current VRAM usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        print(f"[VRAM{' ' + label if label else ''}] Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")


def chunk_by_speaker(text: str) -> List[str]:
    """Split text into chunks by [SPEAKER*] tags."""
    lines = text.strip().split("\n")
    chunks = []
    current_chunk = ""

    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Check if line starts with speaker tag
        if re.match(r'\[SPEAKER\d+\]', line):
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = line
        else:
            current_chunk += " " + line if current_chunk else line

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def is_multi_speaker(text: str) -> bool:
    """Check if text contains multiple speaker tags."""
    speakers = set(re.findall(r'\[SPEAKER\d+\]', text))
    return len(speakers) > 1


@app.get("/health")
async def health():
    """Health check endpoint."""
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2),
            "gpu_memory_used_gb": round(torch.cuda.memory_allocated(0) / 1e9, 2),
        }
    return {
        "status": "ok",
        "higgs_available": HIGGS_AVAILABLE,
        "model_loaded": model is not None,
        "model_id": MODEL_ID,
        **gpu_info
    }


@app.post("/generate")
async def generate(request: GenerateRequest):
    """
    Generate speech from text.

    For multi-speaker dialogue, use [SPEAKER0], [SPEAKER1] tags.
    Each speaker turn will be generated separately and concatenated.
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    engine = get_model()
    tmp_path = None

    try:
        # Check if multi-speaker
        if is_multi_speaker(request.text):
            return await generate_multi_speaker(request, engine)

        # Single speaker generation
        effective_system_prompt = request.system_prompt or DEFAULT_SYSTEM_PROMPT
        messages = [Message(role="system", content=effective_system_prompt)]

        # Handle voice cloning
        if request.reference_audio_base64:
            audio_bytes = base64.b64decode(request.reference_audio_base64)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            ref_text = request.reference_text.strip() if request.reference_text else "Please speak naturally."
            messages.append(Message(role="user", content=ref_text))
            messages.append(Message(role="assistant", content=AudioContent(audio_url=tmp_path)))

        messages.append(Message(role="user", content=request.text))

        print(f"[single] Generating: {request.text[:80]}...")
        log_vram("before")
        start = time.time()

        output: HiggsAudioResponse = engine.generate(
            chat_ml_sample=ChatMLSample(messages=messages),
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            stop_strings=["<|end_of_text|>", "<|eot_id|>"],
        )

        generation_time = time.time() - start
        log_vram("after")
        print(f"Generated in {generation_time:.2f}s")

        wav_bytes = audio_to_wav_bytes(output.audio, output.sampling_rate)

        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={
                "X-Generation-Time": str(round(generation_time, 2)),
                "X-Sample-Rate": str(output.sampling_rate),
                "X-Mode": "single",
            }
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


async def generate_multi_speaker(request: GenerateRequest, engine: HiggsAudioServeEngine):
    """Generate multi-speaker dialogue by chunking and concatenating."""

    # Parse speaker tags to build scene description
    speakers = sorted(set(re.findall(r'\[SPEAKER(\d+)\]', request.text)))
    speaker_desc = []
    for idx, spk_id in enumerate(speakers):
        # Alternate between masculine/feminine for variety
        voice = "feminine" if int(spk_id) % 2 == 0 else "masculine"
        speaker_desc.append(f"SPEAKER{spk_id}: {voice}")

    # Build system prompt for multi-speaker
    scene_desc = request.system_prompt if request.system_prompt else "Two people having a conversation."
    system_prompt = (
        "You are an AI assistant designed to convert text into speech.\n"
        "If the user's message includes a [SPEAKER*] tag, do not read out the tag "
        "and generate speech for the following text, using the specified voice.\n\n"
        f"<|scene_desc_start|>\n{scene_desc}\n\n" + "\n".join(speaker_desc) + "\n<|scene_desc_end|>"
    )

    # Chunk by speaker
    chunks = chunk_by_speaker(request.text)
    print(f"[multi-speaker] Generating {len(chunks)} chunks...")
    log_vram("before")

    all_audio = []
    sample_rate = 24000
    start = time.time()

    for i, chunk in enumerate(chunks):
        print(f"  Chunk {i+1}/{len(chunks)}: {chunk[:50]}...")

        # Generate each chunk independently (serve_engine can't handle audio context)
        messages = [
            Message(role="system", content=system_prompt),
            Message(role="user", content=chunk),
        ]

        output: HiggsAudioResponse = engine.generate(
            chat_ml_sample=ChatMLSample(messages=messages),
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            stop_strings=["<|end_of_text|>", "<|eot_id|>"],
        )

        all_audio.append(output.audio)
        sample_rate = output.sampling_rate

    # Concatenate all audio
    combined_audio = np.concatenate(all_audio)
    generation_time = time.time() - start
    log_vram("after")
    print(f"Generated {len(chunks)} chunks in {generation_time:.2f}s")

    wav_bytes = audio_to_wav_bytes(combined_audio, sample_rate)

    return Response(
        content=wav_bytes,
        media_type="audio/wav",
        headers={
            "X-Generation-Time": str(round(generation_time, 2)),
            "X-Sample-Rate": str(sample_rate),
            "X-Mode": "multi-speaker",
            "X-Chunks": str(len(chunks)),
        }
    )


if __name__ == "__main__":
    print("Starting Higgs Audio v2 POC Server...")
    print(f"Model: {MODEL_ID}")
    print("")
    print("Endpoints:")
    print("  GET  /health     - Health check")
    print("  POST /generate   - Generate speech (JSON)")
    print("  GET  /docs       - API docs")
    print("")
    uvicorn.run(app, host="0.0.0.0", port=8888)
