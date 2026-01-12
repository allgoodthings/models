"""
Higgs Audio Native POC Server

FastAPI server using native boson_multimodal library (not vLLM).
Used to test if system_prompt + voice cloning work together natively.
"""

import base64
import hashlib
import io
import json
import os
import tempfile
import threading
import time
import traceback
import urllib.request
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

# Voice cache directory
VOICE_CACHE_DIR = os.environ.get("VOICE_CACHE_DIR", "/tmp/voice_cache")

app = FastAPI(
    title="Higgs Audio Native POC",
    description="Native POC server for testing Higgs Audio TTS with system_prompt + voice cloning",
    version="0.2.0",
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


# =============================================================================
# Voice Cache (same as vLLM version)
# =============================================================================

class VoiceCache:
    """Thread-safe file-based cache for voice audio and metadata."""

    def __init__(self, cache_dir: str = VOICE_CACHE_DIR):
        self.cache_dir = cache_dir
        self._locks: dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()

    def _get_lock(self, voice_id: str) -> threading.Lock:
        """Get or create a lock for a specific voice_id."""
        with self._global_lock:
            if voice_id not in self._locks:
                self._locks[voice_id] = threading.Lock()
            return self._locks[voice_id]

    def _get_paths(self, voice_id: str) -> tuple[str, str]:
        """Get filesystem paths for voice audio and metadata."""
        safe_id = hashlib.md5(voice_id.encode()).hexdigest()[:16]
        return (
            os.path.join(self.cache_dir, f"{safe_id}.wav"),
            os.path.join(self.cache_dir, f"{safe_id}.json"),
        )

    def _load_metadata(self, meta_path: str) -> Optional[str]:
        """Load reference_text from metadata file."""
        if not os.path.exists(meta_path):
            return None
        try:
            with open(meta_path, "r") as f:
                return json.load(f).get("reference_text")
        except Exception as e:
            print(f"[VoiceCache] Failed to load metadata: {e}")
            return None

    def _save_metadata(self, meta_path: str, voice_id: str, reference_text: str) -> None:
        """Save reference_text to metadata file."""
        try:
            with open(meta_path, "w") as f:
                json.dump({"voice_id": voice_id, "reference_text": reference_text}, f)
        except Exception as e:
            print(f"[VoiceCache] Failed to save metadata: {e}")

    def get(self, voice_id: str) -> tuple[Optional[bytes], Optional[str]]:
        """Get cached voice. Returns (audio_bytes, reference_text) or (None, None)."""
        audio_path, meta_path = self._get_paths(voice_id)
        if not os.path.exists(audio_path):
            return None, None
        with open(audio_path, "rb") as f:
            return f.read(), self._load_metadata(meta_path)

    def download_and_cache(
        self, voice_id: str, voice_url: str, reference_text: Optional[str] = None
    ) -> tuple[bytes, Optional[str]]:
        """
        Download voice from URL and cache. Thread-safe with per-voice locking.
        Returns (audio_bytes, reference_text).
        """
        audio_path, meta_path = self._get_paths(voice_id)

        # Fast path: already cached
        if os.path.exists(audio_path):
            print(f"[VoiceCache] Cache hit: {voice_id}")
            cached_audio, cached_text = self.get(voice_id)
            return cached_audio, reference_text or cached_text

        # Slow path: need to download
        lock = self._get_lock(voice_id)
        with lock:
            # Double-check after acquiring lock
            if os.path.exists(audio_path):
                print(f"[VoiceCache] Cache hit (after lock): {voice_id}")
                cached_audio, cached_text = self.get(voice_id)
                return cached_audio, reference_text or cached_text

            # Download
            os.makedirs(self.cache_dir, exist_ok=True)
            print(f"[VoiceCache] Downloading {voice_id} from {voice_url}")

            try:
                with urllib.request.urlopen(voice_url, timeout=30) as response:
                    audio_data = response.read()
            except Exception as e:
                raise ValueError(f"Failed to download voice: {e}") from e

            # Atomic write
            temp_path = audio_path + f".tmp.{os.getpid()}"
            try:
                with open(temp_path, "wb") as f:
                    f.write(audio_data)
                os.rename(temp_path, audio_path)
            except Exception:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                raise

            if reference_text:
                self._save_metadata(meta_path, voice_id, reference_text)

            print(f"[VoiceCache] Cached {voice_id} ({len(audio_data)} bytes)")
            return audio_data, reference_text


# Global voice cache instance
voice_cache = VoiceCache()


# =============================================================================
# Request/Response Models
# =============================================================================

class GenerateRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize")
    system_prompt: Optional[str] = Field(None, description="Custom system prompt with scene description")

    # Legacy: direct base64 audio
    reference_audio_base64: Optional[str] = Field(None, description="Base64-encoded WAV for voice cloning")
    reference_text: Optional[str] = Field(None, description="Transcript of reference audio")

    # New: voice caching
    voice_id: Optional[str] = Field(None, description="Voice ID for caching")
    voice_url: Optional[str] = Field(None, description="URL to download voice from if not cached")

    # Generation params
    temperature: float = Field(0.3, ge=0.1, le=1.5)
    top_p: float = Field(0.95, ge=0.0, le=1.0)
    top_k: int = Field(50, ge=1, le=100)
    max_tokens: int = Field(2048, ge=256, le=8192)


# =============================================================================
# Utilities
# =============================================================================

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


def resolve_voice_reference(request: GenerateRequest) -> tuple[Optional[bytes], Optional[str]]:
    """
    Resolve voice reference audio.
    Priority: voice_id cache -> voice_url download -> reference_audio_base64 -> None
    """
    # Priority 1: voice_id with caching
    if request.voice_id:
        cached_audio, cached_text = voice_cache.get(request.voice_id)
        if cached_audio:
            print(f"[Voice] Using cached voice: {request.voice_id}")
            return cached_audio, request.reference_text or cached_text

        # Download if URL provided
        if request.voice_url:
            audio_bytes, ref_text = voice_cache.download_and_cache(
                request.voice_id, request.voice_url, request.reference_text
            )
            return audio_bytes, ref_text

        print(f"[Voice] voice_id '{request.voice_id}' not cached and no voice_url provided")

    # Priority 2: direct base64 audio
    if request.reference_audio_base64:
        print("[Voice] Using direct base64 reference audio")
        return base64.b64decode(request.reference_audio_base64), request.reference_text

    # No voice cloning
    return None, None


# =============================================================================
# Endpoints
# =============================================================================

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
        "implementation": "native",
        "higgs_available": HIGGS_AVAILABLE,
        "model_loaded": model is not None,
        "model_id": MODEL_ID,
        **gpu_info
    }


@app.post("/generate")
async def generate(request: GenerateRequest):
    """
    Generate speech from text.

    - system_prompt: Always used when provided (controls emotion/style)
    - voice_id + voice_url: Voice cloning with caching
    - reference_audio_base64: Legacy voice cloning (no caching)

    system_prompt and voice cloning work together.
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    engine = get_model()
    tmp_path = None

    try:
        # System prompt: always use when provided
        effective_system_prompt = request.system_prompt or DEFAULT_SYSTEM_PROMPT

        # Debug logging
        print(f"=== GENERATE DEBUG ===")
        print(f"system_prompt provided: {bool(request.system_prompt)}")
        print(f"effective_system_prompt: {effective_system_prompt[:100]}...")

        messages = [Message(role="system", content=effective_system_prompt)]

        # Voice cloning: resolve reference audio
        ref_audio_bytes, ref_text = resolve_voice_reference(request)

        if ref_audio_bytes:
            # Write to temp file for boson_multimodal
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(ref_audio_bytes)
                tmp_path = tmp.name

            ref_text = ref_text or "Please speak naturally."
            messages.append(Message(role="user", content=ref_text))
            messages.append(Message(role="assistant", content=AudioContent(audio_url=tmp_path)))
            print(f"has_reference_audio: True, ref_text: {ref_text[:50]}...")
        else:
            print(f"has_reference_audio: False")

        messages.append(Message(role="user", content=request.text))

        print(f"num_messages: {len(messages)}")
        for i, msg in enumerate(messages):
            if hasattr(msg.content, 'audio_url'):
                print(f"  msg[{i}] role={msg.role} content=<audio>")
            else:
                content_str = str(msg.content)[:80]
                print(f"  msg[{i}] role={msg.role} content={content_str}...")
        print(f"=== END DEBUG ===")

        print(f"[generate] Text: {request.text[:80]}...")
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

        duration = len(output.audio) / output.sampling_rate
        print(f"Generated {duration:.2f}s audio in {generation_time:.2f}s")

        wav_bytes = audio_to_wav_bytes(output.audio, output.sampling_rate)

        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={
                "X-Generation-Time": str(round(generation_time, 2)),
                "X-Sample-Rate": str(output.sampling_rate),
                "X-Duration": str(round(duration, 2)),
                "X-Has-Voice-Clone": str(bool(ref_audio_bytes)),
                "X-Has-System-Prompt": str(bool(request.system_prompt)),
            }
        )

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


if __name__ == "__main__":
    print("=" * 60)
    print("Higgs Audio Native POC Server")
    print("=" * 60)
    print(f"Model: {MODEL_ID}")
    print(f"Voice cache: {VOICE_CACHE_DIR}")
    print("")
    print("Endpoints:")
    print("  GET  /health     - Health check")
    print("  POST /generate   - Generate speech")
    print("  GET  /docs       - API docs")
    print("")
    print("Testing system_prompt + voice cloning together")
    print("=" * 60)
    uvicorn.run(app, host="0.0.0.0", port=8000)
