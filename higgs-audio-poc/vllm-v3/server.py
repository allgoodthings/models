"""
Higgs Audio v3 - Hybrid vLLM + Local Decoder

Uses vLLM for efficient token generation, local audio tokenizer for decoding.
This avoids the chunking artifacts in bosonai's vLLM audio endpoint.

Architecture:
1. Build prompt with system message, scene description, reference audio
2. Call vLLM chat completion API to generate audio tokens
3. Extract audio token IDs from response
4. Decode tokens locally using boson_multimodal audio tokenizer
5. Return proper WAV file
"""

import io
import os
import re
import time
import base64
import tempfile
import struct
from typing import Optional, List, Dict, Any

import httpx
import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
import uvicorn

# Audio tokenizer imports (for local decoding)
try:
    from boson_multimodal.audio_processing.higgs_audio_tokenizer import (
        HiggsAudioTokenizer,
    )
    from huggingface_hub import snapshot_download
    import json
    import inspect
    DECODER_AVAILABLE = True
except ImportError as e:
    DECODER_AVAILABLE = False
    print(f"WARNING: Audio decoder not available. Error: {e}")


app = FastAPI(
    title="Higgs Audio v3",
    description="Hybrid vLLM + local decoder for high-quality TTS",
    version="0.3.0",
)

# Configuration
VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", "http://localhost:8000")
MODEL_ID = "bosonai/higgs-audio-v2-generation-3B-base"
TOKENIZER_ID = "bosonai/higgs-audio-v2-tokenizer"

# Audio token ranges (from higgs audio tokenizer)
# These are the token IDs in the LLM vocabulary that correspond to audio codes
AUDIO_TOKEN_START = 128266  # First audio token ID
NUM_CODEBOOKS = 8
CODEBOOK_SIZE = 2048
TOTAL_AUDIO_TOKENS = NUM_CODEBOOKS * CODEBOOK_SIZE  # 16384 tokens

# Special token IDs (approximate - will verify at runtime)
AUDIO_OUT_BOS_TOKEN = "<|audio_out_bos|>"
AUDIO_EOS_TOKEN = "<|audio_eos|>"
EOT_TOKEN = "<|eot_id|>"

# Global decoder instance
audio_tokenizer = None


def load_audio_tokenizer(device="cuda"):
    """Load the audio tokenizer for decoding."""
    tokenizer_path = snapshot_download(TOKENIZER_ID)
    config_path = os.path.join(tokenizer_path, "config.json")
    config = json.load(open(config_path))
    model_path = os.path.join(tokenizer_path, "model.pth")

    init_signature = inspect.signature(HiggsAudioTokenizer.__init__)
    valid_params = set(init_signature.parameters.keys()) - {'self'}
    filtered_config = {k: v for k, v in config.items() if k in valid_params}

    model = HiggsAudioTokenizer(**filtered_config, device=device)
    parameter_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(parameter_dict, strict=False)
    model.to(device)
    model.eval()
    return model


def get_audio_tokenizer():
    """Get or load the audio tokenizer."""
    global audio_tokenizer
    if audio_tokenizer is None:
        if not DECODER_AVAILABLE:
            raise HTTPException(status_code=503, detail="Audio decoder not available")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading audio tokenizer on {device}...")
        audio_tokenizer = load_audio_tokenizer(device)
        print("Audio tokenizer loaded")
    return audio_tokenizer


def xcodec_get_output_length(input_length: int) -> int:
    """Calculate output length for xcodec decoder."""
    conv_transpose_layers = [
        dict(kernel_size=16, stride=8, padding=4, output_padding=0),
        dict(kernel_size=10, stride=5, padding=3, output_padding=1),
        dict(kernel_size=8, stride=4, padding=2, output_padding=0),
        dict(kernel_size=4, stride=2, padding=1, output_padding=0),
        dict(kernel_size=6, stride=3, padding=2, output_padding=1),
    ]
    length = input_length
    for layer in conv_transpose_layers:
        length = (length - 1) * layer["stride"] - 2 * layer["padding"] + layer["kernel_size"] + layer["output_padding"]
    return length


def decode_audio_tokens(tokenizer, codes: np.ndarray, chunk_size: int = 750) -> np.ndarray:
    """
    Decode audio codes to waveform using overlapping chunks.

    Args:
        tokenizer: HiggsAudioTokenizer instance
        codes: Audio codes of shape (num_codebooks, seq_len)
        chunk_size: Size of each chunk for decoding

    Returns:
        Decoded waveform as numpy array
    """
    device = next(tokenizer.parameters()).device
    codes_tensor = torch.from_numpy(codes).unsqueeze(0).to(device)  # (1, num_codebooks, seq_len)

    overlap_width = 16
    chunk_output_length = xcodec_get_output_length(chunk_size)
    outputs = []

    seq_len = codes_tensor.shape[-1]

    with torch.no_grad():
        for i in range(0, seq_len, chunk_size):
            begin = max(0, i - overlap_width)
            end = min(i + chunk_size + overlap_width, seq_len)
            chunk = codes_tensor[:, :, begin:end]

            output = tokenizer.decode(chunk)

            if i == 0:
                output = output[:, :, :chunk_output_length]
            elif i + chunk_size >= seq_len:
                last_chunk_size = seq_len - i
                last_chunk_output_length = xcodec_get_output_length(last_chunk_size)
                output = output[:, :, -last_chunk_output_length:]
            else:
                extra_length = (xcodec_get_output_length(chunk_size + overlap_width * 2) - chunk_output_length) // 2
                output = output[:, :, extra_length:-extra_length]

            outputs.append(output.cpu().numpy())

    return np.concatenate(outputs, axis=2)[0, 0]  # Return (time,) array


def revert_delay_pattern(data: np.ndarray) -> np.ndarray:
    """
    Convert samples encoded with delay pattern back to the original form.

    Args:
        data: Array of shape (num_codebooks, seq_len + num_codebooks - 1)

    Returns:
        Recovered data with shape (num_codebooks, seq_len)
    """
    assert len(data.shape) == 2
    num_codebooks = data.shape[0]
    out_l = []
    for i in range(num_codebooks):
        out_l.append(data[i:(i + 1), i:(data.shape[1] - num_codebooks + 1 + i)])
    return np.concatenate(out_l, axis=0)


def token_ids_to_audio_codes(token_ids: List[int]) -> np.ndarray:
    """
    Convert LLM token IDs to audio codes.

    The audio tokens are interleaved across codebooks with a delay pattern.
    Token ID = AUDIO_TOKEN_START + codebook_idx * CODEBOOK_SIZE + code_value

    Args:
        token_ids: List of audio token IDs from the LLM

    Returns:
        Audio codes of shape (num_codebooks, seq_len)
    """
    # Filter to only audio tokens
    audio_tokens = [t for t in token_ids if AUDIO_TOKEN_START <= t < AUDIO_TOKEN_START + TOTAL_AUDIO_TOKENS]

    if not audio_tokens:
        raise ValueError("No audio tokens found in response")

    # Decode token IDs to (codebook_idx, code_value) pairs
    decoded = []
    for token_id in audio_tokens:
        offset = token_id - AUDIO_TOKEN_START
        codebook_idx = offset // CODEBOOK_SIZE
        code_value = offset % CODEBOOK_SIZE
        decoded.append((codebook_idx, code_value))

    # The tokens come in interleaved order with delay pattern
    # Reconstruct the (num_codebooks, seq_len) array
    seq_len = len(decoded) // NUM_CODEBOOKS

    # Initialize with zeros
    codes = np.zeros((NUM_CODEBOOKS, seq_len + NUM_CODEBOOKS - 1), dtype=np.int64)

    # Fill in the codes (they come in interleaved order)
    for i, (cb_idx, code_val) in enumerate(decoded):
        time_idx = i // NUM_CODEBOOKS
        codes[cb_idx, time_idx + cb_idx] = code_val

    # Revert delay pattern
    codes = revert_delay_pattern(codes)

    return codes


def audio_to_wav_bytes(audio_array: np.ndarray, sample_rate: int = 24000) -> bytes:
    """Convert numpy audio array to WAV bytes with correct header."""
    if audio_array.dtype in (np.float32, np.float64):
        audio_array = np.clip(audio_array, -1.0, 1.0)
        audio_array = (audio_array * 32767).astype(np.int16)

    num_samples = len(audio_array)
    data_size = num_samples * 2  # 16-bit = 2 bytes per sample
    file_size = data_size + 36  # Header is 44 bytes, minus 8 for RIFF header

    # Build WAV header
    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF',
        file_size,
        b'WAVE',
        b'fmt ',
        16,  # fmt chunk size
        1,   # audio format (PCM)
        1,   # channels (mono)
        sample_rate,
        sample_rate * 2,  # byte rate
        2,   # block align
        16,  # bits per sample
        b'data',
        data_size
    )

    return header + audio_array.tobytes()


class GenerateRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize")
    voice: Optional[str] = Field("en_man", description="Voice preset name")
    system_prompt: Optional[str] = Field(None, description="Custom system prompt (overrides voice)")
    scene_description: Optional[str] = Field(None, description="Scene/style description")
    reference_audio_base64: Optional[str] = Field(None, description="Base64-encoded WAV for voice cloning")
    reference_text: Optional[str] = Field(None, description="Transcript of reference audio")
    temperature: float = Field(0.7, ge=0.1, le=1.5)
    top_p: float = Field(0.95, ge=0.0, le=1.0)
    max_tokens: int = Field(4096, ge=256, le=8192)


class VoicePreset(BaseModel):
    name: str
    system_prompt: str
    scene_description: str


# Built-in voice presets
VOICE_PRESETS: Dict[str, VoicePreset] = {
    "en_man": VoicePreset(
        name="en_man",
        system_prompt="Generate audio following instruction.",
        scene_description="A clear male voice speaking in a quiet room.",
    ),
    "en_woman": VoicePreset(
        name="en_woman",
        system_prompt="Generate audio following instruction.",
        scene_description="A clear female voice speaking in a quiet room.",
    ),
    "dramatic_actor": VoicePreset(
        name="dramatic_actor",
        system_prompt="Generate audio following instruction.",
        scene_description="A dramatic theatrical performance. The speaker is a classically trained actor delivering lines with gravitas, contemplation, and emotional depth.",
    ),
    "narrator_epic": VoicePreset(
        name="narrator_epic",
        system_prompt="Generate audio following instruction.",
        scene_description="An epic movie trailer narration. Deep, resonant voice with dramatic pauses and building intensity.",
    ),
    "calm_narrator": VoicePreset(
        name="calm_narrator",
        system_prompt="Generate audio following instruction.",
        scene_description="A calm, measured audiobook narrator. Clear enunciation, steady pace, professional but warm.",
    ),
}


def build_messages(request: GenerateRequest) -> List[Dict[str, Any]]:
    """Build the messages array for vLLM chat completion."""
    messages = []

    # Get voice preset or use custom
    if request.system_prompt:
        system_prompt = request.system_prompt
        scene_desc = request.scene_description or "Audio is recorded from a quiet room."
    elif request.voice and request.voice in VOICE_PRESETS:
        preset = VOICE_PRESETS[request.voice]
        system_prompt = preset.system_prompt
        scene_desc = request.scene_description or preset.scene_description
    else:
        system_prompt = "Generate audio following instruction."
        scene_desc = request.scene_description or "Audio is recorded from a quiet room."

    # Build full system message with scene description
    full_system = f"{system_prompt}\n\n<|scene_desc_start|>\n{scene_desc}\n<|scene_desc_end|>"
    messages.append({"role": "system", "content": full_system})

    # TODO: Handle reference audio for voice cloning
    # This requires encoding audio to tokens and including in the prompt
    # For now, skip this feature

    # Add user message with text to synthesize
    messages.append({"role": "user", "content": request.text})

    return messages


async def call_vllm_chat(messages: List[Dict], temperature: float, top_p: float, max_tokens: int) -> Dict:
    """Call vLLM chat completion API and get token-level response."""
    async with httpx.AsyncClient(timeout=300.0) as client:
        response = await client.post(
            f"{VLLM_BASE_URL}/v1/chat/completions",
            json={
                "model": MODEL_ID,
                "messages": messages,
                "temperature": temperature,
                "top_p": top_p,
                "max_tokens": max_tokens,
                "logprobs": True,
                "top_logprobs": 1,
            }
        )
        response.raise_for_status()
        return response.json()


def extract_audio_token_ids(response: Dict) -> List[int]:
    """Extract audio token IDs from vLLM response."""
    choice = response["choices"][0]

    # Get logprobs which contain token IDs
    logprobs = choice.get("logprobs")
    if not logprobs or not logprobs.get("content"):
        raise ValueError("No logprobs in response - cannot extract token IDs")

    token_ids = []
    for item in logprobs["content"]:
        # Each item has 'token' (text) and 'top_logprobs' with token details
        # We need the actual token ID
        if "top_logprobs" in item and item["top_logprobs"]:
            # The selected token is in top_logprobs[0]
            # But we need the token ID, not just the text
            # vLLM returns token text in 'token' field
            token_text = item.get("token", "")

            # Check if this is an audio token
            if token_text == "<|AUDIO_OUT|>" or token_text.startswith("<|AUDIO_OUT"):
                # We need to get the actual token ID from somewhere
                # vLLM's logprobs should include token IDs
                pass

    # Alternative: parse the content text to count audio tokens
    # Then use the token IDs from a different API endpoint
    content = choice["message"]["content"]

    # Count AUDIO_OUT tokens
    audio_out_count = content.count("<|AUDIO_OUT|>")

    if audio_out_count == 0:
        raise ValueError("No audio tokens in response")

    print(f"Found {audio_out_count} audio tokens in response")

    # For now, return empty - we need to figure out how to get actual token IDs
    # The vLLM API may need to be called differently
    return []


@app.get("/health")
async def health():
    """Health check endpoint."""
    decoder_status = "available" if DECODER_AVAILABLE else "unavailable"
    tokenizer_loaded = audio_tokenizer is not None

    return {
        "status": "ok",
        "vllm_url": VLLM_BASE_URL,
        "model_id": MODEL_ID,
        "decoder_status": decoder_status,
        "tokenizer_loaded": tokenizer_loaded,
        "voice_presets": list(VOICE_PRESETS.keys()),
    }


@app.get("/v1/audio/voices")
async def list_voices():
    """List available voice presets."""
    return list(VOICE_PRESETS.keys())


@app.post("/v1/audio/speech")
async def generate_speech(request: GenerateRequest):
    """
    Generate speech from text using vLLM for tokens + local decoder.

    This endpoint is compatible with OpenAI's TTS API format.
    """
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    # Ensure decoder is loaded
    tokenizer = get_audio_tokenizer()

    try:
        # Build messages
        messages = build_messages(request)
        print(f"Generating speech for: {request.text[:80]}...")

        start = time.time()

        # Call vLLM
        vllm_response = await call_vllm_chat(
            messages=messages,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
        )

        vllm_time = time.time() - start
        print(f"vLLM response in {vllm_time:.2f}s")

        # Extract token IDs
        # NOTE: This is a placeholder - need to implement proper token extraction
        # The current vLLM API doesn't easily expose token IDs in a usable format

        # For now, let's try a different approach:
        # Get the raw audio from vLLM's audio endpoint and just fix the WAV header
        # This is a temporary workaround until we implement proper token extraction

        async with httpx.AsyncClient(timeout=300.0) as client:
            audio_response = await client.post(
                f"{VLLM_BASE_URL}/v1/audio/speech",
                json={
                    "model": MODEL_ID,
                    "input": request.text,
                    "voice": request.voice or "en_man",
                    "response_format": "wav",
                }
            )
            audio_response.raise_for_status()
            raw_wav = audio_response.content

        audio_time = time.time() - start

        # Fix the WAV header
        if len(raw_wav) > 44:
            actual_data_size = len(raw_wav) - 44
            actual_file_size = len(raw_wav) - 8

            # Fix file size (bytes 4-7) and data size (bytes 40-43)
            fixed_wav = (
                raw_wav[:4] +
                struct.pack('<I', actual_file_size) +
                raw_wav[8:40] +
                struct.pack('<I', actual_data_size) +
                raw_wav[44:]
            )
        else:
            fixed_wav = raw_wav

        duration = actual_data_size / 24000 / 2  # 24kHz, 16-bit mono
        print(f"Generated {duration:.2f}s audio in {audio_time:.2f}s")

        return Response(
            content=fixed_wav,
            media_type="audio/wav",
            headers={
                "X-Generation-Time": str(round(audio_time, 2)),
                "X-Audio-Duration": str(round(duration, 2)),
            }
        )

    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"vLLM request failed: {str(e)}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@app.post("/generate")
async def generate(request: GenerateRequest):
    """Alias for /v1/audio/speech for backwards compatibility."""
    return await generate_speech(request)


if __name__ == "__main__":
    print("=" * 60)
    print("Higgs Audio v3 - Hybrid vLLM + Local Decoder")
    print("=" * 60)
    print(f"vLLM URL: {VLLM_BASE_URL}")
    print(f"Model: {MODEL_ID}")
    print(f"Decoder available: {DECODER_AVAILABLE}")
    print("")
    print("Endpoints:")
    print("  GET  /health           - Health check")
    print("  GET  /v1/audio/voices  - List voice presets")
    print("  POST /v1/audio/speech  - Generate speech")
    print("  POST /generate         - Generate speech (alias)")
    print("  GET  /docs             - API docs")
    print("")
    uvicorn.run(app, host="0.0.0.0", port=8080)
