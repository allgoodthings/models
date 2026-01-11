"""
Higgs Audio v3 - Integrated vLLM Server

Uses vLLM as a library for token generation with full control over output.
Decodes audio tokens locally using boson_multimodal.

This is the proper v3 solution - no external API calls, full token control.
"""

import io
import os
import time
import struct
from typing import Optional, List, Dict, Any

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
import uvicorn

# vLLM imports
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt
from transformers import AutoTokenizer

# Audio decoder imports
from boson_multimodal.audio_processing.higgs_audio_tokenizer import HiggsAudioTokenizer
from huggingface_hub import snapshot_download
import json
import inspect


app = FastAPI(
    title="Higgs Audio v3 Integrated",
    description="vLLM-integrated server with local audio decoding",
    version="0.3.0",
)

# Configuration
MODEL_ID = "bosonai/higgs-audio-v2-generation-3B-base"
TOKENIZER_ID = "bosonai/higgs-audio-v2-tokenizer"

# Audio token configuration
# These values come from the higgs audio tokenizer
NUM_CODEBOOKS = 8
CODEBOOK_SIZE = 2048
AUDIO_TOKEN_OFFSET = 128266  # Start of audio tokens in vocabulary

# Global instances
llm: Optional[LLM] = None
tokenizer = None
audio_decoder = None


def load_audio_decoder(device="cuda"):
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


def init_models():
    """Initialize vLLM and audio decoder."""
    global llm, tokenizer, audio_decoder

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Initializing models on {device}...")

    # Load tokenizer
    print(f"Loading tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

    # Load vLLM
    print(f"Loading vLLM model: {MODEL_ID}")
    llm = LLM(
        model=MODEL_ID,
        tokenizer=MODEL_ID,
        trust_remote_code=True,
        max_model_len=8192,
        gpu_memory_utilization=0.85,
    )

    # Load audio decoder
    print(f"Loading audio decoder: {TOKENIZER_ID}")
    audio_decoder = load_audio_decoder(device)

    print("All models loaded!")


def xcodec_get_output_length(input_length: int) -> int:
    """Calculate output waveform length from code length."""
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


def decode_audio_tokens(codes: np.ndarray, chunk_size: int = 750) -> np.ndarray:
    """
    Decode audio codes to waveform using overlapping chunks.

    Args:
        codes: Audio codes of shape (num_codebooks, seq_len)
        chunk_size: Chunk size for decoding

    Returns:
        Waveform as numpy array
    """
    global audio_decoder
    device = next(audio_decoder.parameters()).device
    codes_tensor = torch.from_numpy(codes).unsqueeze(0).to(device)

    overlap_width = 16
    chunk_output_length = xcodec_get_output_length(chunk_size)
    outputs = []
    seq_len = codes_tensor.shape[-1]

    with torch.no_grad():
        for i in range(0, seq_len, chunk_size):
            begin = max(0, i - overlap_width)
            end = min(i + chunk_size + overlap_width, seq_len)
            chunk = codes_tensor[:, :, begin:end]
            output = audio_decoder.decode(chunk)

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

    return np.concatenate(outputs, axis=2)[0, 0]


def revert_delay_pattern(data: np.ndarray) -> np.ndarray:
    """Revert the delay pattern encoding used in audio tokens."""
    assert len(data.shape) == 2
    num_codebooks = data.shape[0]
    out_l = []
    for i in range(num_codebooks):
        out_l.append(data[i:(i + 1), i:(data.shape[1] - num_codebooks + 1 + i)])
    return np.concatenate(out_l, axis=0)


def token_ids_to_audio_codes(token_ids: List[int]) -> np.ndarray:
    """
    Convert LLM token IDs to audio codes for decoding.

    Audio tokens are encoded as:
        token_id = AUDIO_TOKEN_OFFSET + codebook_idx * CODEBOOK_SIZE + code_value

    They come interleaved with a delay pattern that needs to be reverted.
    """
    # Filter to only audio tokens
    audio_tokens = [
        t for t in token_ids
        if AUDIO_TOKEN_OFFSET <= t < AUDIO_TOKEN_OFFSET + NUM_CODEBOOKS * CODEBOOK_SIZE
    ]

    if not audio_tokens:
        raise ValueError("No audio tokens found in output")

    # Decode to (codebook_idx, code_value) pairs
    decoded = []
    for token_id in audio_tokens:
        offset = token_id - AUDIO_TOKEN_OFFSET
        codebook_idx = offset // CODEBOOK_SIZE
        code_value = offset % CODEBOOK_SIZE
        decoded.append((codebook_idx, code_value))

    # Reconstruct the codes array with delay pattern
    seq_len = len(decoded) // NUM_CODEBOOKS
    codes = np.zeros((NUM_CODEBOOKS, seq_len + NUM_CODEBOOKS - 1), dtype=np.int64)

    for i, (cb_idx, code_val) in enumerate(decoded):
        time_idx = i // NUM_CODEBOOKS
        codes[cb_idx, time_idx + cb_idx] = code_val

    # Revert delay pattern
    codes = revert_delay_pattern(codes)

    return codes


def audio_to_wav_bytes(audio: np.ndarray, sample_rate: int = 24000) -> bytes:
    """Convert audio array to WAV bytes with correct header."""
    if audio.dtype in (np.float32, np.float64):
        audio = np.clip(audio, -1.0, 1.0)
        audio = (audio * 32767).astype(np.int16)

    data_size = len(audio) * 2
    file_size = data_size + 36

    header = struct.pack(
        '<4sI4s4sIHHIIHH4sI',
        b'RIFF', file_size, b'WAVE', b'fmt ', 16,
        1, 1, sample_rate, sample_rate * 2, 2, 16,
        b'data', data_size
    )

    return header + audio.tobytes()


class GenerateRequest(BaseModel):
    text: str = Field(..., description="Text to synthesize")
    voice: Optional[str] = Field("en_man", description="Voice preset")
    scene_description: Optional[str] = Field(None, description="Scene/style description")
    temperature: float = Field(0.7, ge=0.1, le=1.5)
    top_p: float = Field(0.95, ge=0.0, le=1.0)
    max_tokens: int = Field(4096, ge=256, le=8192)


# Voice presets
VOICE_PRESETS = {
    "en_man": "A clear male voice speaking in a quiet room.",
    "en_woman": "A clear female voice speaking in a quiet room.",
    "dramatic_actor": "A dramatic theatrical performance with gravitas and emotional depth.",
    "narrator_epic": "An epic movie trailer narration with dramatic pauses.",
    "calm_narrator": "A calm, measured audiobook narrator with clear enunciation.",
}


def build_prompt(request: GenerateRequest) -> str:
    """Build the full prompt for audio generation."""
    scene_desc = request.scene_description or VOICE_PRESETS.get(request.voice, VOICE_PRESETS["en_man"])

    system = f"Generate audio following instruction.\n\n<|scene_desc_start|>\n{scene_desc}\n<|scene_desc_end|>"

    # Build ChatML-style prompt
    prompt = (
        f"<|begin_of_text|>"
        f"<|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n{request.text}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n<|audio_out_bos|><|AUDIO_OUT|>"
    )

    return prompt


@app.on_event("startup")
async def startup():
    """Initialize models on startup."""
    init_models()


@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "ok",
        "model": MODEL_ID,
        "llm_loaded": llm is not None,
        "decoder_loaded": audio_decoder is not None,
        "voices": list(VOICE_PRESETS.keys()),
    }


@app.post("/v1/audio/speech")
async def generate_speech(request: GenerateRequest):
    """Generate speech using vLLM + local decoder."""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    if llm is None or audio_decoder is None:
        raise HTTPException(status_code=503, detail="Models not loaded")

    try:
        # Build prompt
        prompt = build_prompt(request)
        print(f"Generating: {request.text[:60]}...")

        start = time.time()

        # Generate with vLLM
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            stop=["<|audio_eos|>", "<|eot_id|>", "<|end_of_text|>"],
        )

        outputs = llm.generate([prompt], sampling_params)
        output = outputs[0]

        # Get token IDs from output
        token_ids = output.outputs[0].token_ids
        gen_time = time.time() - start
        print(f"Generated {len(token_ids)} tokens in {gen_time:.2f}s")

        # Convert to audio codes
        decode_start = time.time()
        audio_codes = token_ids_to_audio_codes(list(token_ids))
        print(f"Audio codes shape: {audio_codes.shape}")

        # Decode to waveform
        waveform = decode_audio_tokens(audio_codes)
        decode_time = time.time() - decode_start
        duration = len(waveform) / 24000

        print(f"Decoded {duration:.2f}s audio in {decode_time:.2f}s")

        # Convert to WAV
        wav_bytes = audio_to_wav_bytes(waveform)

        return Response(
            content=wav_bytes,
            media_type="audio/wav",
            headers={
                "X-Generation-Time": str(round(gen_time, 2)),
                "X-Decode-Time": str(round(decode_time, 2)),
                "X-Audio-Duration": str(round(duration, 2)),
                "X-Tokens": str(len(token_ids)),
            }
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/audio/voices")
async def list_voices():
    """List available voices."""
    return list(VOICE_PRESETS.keys())


if __name__ == "__main__":
    print("=" * 60)
    print("Higgs Audio v3 - Integrated vLLM Server")
    print("=" * 60)
    print(f"Model: {MODEL_ID}")
    print(f"Tokenizer: {TOKENIZER_ID}")
    print("")
    uvicorn.run(app, host="0.0.0.0", port=8080)
