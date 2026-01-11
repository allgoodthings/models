# Higgs Audio v3 - Integrated vLLM + Local Decoder

High-quality TTS using vLLM for efficient token generation and local audio decoding.

## Why v3?

The bosonai vLLM Docker image has two bugs:
1. **WAV header bug** - Writes hardcoded 0.68s header regardless of actual length
2. **Chunk concatenation bug** - No crossfading between 1-second chunks, causing clicks every ~1 second

v3 solves this by:
- Using vLLM as a library (not external API) for full token control
- Decoding audio tokens locally using boson_multimodal's decoder with proper overlap-add
- One continuous decode pass → no chunk boundary artifacts

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    v3 Integrated Server                      │
├─────────────────────────────────────────────────────────────┤
│  1. Build prompt (system + scene desc)                       │
│  2. Generate tokens with vLLM (batched, efficient)           │
│  3. Extract audio token IDs directly from output             │
│  4. Convert token IDs → audio codes (8 codebooks)            │
│  5. Decode with HiggsAudioTokenizer (overlap-add)            │
│  6. Return WAV with correct headers                          │
└─────────────────────────────────────────────────────────────┘
                           │
         ┌─────────────────┴─────────────────┐
         ▼                                   ▼
   ┌───────────────┐                 ┌───────────────┐
   │     vLLM      │                 │ Audio Decoder │
   │  (in-process) │                 │  (in-process) │
   │               │                 │               │
   │ • PagedAttn   │                 │ • Overlap-add │
   │ • Batching    │                 │ • Full context│
   │ • Token IDs   │                 │ • No clicks   │
   └───────────────┘                 └───────────────┘
```

## Files

- `server_integrated.py` - **Main server** (vLLM as library, recommended)
- `server.py` - API client version (calls external vLLM, fallback)
- `Dockerfile` - Container build
- `test-requests.http` - HTTP test file

## Setup

### Option 1: Docker (Recommended)

```bash
docker build -t higgs-audio-v3 .
docker run --gpus all -p 8080:8080 higgs-audio-v3
```

### Option 2: Direct

```bash
pip install -r requirements.txt
pip install vllm
pip install git+https://github.com/boson-ai/higgs-audio.git

python server_integrated.py
```

Server runs on port 8080.

## API

### Generate Speech

```bash
curl -X POST http://localhost:8080/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello world, this is a test.",
    "voice": "en_man"
  }' \
  --output speech.wav
```

### Voice Presets

| Voice | Description |
|-------|-------------|
| `en_man` | Clear male voice |
| `en_woman` | Clear female voice |
| `dramatic_actor` | Theatrical with gravitas |
| `narrator_epic` | Movie trailer style |
| `calm_narrator` | Audiobook narrator |

### Custom Scene Description

```json
{
  "text": "Your text here",
  "scene_description": "A whispered conversation in a dark room with echo."
}
```

## Token → Audio Pipeline

```
LLM Output:  <|audio_out_bos|><|AUDIO_OUT|><|AUDIO_OUT|>...<|audio_eos|>
                                   ↓
Token IDs:   [128266, 128267, 129314, ...]  (vocab IDs for audio tokens)
                                   ↓
Audio Codes: token_id = 128266 + codebook * 2048 + code
             → (codebook_idx, code_value) pairs
                                   ↓
Code Array:  shape (8, seq_len) after reverting delay pattern
                                   ↓
Decoder:     xcodec with overlap-add → continuous waveform
                                   ↓
WAV:         24kHz, 16-bit, mono with correct headers
```

## Comparison

| Feature | v2 (bosonai API) | v3 (integrated) |
|---------|------------------|-----------------|
| Token generation | vLLM | vLLM (same) |
| Audio decoding | Chunked, no crossfade | Overlap-add |
| WAV headers | Hardcoded 0.68s | Correct |
| Chunk artifacts | Clicks every 1s | None |
| Token access | Via API (limited) | Direct |
| Concurrency | Yes | Yes |

## TODO

- [ ] Voice cloning with reference audio
- [ ] Streaming output for long texts
- [ ] Batch inference API
- [ ] GPU memory optimization
