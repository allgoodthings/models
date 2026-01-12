# Higgs Audio v2 POC

POC for evaluating [Higgs Audio v2](https://github.com/boson-ai/higgs-audio) as a potential replacement for Chatterbox TTS.

## Quick Start

```bash
# On a Vast.ai GPU instance (24GB+ VRAM)
./setup.sh
./start.sh
```

## VRAM Requirements

### Full Precision (Default)

| Stage | Allocated | Reserved |
|-------|-----------|----------|
| Model loaded | 16.28 GB | 17.68 GB |
| During generation | 16.36 GB | 17.80 GB |
| **Minimum Required** | - | **~18 GB** |

**Conclusion: 16GB VRAM is NOT enough for full precision.**

### Quantized Versions (Community Forks)

| Quantization | VRAM Usage | Quality | Speed |
|--------------|------------|---------|-------|
| Full (bf16) | ~18 GB | Perfect | Baseline |
| 8-bit | ~7 GB | Near Perfect | ~0.5x realtime |
| 4-bit | ~5 GB | Fair* | Near realtime |

*4-bit requires specific parameters: temperature 0.01-0.3, top_k=0

## Performance Benchmarks

### Validated Test Results (2026-01-10)

Tested on **NVIDIA GeForce RTX 3090 (24GB)** via Vast.ai:

**Server Stats:**
| Metric | Value |
|--------|-------|
| Model Load Time | 74.4s |
| VRAM Allocated | 16.28 GB |
| VRAM Reserved | 17.68 GB |
| Generation Speed | 3.9-4.5s per request |

**Shakespeare Speech Tests:**
| Test | Text | Duration | File Size |
|------|------|----------|-----------|
| Short | "To be, or not to be..." | 4.0s | 192 KB |
| Medium | "All the world's a stage..." | 10.0s | 480 KB |
| Long | Macbeth "Tomorrow" monologue | 37.1s | 1.7 MB |
| Dramatic | Romeo balcony (with emotional prompt) | 14.4s | 693 KB |

**Audio Quality:** Validated as production-ready for all test cases.

### Earlier Benchmarks

Tested on **NVIDIA RTX PRO 4000 Blackwell (24GB)**:

| Mode | Chunks | Time | Per Chunk |
|------|--------|------|-----------|
| Multi-speaker (cold) | 5 | 14.38s | ~2.9s |
| Multi-speaker (warm) | 5 | 9.95s | ~2.0s |

## Comparison: Higgs Audio v2 vs Chatterbox TTS

| Feature | Higgs Audio v2 | Chatterbox TTS |
|---------|----------------|----------------|
| VRAM (full) | ~18 GB | ~8 GB |
| Sample Rate | 24 kHz | 24 kHz |
| Model Size | 3.6B + 2.2B params | ~2B params |
| Voice Cloning | Zero-shot | Zero-shot |
| Multi-speaker | Native support | Not supported |
| Emotional Control | Weak* (base model) | Limited |
| Quantization | Community forks | Not available |

*Official benchmarks claim 75.7% vs GPT-4o-mini, but testing shows the **base model** has weak emotional control. See [Evaluation Results](#evaluation-results-2026-01-11).

## Quantized Alternatives

For lower VRAM requirements, use these community forks:

### [faster-higgs-audio](https://github.com/sorbetstudio/faster-higgs-audio)
- 4-bit and 8-bit quantization via BitsAndBytes
- OpenAI-compatible API server
- Runs on 8GB GPUs with 8-bit quantization

```bash
git clone https://github.com/boson-ai/higgs-audio.git
cd higgs-audio
uv pip install -r requirements.txt -e . bitsandbytes
./run_tts.sh "Text" --quantization_bits 8  # 8-bit mode
```

### [higgs-audio-v2-ui](https://github.com/Paxurux/higgs-audio-v2-ui)
- Gradio web UI
- 4-bit, 6-bit, 8-bit quantization options
- Long-form text generation with chunking

## API Endpoints

### Health Check
```
GET /health
```

### Generate Speech
```
POST /generate
Content-Type: application/json

{
  "text": "Text to synthesize",
  "system_prompt": "Optional custom prompt for style/emotion",
  "temperature": 0.3,
  "max_tokens": 2048
}
```

### Multi-speaker Dialogue
Use `[SPEAKER0]`, `[SPEAKER1]` tags:
```json
{
  "text": "[SPEAKER0] Hello!\n[SPEAKER1] Hi there!",
  "system_prompt": "Two friends having a casual conversation"
}
```

### Voice Cloning
```json
{
  "text": "Text to synthesize",
  "reference_audio_base64": "<base64-encoded WAV>",
  "reference_text": "Transcript of reference audio"
}
```

## System Prompt Examples

### Emotional Speech
```
Generate audio following instruction.

<|scene_desc_start|>
The speaker sounds happy and excited.
<|scene_desc_end|>
```

### Dramatic Performance
```
Generate audio following instruction.

<|scene_desc_start|>
A dramatic theatrical performance. The speaker delivers lines with gravitas and emotional depth.
<|scene_desc_end|>
```

### Whisper
```
Generate audio following instruction.

<|scene_desc_start|>
Audio is recorded as a soft whisper in a quiet room.
<|scene_desc_end|>
```

## Files

- `server.py` - FastAPI server
- `setup.sh` - Installation script
- `start.sh` - Start server with GPU monitoring
- `monitor_gpu.sh` - VRAM monitoring script

## Concurrent Inference with vLLM

The default `HiggsAudioServeEngine` has `max_batch_size=1` - it only processes one request at a time. For concurrent inference, use the **vLLM backend**.

**See [`vllm/`](./vllm/) for a complete high-throughput POC with:**
- Same `/generate` API as this POC
- Concurrent request handling and benchmarking
- Request statistics and monitoring
- Voice cloning support

```bash
cd vllm
./setup.sh
./start-vllm.sh   # Start vLLM Docker container
python server.py  # Start proxy server on port 8888
python benchmark.py --requests 50 --concurrent 10
```

### vLLM Direct Usage

```bash
docker run --gpus all --ipc=host --shm-size=20gb --network=host \
  bosonai/higgs-audio-vllm:latest \
  --served-model-name "higgs-audio-v2-generation-3B-base" \
  --model "bosonai/higgs-audio-v2-generation-3B-base" \
  --audio-tokenizer-type "bosonai/higgs-audio-v2-tokenizer" \
  --limit-mm-per-prompt audio=50 \
  --max-model-len 8192 \
  --port 8000 \
  --gpu-memory-utilization 0.8
```

### vLLM Throughput Benchmarks

| GPU | Throughput | Audio Generation Speed |
|-----|------------|------------------------|
| A100 40GB | 1500 tokens/s | **60s audio per second** |
| RTX 4090 24GB | 600 tokens/s | **24s audio per second** |

This is ~10-20x faster than the single-request serve engine.

### vLLM API Example

```bash
curl -X POST "http://localhost:8000/v1/audio/speech" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "higgs-audio-v2-generation-3B-base",
    "voice": "en_woman",
    "input": "Hello world!",
    "response_format": "pcm"
  }' --output - | ffmpeg -f s16le -ar 24000 -ac 1 -i - speech.wav
```

## Recommendations

1. **For production with 24GB+ VRAM**: Use vLLM backend for concurrent inference
2. **For single-request testing**: Use serve_engine (this POC)
3. **For 16GB VRAM**: Use 8-bit quantization via faster-higgs-audio fork
4. **For 8GB VRAM**: Use 8-bit quantization with careful memory management
5. **For <8GB VRAM**: Not recommended; use CPU offloading (very slow)

## Evaluation Results (2026-01-11)

### Emotional/Characteristic Control: WEAK

**Finding:** The base model (`higgs-audio-v2-generation-3B-base`) has **weak and inconsistent** emotional control despite official benchmarks claiming 75.7% vs GPT-4o-mini.

Per official documentation: *"The base model has not been post-trained."* Emotions are inferred from **text semantics** (the content of words), not from explicit system prompts.

**Tested Formats:**
1. ❌ Emotion names: `"happy"`, `"angry"`, `"scared"` - No effect
2. ❌ Natural language: `"sounds excited and energetic"` - Minimal effect
3. ⚠️ Voice characteristics: `"whispered;breathy;fast pace"` - Inconsistent results

**Correct Scene Description Format:**
```
Generate audio following instruction.

<|scene_desc_start|>
SPEAKER0: whispered;breathy;trembling;slow pace;feminine
<|scene_desc_end|>
```

Available characteristics (from official examples):
- **Volume:** whispered, hushed, soft, loud
- **Pace:** slow pace, moderate pace, fast pace
- **Pitch:** low pitch, moderate pitch, high pitch, deep
- **Tone:** monotone, breathy, calm, energetic, bright, expressive
- **Style:** vocal fry, trembling, articulate, clear
- **Gender:** masculine, feminine
- **Age:** young adult, mature
- **Accent:** british accent, american accent

**Test Results:**
| Prompt | Expected | Actual |
|--------|----------|--------|
| `whispered;breathy;trembling` (scared text) | Whispered, fearful | Neutral delivery |
| `loud;aggressive;fast pace` (angry text) | Loud, intense | Slightly faster, still neutral |
| `energetic;fast pace;high pitch` (excited text) | Upbeat, excited | Starts neutral, ends high-pitched |

**Conclusion:** The base model cannot reliably produce emotional speech. The "75.7% emotional accuracy" benchmark likely applies to a **post-trained** variant not publicly available.

### Voice Cloning: GOOD

Voice identity cloning works well. The model accurately reproduces speaker characteristics from reference audio. Tested with:
- 5-10 second reference clips
- Clear audio with minimal background noise
- Provided transcript matching reference audio

### Concurrency

| Backend | Max Batch | Concurrent Requests | Notes |
|---------|-----------|---------------------|-------|
| serve_engine | 1 | Sequential only | ~4s per request |
| vLLM | Configurable | Yes | 10-20x throughput |

The default `HiggsAudioServeEngine` has `max_batch_size=1`. For production, vLLM is required.

### Memory Usage

| Configuration | VRAM | Notes |
|---------------|------|-------|
| Full precision (bf16) | 17-18 GB | Minimum for RTX 3090/4090 |
| 8-bit quantization | ~7 GB | Community fork (faster-higgs-audio) |
| 4-bit quantization | ~5 GB | Requires careful temperature tuning |

### Recommendation

**For production use with emotional control requirements:**
- ❌ Do NOT use Higgs Audio v2 base model
- Consider alternatives with stronger emotional control:
  - **FishAudio OpenAudio S1** - Apache 2.0, better emotion handling
  - **IndexTTS-2** - Apache 2.0, strong prosody control
  - **Chatterbox TTS** - Current production model, proven reliability
  - **XTTS-v2** - Coqui, multilingual with emotion control

**For production use where voice cloning is primary requirement (emotions not critical):**
- ✅ Higgs Audio v2 with vLLM backend is viable
- Good voice identity cloning
- High throughput with vLLM
- Multilingual support

## Sources

- [Higgs Audio v2 GitHub](https://github.com/boson-ai/higgs-audio)
- [faster-higgs-audio (quantized)](https://github.com/sorbetstudio/faster-higgs-audio)
- [higgs-audio-v2-ui (Gradio)](https://github.com/Paxurux/higgs-audio-v2-ui)
- [VRAM discussion](https://github.com/boson-ai/higgs-audio/issues/117)
- [Boson AI Blog](https://www.boson.ai/blog/higgs-audio-v2)
