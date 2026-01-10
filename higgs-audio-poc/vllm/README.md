# Higgs Audio v2 - vLLM High-Throughput Setup

High-throughput TTS inference using vLLM's continuous batching for concurrent request handling.

## Overview

This setup uses the official `bosonai/higgs-audio-vllm` Docker image which provides:
- **Continuous batching**: Handle multiple requests simultaneously
- **High throughput**: ~600 tokens/s on RTX 4090 (~24 seconds of audio per second)
- **PagedAttention**: Efficient KV cache management
- **Same API**: Compatible with our existing `/generate` endpoint

## Architecture

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Your Client    │────▶│  Proxy Server   │────▶│  vLLM Backend   │
│  (Concurrent)   │     │  (port 8888)    │     │  (port 8000)    │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                              │
                              ▼
                        Request Stats
                        VRAM Monitoring
```

## Requirements

- NVIDIA GPU with **~18GB VRAM** (RTX 3090, RTX 4090, A100, etc.)
- Docker with NVIDIA runtime
- Python 3.10+ (for proxy server and benchmark)

## Quick Start

### 1. Setup (First Time)

```bash
# On your GPU instance
cd vllm
./setup.sh
```

### 2. Start vLLM Backend

```bash
./start-vllm.sh
```

This starts:
- vLLM Docker container on port 8000
- GPU monitoring (logs to `gpu_monitor.log`)

Wait for "Uvicorn running on http://0.0.0.0:8000" message.

### 3. Start Proxy Server

In a new terminal:

```bash
python server.py
```

This starts the proxy on port 8888 with request tracking.

### 4. Test

```bash
# Health check
curl http://localhost:8888/health

# Generate speech
curl -X POST http://localhost:8888/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, this is a test of the high throughput system."}'
```

### 5. Run Benchmark

```bash
# Default: 10 requests, 5 concurrent
python benchmark.py

# Custom settings
python benchmark.py --requests 50 --concurrent 10
```

### 6. Stop

```bash
./stop-vllm.sh
```

## API Reference

### POST /generate

Generate speech from text.

**Request:**
```json
{
  "text": "Text to synthesize",
  "system_prompt": "Optional custom system prompt",
  "reference_audio_base64": "Optional base64 WAV for voice cloning",
  "reference_text": "Optional transcript of reference audio",
  "temperature": 0.5,
  "top_p": 0.95,
  "top_k": 50,
  "max_tokens": 2048
}
```

**Response:** WAV audio binary

**Headers:**
- `X-Generation-Time`: Server-side generation time in seconds
- `X-Concurrent-Requests`: Current concurrent request count

### POST /generate/batch

Generate multiple audio files concurrently.

**Request:**
```json
[
  {"text": "First sentence."},
  {"text": "Second sentence."},
  {"text": "Third sentence."}
]
```

**Response:**
```json
{
  "results": [
    {"index": 0, "success": true, "audio_base64": "...", "generation_time": 1.5},
    {"index": 1, "success": true, "audio_base64": "...", "generation_time": 1.4},
    {"index": 2, "success": true, "audio_base64": "...", "generation_time": 1.6}
  ],
  "total": 3,
  "successful": 3,
  "failed": 0
}
```

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "vllm_backend": "http://localhost:8000",
  "vllm_healthy": true,
  "model": "higgs-audio-v2",
  "stats": {
    "total_requests": 100,
    "successful_requests": 98,
    "failed_requests": 2,
    "concurrent_requests": 5,
    "peak_concurrent": 12
  }
}
```

### GET /stats

Detailed statistics.

**Response:**
```json
{
  "total_requests": 100,
  "successful_requests": 98,
  "failed_requests": 2,
  "total_generation_time": 150.5,
  "concurrent_requests": 5,
  "peak_concurrent": 12,
  "avg_generation_time": 1.54
}
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8000` | vLLM backend port |
| `GPU_MEMORY_UTILIZATION` | `0.85` | Fraction of GPU memory to use |
| `MAX_MODEL_LEN` | `8192` | Maximum sequence length |
| `VLLM_URL` | `http://localhost:8000` | vLLM backend URL (for proxy) |

### GPU Memory Tuning

```bash
# For 24GB GPU (RTX 3090/4090)
GPU_MEMORY_UTILIZATION=0.85 ./start-vllm.sh

# For larger GPUs (A100 40GB/80GB)
GPU_MEMORY_UTILIZATION=0.90 ./start-vllm.sh
```

## Monitoring

### GPU Monitor

Real-time GPU stats are logged to `gpu_monitor.log`:

```bash
# Watch live
tail -f gpu_monitor.log

# Analyze
cat gpu_monitor.log | awk -F',' '{print $3}' | sort -n | tail -1
```

CSV columns: `timestamp, gpu_name, memory_used_mb, memory_total_mb, gpu_util_pct, mem_util_pct, temp_c`

### Request Stats

The proxy server tracks:
- Total/successful/failed requests
- Concurrent requests (current and peak)
- Average generation time

Access via `/stats` endpoint.

## Voice Cloning

Place reference audio files in `voice_presets/` directory:

```bash
voice_presets/
  narrator.wav
  character_a.wav
  character_b.wav
```

Use in requests:

```bash
# Base64 encode the reference
REF_AUDIO=$(base64 -i voice_presets/narrator.wav)

curl -X POST http://localhost:8888/generate \
  -H "Content-Type: application/json" \
  -d "{
    \"text\": \"Hello, I sound like the narrator.\",
    \"reference_audio_base64\": \"$REF_AUDIO\",
    \"reference_text\": \"Transcript of the narrator speaking.\"
  }"
```

## Performance Notes

### Expected Throughput (RTX 4090)

| Concurrency | Throughput | Latency (p50) |
|-------------|------------|---------------|
| 1 | ~1 req/s | ~1s |
| 5 | ~3 req/s | ~1.5s |
| 10 | ~5 req/s | ~2s |
| 20 | ~7 req/s | ~3s |

*Actual performance depends on text length and GPU.*

### VRAM Usage

- **Model loaded**: ~16-17 GB
- **During inference**: ~17-18 GB
- **Minimum recommended**: 20 GB free

### Optimizations

1. **Batch similar-length texts** - vLLM batches more efficiently when sequences are similar length
2. **Use `/generate/batch`** - Reduces HTTP overhead for multiple requests
3. **Tune `max_tokens`** - Lower values for short text reduce compute

## Troubleshooting

### vLLM won't start

```bash
# Check Docker logs
docker logs $(docker ps -q --filter ancestor=bosonai/higgs-audio-vllm:latest)

# Check GPU memory
nvidia-smi
```

### Out of memory

```bash
# Reduce GPU memory utilization
GPU_MEMORY_UTILIZATION=0.75 ./start-vllm.sh
```

### Slow generation

1. Check `nvidia-smi` for GPU utilization
2. Ensure no other processes using GPU
3. Check network latency to vLLM backend

### Connection refused

```bash
# Verify vLLM is running
curl http://localhost:8000/health

# Check container status
docker ps
```

## Files

```
vllm/
├── setup.sh           # Initial setup script
├── start-vllm.sh      # Start vLLM Docker container
├── stop-vllm.sh       # Stop everything
├── monitor_gpu.sh     # GPU monitoring script
├── server.py          # Proxy server (port 8888)
├── benchmark.py       # Concurrent request benchmark
├── voice_presets/     # Voice cloning reference audio
└── README.md          # This file
```

## Comparison: HiggsAudioServeEngine vs vLLM

| Feature | ServeEngine | vLLM |
|---------|-------------|------|
| Concurrent requests | No | Yes |
| Continuous batching | No | Yes |
| Throughput | 1 req/s | 5-10 req/s |
| Memory efficiency | Standard | PagedAttention |
| Setup complexity | Simple | Docker required |
| Voice cloning | Yes | Yes |
| Emotional prompts | Yes | Yes |

## Next Steps for Production

1. **Load balancing**: Deploy multiple vLLM instances behind nginx
2. **Caching**: Cache frequent phrases/voices
3. **Queue**: Add Redis/RabbitMQ for request queuing
4. **Monitoring**: Integrate with Prometheus/Grafana
5. **Auto-scaling**: Scale based on queue depth
