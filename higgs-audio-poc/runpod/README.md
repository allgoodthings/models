# Higgs Audio v2 on RunPod with vLLM

High-throughput TTS using the official `bosonai/higgs-audio-vllm` Docker image on RunPod.

## Why RunPod?

- **Native Docker support** - Run any Docker image with GPU access
- **True vLLM inference** - Concurrent requests with continuous batching
- **No workarounds** - Unlike Vast.ai containers, RunPod pods have full Docker capabilities

## Quick Start

### 1. Create a Pod

1. Go to [RunPod Console](https://www.runpod.io/console/pods)
2. Click **Deploy** → **GPU Pod**
3. Select GPU: **RTX 4090 (24GB)** or **A100 (40GB/80GB)**
4. Choose template: **RunPod Pytorch 2.1** (or any with CUDA)

### 2. Configure the Pod

**Container Image:**
```
bosonai/higgs-audio-vllm:latest
```

**Docker Command (optional - can run after SSH):**
Leave empty for now, we'll start manually.

**Expose Ports:**
```
8000
```

**Volume:**
Mount `/workspace` for persistent storage (optional, for caching models)

### 3. Start the Pod and SSH In

Once the pod is running:

```bash
# SSH into your pod (get command from RunPod dashboard)
ssh root@<pod-ip> -p <port> -i ~/.ssh/<your-key>
```

### 4. Start vLLM Server

```bash
# Pull and run the vLLM container
docker run --gpus all \
    --ipc=host \
    --shm-size=20gb \
    -p 8000:8000 \
    bosonai/higgs-audio-vllm:latest \
    --model bosonai/higgs-audio-v2-generation-3B-base \
    --audio-tokenizer-type bosonai/higgs-audio-v2-tokenizer \
    --served-model-name higgs-audio-v2 \
    --port 8000 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 8192
```

Wait for: `Uvicorn running on http://0.0.0.0:8000`

### 5. Test the API

From your local machine (replace `<pod-url>` with your RunPod pod URL):

```bash
# Health check
curl https://<pod-id>-8000.proxy.runpod.net/health

# Generate speech (OpenAI-compatible endpoint)
curl -X POST https://<pod-id>-8000.proxy.runpod.net/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "higgs-audio-v2",
    "input": "Hello, this is a test of the high throughput system.",
    "voice": "alloy"
  }' --output test.wav
```

## API Reference

### OpenAI-Compatible Endpoint

**POST /v1/audio/speech**

```json
{
  "model": "higgs-audio-v2",
  "input": "Text to synthesize",
  "voice": "alloy",
  "response_format": "wav"
}
```

Response: Audio binary (WAV format)

### Chat Completions (for advanced control)

**POST /v1/chat/completions**

```json
{
  "model": "higgs-audio-v2",
  "messages": [
    {
      "role": "system",
      "content": "Generate audio following instruction.\n\n<|scene_desc_start|>\nThe speaker sounds excited and happy.\n<|scene_desc_end|>"
    },
    {
      "role": "user",
      "content": "Hello, this is amazing!"
    }
  ],
  "max_tokens": 2048,
  "temperature": 0.5
}
```

## Using the Proxy Server (Optional)

If you want the same `/generate` API as our other POCs, upload and run the proxy server:

```bash
# On RunPod pod
cd /workspace
git clone <your-repo> higgs-poc
cd higgs-poc/runpod

# Start vLLM in background
./start-vllm.sh &

# Start proxy server (port 8888)
python server.py
```

## GPU Recommendations

| GPU | VRAM | Throughput | Cost (approx) |
|-----|------|------------|---------------|
| RTX 4090 | 24GB | ~600 tokens/s | $0.44/hr |
| A100 40GB | 40GB | ~1200 tokens/s | $1.64/hr |
| A100 80GB | 80GB | ~1500 tokens/s | $2.21/hr |
| H100 | 80GB | ~2000 tokens/s | $3.89/hr |

*Throughput = audio tokens/second. 25 tokens ≈ 1 second of audio.*

## Performance Tuning

### GPU Memory Utilization

```bash
# For 24GB GPU (RTX 4090)
--gpu-memory-utilization 0.85

# For larger GPUs (A100/H100)
--gpu-memory-utilization 0.90
```

### Max Concurrent Requests

vLLM handles this automatically via continuous batching. Monitor with:

```bash
curl https://<pod-url>/metrics | grep vllm_num_requests
```

### Tensor Parallelism (Multi-GPU)

For pods with multiple GPUs:

```bash
docker run --gpus all \
    --ipc=host \
    --shm-size=20gb \
    -p 8000:8000 \
    bosonai/higgs-audio-vllm:latest \
    --model bosonai/higgs-audio-v2-generation-3B-base \
    --audio-tokenizer-type bosonai/higgs-audio-v2-tokenizer \
    --tensor-parallel-size 2 \
    --port 8000
```

## Troubleshooting

### Container won't start

```bash
# Check Docker is available
docker --version

# Check GPU access
nvidia-smi

# Pull image manually
docker pull bosonai/higgs-audio-vllm:latest
```

### Out of memory

```bash
# Reduce GPU memory utilization
--gpu-memory-utilization 0.75

# Or reduce max model length
--max-model-len 4096
```

### Port not accessible

1. Check pod is running in RunPod dashboard
2. Verify port 8000 is exposed
3. Use the RunPod proxy URL: `https://<pod-id>-8000.proxy.runpod.net`

## Files

```
runpod/
├── README.md          # This file
├── start-vllm.sh      # Start vLLM container (uses official image)
├── build-vllm.sh      # Build custom vLLM image from source
├── server.py          # Optional proxy server (same API as other POCs)
└── benchmark.py       # Concurrent request benchmark
```

## Building Custom vLLM Image

The official `bosonai/higgs-audio-vllm:latest` image has known issues ([#123](https://github.com/boson-ai/higgs-audio/issues/123), [#156](https://github.com/boson-ai/higgs-audio/issues/156)).

You can build from the public [boson-ai/higgs-audio-vllm](https://github.com/boson-ai/higgs-audio-vllm) fork:

```bash
# Clone and build (takes 30-60 mins)
./build-vllm.sh

# Or build and push to your registry
REGISTRY=ghcr.io/your-org ./build-vllm.sh --push
```

Then use your custom image on RunPod:
```
Container Image: ghcr.io/your-org/higgs-audio-vllm:custom
```

The fork is public with 40+ stars and includes the complete Dockerfile.

## Comparison: RunPod vs Vast.ai

| Feature | RunPod | Vast.ai |
|---------|--------|---------|
| Docker support | Native | Requires special instances |
| vLLM image | Works | Doesn't work (no Docker-in-Docker) |
| Concurrent inference | Yes | No (HiggsAudioServeEngine only) |
| Setup complexity | Simple | Complex workarounds |
| Cost | Slightly higher | Lower |
