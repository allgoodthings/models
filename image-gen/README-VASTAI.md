# Image-Gen Service - Vast AI Setup Guide

Quick guide to deploy and test the FLUX.2-klein-9B image generation service on Vast AI.

**Model:** `black-forest-labs/FLUX.2-klein-9B` (~19GB VRAM, fits on RTX 4090 24GB)

## Prerequisites

1. **Vast AI Account**: https://vast.ai
2. **HuggingFace Token**: https://huggingface.co/settings/tokens
   - Must have access to gated models (accept FLUX terms)

## Step 1: Rent a GPU Instance

### Recommended Specs

| GPU | VRAM | Expected Performance |
|-----|------|---------------------|
| RTX 4090 | 24GB | Primary target (~19GB model, 5GB headroom) |
| A6000 | 48GB | More headroom for larger batches |
| A100 | 40/80GB | Best performance |

### Instance Configuration

1. Go to https://vast.ai/console/create/
2. Filter by:
   - GPU: RTX 4090, A6000, or A100
   - CUDA: 12.1+
   - Disk: 50GB+ (for model cache)
3. Select "Jupyter + SSH" launch mode
4. Rent the instance

## Step 2: Connect to Instance

```bash
# SSH into instance (replace with your instance details)
ssh -p PORT root@IP_ADDRESS

# Or use the Vast AI web terminal
```

## Step 3: Clone and Setup

```bash
# Clone the repository
git clone https://github.com/ScenemaAI/models.git
cd models/image-gen

# Set HuggingFace token
export HUGGING_FACE_TOKEN="hf_your_token_here"

# Install dependencies
pip install -e .

# Or with uv (faster)
pip install uv
uv pip install -e .
```

## Step 4: Run VRAM Benchmark

**IMPORTANT**: Run this before starting the server to validate GPU compatibility.

```bash
python scripts/benchmark_vram.py
```

Expected output:
```
============================================================
FLUX.2 klein VRAM Benchmark
============================================================
GPU: NVIDIA GeForce RTX 4090
Total VRAM: 24.0 GB
...

BENCHMARK SUMMARY
============================================================
Max VRAM Peak: ~16-20 GB
Available VRAM: 24.0 GB
Headroom: 4-8 GB

VRAM usage is within safe limits.
```

## Step 5: Start the Server

```bash
# Run the server
uvicorn imagegen.server.main:app --host 0.0.0.0 --port 7000

# Or with Makefile
make run
```

Server will:
1. Check for HUGGING_FACE_TOKEN
2. Download FLUX model (~12GB, takes a few minutes first time)
3. Load model with CPU offload
4. Start accepting requests

## Step 6: Test the API

### Health Check

```bash
curl http://localhost:7000/health | python3 -m json.tool
```

Expected response:
```json
{
    "status": "healthy",
    "flux_loaded": true,
    "gpu_available": true,
    "gpu_name": "NVIDIA GeForce RTX 4090",
    "gpu_memory_gb": 24.0,
    "gpu_memory_used_gb": 15.5
}
```

### Generate Image

```bash
# Generate to local file (for testing without S3)
curl -X POST http://localhost:7000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A beautiful sunset over mountains, photorealistic",
    "upload_url": "https://httpbin.org/put",
    "width": 1024,
    "height": 1024,
    "num_steps": 4,
    "output_format": "png"
  }'
```

### Run Test Script

```bash
python scripts/test_api.py
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HUGGING_FACE_TOKEN` | Yes | - | HF token for gated model |
| `HF_TOKEN` | Alt | - | Alternative name for HF token |
| `PRELOAD_MODELS` | No | `true` | Load model on startup |
| `LOG_LEVEL` | No | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `HF_HOME` | No | `~/.cache/huggingface` | Model cache directory |

## Troubleshooting

### Out of Memory (OOM)

If you get CUDA OOM errors:

1. **Reduce resolution**: Try 768x768 instead of 1024x1024
2. **Check other processes**: `nvidia-smi` to see what's using VRAM
3. **Restart the server**: Clear any fragmented memory

### Model Download Issues

If model download fails:

1. **Check HF token**: Ensure it has access to gated models
2. **Accept model terms**: Visit https://huggingface.co/black-forest-labs/FLUX.1-schnell and accept
3. **Check disk space**: Model needs ~12GB

### Slow First Request

First request is slow because:
- Model components are moved from CPU to GPU on-demand (CPU offload)
- Subsequent requests are faster

### Port Forwarding

To access from your local machine:

```bash
# SSH tunnel (replace with your instance details)
ssh -L 7000:localhost:7000 -p PORT root@IP_ADDRESS

# Then access at http://localhost:7000
```

## Performance Tips

1. **Batch requests**: Model stays on GPU longer with consecutive requests
2. **Use consistent resolutions**: Avoids recompilation overhead
3. **Pre-warm**: Send a test request after startup to warm up the pipeline

## Cleanup

When done testing:

1. Stop the server (Ctrl+C)
2. Delete/stop the Vast AI instance to avoid charges
