# Image-Gen Service - Vast AI Setup Guide

Quick guide to deploy and test the FLUX.2-klein-9B image generation service on Vast AI.

**Model:** `black-forest-labs/FLUX.2-klein-9B`

## Prerequisites

1. **Vast AI Account**: https://vast.ai
2. **HuggingFace Token**: https://huggingface.co/settings/tokens
   - Must have access to gated models (accept FLUX terms)

## Step 1: Rent a GPU Instance

### Recommended Specs

| GPU | VRAM | Mode | Expected Performance |
|-----|------|------|---------------------|
| **32GB+ (A6000, etc.)** | 32GB+ | BF16 (no quantization) | Best quality, 1-2s inference |
| RTX 4090 | 24GB | FP8 quantization | Slightly lower quality, ~1s inference |
| A100 | 40/80GB | BF16 | Best performance |

**Recommended**: 32GB GPU for BF16 (best quality, no quantization)

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

# Install uv (fast package manager)
pip install uv

# Install dependencies (uses torch 2.5+ for TorchAO FP8 support)
uv pip install -e .
```

## Step 4: Run VRAM Benchmark

**IMPORTANT**: Run this before starting the server to validate GPU compatibility.

```bash
python scripts/benchmark_vram.py
```

Expected output (32GB GPU, BF16):
```
============================================================
FLUX.2-klein-9B VRAM Benchmark
============================================================
GPU: NVIDIA A6000
Total VRAM: 48.0 GB
PyTorch: 2.5.x
CUDA: 12.x

48.0GB VRAM is sufficient for BF16 (no quantization)

------------------------------------------------------------
Loading FLUX.2-klein-9B model (BF16, no quantization)...
------------------------------------------------------------

Test: Model Load (BF16)
VRAM Peak: ~29 GB
Duration: ~60s (first load, downloads model)

Test: Generate 1024x1024
VRAM Peak: ~29 GB
Duration: ~1500ms (1.5s)

BENCHMARK SUMMARY
============================================================
Max VRAM Peak: ~29 GB
Available VRAM: 48.0 GB
Headroom: ~19 GB

VRAM usage is within safe limits.
```

## Step 5: Start the Server

```bash
# For 32GB+ GPU (BF16, best quality)
uvicorn imagegen.server.main:app --host 0.0.0.0 --port 7000

# For 24GB GPU (FP8 quantization)
QUANTIZATION=fp8 uvicorn imagegen.server.main:app --host 0.0.0.0 --port 7000

# Or with Makefile
make run
```

Server will:
1. Check for HUGGING_FACE_TOKEN
2. Download FLUX model (~20GB, takes a few minutes first time)
3. Load model to GPU (BF16 or FP8 depending on QUANTIZATION)
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
| `QUANTIZATION` | No | `none` | `none` (BF16, 32GB+), `fp8` (24GB), `int8` (20GB) |
| `LOG_LEVEL` | No | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR) |
| `HF_HOME` | No | `~/.cache/huggingface` | Model cache directory |

### Quantization Modes

- **`none` (default)**: BF16 precision, best quality, requires 32GB+ VRAM
- **`fp8`**: FP8 quantization via TorchAO, minor quality trade-off, fits in 24GB
- **`int8`**: INT8 quantization, more quality trade-off, fits in 20GB

```bash
# For 24GB GPU (RTX 4090), use FP8
export QUANTIZATION=fp8

# For 32GB+ GPU, use default BF16 (no export needed)
```

## Troubleshooting

### Out of Memory (OOM)

If you get CUDA OOM errors:

1. **Enable quantization**: Set `QUANTIZATION=fp8` for 24GB GPUs
2. **Reduce resolution**: Try 768x768 instead of 1024x1024
3. **Check other processes**: `nvidia-smi` to see what's using VRAM
4. **Restart the server**: Clear any fragmented memory

### Model Download Issues

If model download fails:

1. **Check HF token**: Ensure it has access to gated models
2. **Accept model terms**: Visit https://huggingface.co/black-forest-labs/FLUX.2-klein-9B and accept
3. **Check disk space**: Model needs ~20GB

### Slow First Request

First request may be slightly slower due to:
- CUDA kernel warmup
- torch.compile optimization (if enabled)
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
