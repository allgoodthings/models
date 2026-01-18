# Image-Gen Service

FLUX.2-klein-9B image generation service for Scenema AI.

## Overview

FastAPI service providing text-to-image generation and multi-reference image editing using the FLUX.2-klein-9B model from Black Forest Labs. Designed for deployment on Vast AI GPU instances.

**Model Specs:**
- 9B parameter rectified flow transformer
- Step-distilled to 4 inference steps
- Sub-second generation capability
- VRAM: ~19GB (fits on RTX 4090 24GB)

**Server Port:** `7000`

## Features

- **Text-to-Image**: Generate images from text prompts
- **Multi-Reference Editing**: Edit images using 1-4 reference images (native FLUX.2-klein support)
- **S3 Upload**: Direct upload to presigned S3 URLs
- **GPU Optimized**: Runs on RTX 4090 (24GB) with headroom

## Quick Start

### Prerequisites

- Python 3.10+
- CUDA 12.1+ compatible GPU
- HuggingFace token with access to gated models

### Installation

```bash
# Clone and enter directory
cd models/image-gen

# Install dependencies
pip install -e .

# Set HuggingFace token
export HUGGING_FACE_TOKEN="hf_your_token_here"

# Run the server
uvicorn imagegen.server.main:app --host 0.0.0.0 --port 7000
```

### Using Makefile

```bash
make install      # Install dependencies
make run          # Start server
make benchmark    # Run VRAM benchmark
make test         # Run tests
make health       # Check server health
```

## API Endpoints

### GET /health

Health check with model status and GPU info.

```json
{
  "status": "healthy",
  "flux_loaded": true,
  "gpu_available": true,
  "gpu_name": "NVIDIA RTX 4090",
  "gpu_memory_gb": 24.0,
  "gpu_memory_used_gb": 15.5
}
```

### POST /generate

Generate image from text prompt.

```json
// Request
{
  "prompt": "A beautiful sunset over mountains",
  "upload_url": "https://s3.../presigned-put-url",
  "width": 1024,
  "height": 1024,
  "num_steps": 4,
  "seed": 42,
  "output_format": "png"
}

// Response
{
  "success": true,
  "output_url": "https://s3.../output.png",
  "width": 1024,
  "height": 1024,
  "seed": 42,
  "timing_inference_ms": 2100,
  "timing_total_ms": 2450
}
```

### POST /edit

Edit image using reference images.

```json
// Request
{
  "prompt": "Replace background with beach scene",
  "reference_images": [
    {"url": "https://...", "weight": 1.0}
  ],
  "upload_url": "https://s3.../presigned-put-url"
}
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HUGGING_FACE_TOKEN` | Yes | - | HF token for gated model |
| `PRELOAD_MODELS` | No | `true` | Load model on startup |
| `LOG_LEVEL` | No | `INFO` | Logging level |

## Development

### Run Tests

```bash
pytest tests/ -v
```

### VRAM Benchmark

Before deploying, run the benchmark script to validate GPU compatibility:

```bash
python scripts/benchmark_vram.py
```

## Vast AI Deployment

See [README-VASTAI.md](README-VASTAI.md) for detailed deployment instructions.

## Directory Structure

```
image-gen/
├── imagegen/
│   ├── __init__.py
│   └── server/
│       ├── __init__.py
│       ├── main.py           # FastAPI entry point
│       ├── schemas.py        # Pydantic models
│       └── flux_pipeline.py  # FLUX wrapper
├── scripts/
│   ├── benchmark_vram.py     # VRAM profiling
│   └── test_api.py           # Manual API testing
├── tests/
│   ├── conftest.py
│   └── test_schemas.py
├── pyproject.toml
├── Makefile
└── README.md
```

## Phase 2 (Future)

- Dockerfile for containerized deployment
- GitHub Actions CI/CD
- Docker image push to GHCR
