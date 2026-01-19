# Image-Gen Service

FLUX.2-klein image generation service for Scenema AI.

## Overview

FastAPI service providing text-to-image generation and multi-reference image editing using FLUX.2-klein models from Black Forest Labs. Designed for deployment on Vast AI GPU instances.

**Server Port:** `7000`

## Model Variants

| Model | Parameters | VRAM (BF16) | VRAM (FP8) | License |
|-------|------------|-------------|------------|---------|
| FLUX.2-klein-4B | 4B | ~13GB | ~8GB | Apache 2.0 |
| FLUX.2-klein-9B | 9B | ~30GB | ~24GB | Non-commercial |

**Recommendation:** Use 4B for 24GB GPUs (RTX 4090/3090), 9B FP8 for 32GB+ GPUs.

## Benchmark Results (RTX 5090 32GB)

### 9B Model (FP8 Quantization)

**Model Loading:**
| Metric | Value |
|--------|-------|
| VRAM at Rest | 23.88 GB |
| Peak VRAM (1024x1024) | 26.28 GB |
| Quantization | TorchAO Float8WeightOnly |

**Image Generation Performance:**
| Resolution | Inference Time | Peak VRAM |
|------------|----------------|-----------|
| 512x512 | ~1.2s | ~25 GB |
| 768x768 | ~1.8s | ~26 GB |
| 1024x768 | ~1.9s | ~26 GB |
| 768x1024 | ~1.9s | ~26 GB |
| 1024x1024 | ~2.4s | 26.28 GB |

**Image Editing (Single Reference):**
| Operation | Time | Peak VRAM |
|-----------|------|-----------|
| Style transfer | ~4.4s | 26.28 GB |

**Multi-Reference Editing (VRAM Scaling):**

1024x768 output:
| # References | Peak VRAM | Time |
|--------------|-----------|------|
| 2 | 26.36 GB | 6.1s |
| 4 | 27.63 GB | 11.7s |
| 6 | 28.90 GB | 18.6s |
| 8 | OOM | - |

768x768 output:
| # References | Peak VRAM | Time |
|--------------|-----------|------|
| 2 | 26.24 GB | 6.2s |
| 4 | 27.51 GB | 11.2s |
| 6 | 28.79 GB | 18.0s |
| 8 | 30.06 GB | 26.2s |
| 10 | OOM | - |

**Key Findings:**
- VRAM per reference image: ~635 MB additional
- Max references on 32GB: 6-8 depending on output resolution
- 24GB GPU (RTX 4090/3090): Won't work with 9B - peak 26.28GB exceeds capacity

### 4B Model (BF16)

**Model Loading:**
| Metric | Value |
|--------|-------|
| VRAM at Rest | 14.93 GB |
| Peak VRAM (1024x1024) | 17.32 GB |
| Quantization | None (native BF16) |

**Image Generation Performance:**
| Resolution | Inference Time | Peak VRAM |
|------------|----------------|-----------|
| 512x512 | ~1.2s | 15.54 GB |
| 768x768 | ~630ms | 16.28 GB |
| 1024x768 | ~760ms | 16.73 GB |
| 768x1024 | ~760ms | 16.73 GB |
| 1024x1024 | ~1.0s | 17.32 GB |

**Multi-Reference Editing (VRAM Scaling):**

1024x768 output:
| # References | Peak VRAM | Time |
|--------------|-----------|------|
| 2 | 16.73 GB | 2.0s |
| 4 | 16.74 GB | 3.6s |
| 6 | 17.08 GB | 5.2s |

**Multi-Reference Stress Test (pushed to limits):**

1024x768 output:
| # References | Peak VRAM | Time |
|--------------|-----------|------|
| 2 | 16.73 GB | 1.3s |
| 4 | 16.73 GB | 1.9s |
| 6 | 16.74 GB | 2.5s |
| 8 | 16.74 GB | 3.1s |
| 10 | 16.74 GB | 4.0s |
| 12 | 16.90 GB | 4.6s |
| 14 | 17.15 GB | 5.5s |
| 16 | 17.40 GB | 6.5s |

768x768 output:
| # References | Peak VRAM | Time |
|--------------|-----------|------|
| 2 | 16.29 GB | 1.1s |
| 4 | 16.29 GB | 1.7s |
| 6 | 16.29 GB | 2.3s |
| 8 | 16.30 GB | 3.0s |
| 10 | 16.55 GB | 3.8s |
| 12 | 16.80 GB | 4.5s |
| 14 | 17.05 GB | 5.3s |
| 16 | 17.31 GB | 6.1s |

512x512 output:
| # References | Peak VRAM | Time |
|--------------|-----------|------|
| 2 | 15.54 GB | 0.7s |
| 8 | 16.15 GB | 2.5s |
| 16 | 17.15 GB | 5.7s |

**Key Findings:**
- **16 references without OOM** on 32GB GPU
- VRAM scales ~40-50 MB per additional reference (very efficient)
- Peak VRAM stays under 17.5 GB even with 16 refs at 1024x768
- Fits comfortably on 24GB GPUs (RTX 4090/3090) with 10+ references
- ~2x faster than 9B FP8 for text-to-image
- Time scales ~400ms per additional reference

### Model Comparison Summary

| Feature | 4B BF16 | 9B FP8 |
|---------|---------|--------|
| VRAM at Rest | 14.93 GB | 23.88 GB |
| Peak VRAM (1024x1024) | 17.32 GB | 26.28 GB |
| Inference (1024x1024) | ~1.0s | ~2.4s |
| VRAM per Reference | ~40-50 MB | ~635 MB |
| Max Refs (32GB) | 16+ | 6-8 |
| Max Refs (24GB) | 10+ | N/A |
| Min GPU | 24GB | 32GB |
| License | Apache 2.0 | Non-commercial |

**Recommendation:**
- **Production (24GB GPUs)**: Use 4B BF16 - excellent quality, fast, commercially usable
- **Research/Quality (32GB+ GPUs)**: Use 9B FP8 - best quality for complex scenes

## Features

- **Batch Generation**: Generate multiple images in one request for scene consistency
- **Reference Images**: Use up to 16 reference images to guide generation
- **Upscaling**: Optional 2x/4x Real-ESRGAN upscaling
- **S3 Upload**: Direct upload to presigned S3 URLs
- **GPU Optimized**: BF16 on 32GB+ GPUs, FP8 quantization for 24GB GPUs

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

Batch image generation with optional reference images and upscaling.

```json
// Request
{
  "prompts": [
    "Scene 1: A hero enters a dark castle",
    "Scene 2: The hero discovers a treasure room"
  ],
  "upload_urls": [
    "https://s3.../scene1.png",
    "https://s3.../scene2.png"
  ],
  "images": [  // Optional: reference images for consistency
    {"url": "https://...", "weight": 1.0}
  ],
  "width": 1024,
  "height": 768,
  "num_steps": 4,
  "seed": 42,
  "upscale": 4,  // Optional: 2 or 4
  "output_format": "png"
}

// Response
{
  "success": true,
  "results": [
    {
      "index": 0,
      "success": true,
      "output_url": "https://s3.../scene1.png",
      "width": 4096,
      "height": 3072,
      "seed": 42,
      "timing_inference_ms": 1000,
      "timing_upscale_ms": 1500,
      "timing_upload_ms": 200
    },
    {
      "index": 1,
      "success": true,
      "output_url": "https://s3.../scene2.png",
      "width": 4096,
      "height": 3072,
      "seed": 43,
      "timing_inference_ms": 1000,
      "timing_upscale_ms": 1500,
      "timing_upload_ms": 200
    }
  ],
  "timing_total_ms": 5500
}
```

**Notes:**
- `prompts` and `upload_urls` must have the same length
- `images` are shared across all prompts for scene consistency
- Seeds increment sequentially (seed, seed+1, seed+2, ...)

### POST /upscale

Upscale an image using Real-ESRGAN (2x or 4x).

Model is loaded on-demand and unloaded after each request to minimize VRAM usage.

```json
// Request
{
  "image_url": "https://s3.../input.png",
  "upload_url": "https://s3.../presigned-put-url",
  "scale": 4,
  "output_format": "png"
}

// Response
{
  "success": true,
  "output_url": "https://s3.../output.png",
  "input_width": 1024,
  "input_height": 1024,
  "output_width": 4096,
  "output_height": 4096,
  "scale": 4,
  "timing_download_ms": 150,
  "timing_load_ms": 200,
  "timing_upscale_ms": 1500,
  "timing_upload_ms": 500,
  "timing_total_ms": 2350
}
```

**Performance (RTX 5090):**
- Model load: ~100-200ms
- Upscale 1024→4096: ~1-2s (with tiling)
- VRAM: ~2GB during inference

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HUGGING_FACE_TOKEN` | Yes | - | HF token for gated model |
| `PRELOAD_MODELS` | No | `true` | Load model on startup |
| `QUANTIZATION` | No | `none` | Quantization mode: `none`, `fp8`, `int8` |
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
│       ├── flux_pipeline.py  # FLUX wrapper
│       └── upscaler.py       # Real-ESRGAN upscaler
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
