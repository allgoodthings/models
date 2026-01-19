# Video Upscaler

Real-ESRGAN based video upscaling service with optional temporal deflicker.

## Features

- **Real-ESRGAN upscaling**: State-of-the-art video super-resolution
- **Two models**: `realesr-animevideov3` (fast, anime/AI optimized) and `realesrgan-x4plus` (general)
- **Multi-pass upscaling**: Optional high-quality two-pass mode for 4K output
- **Temporal deflicker**: Optional post-processing to reduce frame-to-frame flickering
- **Audio preservation**: Keeps original audio track in output
- **FastAPI REST interface**: Simple HTTP API for integration

## Quick Start

### Docker (Recommended)

```bash
# Build
docker build -t video-upscaler:latest .

# Run
docker run --gpus all -p 8000:8000 video-upscaler:latest
```

### Local Development

```bash
# Install
pip install -e ".[dev]"

# Download models
make download-models

# Run (GPU)
make dev

# Run (CPU - for testing only)
make dev-cpu
```

## API

### POST /upscale

Upscale a video.

```json
{
  "video_url": "https://example.com/input.mp4",
  "upload_url": "https://example.com/presigned-upload-url",
  "target_resolution": "4k",
  "high_quality": true,
  "deflicker": false,
  "model": "realesr-animevideov3",
  "preserve_audio": true
}
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `video_url` | string | required | URL to input video |
| `upload_url` | string | required | Presigned PUT URL for output |
| `target_resolution` | "1080p" \| "4k" | "1080p" | Target output resolution |
| `high_quality` | bool | false | Two-pass upscaling for better quality |
| `deflicker` | bool | false | Apply temporal deflicker post-processing |
| `model` | string | "realesr-animevideov3" | Upscaling model |
| `preserve_audio` | bool | true | Keep original audio track |

**Response:**

```json
{
  "success": true,
  "output_url": "https://example.com/output.mp4",
  "input_resolution": "854x480",
  "output_resolution": "3840x2160",
  "frames_processed": 720,
  "timing_upscale_ms": 180000,
  "timing_total_ms": 195000
}
```

### GET /health

Health check endpoint.

```json
{
  "status": "healthy",
  "model_loaded": true,
  "deflicker_available": false,
  "gpu_available": true,
  "gpu_name": "NVIDIA RTX 4090",
  "gpu_memory_gb": 24.0,
  "version": "0.1.0"
}
```

## Models

### realesr-animevideov3 (Default)

- **Best for**: AI-generated video, anime, cartoons
- **Speed**: ~2-4 fps on RTX 3090
- **Quality**: Excellent for stylized content

### realesrgan-x4plus

- **Best for**: Photorealistic content, live action
- **Speed**: ~1-2 fps on RTX 3090
- **Quality**: Excellent detail preservation

## Processing Modes

### Standard Mode (default)

Single-pass upscaling. Fast, good quality.

```json
{
  "high_quality": false
}
```

### High Quality Mode

Two-pass upscaling. First pass: 4x, Second pass: remaining scale.
Better results for large upscale factors (e.g., 480p → 4K).

```json
{
  "high_quality": true
}
```

### With Deflicker

Applies temporal smoothing after upscaling to reduce frame-to-frame flickering.
Adds ~50% processing time.

```json
{
  "deflicker": true
}
```

## Performance

| Input | Output | Model | High Quality | Time (RTX 3090) |
|-------|--------|-------|--------------|-----------------|
| 480p 1min | 1080p | animevideov3 | No | ~3 min |
| 480p 1min | 1080p | animevideov3 | Yes | ~5 min |
| 480p 1min | 4K | animevideov3 | No | ~4 min |
| 480p 1min | 4K | animevideov3 | Yes | ~7 min |

*Times exclude download/upload. Add ~1 min for deflicker if enabled.*

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODELS_DIR` | `/app/models` | Directory for model weights |
| `PRELOAD_MODELS` | `true` | Load models on startup |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU device to use |
| `LOG_LEVEL` | `INFO` | Logging level |

## Development

```bash
# Run tests
make test

# Lint
make lint

# Format
make format
```

## License

MIT
