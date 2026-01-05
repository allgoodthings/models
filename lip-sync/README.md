# Multi-Face Lip-Sync

A multi-face lip-sync pipeline using MuseTalk, LivePortrait, and CodeFormer.

## Features

- **Multi-face support**: Process multiple speaking characters in a single video
- **Targeted face selection**: Use bounding boxes to specify which faces to process
- **High-quality output**: LivePortrait neutralization + MuseTalk lip-sync + CodeFormer enhancement
- **Seamless compositing**: Feathered blending for natural results
- **FastAPI server**: Easy HTTP API for testing and integration
- **Qwen-VL integration**: Optional character detection via OpenRouter

## Quick Start

### Docker (Recommended)

```bash
# Pull the image
docker pull ghcr.io/allgoodthings/lip-sync:latest

# Run with GPU
docker run --gpus all -p 8000:8000 ghcr.io/allgoodthings/lip-sync:latest

# With Qwen-VL face detection enabled
docker run --gpus all -p 8000:8000 \
  -e OPENROUTER_API_KEY=sk-or-... \
  -e PRELOAD_MODELS=true \
  ghcr.io/allgoodthings/lip-sync:latest
```

### Build from Source

```bash
# Build the image locally (from the lip-sync directory)
cd lip-sync
docker build -t lip-sync:dev .

# Run
docker run --gpus all -p 8000:8000 lip-sync:dev
```

### Local Development (No Docker)

```bash
# Clone the repo
git clone https://github.com/allgoodthings/models.git
cd models/lip-sync

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Download model weights (~7GB)
python scripts/download_models.py

# Run the server
uvicorn lipsync.server.main:app --host 0.0.0.0 --port 8000 --reload
```

## API Endpoints

### POST /lipsync

Process multi-face lip-sync on a video. Returns the processed video directly.

**Request (multipart/form-data):**
- `video`: Video file (mp4, mov, etc.)
- `audio`: Audio file (wav, mp3, etc.)
- `request`: JSON string with configuration

**Request JSON:**
```json
{
  "faces": [
    {
      "character_id": "alice",
      "bbox": [100, 50, 200, 250],
      "start_time_ms": 0,
      "end_time_ms": 5000
    },
    {
      "character_id": "bob",
      "bbox": [400, 60, 180, 220],
      "start_time_ms": 2000,
      "end_time_ms": 5000
    }
  ],
  "enhance_quality": true,
  "fidelity_weight": 0.7
}
```

**Response:** MP4 video bytes (Content-Type: video/mp4)

**Example:**
```bash
curl -X POST http://localhost:8000/lipsync \
  -F "video=@input.mp4" \
  -F "audio=@speech.wav" \
  -F 'request={"faces":[{"character_id":"alice","bbox":[100,50,200,250],"start_time_ms":0,"end_time_ms":5000}]}' \
  --output result.mp4
```

### POST /lipsync/json

Same as `/lipsync` but returns JSON with base64-encoded video and metadata.

**Response:**
```json
{
  "success": true,
  "faces_processed": 2,
  "face_results": [
    {"character_id": "alice", "success": true},
    {"character_id": "bob", "success": true}
  ],
  "processing_time_ms": 45000,
  "output_url": "data:video/mp4;base64,..."
}
```

### POST /detect-faces

Detect character faces in an image using Qwen-VL (requires OPENROUTER_API_KEY).

**Request (multipart/form-data):**
- `frame`: Image file (jpg, png)
- `request`: JSON with character definitions

**Request JSON:**
```json
{
  "characters": [
    {"id": "alice", "name": "Alice", "description": "Woman with red hair"},
    {"id": "bob", "name": "Bob", "description": "Man with glasses"}
  ]
}
```

**Response:**
```json
{
  "faces": [
    {"character_id": "alice", "bbox": [100, 50, 200, 250], "confidence": 0.95},
    {"character_id": "bob", "bbox": [400, 60, 180, 220], "confidence": 0.87}
  ],
  "frame_width": 1920,
  "frame_height": 1080
}
```

### GET /health

Health check with GPU info.

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": true,
  "gpu_available": true,
  "gpu_name": "NVIDIA RTX 4090",
  "gpu_memory_gb": 24.0
}
```

### GET /

API info and available endpoints.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    LipSyncPipeline                       │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  1. Extract frames at target bbox regions                │
│  2. Align faces (MediaPipe landmarks)                    │
│  3. Neutralize expressions (LivePortrait)                │
│  4. Generate lip-sync (MuseTalk)                         │
│  5. Enhance quality (CodeFormer)                         │
│  6. Composite back with feathered blending               │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## GPU Requirements

- **Minimum**: RTX 3090 (24GB) or RTX 4090 (24GB)
- **VRAM Usage**: ~9GB per concurrent job

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENROUTER_API_KEY` | API key for Qwen-VL face detection | - |
| `CUDA_VISIBLE_DEVICES` | GPU device(s) to use | `0` |
| `LOG_LEVEL` | Logging level | `INFO` |

## License

MIT
