# Multi-Face Lip-Sync Pipeline

Local face detection and lip-sync processing using InsightFace + MuseTalk + LivePortrait + CodeFormer.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              CLIENT REQUEST                                  │
│                                                                             │
│  Video URL + Audio URL + Character References (with headshot URLs)          │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FACE DETECTION PHASE                                 │
│                         POST /detect-faces                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────────┐  │
│  │ Download     │    │ Download     │    │ Extract Frames               │  │
│  │ Reference    │───▶│ Video        │───▶│ at N FPS                     │  │
│  │ Images       │    │              │    │ (e.g., 3 frames/sec)         │  │
│  └──────────────┘    └──────────────┘    └──────────────────────────────┘  │
│         │                                              │                    │
│         ▼                                              ▼                    │
│  ┌──────────────────────┐              ┌──────────────────────────────┐    │
│  │ INSIGHTFACE          │              │ For Each Frame:              │    │
│  │ Extract embeddings   │              │                              │    │
│  │ from headshots       │              │  • Detect all faces          │    │
│  │ (512-dim vectors)    │              │  • Extract embeddings        │    │
│  └──────────────────────┘              │  • Get head pose (yaw/pitch) │    │
│         │                              │  • Get landmarks             │    │
│         │                              └──────────────────────────────┘    │
│         │                                              │                    │
│         └──────────────┬───────────────────────────────┘                    │
│                        ▼                                                    │
│              ┌─────────────────────────────────────────┐                    │
│              │ MATCH & ANALYZE                         │                    │
│              │                                         │                    │
│              │ • Compare embeddings (cosine sim >0.5)  │                    │
│              │ • Assign character IDs or face_1/2/etc  │                    │
│              │ • Check syncability:                    │                    │
│              │   - Yaw > 45° → skip (profile)          │                    │
│              │   - Face < 64px → skip (too small)      │                    │
│              │   - Low confidence → skip               │                    │
│              │ • Calculate sync_quality (0-1)          │                    │
│              └─────────────────────────────────────────┘                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DETECTION RESPONSE                                   │
│                                                                             │
│  Per-frame data with syncability metadata:                                  │
│  ┌────────────────────────────────────────────────────────────────────┐     │
│  │ Frame @0ms:    alice bbox=[100,50,200,250] syncable=true  q=0.92   │     │
│  │                bob   bbox=[400,60,180,220] syncable=false (profile)│     │
│  │ Frame @333ms:  alice bbox=[102,51,198,248] syncable=true  q=0.90   │     │
│  └────────────────────────────────────────────────────────────────────┘     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           LIP-SYNC PHASE                                     │
│                           POST /lipsync                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Input: Video URL + Audio URL + Upload URL + Face configs                   │
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌────────────┐   │
│  │  MUSETALK   │    │ LIVEPORTRAIT│    │ COMPOSITOR  │    │ CODEFORMER │   │
│  │             │    │             │    │             │    │            │   │
│  │ Audio →     │───▶│ Mouth shape │───▶│ Blend into  │───▶│ Enhance    │   │
│  │ Lip motion  │    │ animation   │    │ original    │    │ face       │   │
│  │ features    │    │             │    │ video       │    │ quality    │   │
│  └─────────────┘    └─────────────┘    └─────────────┘    └────────────┘   │
│                                                                             │
│  For each face in faces[]:                                                  │
│    • Crop face region using bbox                                            │
│    • Apply lip-sync for time range                                          │
│    • Composite back into video                                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              UPLOAD & RESPOND                                │
│                                                                             │
│  • Upload MP4 to presigned URL (PUT request)                                │
│  • Return metadata: duration, dimensions, file size, processing time        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Flow Summary

| Step | Component | What Happens |
|------|-----------|--------------|
| 1 | **Reference Loading** | Download character headshots → InsightFace extracts 512-dim embeddings |
| 2 | **Frame Sampling** | Download video → Extract frames at N FPS using ffmpeg |
| 3 | **Face Detection** | InsightFace finds all faces per frame → bboxes, embeddings, head pose |
| 4 | **Identity Matching** | Compare face embeddings to reference embeddings (cosine similarity) |
| 5 | **Syncability Check** | Filter faces: skip profiles (yaw>45°), tiny faces (<64px), low quality |
| 6 | **Lip-Sync** | MuseTalk generates lip motion → LivePortrait animates → Composite into video |
| 7 | **Enhancement** | CodeFormer upscales/restores face quality |
| 8 | **Upload** | PUT result to presigned URL, return metadata |

## Models & VRAM

| Model | VRAM | Purpose |
|-------|------|---------|
| InsightFace (buffalo_l) | ~500MB | Face detection & embeddings |
| MuseTalk | ~4GB | Audio-to-lip features |
| LivePortrait | ~2GB | Face animation |
| CodeFormer | ~1GB | Face enhancement |
| **Total** | **~7.5GB** | Fits on RTX 3080+ |

## API Endpoints

### `GET /health`

Health check with per-model status.

```json
{
  "status": "healthy",
  "insightface_loaded": true,
  "musetalk_loaded": true,
  "liveportrait_loaded": true,
  "codeformer_loaded": true,
  "gpu_available": true,
  "gpu_name": "NVIDIA RTX 4090",
  "gpu_memory_gb": 24.0
}
```

### `POST /detect-faces`

Detect and track faces across video frames.

**Request:**
```json
{
  "video_url": "https://example.com/video.mp4",
  "sample_fps": 3,
  "characters": [
    {
      "id": "alice",
      "name": "Alice",
      "reference_image_url": "https://example.com/alice-headshot.jpg"
    }
  ],
  "similarity_threshold": 0.5
}
```

**Response:**
```json
{
  "frames": [
    {
      "timestamp_ms": 0,
      "faces": [
        {
          "character_id": "alice",
          "bbox": [100, 50, 200, 250],
          "confidence": 0.95,
          "head_pose": [5.2, -12.3, 2.1],
          "syncable": true,
          "sync_quality": 0.92,
          "skip_reason": null
        }
      ]
    }
  ],
  "frame_width": 1920,
  "frame_height": 1080,
  "sample_fps": 3,
  "video_duration_ms": 10000,
  "characters_detected": ["alice"]
}
```

### `POST /lipsync`

Process lip-sync and upload result to presigned URL.

**Request:**
```json
{
  "video_url": "https://example.com/video.mp4",
  "audio_url": "https://example.com/audio.mp3",
  "upload_url": "https://storage.example.com/presigned-put-url",
  "faces": [
    {
      "character_id": "alice",
      "bbox": [100, 50, 200, 250],
      "start_time_ms": 0,
      "end_time_ms": 5000
    }
  ],
  "enhance_quality": true,
  "fidelity_weight": 0.7
}
```

**Response:**
```json
{
  "success": true,
  "processing_time_ms": 45000,
  "face_results": [
    {"character_id": "alice", "success": true}
  ],
  "output": {
    "duration_ms": 5000,
    "width": 1920,
    "height": 1080,
    "file_size_bytes": 2500000,
    "fps": 30.0
  }
}
```

## Quick Start

### Docker

```bash
# Build
docker build -t lip-sync:latest .

# Run
docker run --gpus all -p 8000:8000 lip-sync:latest
```

### Local Development

```bash
# Install dependencies
pip install -e ".[dev]"

# Download models
python scripts/download_models.py --models-dir ./models

# Run server
MODELS_DIR=./models uvicorn lipsync.server.main:app --reload
```

## Testing

```bash
# Auto-detect faces and process
node scripts/test-api.mjs \
  --video https://example.com/video.mp4 \
  --audio https://example.com/audio.mp3 \
  --auto

# With character references for identity matching
node scripts/test-api.mjs \
  --video https://example.com/video.mp4 \
  --audio https://example.com/audio.mp3 \
  --refs characters.json \
  --auto
```

## License

MIT
