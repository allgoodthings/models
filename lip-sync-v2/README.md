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
  "faces": {
    "total_detected": 2,
    "processed": 1,
    "unknown": 1,
    "results": [
      {
        "character_id": "alice",
        "success": true,
        "segments": [
          {
            "start_ms": 0,
            "end_ms": 3000,
            "synced": true,
            "skip_reason": null,
            "avg_quality": 0.92,
            "avg_bbox": [102, 51, 198, 248],
            "avg_head_pose": [5.2, -12.3, 2.1]
          },
          {
            "start_ms": 3000,
            "end_ms": 4200,
            "synced": false,
            "skip_reason": "profile_view",
            "avg_quality": null,
            "avg_bbox": [95, 48, 190, 240],
            "avg_head_pose": [8.1, 52.4, -3.2]
          },
          {
            "start_ms": 4200,
            "end_ms": 5000,
            "synced": true,
            "skip_reason": null,
            "avg_quality": 0.88,
            "avg_bbox": [105, 52, 195, 245],
            "avg_head_pose": [4.5, -8.2, 1.8]
          }
        ],
        "summary": {
          "total_ms": 5000,
          "synced_ms": 3800,
          "skipped_ms": 1200
        }
      }
    ],
    "unknown_faces": [
      {
        "character_id": "face_1",
        "segments": [
          {
            "start_ms": 1000,
            "end_ms": 3500,
            "synced": true,
            "skip_reason": null,
            "avg_quality": 0.85,
            "avg_bbox": [450, 80, 160, 200],
            "avg_head_pose": [2.1, 15.3, -1.5]
          }
        ]
      }
    ]
  },
  "output": {
    "duration_ms": 5000,
    "width": 1920,
    "height": 1080,
    "fps": 30.0,
    "file_size_bytes": 2500000
  },
  "timing": {
    "total_ms": 45000,
    "download_ms": 3200,
    "detection_ms": 5500,
    "lipsync_ms": 28000,
    "enhancement_ms": 4500,
    "encoding_ms": 2300,
    "upload_ms": 1500
  }
}
```

#### Response Fields

| Field | Description |
|-------|-------------|
| `success` | Whether processing succeeded |
| `error_message` | Error message if failed |
| `faces.total_detected` | Total unique faces detected during processing |
| `faces.processed` | Number of faces from request that were processed |
| `faces.unknown` | Number of faces detected but not in request |
| `faces.results[]` | Results for each requested face |
| `faces.results[].segments[]` | Time segments where face had same sync state |
| `faces.results[].summary` | Summary of sync coverage (total/synced/skipped ms) |
| `faces.unknown_faces[]` | Faces detected but not in request (for validation) |
| `output` | Output video metadata |
| `timing` | Processing time breakdown by stage |

#### Segment States

Each segment represents a contiguous time range where a face had the same sync state:

- **synced=true**: Lip-sync was applied (face visible, suitable pose, good quality)
- **synced=false**: Lip-sync was skipped, with `skip_reason`:
  - `profile_view`: Face turned too far (yaw > 45°)
  - `face_too_small`: Face bbox < 64px wide
  - `low_detection_quality`: Detection confidence < 0.5

The `avg_*` fields show averaged values across all frames in the segment (for display/validation only - actual processing uses per-frame values).

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
# Auto-detect faces and process (requires presigned upload URL)
node scripts/test-api.mjs \
  --video https://example.com/video.mp4 \
  --audio https://example.com/audio.mp3 \
  --upload-url https://storage.example.com/presigned-put-url \
  --auto

# With character references for identity matching
node scripts/test-api.mjs \
  --video https://example.com/video.mp4 \
  --audio https://example.com/audio.mp3 \
  --upload-url https://storage.example.com/presigned-put-url \
  --refs characters.json \
  --auto

# Download output after processing (optional)
node scripts/test-api.mjs \
  --video https://example.com/video.mp4 \
  --audio https://example.com/audio.mp3 \
  --upload-url https://storage.example.com/presigned-put-url \
  --download-url https://storage.example.com/output.mp4 \
  --output result.mp4 \
  --auto
```

## License

MIT
