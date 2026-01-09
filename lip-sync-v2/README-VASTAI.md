# Lip-Sync V2 - Wav2Lip + GFPGAN

> **Production-ready lip-sync API with multi-face tracking, seamless blending, and video looping**

## What is this template?

This template gives you a **production-ready lip-sync API** running on GPU. It uses Wav2Lip for lip synchronization, InsightFace for multi-face tracking, and GFPGAN for face enhancement - all wrapped in a fast FastAPI server.

**Think:** *"Upload a video + audio, get back a perfectly lip-synced video in seconds."*

---

## What can I do with this?

- **Lip-sync any video** to new audio with natural-looking results
- **Track multiple faces** using reference images for identity matching
- **Seamless face compositing** with Poisson blending (no visible rectangular artifacts)
- **Video looping** for talking heads - extend short clips to any audio length
- **Face enhancement** with GFPGAN for higher quality output
- **REST API** for easy integration into your pipeline

---

## Who is this for?

This is **perfect** if you:
- Need to dub videos into different languages
- Want to create AI-powered talking head videos
- Are building an avatar or virtual presenter product
- Need automated dialogue replacement (ADR) for film production
- Want real-time lip-sync for UGC platforms

---

## Quick Start Guide

### **Step 1: Launch Your Instance**
Select a GPU with at least **8GB VRAM** (16GB recommended for concurrent requests)

### **Step 2: Wait for Model Download**
On first boot, models are downloaded automatically (~1.5GB total):
- `wav2lip_gan.pth` (~400MB) - Wav2Lip model
- `GFPGANv1.4.pth` (~350MB) - Face enhancement
- `s3fd.pth` (~85MB) - Face detection
- InsightFace `buffalo_l` (~500MB) - Face recognition

### **Step 3: Check Health**
```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "insightface_loaded": true,
  "wav2lip_loaded": true,
  "gpu_available": true,
  "gpu_name": "NVIDIA RTX 4090",
  "gpu_memory_gb": 24.0
}
```

### **Step 4: Start Lip-Syncing!**
```bash
curl -X POST http://localhost:8000/lipsync \
  -H "Content-Type: application/json" \
  -d '{
    "video_url": "https://example.com/video.mp4",
    "audio_url": "https://example.com/audio.mp3",
    "upload_url": "https://your-storage.com/presigned-upload-url",
    "characters": [{
      "id": "speaker",
      "name": "Main Speaker",
      "reference_image_url": "https://example.com/face-reference.jpg"
    }]
  }'
```

---

## API Endpoints

### **GET /health**
Health check - returns model loading status and GPU info.

### **GET /**
API info and available endpoints.

### **POST /face-tracking**
Generate a visualization video with colored bounding boxes to verify face detection before running lip-sync.

### **POST /lipsync**
Full lip-sync pipeline: face tracking → Wav2Lip → optional GFPGAN enhancement.

---

## API Reference

### **POST /lipsync**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `video_url` | string | *required* | URL to input video |
| `audio_url` | string | *required* | URL to audio for lip-sync |
| `upload_url` | string | *required* | Presigned URL for uploading output (PUT) |
| `characters` | array | *required* | Characters to lip-sync (see below) |
| `similarity_threshold` | float | `0.5` | Face matching threshold (0-1) |
| `enhance_quality` | bool | `true` | Apply GFPGAN face enhancement |
| `loop_mode` | string | `"crossfade"` | How to handle audio longer than video |
| `crossfade_frames` | int | `10` | Frames to blend at loop boundary |
| `temporal_smoothing` | float | `0.3` | Bbox smoothing factor (0-1) |

**Character object:**
```json
{
  "id": "unique_id",
  "name": "Display Name",
  "reference_image_url": "https://example.com/face.jpg"
}
```

**Loop modes:**
| Mode | Behavior | Best For |
|------|----------|----------|
| `"none"` | Trim to shorter of video/audio | Action films, ADR |
| `"repeat"` | Hard loop (visible jump) | Testing only |
| `"pingpong"` | Forward-backward oscillation | Portraits, talking heads |
| `"crossfade"` | Smooth blend at boundaries | General use (default) |

---

## Key Features

### **Multi-Face Tracking**
- Uses InsightFace (`buffalo_l`) for face detection and recognition
- Match faces to characters using reference images
- Per-frame bounding box tracking with temporal smoothing
- Handles multiple characters in the same video

### **Seamless Face Compositing**
- **Poisson blending** (`cv2.seamlessClone`) eliminates rectangular artifacts
- Elliptical mask with Gaussian blur for natural edges
- Automatic fallback to feathered blending if needed

### **Video Looping**
- Extend short video clips to match longer audio
- Crossfade blending at loop boundaries for smooth transitions
- Ping-pong mode for natural-looking talking heads

### **Face Enhancement**
- Optional GFPGAN post-processing
- Restores fine details (teeth, skin texture)
- Reduces artifacts from Wav2Lip generation

### **Performance**
- ~2x realtime on RTX 4090 (5 second video in ~10 seconds)
- Low VRAM footprint (~2GB for Wav2Lip alone)
- Concurrent request support on 16GB+ GPUs

---

## Performance Benchmarks

| GPU | VRAM | Speed | Notes |
|-----|------|-------|-------|
| RTX 4090 | 24GB | ~2x realtime | Best performance |
| RTX 3090 | 24GB | ~1.5x realtime | Great value |
| RTX 3080 | 10GB | ~1.2x realtime | Good budget option |
| RTX 3060 | 12GB | ~1x realtime | Minimum recommended |

*Speed measured without GFPGAN enhancement. Enable `enhance_quality: true` for ~2x slower but higher quality.*

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MODELS_DIR` | `/app/models` | Directory for model weights |
| `PRELOAD_MODELS` | `true` | Load models on startup |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `CUDA_VISIBLE_DEVICES` | `0` | GPU device index |

---

## Customization

### **Adjusting Face Padding**
The default padding `[0, 20, 0, 0]` (top, bottom, left, right) captures the chin area. Modify in the API request or adjust defaults in code.

### **Crossfade Duration**
Default is 10 frames (~0.4s at 25fps). Increase for smoother transitions, decrease for faster loops.

### **Disabling Enhancement**
Set `enhance_quality: false` for ~2x faster processing. Useful for:
- Testing/iteration
- Videos where quality is already good
- Real-time applications

---

## Troubleshooting

### **"Face not detected" error**
- Ensure faces are clearly visible in all frames
- Try lowering `similarity_threshold` for face matching
- Check reference image has a clear frontal face

### **Rectangular artifacts visible**
- This shouldn't happen with Poisson blending enabled
- If it does, the fallback feathered blending is being used
- Check logs for `cv2.error` messages

### **Out of memory (OOM)**
- Reduce `wav2lip_batch_size` (default: 128)
- Process shorter video segments
- Use a GPU with more VRAM

### **Models not downloading**
- Check network connectivity
- Verify HuggingFace/GitHub URLs are accessible
- Models download on first startup (~2 minutes)

---

## Technical Details

### **Model Stack**
| Component | Model | Size | Purpose |
|-----------|-------|------|---------|
| Face Detection | InsightFace `buffalo_l` | ~500MB | Multi-face tracking + recognition |
| Lip Sync | Wav2Lip GAN | ~400MB | Audio-to-lip generation |
| Face Enhancement | GFPGAN v1.4 | ~350MB | Post-processing quality boost |
| S3FD | S3FD | ~85MB | Wav2Lip internal face detection |

### **Processing Pipeline**
```
Input Video + Audio
       ↓
[1] Download assets
       ↓
[2] Extract reference face embeddings (InsightFace)
       ↓
[3] Track faces across video frames (sampled + interpolated)
       ↓
[4] Generate lip-synced frames (Wav2Lip @ 96x96)
       ↓
[5] Composite with Poisson blending
       ↓
[6] Optional GFPGAN enhancement
       ↓
[7] Upload to presigned URL
```

### **VRAM Usage**
| Component | VRAM |
|-----------|------|
| InsightFace | ~1GB |
| Wav2Lip | ~1GB |
| GFPGAN (if enabled) | ~2GB |
| **Total** | **~4GB** |

---

## Links

- **GitHub:** [ScenemaAI/models](https://github.com/allgoodthings/models)
- **Wav2Lip Paper:** [A Lip Sync Expert Is All You Need](https://arxiv.org/abs/2008.10010)
- **GFPGAN:** [Towards Real-World Blind Face Restoration](https://github.com/TencentARC/GFPGAN)
- **InsightFace:** [Deep Face Analysis](https://github.com/deepinsight/insightface)

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 2.0.0 | 2026-01-09 | Initial release with Wav2Lip + GFPGAN |
| 2.1.0 | 2026-01-09 | Added Poisson blending + loop modes |

---

*Built with Wav2Lip, InsightFace, and GFPGAN. Optimized for production deployment.*
