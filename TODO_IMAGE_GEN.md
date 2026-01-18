# Image-Gen Service Implementation Progress

## Overview
FLUX.2 [klein] 9B distilled image generation service following the lip-sync-v2 pattern.

**Two-phase approach:**
- Phase 1: Test directly on Vast AI (no Docker) - benchmark VRAM with reference images
- Phase 2: Dockerize once benchmarks confirm GPU requirements

---

## Phase 1: Vast AI Testing

### Step 1: Project Scaffolding
- [x] Create `pyproject.toml`
- [x] Create `Makefile`
- [x] Create `.gitignore`
- [x] Create `imagegen/__init__.py`
- [x] Create `imagegen/server/__init__.py`

### Step 2: Pydantic Schemas
- [x] Create `imagegen/server/schemas.py`
  - HealthResponse
  - GenerateRequest/Response
  - EditRequest/Response

### Step 3: VRAM Benchmark Script
- [x] Create `scripts/benchmark_vram.py`
  - Model loading with/without CPU offload
  - Text-to-image at various resolutions
  - Multi-reference editing tests
  - Peak VRAM logging

### Step 4: FLUX Pipeline Wrapper
- [x] Create `imagegen/server/flux_pipeline.py`
  - FluxConfig dataclass
  - FluxPipeline class with load/unload/generate/edit methods
  - CPU offload for RTX 4090 compatibility

### Step 5: FastAPI Server
- [x] Create `imagegen/server/main.py`
  - Lifespan for model loading
  - GET /health endpoint
  - POST /generate endpoint
  - POST /edit endpoint
  - GET / root endpoint

### Step 6: Documentation
- [x] Create `README-VASTAI.md` - Setup instructions for Vast AI
- [x] Create `scripts/test_api.py` - API testing script

### Step 7: Tests
- [x] Create `tests/conftest.py` - Pytest fixtures
- [x] Create `tests/test_schemas.py` - Schema validation tests
- [x] Create `README.md` - Main project documentation

---

## Phase 2: Dockerization (After Benchmarks Pass)

### Docker Setup
- [ ] Create `Dockerfile` based on lip-sync-v2
- [ ] Test Docker build locally
- [ ] Test on Vast AI with Docker

### CI/CD
- [ ] Create `.github/workflows/image-gen.yml`
- [ ] Unit tests for schemas
- [ ] Docker build and push to GHCR

---

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HUGGING_FACE_TOKEN` | Yes | - | HF token for gated model |
| `PRELOAD_MODELS` | No | `true` | Load model on startup |
| `LOG_LEVEL` | No | `INFO` | Logging level |
| `HF_HOME` | No | `~/.cache/huggingface` | Model cache dir |

---

## Verification Checklist

### Phase 1 (Vast AI)
- [ ] Run `scripts/benchmark_vram.py` on RTX 4090
- [ ] Confirm model loads within 24GB with CPU offload
- [ ] Record max reference images supported before OOM
- [ ] Start server: `uvicorn imagegen.server.main:app --host 0.0.0.0 --port 7000`
- [ ] Test `/health` endpoint
- [ ] Test `/generate` with curl/httpie
- [ ] Test `/edit` with 1, 2, 3, 4 reference images

### Phase 2 (Docker)
- [ ] Build: `docker build -t image-gen:latest .`
- [ ] Run: `docker run --gpus all -p 7000:7000 -e HUGGING_FACE_TOKEN=xxx image-gen:latest`
- [ ] Wait for health check to pass
- [ ] Test all endpoints

---

## Notes

- **Model**: `black-forest-labs/FLUX.2-klein-9B`
  - 9B parameter rectified flow transformer
  - Step-distilled to 4 steps, sub-second generation
  - VRAM: ~19GB (fits on RTX 4090 24GB with headroom)
  - Supports both text-to-image and multi-reference image editing
- Reference files: `/Users/biz/Documents/projects/ScenemaAI/models/lip-sync-v2/`
- GPU target: RTX 4090 (24GB)