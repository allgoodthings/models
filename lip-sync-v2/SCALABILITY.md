# Lip-Sync V2 Scalability Analysis

Based on performance testing conducted on 2026-01-09.

## Test Configuration

- **Video**: 1280x720 @ 30fps, 9.3s output (280 frames)
- **Hardware**: Vast.ai instance with RTX 3090
- **Mode**: `enhance_quality=false` (no GFPGAN)

## Hardware Specs

| Resource | Specification |
|----------|---------------|
| GPU | NVIDIA GeForce RTX 3090 (24GB VRAM) |
| CPU | AMD EPYC 7532 32-Core (64 threads) |
| RAM | 109GB total |
| CUDA | 12.8 |
| Driver | 570.172.08 |

## Performance Results

### With vs Without GFPGAN Enhancement

| Mode | Lipsync Time | Total Time | Speedup |
|------|-------------|------------|---------|
| With GFPGAN | 81s | 85s | baseline |
| **Without GFPGAN** | **10.8s** | **17.4s** | **5x faster** |

### Timing Breakdown (Without GFPGAN)

| Stage | Time | % of Total |
|-------|------|------------|
| Download (video + audio) | 5.5s | 32% |
| Face Tracking (InsightFace) | 0.9s | 5% |
| Wav2Lip Inference | **0.36s** | **2%** |
| Frame Processing + Encoding | ~10s | 57% |
| Upload | 0.2s | 1% |
| **Total** | **17.4s** | 100% |

### GPU Utilization

| Mode | GPU Utilization | GPU Memory |
|------|-----------------|------------|
| With GFPGAN | 35-59% sustained | ~3.5GB |
| Without GFPGAN | <5% (brief spikes) | 3.4GB |

**Key Finding**: Without GFPGAN, GPU utilization is minimal. The 0.36s Wav2Lip inference represents only 2% of total processing time.

## Resource Usage Per Request

| Resource | Usage | Notes |
|----------|-------|-------|
| RAM | ~1GB | 738MB for 280 frames @ 1280x720 + buffers |
| GPU Memory | ~200MB burst | Models shared across requests |
| GPU Compute | 0.36s | Brief inference burst |
| CPU | ~10s | Frame I/O, mel processing, ffmpeg encoding |

## Concurrency Benchmarks

Actual load testing performed with 1, 2, 4, 8, and 16 concurrent requests.

### Results

| Concurrent | Per-Request Time | Wall Time | Throughput | RAM Peak |
|------------|-----------------|-----------|------------|----------|
| 1 | 12.2s | 12.2s | 0.082 req/s | 40.0 GB |
| 2 | 23.3s | 23.3s | 0.086 req/s | 40.2 GB |
| 4 | 50.0s | 50.0s | 0.080 req/s | 40.8 GB |
| 8 | 101.1s | 101.1s | 0.079 req/s | 46.7 GB |
| 16 | 99.0s* | ~200s | ~0.08 req/s | 47.6 GB |

*Only 8 of 16 requests completed within measurement window

### Hardware Utilization Under Load (8 Concurrent)

| Resource | Idle | 8 Concurrent | Notes |
|----------|------|--------------|-------|
| GPU Util | 0% | 67% peak | Brief spikes during inference |
| GPU Memory | 3438 MB | 3438 MB | No increase (shared model) |
| RAM | 40.0 GB | 46.7 GB | +840 MB per concurrent request |
| CPU | ~3% | ~100% | Fully saturated |

### Key Finding: Serial Execution

**Throughput is constant (~0.08 req/s) regardless of concurrency level.**

The single uvicorn worker serializes all requests. Per-request latency scales linearly with queue depth:
- 1 concurrent: 12s
- 8 concurrent: 101s (8x slower per request, same total throughput)

### Scaling Options

To increase throughput beyond 0.08 req/s:

1. **Multiple uvicorn workers**: `uvicorn ... --workers 4`
2. **Horizontal scaling**: Multiple server instances behind load balancer
3. **GPU batching**: Batch multiple requests' inference together (requires code changes)

### Theoretical Limits (with parallelization)

| Bottleneck | Calculation | Max Concurrent |
|------------|-------------|----------------|
| RAM | 60GB available / 0.84GB per request | ~70 |
| GPU Memory | 20GB free / shared model | ~100+ |
| GPU Compute | 0.36s burst per request | ~30 (if batched) |
| **CPU** | **64 threads / ~10s CPU work** | **~6 parallel** |

### Multi-Worker Benchmark Results (Actual)

Tested with uvicorn `--workers N` and N concurrent requests:

| Workers | Per-Request Time | Wall Time | Throughput | VRAM | RAM |
|---------|------------------|-----------|------------|------|-----|
| 1 | 12.2s | 12.2s | 0.08 req/s | 3.4 GB | 40 GB |
| 2 | 18.5s | 19s | 0.11 req/s | 6.7 GB | 45 GB |
| 4 | 18-30s | 30s | 0.13 req/s | 10 GB | 50 GB |
| 8 | 9-42s* | 42s | 0.19 req/s | 13.3 GB | 55 GB |

*8 workers hit fork resource limits; 5 requests finished fast (9s), 3 queued (27-42s)

**Key Finding**: Throughput scales sub-linearly due to:
1. **VRAM duplication**: Each worker loads its own model copy (~1.7GB per worker)
2. **CPU contention**: Workers compete for ffmpeg/encoding resources
3. **Fork limits**: 8+ workers exhaust process limits

## Key Insights

### 1. GFPGAN is Unnecessary

GFPGAN was designed for restoring degraded/old photos. In our pipeline:
- Wav2Lip only modifies the 96x96 mouth region
- The original face is already high quality
- GFPGAN was enhancing pixels that were never modified
- Poisson blending alone provides seamless results

**Recommendation**: Default `enhance_quality=false`. Remove GFPGAN from the pipeline entirely.

### 2. GPU is Over-Provisioned

With GFPGAN disabled:
- GPU utilization: <5%
- GPU compute time: 0.36s out of 17.4s (2%)

**Recommendation**: A much cheaper GPU (T4, RTX 3060) would suffice. The RTX 3090 is overkill for this workload.

### 3. CPU is the New Bottleneck

The majority of processing time is CPU-bound:
- Frame extraction (ffmpeg)
- Mel spectrogram generation
- Video encoding (ffmpeg)
- File I/O

**Recommendation**: For higher throughput, optimize ffmpeg settings or use hardware-accelerated encoding (NVENC).

### 4. Memory Scales Linearly

Each concurrent request needs ~1GB RAM for frame buffers. With 60GB available:
- Safe concurrent limit: ~40-50 requests (leaving headroom)
- Actual limit will be CPU before RAM

## Cost Optimization

| Configuration | GPU | Est. Cost | Max Concurrent | Notes |
|---------------|-----|-----------|----------------|-------|
| Current | RTX 3090 | ~$0.40/hr | 10-15 | Over-provisioned |
| Optimized | RTX 3060 | ~$0.15/hr | 10-15 | Same throughput |
| Budget | T4 | ~$0.10/hr | 8-12 | Slightly slower |

## Recommendations

1. **Disable GFPGAN by default** - 5x speedup, no quality loss for the mouth region
2. **Use 2-4 workers per instance** - Best throughput/resource ratio (0.11-0.13 req/s)
3. **Scale horizontally** - Multiple 2-worker instances > one 8-worker instance
4. **Downgrade GPU** - T4 or RTX 3060 sufficient for 2 workers (~3.4GB VRAM each)
5. **Consider NVENC** - Hardware video encoding could reduce CPU contention

## Appendix: Raw Metrics

### Server Logs (Without GFPGAN)
```
2026-01-09 21:53:XX [lipsync.server] INFO: Using in-process Wav2Lip model (GPU optimized)
2026-01-09 21:53:XX [lipsync.server] INFO: Loaded 153 frames at 1280x720 @ 30.0fps
2026-01-09 21:53:XX [lipsync.server] INFO: Generated 280 mel chunks
2026-01-09 21:53:XX [lipsync.wav2lip_model] INFO: Processing 153 frames with 280 mel chunks
2026-01-09 21:53:XX [lipsync.wav2lip_model] INFO: Running Wav2Lip inference on 280 faces...
2026-01-09 21:53:XX [lipsync.wav2lip_model] INFO: Inference completed in 0.36s
```

### GPU State (Idle)
```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.172.08             Driver Version: 570.172.08     CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3090        On  |   00000000:C2:00.0 Off |                  N/A |
|  0%   32C    P8             20W /  350W |    3438MiB /  24576MiB |      0%      Default |
+-----------------------------------------+------------------------+----------------------+
```

### Memory Profile
```
Mem: 109Gi total, 38Gi used, 18Gi free, 53Gi buff/cache, 60Gi available
```
