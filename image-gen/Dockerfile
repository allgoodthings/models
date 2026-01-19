# Image-Gen Service
# FLUX.2-klein image generation with Real-ESRGAN upscaling
#
# Build:
#   docker build -t image-gen:latest .
#
# Run:
#   docker run --gpus all -p 7000:7000 -e HUGGING_FACE_TOKEN=hf_xxx image-gen:latest
#
# Models:
#   - FLUX.2-klein-4B (default, 24GB GPUs) - set MODEL_ID=4B
#   - FLUX.2-klein-9B (32GB+ GPUs) - set MODEL_ID=9B
#
# GPU Support:
#   - Uses PyTorch nightly for Blackwell (sm_120) support
#   - Compatible with RTX 4090, A100, H100, RTX PRO 4000 Blackwell, etc.

FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    libgl1 \
    libglib2.0-0 \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Install uv for fast package management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Install PyTorch nightly with CUDA 12.6 (supports Blackwell sm_120)
RUN uv pip install --system --no-cache \
    --pre torch torchvision \
    --index-url https://download.pytorch.org/whl/nightly/cu126

# Install diffusers from git (needed for Flux2KleinPipeline)
RUN uv pip install --system --no-cache \
    "git+https://github.com/huggingface/diffusers.git" \
    transformers>=4.40.0 \
    accelerate>=0.30.0 \
    safetensors>=0.4.0 \
    sentencepiece>=0.2.0 \
    protobuf>=4.0.0 \
    "huggingface-hub>=0.23.0,<1.0"

# Install TorchAO nightly for FP8 quantization (optional, for 9B model on 24GB)
RUN uv pip install --system --no-cache --pre torchao --index-url https://download.pytorch.org/whl/nightly/cu126

# Install Real-ESRGAN dependencies (skip facexlib which has Python 3.11 incompatible deps)
# basicsr from git has torchvision compatibility fixes
RUN uv pip install --system --no-cache \
    "git+https://github.com/XPixelGroup/BasicSR.git" \
    opencv-python>=4.8.0

# Install realesrgan without deps (we manually installed what we need above)
RUN uv pip install --system --no-cache --no-deps realesrgan>=0.3.0

# Install server dependencies
RUN uv pip install --system --no-cache \
    fastapi>=0.104.0 \
    "uvicorn[standard]>=0.24.0" \
    httpx>=0.25.0 \
    pydantic>=2.5.0 \
    pillow>=10.0.0 \
    numpy>=1.24.0

# Copy source code
COPY imagegen/ ./imagegen/
COPY scripts/ ./scripts/
COPY pyproject.toml ./

# Create cache directories
RUN mkdir -p /root/.cache/huggingface /root/.cache/realesrgan

# Pre-download Real-ESRGAN model weights (~64MB)
RUN mkdir -p /root/.cache/realesrgan && \
    wget -q "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth" \
    -O /root/.cache/realesrgan/x4plus.pth && \
    echo "Real-ESRGAN model downloaded"

# Environment
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# HuggingFace token (required - set at runtime)
# docker run -e HUGGING_FACE_TOKEN=hf_xxx ...
ENV HUGGING_FACE_TOKEN=""

# Model selection (4B default for 24GB GPUs, 9B for 32GB+)
ENV MODEL_ID="black-forest-labs/FLUX.2-klein-4B"

# Quantization (none for 4B, fp8 for 9B on 24GB)
ENV QUANTIZATION="none"

# Preload model on startup
ENV PRELOAD_MODELS="true"

# Server port
EXPOSE 7000

# Health check (longer start-period for model download)
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:7000/health || exit 1

# Run server
CMD ["uvicorn", "imagegen.server.main:app", "--host", "0.0.0.0", "--port", "7000"]
