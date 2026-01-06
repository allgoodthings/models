# Multi-Face Lip-Sync Pipeline
# Based on InsightFace + MuseTalk + LivePortrait + CodeFormer
#
# Build:
#   docker build -t lip-sync:latest .
#
# Run:
#   docker run --gpus all -p 8000:8000 lip-sync:latest

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Prevent interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    git-lfs \
    curl \
    wget \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Clone official repos first (separate layer for caching)
RUN git clone --depth 1 https://github.com/TMElyralab/MuseTalk.git /app/MuseTalk \
    && git clone --depth 1 https://github.com/KwaiVGI/LivePortrait.git /app/LivePortrait

# Install PyTorch with CUDA support FIRST (pinned version for CUDA 12.4)
RUN uv pip install --system --no-cache \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

# Install other Python dependencies
COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

# Install face-related packages (after torch is installed)
RUN uv pip install --system --no-cache \
    onnxruntime-gpu \
    basicsr \
    facexlib \
    gfpgan \
    insightface

# Install MuseTalk dependencies from their requirements
RUN uv pip install --system --no-cache \
    openai-whisper \
    einops \
    diffusers \
    accelerate \
    transformers

# Install LivePortrait dependencies
RUN uv pip install --system --no-cache \
    tyro \
    dlib \
    pyyaml

# Install scikit-image for histogram matching
RUN uv pip install --system --no-cache scikit-image

# Copy source code
COPY lipsync/ ./lipsync/
COPY scripts/ ./scripts/

# Download and bake model weights (~7GB total for lip-sync models)
# This is done as a separate layer so code changes don't trigger re-download
ARG SKIP_MODEL_DOWNLOAD=false
RUN if [ "$SKIP_MODEL_DOWNLOAD" = "false" ]; then \
    python scripts/download_models.py --models-dir /app/models; \
    fi

# Pre-download InsightFace buffalo_l model (~500MB)
# This ensures the model is cached and ready at startup
RUN python3 -c "\
from insightface.app import FaceAnalysis; \
print('Downloading InsightFace buffalo_l model...'); \
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider']); \
app.prepare(ctx_id=-1); \
print('InsightFace model downloaded successfully')"

# Pre-download Whisper tiny model (~75MB)
RUN python3 -c "\
import whisper; \
print('Downloading Whisper tiny model...'); \
whisper.load_model('tiny'); \
print('Whisper model downloaded successfully')"

# Pre-download VAE model
RUN python3 -c "\
from diffusers import AutoencoderKL; \
print('Downloading SD VAE...'); \
AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-mse'); \
print('VAE downloaded successfully')"

# Set environment
ENV PYTHONPATH=/app:/app/MuseTalk:/app/LivePortrait
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Model paths
ENV MODELS_DIR=/app/models
ENV MUSETALK_PATH=/app/MuseTalk
ENV LIVEPORTRAIT_PATH=/app/LivePortrait
ENV CODEFORMER_PATH=/app/models/codeformer

# Server port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Preload models on startup for faster first request
ENV PRELOAD_MODELS=true

# Run the server
CMD ["uvicorn", "lipsync.server.main:app", "--host", "0.0.0.0", "--port", "8000"]
