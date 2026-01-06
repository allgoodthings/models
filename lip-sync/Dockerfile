# Multi-Face Lip-Sync Pipeline
# Based on InsightFace + MuseTalk + LivePortrait + CodeFormer
#
# Build:
#   docker build -t lip-sync:latest .
#
# Run:
#   docker run --gpus all -p 8000:8000 lip-sync:latest

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# Prevent interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-venv \
    python3.10-dev \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    git \
    git-lfs \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

WORKDIR /app

# Install PyTorch with CUDA support FIRST (pinned version)
RUN pip install --no-cache-dir \
    torch==2.1.2 \
    torchvision==0.16.2 \
    torchaudio==2.1.2 \
    --index-url https://download.pytorch.org/whl/cu121

# Install other Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install face-related packages (after torch is installed)
RUN pip install --no-cache-dir \
    mediapipe \
    onnxruntime-gpu \
    basicsr \
    facexlib \
    gfpgan \
    insightface

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

# Set environment
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Model paths
ENV MODELS_DIR=/app/models
ENV MUSETALK_PATH=/app/models/musetalk
ENV LIVEPORTRAIT_PATH=/app/models/liveportrait
ENV CODEFORMER_PATH=/app/models/codeformer

# Server port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the server
CMD ["uvicorn", "lipsync.server.main:app", "--host", "0.0.0.0", "--port", "8000"]
