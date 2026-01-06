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

# Install uv for fast package management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Clone official repos first (separate layer for caching)
# These provide the actual model code that we import
RUN git clone --depth 1 https://github.com/TMElyralab/MuseTalk.git /app/MuseTalk \
    && git clone --depth 1 https://github.com/KwaiVGI/LivePortrait.git /app/LivePortrait

# Install PyTorch with CUDA support FIRST (pinned version for CUDA 12.4)
RUN uv pip install --system --no-cache \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

# Install MuseTalk dependencies (from their requirements.txt)
RUN uv pip install --system --no-cache \
    diffusers==0.30.2 \
    accelerate==0.28.0 \
    numpy==1.26.4 \
    opencv-python==4.9.0.80 \
    soundfile==0.12.1 \
    transformers==4.39.2 \
    huggingface_hub>=0.20.0 \
    librosa==0.11.0 \
    einops==0.8.1 \
    omegaconf \
    ffmpeg-python \
    moviepy \
    imageio[ffmpeg]

# Install LivePortrait dependencies (from their requirements_base.txt)
RUN uv pip install --system --no-cache \
    pyyaml==6.0.1 \
    scipy==1.13.1 \
    imageio==2.34.2 \
    lmdb==1.4.1 \
    tqdm==4.66.4 \
    rich==13.7.1 \
    onnx==1.16.1 \
    scikit-image==0.24.0 \
    albumentations==1.4.10 \
    matplotlib==3.9.0 \
    imageio-ffmpeg==0.5.1 \
    tyro==0.8.5 \
    pykalman==0.9.7 \
    pillow>=10.2.0

# Install face-related packages (after torch is installed)
RUN uv pip install --system --no-cache \
    onnxruntime-gpu==1.18.0 \
    basicsr>=1.4.2 \
    facexlib>=0.3.0 \
    gfpgan>=1.3.8 \
    insightface>=0.7.3

# Install Whisper for audio feature extraction
RUN uv pip install --system --no-cache openai-whisper>=20231117

# Install server dependencies
RUN uv pip install --system --no-cache \
    fastapi>=0.104.0 \
    "uvicorn[standard]>=0.24.0" \
    httpx>=0.25.0 \
    pydantic>=2.5.0

# Install MMPose for MuseTalk face detection (optional, InsightFace is primary)
# Note: MuseTalk uses mmpose for DWPose but we use InsightFace instead
# RUN uv pip install --system --no-cache mmcv mmpose

# Copy source code
COPY lipsync/ ./lipsync/
COPY scripts/ ./scripts/

# Copy requirements.txt if it exists (for any additional dependencies)
COPY requirements.txt* ./
RUN if [ -f requirements.txt ]; then uv pip install --system --no-cache -r requirements.txt; fi

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

# Pre-download VAE model from HuggingFace
RUN python3 -c "\
from diffusers import AutoencoderKL; \
print('Downloading SD VAE...'); \
AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-mse'); \
print('VAE downloaded successfully')"

# Set environment - PYTHONPATH must include repo roots for imports
ENV PYTHONPATH=/app:/app/MuseTalk:/app/LivePortrait
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Model paths - where downloaded weights are stored
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
