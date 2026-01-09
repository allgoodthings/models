# Lip-Sync V2 Pipeline
# InsightFace face tracking + Wav2Lip lip-sync + GFPGAN enhancement
#
# Build:
#   docker build -t lip-sync-v2:latest .
#
# Run:
#   docker run --gpus all -p 8000:8000 lip-sync-v2:latest

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Prevent interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies (matching v1)
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

# Clone Wav2Lip and GFPGAN repositories FIRST (better layer caching)
RUN mkdir -p /app/vendors/wav2lip-hd && \
    git clone --depth 1 https://github.com/Rudrabha/Wav2Lip.git /app/vendors/wav2lip-hd/Wav2Lip-master && \
    git clone --depth 1 https://github.com/TencentARC/GFPGAN.git /app/vendors/wav2lip-hd/GFPGAN-master

# Copy patched files:
# - inference.py: --bbox_file support for per-frame bounding boxes
# - audio.py: librosa 0.10+ compatibility (keyword-only args)
COPY patches/wav2lip_inference.py /app/vendors/wav2lip-hd/Wav2Lip-master/inference.py
COPY patches/wav2lip_audio.py /app/vendors/wav2lip-hd/Wav2Lip-master/audio.py

# Install PyTorch with CUDA support (pinned version for CUDA 12.4)
RUN uv pip install --system --no-cache \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

# Install face detection and processing packages
# Note: onnxruntime-gpu 1.19+ required for CUDA 12
RUN uv pip install --system --no-cache \
    onnxruntime-gpu>=1.19.0 \
    insightface>=0.7.3 \
    opencv-python-headless>=4.8.0 \
    numpy>=1.24.0

# Install Wav2Lip dependencies
RUN uv pip install --system --no-cache \
    librosa>=0.10.0 \
    scipy>=1.11.0 \
    tqdm>=4.66.0 \
    batch_face>=1.4.0

# Install GFPGAN and dependencies (for face enhancement)
RUN uv pip install --system --no-cache \
    basicsr>=1.4.2 \
    facexlib>=0.3.0 \
    gfpgan>=1.3.8 \
    realesrgan>=0.3.0

# Fix basicsr/torchvision compatibility issue
# basicsr imports from torchvision.transforms.functional_tensor which was removed in torchvision 0.18+
# This patches the import at the package level so it works for all processes (including subprocesses)
RUN BASICSR_PATH=$(python3 -c "import basicsr; print(basicsr.__path__[0])") && \
    sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' \
    "$BASICSR_PATH/data/degradations.py" && \
    echo "Patched basicsr for torchvision compatibility"

# Install server dependencies
RUN uv pip install --system --no-cache \
    fastapi>=0.104.0 \
    "uvicorn[standard]>=0.24.0" \
    httpx>=0.25.0 \
    pydantic>=2.5.0 \
    pillow>=10.0.0 \
    av>=10.0.0 \
    ffmpeg-python>=0.2.0 \
    huggingface-hub>=0.19.0 \
    requests>=2.31.0

# Clone RIFE for frame interpolation (used for smooth loop transitions)
RUN git clone --depth 1 https://github.com/hzwer/ECCV2022-RIFE.git /app/vendors/rife

# Copy source code and scripts
COPY lipsync/ ./lipsync/
COPY scripts/ ./scripts/

# Copy requirements.txt if it exists (for any additional dependencies)
COPY requirements.txt* ./
RUN if [ -f requirements.txt ]; then uv pip install --system --no-cache -r requirements.txt; fi

# Create models directory
RUN mkdir -p /app/models

# Model downloads
#
# Models are downloaded at RUNTIME on first startup (see main.py ensure_models_downloaded).
# This allows faster CI builds with SKIP_MODEL_DOWNLOAD=true.
#
# Models downloaded at runtime:
#   - wav2lip_gan.pth (~400MB) - from HuggingFace
#   - GFPGANv1.4.pth (~350MB) - from GitHub
#   - s3fd.pth (~85MB) - S3FD face detection for Wav2Lip
#
# For faster cold starts, you can pre-download during build:
ARG SKIP_MODEL_DOWNLOAD=true

RUN if [ "$SKIP_MODEL_DOWNLOAD" = "false" ]; then \
    echo "Pre-downloading models during build..." && \
    # Wav2Lip (from HuggingFace - more reliable than SharePoint)
    wget -q "https://huggingface.co/numz/wav2lip_studio/resolve/main/Wav2lip/wav2lip_gan.pth" -O /app/models/wav2lip_gan.pth && \
    # GFPGAN
    wget -q "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth" -O /app/models/GFPGANv1.4.pth && \
    # S3FD face detection
    mkdir -p /app/vendors/wav2lip-hd/Wav2Lip-master/face_detection/detection/sfd && \
    wget -q "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth" -O /app/vendors/wav2lip-hd/Wav2Lip-master/face_detection/detection/sfd/s3fd.pth && \
    echo "Models pre-downloaded successfully"; \
    fi

# Pre-download InsightFace buffalo_l model (~500MB)
RUN python3 -c "\
from insightface.app import FaceAnalysis; \
print('Downloading InsightFace buffalo_l model...'); \
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider']); \
app.prepare(ctx_id=-1); \
print('InsightFace model downloaded successfully')"

# Download RIFE model weights (~120MB)
RUN mkdir -p /app/vendors/rife/train_log && \
    wget -q "https://github.com/hzwer/Practical-RIFE/releases/download/v4.6/flownet-v46.pkl" \
    -O /app/vendors/rife/train_log/flownet.pkl && \
    echo "RIFE model downloaded successfully"

# Set environment - PYTHONPATH must include Wav2Lip and RIFE for imports
ENV PYTHONPATH=/app:/app/vendors/wav2lip-hd/Wav2Lip-master:/app/vendors/rife
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Model paths
ENV MODELS_DIR=/app/models

# Server port
EXPOSE 8000

# Health check (longer start-period for model loading)
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Preload models on startup
ENV PRELOAD_MODELS=true

# Run the server
CMD ["uvicorn", "lipsync.server.main:app", "--host", "0.0.0.0", "--port", "8000"]
