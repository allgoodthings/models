# Lip-Sync V2 Pipeline
# InsightFace face tracking + Wav2Lip-HD lip-sync + GFPGAN enhancement
#
# Build:
#   docker build -t lip-sync-v2:latest .
#
# Run:
#   docker run --gpus all -p 8000:8000 lip-sync-v2:latest

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

# Install PyTorch with CUDA support (pinned version for CUDA 12.4)
RUN uv pip install --system --no-cache \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

# Install face detection and processing packages
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

# Install server dependencies
RUN uv pip install --system --no-cache \
    fastapi>=0.104.0 \
    "uvicorn[standard]>=0.24.0" \
    httpx>=0.25.0 \
    pydantic>=2.5.0 \
    pillow>=10.0.0 \
    av>=10.0.0 \
    ffmpeg-python>=0.2.0

# Copy Wav2Lip-HD vendor code (includes modifications for per-frame bbox)
COPY vendors/ ./vendors/

# Copy source code
COPY lipsync/ ./lipsync/

# Create models directory
RUN mkdir -p /app/models

# Download Wav2Lip checkpoint (~400MB)
ARG SKIP_MODEL_DOWNLOAD=false
RUN if [ "$SKIP_MODEL_DOWNLOAD" = "false" ]; then \
    echo "Downloading Wav2Lip checkpoint..." && \
    wget -q "https://iiitaphyd-my.sharepoint.com/personal/radrabha_m_research_iiit_ac_in/_layouts/15/download.aspx?SourceUrl=%2Fpersonal%2Fradrabha%5Fm%5Fresearch%5Fiiit%5Fac%5Fin%2FDocuments%2FWav2Lip%5FModels%2Fwav2lip%5Fgan%2Epth" -O /app/models/wav2lip_gan.pth || \
    echo "WARNING: Could not download Wav2Lip model from SharePoint. Please download manually."; \
    fi

# Download GFPGAN checkpoint (~350MB)
RUN if [ "$SKIP_MODEL_DOWNLOAD" = "false" ]; then \
    echo "Downloading GFPGAN checkpoint..." && \
    wget -q "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth" -O /app/models/GFPGANv1.4.pth; \
    fi

# Download face detection model for Wav2Lip's S3FD
RUN if [ "$SKIP_MODEL_DOWNLOAD" = "false" ]; then \
    echo "Downloading S3FD face detection model..." && \
    mkdir -p /app/vendors/wav2lip-hd/Wav2Lip-master/face_detection/detection/sfd && \
    wget -q "https://www.adrianbulat.com/downloads/python-fan/s3fd-619a316812.pth" -O /app/vendors/wav2lip-hd/Wav2Lip-master/face_detection/detection/sfd/s3fd.pth; \
    fi

# Pre-download InsightFace buffalo_l model (~500MB)
RUN python3 -c "\
from insightface.app import FaceAnalysis; \
print('Downloading InsightFace buffalo_l model...'); \
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider']); \
app.prepare(ctx_id=-1); \
print('InsightFace model downloaded successfully')"

# Set environment
ENV PYTHONPATH=/app:/app/vendors/wav2lip-hd/Wav2Lip-master
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0

# Model paths
ENV MODELS_DIR=/app/models

# Server port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Preload models on startup
ENV PRELOAD_MODELS=true

# Run the server
CMD ["uvicorn", "lipsync.server.main:app", "--host", "0.0.0.0", "--port", "8000"]
