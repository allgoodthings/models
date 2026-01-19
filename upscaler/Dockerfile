# Video Upscaler - Real-ESRGAN based video upscaling service
#
# Build:
#   docker build -t video-upscaler:latest .
#
# Run:
#   docker run --gpus all -p 8000:8000 video-upscaler:latest

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
    --index-url https://download.pytorch.org/whl/cu124

# Install Real-ESRGAN and dependencies
RUN uv pip install --system --no-cache \
    realesrgan>=0.3.0 \
    basicsr>=1.4.2 \
    numpy>=1.24.0 \
    opencv-python-headless>=4.8.0 \
    pillow>=10.0.0 \
    scipy>=1.11.0 \
    tqdm>=4.66.0

# Fix basicsr/torchvision compatibility issue
# basicsr imports from torchvision.transforms.functional_tensor which was removed in torchvision 0.18+
RUN BASICSR_PATH=$(uv pip show basicsr | grep Location | cut -d' ' -f2)/basicsr && \
    sed -i 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' \
    "$BASICSR_PATH/data/degradations.py" && \
    echo "Patched basicsr at $BASICSR_PATH for torchvision compatibility"

# Install server dependencies
RUN uv pip install --system --no-cache \
    fastapi>=0.104.0 \
    "uvicorn[standard]>=0.24.0" \
    httpx>=0.25.0 \
    pydantic>=2.5.0 \
    av>=10.0.0 \
    ffmpeg-python>=0.2.0 \
    huggingface-hub>=0.19.0

# Clone All-In-One-Deflicker (optional - for temporal deflicker)
ARG INCLUDE_DEFLICKER=false
RUN if [ "$INCLUDE_DEFLICKER" = "true" ]; then \
    git clone --depth 1 https://github.com/ChenyangLEI/All-In-One-Deflicker.git /app/vendors/All-In-One-Deflicker && \
    cd /app/vendors/All-In-One-Deflicker && \
    uv pip install --system --no-cache -r requirements.txt && \
    echo "Deflicker installed"; \
    fi

# Copy source code
COPY upscaler/ ./upscaler/
COPY scripts/ ./scripts/

# Create models directory
RUN mkdir -p /app/models

# Pre-download Real-ESRGAN models
ARG SKIP_MODEL_DOWNLOAD=false
RUN if [ "$SKIP_MODEL_DOWNLOAD" = "false" ]; then \
    echo "Pre-downloading Real-ESRGAN models..." && \
    # AnimevideV3 (optimized for anime/AI video)
    wget -q "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth" \
    -O /app/models/realesr-animevideov3.pth && \
    # X4plus (general purpose)
    wget -q "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth" \
    -O /app/models/realesrgan-x4plus.pth && \
    echo "Models downloaded successfully"; \
    fi

# Set environment
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV MODELS_DIR=/app/models

# Server port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Preload models on startup
ENV PRELOAD_MODELS=true

# Run the server
CMD ["uvicorn", "upscaler.server.main:app", "--host", "0.0.0.0", "--port", "8000"]
