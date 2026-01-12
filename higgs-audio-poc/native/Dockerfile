# Higgs Audio Native POC
# Uses boson_multimodal library directly (not vLLM)
# For testing if system_prompt + voice cloning work together

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Avoid interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3-pip \
    git \
    curl \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 \
    && update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

WORKDIR /workspace

# Clone higgs-audio repository
RUN git clone https://github.com/boson-ai/higgs-audio.git

WORKDIR /workspace/higgs-audio

# Install higgs-audio dependencies (no venv - we're in a container)
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -e .

# Install server dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    python-multipart \
    scipy \
    numpy \
    torchaudio

# Copy server code and start script to /app (not /workspace - RunPod mounts volume there)
COPY server.py /app/server.py
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# Create voice cache directory
RUN mkdir -p /tmp/voice_cache

# Expose port
EXPOSE 8000

# Set environment
ENV VOICE_CACHE_DIR=/tmp/voice_cache

# Run server
WORKDIR /app
CMD ["/app/start.sh"]
