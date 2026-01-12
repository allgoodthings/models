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
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Install uv for fast package management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /workspace

# Clone higgs-audio repository
RUN git clone https://github.com/boson-ai/higgs-audio.git

WORKDIR /workspace/higgs-audio

# Create virtual environment and install dependencies
RUN uv venv --python 3.10

# Install higgs-audio dependencies
RUN . .venv/bin/activate && \
    uv pip install -r requirements.txt && \
    uv pip install -e .

# Install server dependencies
RUN . .venv/bin/activate && \
    uv pip install \
        fastapi \
        uvicorn \
        python-multipart \
        scipy \
        numpy \
        torchaudio

# Copy server code
COPY server.py /workspace/server.py

# Create voice cache directory
RUN mkdir -p /tmp/voice_cache

# Expose port
EXPOSE 8000

# Set environment
ENV VOICE_CACHE_DIR=/tmp/voice_cache

# Run server - override nvidia entrypoint
WORKDIR /workspace
ENTRYPOINT ["/workspace/higgs-audio/.venv/bin/python"]
CMD ["server.py"]
