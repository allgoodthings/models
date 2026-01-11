# Higgs Audio v3 - Integrated vLLM Server
# Uses vLLM for token generation + local audio decoding

FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Create app directory
WORKDIR /app

# Install PyTorch with CUDA support
RUN pip3 install --no-cache-dir \
    torch==2.4.0 \
    torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu124

# Install vLLM
RUN pip3 install --no-cache-dir vllm==0.6.0

# Install audio processing dependencies
RUN pip3 install --no-cache-dir \
    numpy \
    scipy \
    librosa \
    soundfile \
    huggingface_hub

# Install FastAPI and web dependencies
RUN pip3 install --no-cache-dir \
    fastapi \
    uvicorn \
    pydantic \
    httpx

# Install boson_multimodal for audio decoder
RUN pip3 install --no-cache-dir git+https://github.com/boson-ai/higgs-audio.git

# Copy server code
COPY server_integrated.py /app/server.py

# Download models at build time (optional - can also download at runtime)
# RUN python3 -c "from huggingface_hub import snapshot_download; snapshot_download('bosonai/higgs-audio-v2-tokenizer')"

# Expose port
EXPOSE 8080

# Run server
CMD ["python3", "server.py"]
