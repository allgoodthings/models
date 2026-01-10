#!/bin/bash
# Higgs Audio v2 High-Throughput Setup
# Run this on a GPU instance with 24GB+ VRAM
#
# Supports two modes:
#   1. Docker mode (preferred): Uses pre-built bosonai/higgs-audio-vllm image
#      - Supports concurrent inference via continuous batching
#   2. Direct mode: Uses HiggsAudioServeEngine
#      - Single request at a time (max_batch_size=1)
#      - Still useful for testing quality and API compatibility

set -e

echo "=== Higgs Audio v2 Setup ==="

# Check for NVIDIA GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. This requires an NVIDIA GPU."
    exit 1
fi

echo "GPU detected:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Determine setup mode
USE_DOCKER=false
if command -v docker &> /dev/null; then
    # Check if Docker daemon is running and we can use it
    if docker info &> /dev/null; then
        USE_DOCKER=true
        echo "Docker available - will use vLLM Docker mode (concurrent inference)"
    else
        echo "Docker installed but daemon not accessible - using direct mode"
    fi
else
    echo "Docker not available - using HiggsAudioServeEngine mode (single request)"
fi

if [ "$USE_DOCKER" = true ]; then
    # ============================================
    # Docker Mode Setup (vLLM with concurrent inference)
    # ============================================

    # Check for nvidia-container-toolkit
    if ! docker info 2>/dev/null | grep -q "nvidia"; then
        echo "WARNING: NVIDIA container toolkit may not be installed."
        echo "If docker run fails, install it with:"
        echo "  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
        echo "  distribution=\$(. /etc/os-release;echo \$ID\$VERSION_ID)"
        echo "  curl -s -L https://nvidia.github.io/libnvidia-container/\$distribution/libnvidia-container.list | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list"
        echo "  sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit"
        echo "  sudo systemctl restart docker"
        echo ""
    fi

    # Pull the vLLM image
    echo "Pulling Higgs Audio vLLM Docker image..."
    docker pull bosonai/higgs-audio-vllm:latest

    # Install Python dependencies for proxy server and benchmarking
    echo "Installing Python dependencies for proxy server..."
    pip install fastapi uvicorn httpx aiohttp numpy scipy requests --quiet 2>/dev/null || \
    pip install fastapi uvicorn httpx aiohttp numpy scipy requests --quiet --break-system-packages

    # Write mode indicator
    echo "docker" > .vllm_mode

else
    # ============================================
    # Direct Mode Setup (HiggsAudioServeEngine)
    # Note: vLLM stock doesn't support higgs_audio model type
    # The custom vLLM fork is only in the Docker image
    # ============================================

    echo "Installing Higgs Audio and dependencies..."
    echo "This may take several minutes..."

    # Clone higgs-audio repo if not exists
    if [ ! -d "higgs-audio" ]; then
        echo "Cloning Higgs Audio repository..."
        git clone https://github.com/boson-ai/higgs-audio.git
    fi

    # Install higgs-audio package and its dependencies
    cd higgs-audio
    pip install -r requirements.txt --break-system-packages 2>/dev/null || pip install -r requirements.txt
    pip install -e . --break-system-packages 2>/dev/null || pip install -e .
    pip install librosa fastapi uvicorn httpx aiohttp numpy scipy requests --break-system-packages 2>/dev/null || \
    pip install librosa fastapi uvicorn httpx aiohttp numpy scipy requests
    cd ..

    echo ""
    echo "Verifying Higgs Audio installation..."
    python3 -c "from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine; print('HiggsAudioServeEngine: OK')"

    # Write mode indicator
    echo "direct" > .vllm_mode
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Mode: $(cat .vllm_mode)"
if [ "$(cat .vllm_mode)" = "direct" ]; then
    echo ""
    echo "NOTE: Direct mode uses HiggsAudioServeEngine (max_batch_size=1)"
    echo "      For concurrent inference, use a Docker-enabled instance."
fi
echo ""
echo "Next steps:"
echo "  1. Start backend:       ./start-vllm.sh"
echo "  2. Start proxy server:  python server.py"
echo "  3. Run benchmarks:      python benchmark.py"
echo ""
