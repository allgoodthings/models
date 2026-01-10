#!/bin/bash
# Higgs Audio v2 POC Setup Script
# Run this on a Vast.ai GPU instance with 24GB+ VRAM

set -e

echo "=== Higgs Audio v2 POC Setup ==="

# Check for NVIDIA GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. This requires an NVIDIA GPU."
    exit 1
fi

echo "GPU detected:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Clone higgs-audio if not present
if [ ! -d "higgs-audio" ]; then
    echo "Cloning higgs-audio repository..."
    git clone https://github.com/boson-ai/higgs-audio.git
fi

cd higgs-audio

# Create venv with Python 3.10
echo "Creating Python 3.10 virtual environment..."
uv venv --python 3.10
source .venv/bin/activate

# Install higgs-audio dependencies
echo "Installing higgs-audio dependencies..."
uv pip install -r requirements.txt
uv pip install -e .

# Install server dependencies
echo "Installing FastAPI server dependencies..."
uv pip install fastapi uvicorn python-multipart scipy numpy torchaudio

# Copy scripts
cd ..
cp server.py higgs-audio/
chmod +x monitor_gpu.sh

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To start the server:"
echo "  cd /workspace/higgs-audio-poc/higgs-audio"
echo "  source .venv/bin/activate"
echo "  python /workspace/higgs-audio-poc/server.py"
echo ""
echo "To start GPU monitoring (in separate terminal):"
echo "  ./monitor_gpu.sh"
echo ""
echo "Server will run on http://0.0.0.0:8888"
echo ""
echo "Endpoints:"
echo "  GET  /health     - Health check"
echo "  POST /generate   - Basic TTS (smart voice)"
echo "  POST /clone      - Voice cloning"
echo "  POST /emotional  - Emotional speech"
echo "  POST /dialogue   - Multi-speaker dialogue"
echo "  GET  /docs       - API documentation"
