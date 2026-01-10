#!/bin/bash
# Start Higgs Audio vLLM on RunPod
#
# Usage:
#   ./start-vllm.sh              # Default settings
#   ./start-vllm.sh --background # Run in background

set -e

# Configuration
PORT=${PORT:-8000}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.85}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
MODEL_NAME="higgs-audio-v2"
MODEL_PATH="bosonai/higgs-audio-v2-generation-3B-base"
TOKENIZER_PATH="bosonai/higgs-audio-v2-tokenizer"

echo "=== Starting Higgs Audio vLLM Server ==="
echo "Port: $PORT"
echo "GPU Memory Utilization: $GPU_MEMORY_UTILIZATION"
echo "Max Model Length: $MAX_MODEL_LEN"
echo ""

# Check for GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. GPU required."
    exit 1
fi

echo "GPU detected:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# Check for Docker
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker not found."
    exit 1
fi

# Pull image if needed
echo "Ensuring vLLM image is available..."
docker pull bosonai/higgs-audio-vllm:latest

echo ""
echo "Starting vLLM container..."
echo ""

# Build docker run command
DOCKER_CMD="docker run --gpus all \
    --ipc=host \
    --shm-size=20gb \
    -p ${PORT}:${PORT} \
    --rm"

# Add background flag if requested
if [ "$1" = "--background" ] || [ "$1" = "-d" ]; then
    DOCKER_CMD="$DOCKER_CMD -d"
    echo "Running in background..."
fi

# Run the container
$DOCKER_CMD \
    bosonai/higgs-audio-vllm:latest \
    --model "$MODEL_PATH" \
    --audio-tokenizer-type "$TOKENIZER_PATH" \
    --served-model-name "$MODEL_NAME" \
    --port $PORT \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --max-model-len $MAX_MODEL_LEN

if [ "$1" = "--background" ] || [ "$1" = "-d" ]; then
    echo ""
    echo "Container started in background."
    echo "Check status: docker ps"
    echo "View logs: docker logs -f \$(docker ps -q --filter ancestor=bosonai/higgs-audio-vllm:latest)"
    echo ""
    echo "API available at: http://localhost:$PORT"
fi
