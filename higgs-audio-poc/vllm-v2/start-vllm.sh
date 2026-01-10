#!/bin/bash
# Start Higgs Audio Server
# High-throughput inference with concurrent request support (Docker mode)
# or single-request mode (direct mode with HiggsAudioServeEngine)

set -e

# Configuration
PORT=${PORT:-8000}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.85}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
MODEL_NAME="higgs-audio-v2"
MODEL_PATH="bosonai/higgs-audio-v2-generation-3B-base"
TOKENIZER_PATH="bosonai/higgs-audio-v2-tokenizer"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Starting Higgs Audio Server ==="
echo "Port: $PORT"
echo ""

# Determine mode
if [ -f .vllm_mode ]; then
    MODE=$(cat .vllm_mode)
else
    # Auto-detect
    if command -v docker &> /dev/null && docker info &> /dev/null; then
        MODE="docker"
    else
        MODE="direct"
    fi
fi

echo "Mode: $MODE"
echo ""

# Start GPU monitoring in background
echo "Starting GPU monitoring..."
nohup ./monitor_gpu.sh > /dev/null 2>&1 &
MONITOR_PID=$!
echo "GPU monitor PID: $MONITOR_PID"
echo ""

if [ "$MODE" = "docker" ]; then
    # ============================================
    # Docker Mode - vLLM with concurrent inference
    # ============================================
    echo "Starting vLLM Docker container..."
    echo "GPU Memory Utilization: $GPU_MEMORY_UTILIZATION"
    echo "Max Model Length: $MAX_MODEL_LEN"
    echo ""

    docker run --gpus all \
        --ipc=host \
        --shm-size=20gb \
        --network=host \
        --rm \
        -v "$(pwd)/voice_presets:/voice_presets:ro" \
        bosonai/higgs-audio-vllm:latest \
        --served-model-name "$MODEL_NAME" \
        --model "$MODEL_PATH" \
        --audio-tokenizer-type "$TOKENIZER_PATH" \
        --limit-mm-per-prompt audio=50 \
        --max-model-len $MAX_MODEL_LEN \
        --port $PORT \
        --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
        --disable-mm-preprocessor-cache

else
    # ============================================
    # Direct Mode - HiggsAudioServeEngine
    # Single request at a time (max_batch_size=1)
    # ============================================
    echo "Starting HiggsAudioServeEngine server..."
    echo ""
    echo "NOTE: Direct mode processes one request at a time."
    echo "      For concurrent inference, use Docker mode."
    echo ""
    echo "Loading model (first run downloads ~6GB)..."
    echo ""

    # Start the direct server
    python3 server_direct.py
fi
