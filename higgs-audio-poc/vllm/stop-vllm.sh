#!/bin/bash
# Stop Higgs Audio Server
#
# Handles both Docker and direct modes

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Determine mode
if [ -f .vllm_mode ]; then
    MODE=$(cat .vllm_mode)
else
    MODE="unknown"
fi

echo "Stopping Higgs Audio server..."
echo "Mode: $MODE"
echo ""

if [ "$MODE" = "docker" ]; then
    echo "Stopping vLLM container..."
    docker stop $(docker ps -q --filter ancestor=bosonai/higgs-audio-vllm:latest) 2>/dev/null || echo "No running container found"
else
    echo "Stopping direct server..."
    pkill -f "server_direct.py" 2>/dev/null || echo "No direct server running"
fi

echo "Stopping proxy server..."
pkill -f "python.*server.py" 2>/dev/null || echo "No proxy server running"

echo "Stopping GPU monitor..."
pkill -f "monitor_gpu.sh" 2>/dev/null || echo "No monitor running"

echo ""
echo "Done."
