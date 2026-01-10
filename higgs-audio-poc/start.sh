#!/bin/bash
# Start Higgs Audio v2 POC Server with GPU monitoring
# Run this after setup.sh

set -e

cd /workspace/higgs-audio-poc

# Start GPU monitoring in background
echo "Starting GPU monitoring..."
nohup ./monitor_gpu.sh > /dev/null 2>&1 &
echo "GPU monitoring started (logging to gpu_monitor.log)"

# Start the server
echo "Starting Higgs Audio v2 server on port 8888..."
cd higgs-audio
source .venv/bin/activate
python /workspace/higgs-audio-poc/server.py
