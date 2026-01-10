#!/bin/bash
# GPU monitoring script - logs to gpu_monitor.log every 5 seconds

LOG_FILE="/workspace/higgs-audio-poc/gpu_monitor.log"

echo "Starting GPU monitoring - logging to $LOG_FILE"
echo "Press Ctrl+C to stop"

while true; do
    echo "=== $(date '+%Y-%m-%d %H:%M:%S') ===" >> $LOG_FILE
    nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu,utilization.memory,temperature.gpu --format=csv,noheader >> $LOG_FILE
    sleep 5
done
