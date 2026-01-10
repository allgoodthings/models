#!/bin/bash
# GPU monitoring script - logs to gpu_monitor.log every 2 seconds

LOG_FILE="$(dirname "$0")/gpu_monitor.log"

echo "Starting GPU monitoring - logging to $LOG_FILE"
echo "timestamp,gpu_name,memory_used_mb,memory_total_mb,gpu_util_pct,mem_util_pct,temp_c" > $LOG_FILE

while true; do
    nvidia-smi --query-gpu=timestamp,name,memory.used,memory.total,utilization.gpu,utilization.memory,temperature.gpu --format=csv,noheader,nounits >> $LOG_FILE
    sleep 2
done
