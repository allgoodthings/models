#!/bin/bash
# Hardware monitoring script for GPU workers
# Logs GPU, CPU, RAM, and disk metrics every 2 seconds
#
# Usage:
#   ./monitor-hardware.sh              # Output to stdout
#   ./monitor-hardware.sh > metrics.log &  # Run in background
#
# Output format (CSV):
#   timestamp,gpu_util,gpu_mem_used_mb,gpu_mem_total_mb,cpu_util,ram_used_gb,ram_total_gb,disk_used_gb

set -e

INTERVAL=${MONITOR_INTERVAL:-2}

echo "timestamp,gpu_util,gpu_mem_used_mb,gpu_mem_total_mb,cpu_util,ram_used_gb,ram_total_gb,disk_used_gb"

while true; do
    timestamp=$(date +%Y-%m-%dT%H:%M:%S)

    # GPU metrics (nvidia-smi)
    if command -v nvidia-smi &> /dev/null; then
        gpu_info=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
        gpu_util=$(echo "$gpu_info" | cut -d, -f1 | tr -d " ")
        gpu_mem_used=$(echo "$gpu_info" | cut -d, -f2 | tr -d " ")
        gpu_mem_total=$(echo "$gpu_info" | cut -d, -f3 | tr -d " ")
    else
        gpu_util="N/A"
        gpu_mem_used="N/A"
        gpu_mem_total="N/A"
    fi

    # CPU utilization (Linux: top, macOS: ps)
    if [[ "$(uname)" == "Darwin" ]]; then
        cpu_util=$(ps -A -o %cpu | awk '{s+=$1} END {print s}')
    else
        cpu_util=$(top -bn1 | grep "Cpu(s)" | awk '{print 100 - $8}')
    fi

    # RAM metrics
    if [[ "$(uname)" == "Darwin" ]]; then
        # macOS
        ram_info=$(vm_stat | grep -E "Pages (free|active|inactive|speculative|wired)")
        page_size=4096
        pages_free=$(echo "$ram_info" | grep "free" | awk '{print $3}' | tr -d '.')
        pages_active=$(echo "$ram_info" | grep "active" | awk '{print $3}' | tr -d '.')
        ram_total=$(sysctl -n hw.memsize)
        ram_total_gb=$((ram_total / 1024 / 1024 / 1024))
        ram_used_gb=$(((pages_active * page_size) / 1024 / 1024 / 1024))
    else
        # Linux
        ram_info=$(free -g | grep Mem)
        ram_total=$(echo "$ram_info" | awk '{print $2}')
        ram_used=$(echo "$ram_info" | awk '{print $3}')
        ram_total_gb=$ram_total
        ram_used_gb=$ram_used
    fi

    # Disk usage (root filesystem)
    disk_used=$(df -BG / 2>/dev/null | tail -1 | awk '{print $3}' | tr -d 'G' || echo "N/A")

    echo "$timestamp,$gpu_util,$gpu_mem_used,$gpu_mem_total,$cpu_util,$ram_used_gb,$ram_total_gb,$disk_used"

    sleep "$INTERVAL"
done
