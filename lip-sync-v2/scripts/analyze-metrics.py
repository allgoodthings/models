#!/usr/bin/env python3
"""
Analyze hardware metrics from monitor-hardware.sh output.

Usage:
    python analyze-metrics.py metrics.log
    python analyze-metrics.py < metrics.log
"""

import sys
import csv
from datetime import datetime
from typing import List, Dict, Any


def parse_metrics(lines: List[str]) -> List[Dict[str, Any]]:
    """Parse CSV metrics into list of dicts."""
    reader = csv.DictReader(lines)
    metrics = []
    for row in reader:
        try:
            metrics.append({
                'timestamp': row['timestamp'],
                'gpu_util': float(row['gpu_util']) if row['gpu_util'] != 'N/A' else None,
                'gpu_mem_used_mb': float(row['gpu_mem_used_mb']) if row['gpu_mem_used_mb'] != 'N/A' else None,
                'gpu_mem_total_mb': float(row['gpu_mem_total_mb']) if row['gpu_mem_total_mb'] != 'N/A' else None,
                'cpu_util': float(row['cpu_util']) if row['cpu_util'] != 'N/A' else None,
                'ram_used_gb': float(row['ram_used_gb']) if row['ram_used_gb'] != 'N/A' else None,
                'ram_total_gb': float(row['ram_total_gb']) if row['ram_total_gb'] != 'N/A' else None,
                'disk_used_gb': float(row['disk_used_gb']) if row['disk_used_gb'] != 'N/A' else None,
            })
        except (ValueError, KeyError):
            continue
    return metrics


def analyze(metrics: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute summary statistics."""
    if not metrics:
        return {}

    def stats(values):
        values = [v for v in values if v is not None]
        if not values:
            return {'min': None, 'max': None, 'avg': None}
        return {
            'min': min(values),
            'max': max(values),
            'avg': sum(values) / len(values),
        }

    # Parse timestamps for duration
    try:
        start = datetime.fromisoformat(metrics[0]['timestamp'])
        end = datetime.fromisoformat(metrics[-1]['timestamp'])
        duration_sec = (end - start).total_seconds()
    except:
        duration_sec = len(metrics) * 2  # Assume 2 sec intervals

    gpu_mem_total = metrics[0].get('gpu_mem_total_mb')
    ram_total = metrics[0].get('ram_total_gb')

    return {
        'duration_sec': duration_sec,
        'samples': len(metrics),
        'gpu_util': stats([m['gpu_util'] for m in metrics]),
        'gpu_mem_used_mb': stats([m['gpu_mem_used_mb'] for m in metrics]),
        'gpu_mem_total_mb': gpu_mem_total,
        'cpu_util': stats([m['cpu_util'] for m in metrics]),
        'ram_used_gb': stats([m['ram_used_gb'] for m in metrics]),
        'ram_total_gb': ram_total,
        'disk_used_gb': stats([m['disk_used_gb'] for m in metrics]),
    }


def print_report(analysis: Dict[str, Any]):
    """Print formatted report."""
    print("\n" + "=" * 60)
    print("HARDWARE METRICS REPORT")
    print("=" * 60)

    print(f"\nDuration: {analysis.get('duration_sec', 0):.1f} seconds ({analysis.get('samples', 0)} samples)")

    print("\n--- GPU ---")
    gpu_util = analysis.get('gpu_util', {})
    gpu_mem = analysis.get('gpu_mem_used_mb', {})
    gpu_total = analysis.get('gpu_mem_total_mb')
    if gpu_util.get('avg') is not None:
        print(f"Utilization:  min={gpu_util['min']:.1f}%  max={gpu_util['max']:.1f}%  avg={gpu_util['avg']:.1f}%")
    if gpu_mem.get('avg') is not None and gpu_total:
        print(f"Memory Used:  min={gpu_mem['min']:.0f}MB  max={gpu_mem['max']:.0f}MB  avg={gpu_mem['avg']:.0f}MB  (total={gpu_total:.0f}MB)")
        peak_pct = (gpu_mem['max'] / gpu_total) * 100
        print(f"Peak Memory:  {peak_pct:.1f}% of {gpu_total/1024:.1f}GB")

    print("\n--- CPU ---")
    cpu_util = analysis.get('cpu_util', {})
    if cpu_util.get('avg') is not None:
        print(f"Utilization:  min={cpu_util['min']:.1f}%  max={cpu_util['max']:.1f}%  avg={cpu_util['avg']:.1f}%")

    print("\n--- RAM ---")
    ram = analysis.get('ram_used_gb', {})
    ram_total = analysis.get('ram_total_gb')
    if ram.get('avg') is not None and ram_total:
        print(f"Used:         min={ram['min']:.1f}GB  max={ram['max']:.1f}GB  avg={ram['avg']:.1f}GB  (total={ram_total:.0f}GB)")
        peak_pct = (ram['max'] / ram_total) * 100
        print(f"Peak Memory:  {peak_pct:.1f}% of {ram_total:.0f}GB")

    print("\n--- Disk ---")
    disk = analysis.get('disk_used_gb', {})
    if disk.get('avg') is not None:
        print(f"Used:         min={disk['min']:.1f}GB  max={disk['max']:.1f}GB  avg={disk['avg']:.1f}GB")

    print("\n" + "=" * 60)


def main():
    if len(sys.argv) > 1:
        with open(sys.argv[1], 'r') as f:
            lines = f.readlines()
    else:
        lines = sys.stdin.readlines()

    metrics = parse_metrics(lines)
    if not metrics:
        print("No valid metrics found")
        sys.exit(1)

    analysis = analyze(metrics)
    print_report(analysis)


if __name__ == '__main__':
    main()
