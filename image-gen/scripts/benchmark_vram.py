#!/usr/bin/env python3
"""
VRAM Benchmark Script for FLUX.2-klein-9B on Vast AI

Run this BEFORE building the full service to validate GPU requirements.

Model: black-forest-labs/FLUX.2-klein-9B
- 9B parameters, BF16 precision
- Step-distilled to 4 inference steps
- BF16 requires ~29GB VRAM (fits on 32GB+ GPUs)

Test matrix:
1. Model loading
2. Text-to-image at 512, 768, 1024, 1280 resolution
3. Peak VRAM logging for each test

Usage:
    python scripts/benchmark_vram.py

Environment:
    HUGGING_FACE_TOKEN - HuggingFace token for gated model access
"""

import gc
import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

import torch

# Check for HF token early
HF_TOKEN = os.environ.get("HUGGING_FACE_TOKEN") or os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    print("ERROR: HUGGING_FACE_TOKEN or HF_TOKEN environment variable required")
    print("Get your token from: https://huggingface.co/settings/tokens")
    sys.exit(1)


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""

    test_name: str
    success: bool
    error: Optional[str] = None
    vram_before_gb: float = 0.0
    vram_after_gb: float = 0.0
    vram_peak_gb: float = 0.0
    duration_ms: int = 0


def get_gpu_info():
    """Get GPU information."""
    if not torch.cuda.is_available():
        return None, None, None

    gpu_name = torch.cuda.get_device_name(0)
    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    return gpu_name, total_memory, "cuda"


def get_vram_used():
    """Get current VRAM usage in GB."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated(0) / (1024**3)


def get_vram_reserved():
    """Get reserved VRAM in GB (includes fragmentation)."""
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_reserved(0) / (1024**3)


def clear_vram():
    """Clear VRAM and run garbage collection."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def run_benchmark(test_name: str, test_func) -> BenchmarkResult:
    """Run a benchmark test and capture VRAM stats."""
    clear_vram()
    vram_before = get_vram_used()

    result = BenchmarkResult(
        test_name=test_name,
        success=False,
        vram_before_gb=vram_before,
    )

    start_time = time.time()

    try:
        test_func()
        result.success = True
    except Exception as e:
        result.error = str(e)

    result.duration_ms = int((time.time() - start_time) * 1000)
    result.vram_after_gb = get_vram_used()

    if torch.cuda.is_available():
        result.vram_peak_gb = torch.cuda.max_memory_allocated(0) / (1024**3)

    return result


def print_result(result: BenchmarkResult):
    """Print benchmark result in a formatted way."""
    status = "PASS" if result.success else "FAIL"
    print(f"\n{'='*60}")
    print(f"Test: {result.test_name}")
    print(f"{'='*60}")
    print(f"Status: {status}")

    if result.error:
        print(f"Error: {result.error}")

    print(f"VRAM Before: {result.vram_before_gb:.2f} GB")
    print(f"VRAM After:  {result.vram_after_gb:.2f} GB")
    print(f"VRAM Peak:   {result.vram_peak_gb:.2f} GB")
    print(f"Duration:    {result.duration_ms} ms ({result.duration_ms/1000:.2f}s)")


def main():
    """Run all benchmarks."""
    print("=" * 60)
    print("FLUX.2-klein-9B VRAM Benchmark")
    print("=" * 60)

    # GPU info
    gpu_name, total_memory, device = get_gpu_info()

    if device is None:
        print("ERROR: No GPU available!")
        sys.exit(1)

    print(f"GPU: {gpu_name}")
    print(f"Total VRAM: {total_memory:.1f} GB")
    print(f"Device: {device}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")

    # Check VRAM sufficiency
    if total_memory < 29:
        print(f"\nWARNING: {total_memory:.1f}GB VRAM may be insufficient for BF16!")
        print("BF16 requires ~29GB. Consider FP8 quantization for this GPU.")
        use_quantization = True
    else:
        print(f"\n{total_memory:.1f}GB VRAM is sufficient for BF16 (no quantization)")
        use_quantization = False

    results = []

    # =========================================================================
    # Test 1: Model Loading
    # =========================================================================
    print("\n" + "-" * 60)
    if use_quantization:
        print("Loading FLUX.2-klein-9B model (FP8 quantization)...")
    else:
        print("Loading FLUX.2-klein-9B model (BF16, no quantization)...")
    print("-" * 60)

    pipe = None

    def test_load():
        nonlocal pipe
        from diffusers import Flux2KleinPipeline

        if use_quantization:
            from diffusers import PipelineQuantizationConfig, TorchAoConfig

            pipeline_quant_config = PipelineQuantizationConfig(
                quant_mapping={"transformer": TorchAoConfig("float8wo")}
            )
            pipe = Flux2KleinPipeline.from_pretrained(
                "black-forest-labs/FLUX.2-klein-9B",
                quantization_config=pipeline_quant_config,
                torch_dtype=torch.bfloat16,
                token=HF_TOKEN,
                device_map="cuda",
            )
        else:
            pipe = Flux2KleinPipeline.from_pretrained(
                "black-forest-labs/FLUX.2-klein-9B",
                torch_dtype=torch.bfloat16,
                token=HF_TOKEN,
            )
            pipe = pipe.to("cuda")

    mode_name = "FP8" if use_quantization else "BF16"
    result = run_benchmark(f"Model Load ({mode_name})", test_load)
    print_result(result)
    results.append(result)

    if not result.success:
        print("\nFATAL: Could not load model. Cannot continue benchmarks.")
        sys.exit(1)

    # =========================================================================
    # Test 2: Text-to-Image at Various Resolutions
    # =========================================================================
    resolutions = [(512, 512), (768, 768), (1024, 1024), (1280, 1280)]

    for width, height in resolutions:
        def test_generate(w=width, h=height):
            _ = pipe(
                prompt="A beautiful sunset over mountains",
                width=w,
                height=h,
                num_inference_steps=4,
                guidance_scale=1.0,
                generator=torch.Generator("cpu").manual_seed(42),
            ).images[0]

        result = run_benchmark(f"Generate {width}x{height}", test_generate)
        print_result(result)
        results.append(result)

        # Clear VRAM between tests
        clear_vram()

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)

    max_vram = max(r.vram_peak_gb for r in results)
    all_passed = all(r.success for r in results)

    print(f"\nTotal Tests: {len(results)}")
    print(f"Passed: {sum(1 for r in results if r.success)}")
    print(f"Failed: {sum(1 for r in results if not r.success)}")
    print(f"\nMax VRAM Peak: {max_vram:.2f} GB")
    print(f"Available VRAM: {total_memory:.1f} GB")
    print(f"Headroom: {total_memory - max_vram:.2f} GB")

    # Timing summary
    gen_results = [r for r in results if "Generate" in r.test_name and r.success]
    if gen_results:
        print("\nInference Timing:")
        for r in gen_results:
            print(f"  {r.test_name}: {r.duration_ms}ms ({r.duration_ms/1000:.2f}s)")

    if max_vram > total_memory:
        print("\nWARNING: Peak VRAM exceeds available memory!")
        print("Consider reducing resolution or using quantization.")
    elif max_vram > total_memory * 0.9:
        print("\nWARNING: Peak VRAM is close to limit (>90%)!")
        print("May encounter OOM errors under heavy load.")
    else:
        print("\nVRAM usage is within safe limits.")

    # Recommendations
    print("\n" + "-" * 60)
    print("RECOMMENDATIONS")
    print("-" * 60)

    if total_memory >= 32:
        print(f"- {total_memory:.0f}GB VRAM: BF16 without quantization (best quality)")
        print("- Can handle 1024x1024 and 1280x1280 comfortably")
    elif total_memory >= 24:
        print("- RTX 4090 (24GB): Use FP8 quantization")
        print("- 1024x1024 should work, 1280x1280 may OOM")
    else:
        print(f"- {total_memory:.0f}GB VRAM: Use FP8 or INT8 quantization")
        print("- May need to limit to 768x768")

    # Cleanup
    if pipe is not None:
        del pipe
    clear_vram()

    print("\nBenchmark complete!")

    if not all_passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
