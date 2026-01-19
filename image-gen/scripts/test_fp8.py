#!/usr/bin/env python3
"""Quick FP8 benchmark test."""
import os
import time
import torch

HF_TOKEN = os.environ.get("HUGGING_FACE_TOKEN") or os.environ.get("HF_TOKEN")

print("=" * 60)
print("FLUX.2-klein-9B FP8 Benchmark")
print("=" * 60)
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

print("\nLoading model with FP8 quantization...")
from diffusers import Flux2KleinPipeline, PipelineQuantizationConfig, TorchAoConfig

pipeline_quant_config = PipelineQuantizationConfig(
    quant_mapping={"transformer": TorchAoConfig("float8wo")}
)

start = time.time()
pipe = Flux2KleinPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-9B",
    quantization_config=pipeline_quant_config,
    torch_dtype=torch.bfloat16,
    token=HF_TOKEN,
    device_map="cuda",
)
load_time = time.time() - start
print(f"Model loaded in {load_time:.1f}s")
print(f"VRAM used: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

# Test generations
resolutions = [(512, 512), (768, 768), (1024, 1024)]

for w, h in resolutions:
    torch.cuda.reset_peak_memory_stats()
    start = time.time()

    image = pipe(
        prompt="A beautiful sunset over mountains",
        width=w,
        height=h,
        num_inference_steps=4,
        guidance_scale=1.0,
        generator=torch.Generator("cpu").manual_seed(42),
    ).images[0]

    gen_time = (time.time() - start) * 1000
    peak_vram = torch.cuda.max_memory_allocated() / 1024**3

    print(f"\n{w}x{h}: {gen_time:.0f}ms, peak VRAM: {peak_vram:.2f} GB")
    image.save(f"/workspace/test_{w}x{h}.png")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
