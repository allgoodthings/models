#!/usr/bin/env python3
"""Benchmark FLUX.2-klein-4B BF16 on RTX 5090."""
import os
import time
import torch

HF_TOKEN = os.environ.get("HUGGING_FACE_TOKEN") or os.environ.get("HF_TOKEN")

print("=" * 60)
print("FLUX.2-klein-4B BF16 Benchmark")
print("=" * 60)
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

print("\nLoading model (BF16, no quantization)...")
from diffusers import Flux2KleinPipeline

start = time.time()
pipe = Flux2KleinPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-4B",
    torch_dtype=torch.bfloat16,
    token=HF_TOKEN,
)
pipe = pipe.to("cuda")
load_time = time.time() - start
print(f"Model loaded in {load_time:.1f}s")
print(f"VRAM used: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

# Shared prompts for all tests
PROMPTS = [
    ("portrait", "A photorealistic portrait of an elderly fisherman with weathered skin and kind eyes, golden hour lighting"),
    ("cyberpunk", "A futuristic cyberpunk city street at night with neon signs and rain reflections"),
    ("interior", "A cozy coffee shop interior with warm lighting, wooden furniture, and plants"),
    ("landscape", "A majestic snow-capped mountain reflected in a crystal clear alpine lake at sunrise"),
    ("steampunk", "A detailed steampunk mechanical owl with brass gears and glowing eyes"),
    ("japanese", "A serene Japanese garden with cherry blossoms, a red bridge, and koi pond"),
    ("seascape", "A dramatic stormy seascape with crashing waves against rocky cliffs"),
    ("fantasy", "A whimsical fairy tale cottage in an enchanted forest with mushrooms and fireflies"),
]

# Entity extraction test prompts (product photography for extraction)
ENTITY_EXTRACTION_PROMPTS = [
    ("product_headphones", "Professional product photography of premium wireless headphones with leather ear cups and metal accents, isolated on pure white background, studio lighting, 8k detail"),
    ("product_watch", "Luxury wristwatch product shot showing intricate dial details and metal bracelet, isolated on white background, high-end commercial photography"),
    ("product_sneakers", "Nike Air Jordan sneakers product photography, showing side profile with visible air unit and laces, isolated on white background, crisp studio lighting"),
    ("product_laptop", "MacBook Pro laptop product shot showing sleek aluminum body and screen, isolated on white background, professional commercial photography"),
    ("product_perfume", "Elegant perfume bottle with amber liquid and gold cap, product photography isolated on white background, luxury aesthetic"),
    ("product_camera", "Professional DSLR camera with attached lens, product photography showing body details, isolated on white background, studio lighting"),
    ("product_sunglasses", "Designer aviator sunglasses with gold metal frame, product photography isolated on white background, showing reflection and detail"),
    ("product_handbag", "Luxury leather handbag with gold hardware and stitching details, product photography isolated on white background, high fashion aesthetic"),
]

os.makedirs("/workspace/4b_images", exist_ok=True)
os.makedirs("/workspace/4b_entity_extraction", exist_ok=True)
os.makedirs("/workspace/4b_multiref", exist_ok=True)

# Test generations at various resolutions
print("\n" + "=" * 60)
print("TEXT-TO-IMAGE GENERATION (Resolution Test)")
print("=" * 60)

resolutions = [(512, 512), (768, 768), (1024, 768), (768, 1024), (1024, 1024)]

for w, h in resolutions:
    torch.cuda.reset_peak_memory_stats()
    start = time.time()

    image = pipe(
        prompt="A beautiful sunset over mountains, professional photography",
        width=w,
        height=h,
        num_inference_steps=4,
        guidance_scale=1.0,
        generator=torch.Generator("cuda").manual_seed(42),
    ).images[0]

    gen_time = (time.time() - start) * 1000
    peak_vram = torch.cuda.max_memory_allocated() / 1024**3

    print(f"{w}x{h}: {gen_time:.0f}ms, peak VRAM: {peak_vram:.2f} GB")
    image.save(f"/workspace/4b_test_{w}x{h}.png")

# Generate diverse test images (1024x1024)
print("\n" + "=" * 60)
print("DIVERSE IMAGE GENERATION (1024x1024)")
print("=" * 60)

with open("/workspace/4b_images/prompts.txt", "w") as f:
    for i, (name, prompt) in enumerate(PROMPTS):
        torch.cuda.reset_peak_memory_stats()
        start = time.time()

        image = pipe(
            prompt=prompt,
            width=1024,
            height=1024,
            num_inference_steps=4,
            guidance_scale=1.0,
            generator=torch.Generator("cuda").manual_seed(42 + i),
        ).images[0]

        gen_time = (time.time() - start) * 1000
        peak_vram = torch.cuda.max_memory_allocated() / 1024**3

        filename = f"{name}.png"
        image.save(f"/workspace/4b_images/{filename}")
        f.write(f"{filename}: {prompt}\n")
        print(f"{name}: {gen_time:.0f}ms, peak: {peak_vram:.2f} GB")

# Landscape and portrait versions
print("\n" + "=" * 60)
print("LANDSCAPE & PORTRAIT (1024 max dimension)")
print("=" * 60)

for i, (name, prompt) in enumerate(PROMPTS[:4]):
    # Landscape version
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    image = pipe(
        prompt=prompt,
        width=1024,
        height=768,
        num_inference_steps=4,
        guidance_scale=1.0,
        generator=torch.Generator("cuda").manual_seed(100 + i),
    ).images[0]
    gen_time = (time.time() - start) * 1000
    peak_vram = torch.cuda.max_memory_allocated() / 1024**3
    image.save(f"/workspace/4b_images/{name}_landscape.png")
    print(f"{name}_landscape (1024x768): {gen_time:.0f}ms, peak: {peak_vram:.2f} GB")

    # Portrait version
    torch.cuda.reset_peak_memory_stats()
    start = time.time()
    image = pipe(
        prompt=prompt,
        width=768,
        height=1024,
        num_inference_steps=4,
        guidance_scale=1.0,
        generator=torch.Generator("cuda").manual_seed(200 + i),
    ).images[0]
    gen_time = (time.time() - start) * 1000
    peak_vram = torch.cuda.max_memory_allocated() / 1024**3
    image.save(f"/workspace/4b_images/{name}_portrait.png")
    print(f"{name}_portrait (768x1024): {gen_time:.0f}ms, peak: {peak_vram:.2f} GB")

# Entity extraction test
print("\n" + "=" * 60)
print("ENTITY EXTRACTION TEST (Product Photography)")
print("=" * 60)
print("Generating detailed product images for entity extraction...")

with open("/workspace/4b_entity_extraction/prompts.txt", "w") as f:
    for i, (name, prompt) in enumerate(ENTITY_EXTRACTION_PROMPTS):
        torch.cuda.reset_peak_memory_stats()
        start = time.time()

        image = pipe(
            prompt=prompt,
            width=1024,
            height=1024,
            num_inference_steps=4,
            guidance_scale=1.0,
            generator=torch.Generator("cuda").manual_seed(500 + i),
        ).images[0]

        gen_time = (time.time() - start) * 1000
        peak_vram = torch.cuda.max_memory_allocated() / 1024**3

        filename = f"{name}.png"
        image.save(f"/workspace/4b_entity_extraction/{filename}")
        f.write(f"{filename}: {prompt}\n")
        print(f"{name}: {gen_time:.0f}ms, peak: {peak_vram:.2f} GB")

# Multi-reference test
print("\n" + "=" * 60)
print("MULTI-REFERENCE IMAGE EDITING")
print("=" * 60)
print("Generating reference images...")

# Generate reference images (reusing some prompts)
ref_prompts = [
    ("ref_dog", "A golden retriever dog sitting in a park, photorealistic, isolated subject"),
    ("ref_cat", "A fluffy orange tabby cat, photorealistic portrait, isolated subject"),
    ("ref_tree", "A large oak tree with autumn leaves, dramatic lighting, isolated subject"),
    ("ref_car", "A red vintage sports car, studio photography, isolated subject"),
    ("ref_castle", "A medieval castle on a hilltop at sunset, isolated subject"),
    ("ref_woman", "A portrait of a young woman with curly hair, professional headshot"),
]

ref_images = []
for name, prompt in ref_prompts:
    image = pipe(
        prompt=prompt,
        width=768,
        height=768,
        num_inference_steps=4,
        guidance_scale=1.0,
        generator=torch.Generator("cuda").manual_seed(999),
    ).images[0]
    image.save(f"/workspace/4b_multiref/{name}.png")
    ref_images.append(image)
    print(f"  Generated {name}")

# Test VRAM scaling with references using image parameter (list of images)
print("\nTesting VRAM scaling with multiple references...")

for output_size in [(1024, 768), (768, 768)]:
    w, h = output_size
    print(f"\nOutput size: {w}x{h}")

    for num_refs in [2, 4, 6]:
        refs_to_use = ref_images[:num_refs]

        torch.cuda.reset_peak_memory_stats()
        start = time.time()

        try:
            # Use image parameter with list of images
            result = pipe(
                prompt=f"A magical fantasy scene combining elements from image 1 through image {num_refs} in a dreamlike landscape, epic composition",
                image=refs_to_use,  # List of reference images
                width=w,
                height=h,
                num_inference_steps=4,
                guidance_scale=2.5,  # Higher guidance for multi-ref
                generator=torch.Generator("cuda").manual_seed(777),
            ).images[0]

            gen_time = (time.time() - start) * 1000
            peak_vram = torch.cuda.max_memory_allocated() / 1024**3

            result.save(f"/workspace/4b_multiref/composed_{w}x{h}_{num_refs}refs.png")
            print(f"  {num_refs} refs: {peak_vram:.2f} GB, {gen_time:.0f}ms")
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  {num_refs} refs: OOM")
                torch.cuda.empty_cache()
            else:
                print(f"  {num_refs} refs: ERROR - {e}")
                torch.cuda.empty_cache()
        except TypeError as e:
            print(f"  {num_refs} refs: API ERROR - {e}")
            break

print("\n" + "=" * 60)
print("BENCHMARK COMPLETE")
print("=" * 60)
print(f"Final VRAM: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print("\nImages saved to:")
print("  /workspace/4b_images/ - diverse text-to-image")
print("  /workspace/4b_entity_extraction/ - product photography for extraction")
print("  /workspace/4b_multiref/ - multi-reference compositing")
