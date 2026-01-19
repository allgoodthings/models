"""
FLUX.2 klein Pipeline Wrapper

Encapsulates model loading, configuration, and inference.

Model: black-forest-labs/FLUX.2-klein-9B
- 9B parameter rectified flow transformer
- Step-distilled to 4 steps
- Uses TorchAO FP8 quantization to fit on 24GB VRAM (~18-20GB)
- Supports text-to-image and multi-reference editing
"""

import gc
import io
import logging
import os
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple

import torch
from PIL import Image

logger = logging.getLogger("imagegen.flux")

# Quantization mode type
QuantizationMode = Literal["none", "fp8", "int8"]


@dataclass
class FluxConfig:
    """Configuration for FLUX pipeline.

    Models:
    - FLUX.2-klein-4B: ~17GB peak, Apache 2.0, fast (~1s for 1024x1024)
    - FLUX.2-klein-9B: ~26GB peak (BF16), ~18GB (FP8), non-commercial

    Set MODEL_ID env var to override: "4B" or "9B" (default: 4B for 24GB GPUs)
    """

    model_id: str = field(default_factory=lambda: os.environ.get(
        "MODEL_ID", "black-forest-labs/FLUX.2-klein-4B"
    ).replace("4B", "black-forest-labs/FLUX.2-klein-4B").replace("9B", "black-forest-labs/FLUX.2-klein-9B"))
    torch_dtype: torch.dtype = field(default_factory=lambda: torch.bfloat16)
    quantization: QuantizationMode = "none"
    enable_cpu_offload: bool = False
    enable_compile: bool = False
    hf_token: Optional[str] = None

    def __post_init__(self):
        if self.hf_token is None:
            self.hf_token = os.environ.get("HUGGING_FACE_TOKEN") or os.environ.get("HF_TOKEN")
        # Normalize model_id shortcuts
        if self.model_id == "4B":
            self.model_id = "black-forest-labs/FLUX.2-klein-4B"
        elif self.model_id == "9B":
            self.model_id = "black-forest-labs/FLUX.2-klein-9B"


class FluxPipeline:
    """
    FLUX.2 klein image generation pipeline.

    Handles model loading, text-to-image generation, and image editing
    with reference images.
    """

    def __init__(self, config: Optional[FluxConfig] = None):
        self.config = config or FluxConfig()
        self._pipe = None
        self._is_loaded = False

    @property
    def is_loaded(self) -> bool:
        """Whether the model is loaded."""
        return self._is_loaded and self._pipe is not None

    def load(self) -> None:
        """Load the FLUX model."""
        if self._is_loaded:
            logger.info("Model already loaded, skipping")
            return

        logger.info(f"Loading FLUX model: {self.config.model_id}")
        logger.info(f"  torch_dtype: {self.config.torch_dtype}")
        logger.info(f"  quantization: {self.config.quantization}")
        logger.info(f"  cpu_offload: {self.config.enable_cpu_offload}")
        logger.info(f"  compile: {self.config.enable_compile}")

        # Import here to avoid slow startup when not using pipeline
        from diffusers import Flux2KleinPipeline

        if self.config.quantization != "none":
            # Use TorchAO quantization for memory efficiency
            from diffusers import PipelineQuantizationConfig, TorchAoConfig

            quant_type = "float8wo" if self.config.quantization == "fp8" else "int8wo"
            logger.info(f"Using TorchAO {quant_type} quantization")

            pipeline_quant_config = PipelineQuantizationConfig(
                quant_mapping={"transformer": TorchAoConfig(quant_type)}
            )

            self._pipe = Flux2KleinPipeline.from_pretrained(
                self.config.model_id,
                quantization_config=pipeline_quant_config,
                torch_dtype=self.config.torch_dtype,
                token=self.config.hf_token,
                device_map="cuda",
            )
        else:
            # Load without quantization (requires 48GB+ VRAM)
            self._pipe = Flux2KleinPipeline.from_pretrained(
                self.config.model_id,
                torch_dtype=self.config.torch_dtype,
                token=self.config.hf_token,
            )

            # Enable CPU offload as fallback for memory efficiency
            if self.config.enable_cpu_offload:
                logger.info("Enabling model CPU offload")
                self._pipe.enable_model_cpu_offload()
            else:
                # Move to GPU directly
                self._pipe = self._pipe.to("cuda")

        # Optional torch.compile for extra speed (slower first inference)
        if self.config.enable_compile:
            logger.info("Compiling transformer with torch.compile")
            self._pipe.transformer = torch.compile(
                self._pipe.transformer, mode="max-autotune", fullgraph=True
            )

        self._is_loaded = True
        logger.info("FLUX model loaded successfully")

    def unload(self) -> None:
        """Unload the model and free VRAM."""
        if self._pipe is not None:
            del self._pipe
            self._pipe = None

        self._is_loaded = False

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("FLUX model unloaded")

    def generate(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        num_steps: int = 4,
        guidance_scale: float = 0.0,
        seed: Optional[int] = None,
    ) -> Tuple[Image.Image, int]:
        """
        Generate an image from a text prompt.

        Args:
            prompt: Text description of the desired image
            width: Output width in pixels
            height: Output height in pixels
            num_steps: Number of inference steps
            guidance_scale: CFG scale (0.0 for distilled models)
            seed: Random seed for reproducibility

        Returns:
            Tuple of (PIL Image, seed used)
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Create generator with seed
        if seed is None:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()

        generator = torch.Generator("cpu").manual_seed(seed)

        logger.info(f"Generating image: {width}x{height}, steps={num_steps}, seed={seed}")

        # Run inference
        result = self._pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )

        image = result.images[0]
        logger.info(f"Generated image: {image.size}")

        return image, seed

    def edit(
        self,
        prompt: str,
        reference_images: List[Tuple[Image.Image, float]],
        width: int = 1024,
        height: int = 1024,
        num_steps: int = 4,
        guidance_scale: float = 1.0,
        seed: Optional[int] = None,
    ) -> Tuple[Image.Image, int]:
        """
        Edit/generate an image guided by reference images.

        FLUX.2-klein-9B supports multi-reference image editing natively.
        Reference images guide the generation while the prompt describes
        the desired output or transformation.

        Args:
            prompt: Text description of the desired edit/output
            reference_images: List of (image, weight) tuples (1-4 images)
            width: Output width in pixels
            height: Output height in pixels
            num_steps: Number of inference steps
            guidance_scale: CFG scale (1.0 recommended for editing)
            seed: Random seed for reproducibility

        Returns:
            Tuple of (PIL Image, seed used)
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Create generator with seed
        if seed is None:
            seed = torch.randint(0, 2**32 - 1, (1,)).item()

        generator = torch.Generator("cpu").manual_seed(seed)

        # Log reference images info
        logger.info(f"Edit request with {len(reference_images)} reference images")
        for i, (img, weight) in enumerate(reference_images):
            logger.info(f"  Reference {i+1}: {img.size}, weight={weight}")

        # Extract just the images for the pipeline
        # FLUX.2-klein accepts reference images via the `image` parameter
        images = [img for img, _ in reference_images]

        logger.info(f"Editing image: {width}x{height}, steps={num_steps}, seed={seed}")

        # Run inference with reference images
        # The pipeline accepts `image` parameter for conditioning
        result = self._pipe(
            prompt=prompt,
            image=images if len(images) > 1 else images[0],
            width=width,
            height=height,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )

        image = result.images[0]
        logger.info(f"Edited image: {image.size}")

        return image, seed


def image_to_bytes(image: Image.Image, format: str = "PNG") -> bytes:
    """Convert PIL Image to bytes."""
    buffer = io.BytesIO()
    image.save(buffer, format=format.upper())
    buffer.seek(0)
    return buffer.getvalue()


def get_content_type(format: str) -> str:
    """Get MIME content type for image format."""
    format_map = {
        "png": "image/png",
        "jpeg": "image/jpeg",
        "jpg": "image/jpeg",
        "webp": "image/webp",
    }
    return format_map.get(format.lower(), "image/png")
