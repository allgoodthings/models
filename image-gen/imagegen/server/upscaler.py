"""
Real-ESRGAN Upscaler

Load-on-demand image upscaling using Real-ESRGAN.
Designed to be loaded/unloaded per request to minimize VRAM usage.

Model: RealESRGAN_x4plus (~64MB, ~2GB VRAM with tiling)
- 4x upscaling with detail enhancement
- Tile-based processing for memory efficiency
- FP16 inference for speed
"""

import gc
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger("imagegen.upscaler")

ScaleFactor = Literal[2, 4]

# Model configurations
MODELS = {
    "x4plus": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "scale": 4,
        "num_block": 23,
    },
    "x4plus-anime": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        "scale": 4,
        "num_block": 6,
    },
}


@dataclass
class UpscalerConfig:
    """Configuration for Real-ESRGAN upscaler."""

    model_name: str = "x4plus"
    tile_size: int = 256  # Smaller = less VRAM, more passes
    tile_pad: int = 10
    half_precision: bool = True  # FP16 for speed
    cache_dir: Optional[Path] = None

    def __post_init__(self):
        if self.cache_dir is None:
            self.cache_dir = Path.home() / ".cache" / "realesrgan"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def model_path(self) -> Path:
        return self.cache_dir / f"{self.model_name}.pth"


class Upscaler:
    """
    Real-ESRGAN image upscaler.

    Designed for load-on-demand usage - call load() before upscale(),
    then unload() to free VRAM for other models.
    """

    def __init__(self, config: Optional[UpscalerConfig] = None):
        self.config = config or UpscalerConfig()
        self._upsampler = None

    @property
    def is_loaded(self) -> bool:
        return self._upsampler is not None

    def load(self) -> None:
        """Load the upscaler model to GPU."""
        if self.is_loaded:
            return

        self._ensure_model_downloaded()
        self._create_upsampler()
        logger.info("Upscaler loaded")

    def unload(self) -> None:
        """Unload model and free VRAM."""
        if self._upsampler is not None:
            del self._upsampler
            self._upsampler = None

        self._clear_memory()
        logger.info("Upscaler unloaded")

    def upscale(self, image: Image.Image, scale: ScaleFactor = 4) -> Image.Image:
        """
        Upscale an image.

        Args:
            image: Input PIL Image
            scale: Output scale factor (2 or 4)

        Returns:
            Upscaled PIL Image
        """
        if not self.is_loaded:
            raise RuntimeError("Upscaler not loaded. Call load() first.")

        img_array = self._pil_to_numpy(image)
        output, _ = self._upsampler.enhance(img_array, outscale=scale)
        return self._numpy_to_pil(output)

    def _ensure_model_downloaded(self) -> None:
        """Download model weights if not cached."""
        if self.config.model_path.exists():
            return

        model_info = MODELS.get(self.config.model_name)
        if not model_info:
            raise ValueError(f"Unknown model: {self.config.model_name}")

        logger.info(f"Downloading {self.config.model_name} model...")
        self._download_file(model_info["url"], self.config.model_path)

    def _download_file(self, url: str, dest: Path) -> None:
        """Download a file from URL."""
        import urllib.request
        urllib.request.urlretrieve(url, dest)
        logger.info(f"Downloaded to {dest}")

    def _create_upsampler(self) -> None:
        """Create the RealESRGAN upsampler instance."""
        from basicsr.archs.rrdbnet_arch import RRDBNet
        from realesrgan import RealESRGANer

        model_info = MODELS[self.config.model_name]

        model = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_block=model_info["num_block"],
            num_grow_ch=32,
            scale=model_info["scale"],
        )

        self._upsampler = RealESRGANer(
            scale=model_info["scale"],
            model_path=str(self.config.model_path),
            model=model,
            tile=self.config.tile_size,
            tile_pad=self.config.tile_pad,
            pre_pad=0,
            half=self.config.half_precision,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    def _pil_to_numpy(self, image: Image.Image) -> np.ndarray:
        """Convert PIL Image to numpy array (BGR for OpenCV)."""
        if image.mode != "RGB":
            image = image.convert("RGB")
        rgb = np.array(image)
        return rgb[:, :, ::-1]  # RGB to BGR

    def _numpy_to_pil(self, array: np.ndarray) -> Image.Image:
        """Convert numpy array (BGR) to PIL Image."""
        rgb = array[:, :, ::-1]  # BGR to RGB
        return Image.fromarray(rgb)

    def _clear_memory(self) -> None:
        """Clear GPU memory."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def upscale_image(
    image: Image.Image,
    scale: ScaleFactor = 4,
    config: Optional[UpscalerConfig] = None,
) -> Image.Image:
    """
    Convenience function for one-shot upscaling.

    Loads model, upscales, and unloads automatically.
    Use this when upscaling is infrequent and VRAM should be freed.

    Args:
        image: Input PIL Image
        scale: Output scale factor (2 or 4)
        config: Optional upscaler configuration

    Returns:
        Upscaled PIL Image
    """
    upscaler = Upscaler(config)
    try:
        upscaler.load()
        return upscaler.upscale(image, scale)
    finally:
        upscaler.unload()
