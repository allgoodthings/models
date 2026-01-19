"""
Real-ESRGAN Video Upscaler.

Supports single-pass and two-pass (high quality) upscaling modes.
"""

import gc
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import cv2
import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger("upscaler.realesrgan")

ModelName = Literal["realesr-animevideov3", "realesrgan-x4plus"]

# Model configurations
MODELS = {
    "realesr-animevideov3": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
        "scale": 4,
        "num_block": 6,
        "num_feat": 64,
        "num_conv": 16,  # VGG-style for anime model
        "arch": "animevideov3",
    },
    "realesrgan-x4plus": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "scale": 4,
        "num_block": 23,
        "num_feat": 64,
        "num_grow_ch": 32,
        "arch": "rrdbnet",
    },
}


@dataclass
class UpscalerConfig:
    """Configuration for Real-ESRGAN upscaler."""

    model_name: ModelName = "realesr-animevideov3"
    tile_size: int = 256  # Smaller = less VRAM, more passes
    tile_pad: int = 10
    half_precision: bool = True  # FP16 for speed
    cache_dir: Optional[Path] = None

    def __post_init__(self):
        if self.cache_dir is None:
            self.cache_dir = Path(os.environ.get("MODELS_DIR", "/app/models"))
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @property
    def model_path(self) -> Path:
        return self.cache_dir / f"{self.model_name}.pth"


class VideoUpscaler:
    """
    Real-ESRGAN video upscaler.

    Supports:
    - Single-pass upscaling (fast, good quality)
    - Two-pass upscaling (slower, better quality for large scale factors)
    """

    def __init__(self, config: Optional[UpscalerConfig] = None):
        self.config = config or UpscalerConfig()
        self._upsampler = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def is_loaded(self) -> bool:
        return self._upsampler is not None

    def load(self) -> None:
        """Load the upscaler model to GPU."""
        if self.is_loaded:
            return

        self._ensure_model_downloaded()
        self._create_upsampler()
        logger.info(f"Upscaler loaded: {self.config.model_name} on {self._device}")

    def unload(self) -> None:
        """Unload model and free VRAM."""
        if self._upsampler is not None:
            del self._upsampler
            self._upsampler = None

        self._clear_memory()
        logger.info("Upscaler unloaded")

    def upscale_frame(self, frame: np.ndarray, outscale: float = 4.0) -> np.ndarray:
        """
        Upscale a single frame.

        Args:
            frame: Input frame (BGR numpy array)
            outscale: Output scale factor

        Returns:
            Upscaled frame (BGR numpy array)
        """
        if not self.is_loaded:
            raise RuntimeError("Upscaler not loaded. Call load() first.")

        output, _ = self._upsampler.enhance(frame, outscale=outscale)
        return output

    def upscale_directory(
        self,
        input_dir: str,
        output_dir: str,
        outscale: float = 4.0,
        high_quality: bool = False,
    ) -> int:
        """
        Upscale all frames in a directory.

        Args:
            input_dir: Directory containing input frames (frame_00001.png, ...)
            output_dir: Directory to save upscaled frames
            outscale: Total output scale factor
            high_quality: Use two-pass upscaling for better quality

        Returns:
            Number of frames processed
        """
        if not self.is_loaded:
            raise RuntimeError("Upscaler not loaded. Call load() first.")

        os.makedirs(output_dir, exist_ok=True)

        frame_paths = sorted(Path(input_dir).glob("frame_*.png"))
        total_frames = len(frame_paths)

        if total_frames == 0:
            raise ValueError(f"No frames found in {input_dir}")

        logger.info(f"Upscaling {total_frames} frames (outscale={outscale}, high_quality={high_quality})")

        if high_quality and outscale > 4.0:
            # Two-pass: first 4x, then remaining scale
            return self._upscale_two_pass(frame_paths, output_dir, outscale)
        else:
            # Single pass
            return self._upscale_single_pass(frame_paths, output_dir, outscale)

    def _upscale_single_pass(
        self,
        frame_paths: list,
        output_dir: str,
        outscale: float,
    ) -> int:
        """Single-pass upscaling."""
        processed = 0

        for frame_path in tqdm(frame_paths, desc="Upscaling", unit="frame"):
            frame = cv2.imread(str(frame_path))
            if frame is None:
                logger.warning(f"Failed to read {frame_path}")
                continue

            upscaled = self.upscale_frame(frame, outscale=outscale)

            output_path = Path(output_dir) / frame_path.name
            cv2.imwrite(str(output_path), upscaled)
            processed += 1

        return processed

    def _upscale_two_pass(
        self,
        frame_paths: list,
        output_dir: str,
        outscale: float,
    ) -> int:
        """
        Two-pass upscaling for better quality.

        Pass 1: 4x upscale
        Pass 2: Remaining scale (e.g., 1.125x for 4.5x total)
        """
        first_pass_scale = 4.0
        second_pass_scale = outscale / first_pass_scale

        logger.info(f"Two-pass mode: {first_pass_scale}x -> {second_pass_scale:.3f}x = {outscale}x total")

        processed = 0

        for frame_path in tqdm(frame_paths, desc="Upscaling (2-pass)", unit="frame"):
            frame = cv2.imread(str(frame_path))
            if frame is None:
                logger.warning(f"Failed to read {frame_path}")
                continue

            # Pass 1: 4x upscale
            intermediate = self.upscale_frame(frame, outscale=first_pass_scale)

            # Pass 2: Final scale
            if second_pass_scale > 1.01:  # Only if meaningful scale needed
                final = self.upscale_frame(intermediate, outscale=second_pass_scale)
            else:
                final = intermediate

            output_path = Path(output_dir) / frame_path.name
            cv2.imwrite(str(output_path), final)
            processed += 1

        return processed

    def _ensure_model_downloaded(self) -> None:
        """Download model weights if not cached."""
        if self.config.model_path.exists():
            size_mb = self.config.model_path.stat().st_size / (1024 * 1024)
            logger.info(f"Model exists: {self.config.model_name} ({size_mb:.1f}MB)")
            return

        model_info = MODELS.get(self.config.model_name)
        if not model_info:
            raise ValueError(f"Unknown model: {self.config.model_name}")

        logger.info(f"Downloading {self.config.model_name}...")
        self._download_file(model_info["url"], self.config.model_path)

    def _download_file(self, url: str, dest: Path) -> None:
        """Download a file from URL."""
        import urllib.request

        urllib.request.urlretrieve(url, dest)
        size_mb = dest.stat().st_size / (1024 * 1024)
        logger.info(f"Downloaded {self.config.model_name} ({size_mb:.1f}MB)")

    def _create_upsampler(self) -> None:
        """Create the RealESRGAN upsampler instance."""
        from realesrgan import RealESRGANer

        model_info = MODELS[self.config.model_name]

        if model_info["arch"] == "animevideov3":
            from basicsr.archs.rrdbnet_arch import RRDBNet

            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=model_info["num_feat"],
                num_block=model_info["num_block"],
                num_grow_ch=32,
                scale=model_info["scale"],
            )
        else:
            # Standard RRDBNet for x4plus
            from basicsr.archs.rrdbnet_arch import RRDBNet

            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=model_info["num_feat"],
                num_block=model_info["num_block"],
                num_grow_ch=model_info["num_grow_ch"],
                scale=model_info["scale"],
            )

        self._upsampler = RealESRGANer(
            scale=model_info["scale"],
            model_path=str(self.config.model_path),
            model=model,
            tile=self.config.tile_size,
            tile_pad=self.config.tile_pad,
            pre_pad=0,
            half=self.config.half_precision and self._device == "cuda",
            device=self._device,
        )

    def _clear_memory(self) -> None:
        """Clear GPU memory."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def resize_frames_to_target(
    input_dir: str,
    output_dir: str,
    target_width: int,
    target_height: int,
) -> int:
    """
    Resize frames to exact target resolution using high-quality interpolation.

    Used as final step after upscaling to ensure exact output dimensions.

    Args:
        input_dir: Directory containing upscaled frames
        output_dir: Directory to save resized frames
        target_width: Target width in pixels
        target_height: Target height in pixels

    Returns:
        Number of frames processed
    """
    os.makedirs(output_dir, exist_ok=True)

    frame_paths = sorted(Path(input_dir).glob("frame_*.png"))
    processed = 0

    for frame_path in tqdm(frame_paths, desc="Resizing", unit="frame"):
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue

        # Use LANCZOS for high-quality downscaling, CUBIC for upscaling
        h, w = frame.shape[:2]
        if target_width < w or target_height < h:
            interpolation = cv2.INTER_LANCZOS4
        else:
            interpolation = cv2.INTER_CUBIC

        resized = cv2.resize(frame, (target_width, target_height), interpolation=interpolation)

        output_path = Path(output_dir) / frame_path.name
        cv2.imwrite(str(output_path), resized)
        processed += 1

    return processed
