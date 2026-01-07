"""
CodeFormer model wrapper for face restoration/enhancement.

CodeFormer enhances face quality after lip-sync processing,
reducing artifacts and improving visual fidelity.
"""

# =============================================================================
# TORCHVISION COMPATIBILITY FIX
# =============================================================================
# basicsr 1.4.2 imports from torchvision.transforms.functional_tensor, which was
# removed in torchvision 0.17+. This fix creates a dummy module so basicsr works.
# See: https://github.com/XPixelGroup/BasicSR/pull/677
# See: https://github.com/xinntao/Real-ESRGAN/issues/859
import sys
import types

try:
    from torchvision.transforms.functional_tensor import rgb_to_grayscale
except ImportError:
    from torchvision.transforms.functional import rgb_to_grayscale
    # Create a fake module for backward compatibility
    _functional_tensor = types.ModuleType("torchvision.transforms.functional_tensor")
    _functional_tensor.rgb_to_grayscale = rgb_to_grayscale
    sys.modules["torchvision.transforms.functional_tensor"] = _functional_tensor
# =============================================================================

import logging
import os
import time
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
import cv2

logger = logging.getLogger('lipsync.codeformer')


@dataclass
class CodeFormerConfig:
    """Configuration for CodeFormer model."""
    model_path: str = "models/codeformer"
    device: str = "cuda"
    fp16: bool = True
    fidelity_weight: float = 0.7
    upscale: int = 1


class CodeFormer:
    """
    CodeFormer face restoration model wrapper.

    Enhances face quality in lip-synced video by:
    - Reducing compression artifacts
    - Improving skin texture
    - Sharpening details
    """

    def __init__(self, config: Optional[CodeFormerConfig] = None):
        """
        Initialize CodeFormer wrapper.

        Args:
            config: Model configuration
        """
        self.config = config or CodeFormerConfig()
        self.net = None
        self.device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")
        self._loaded = False
        logger.info(f"CodeFormer initialized with device={self.device}")

    def load(self) -> None:
        """Load model weights to GPU."""
        if self._loaded:
            logger.debug("CodeFormer already loaded, skipping")
            return

        logger.info("=" * 50)
        logger.info("CODEFORMER - Loading model")
        logger.info("=" * 50)
        logger.info(f"  Model path: {self.config.model_path}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  FP16: {self.config.fp16}")
        logger.info(f"  Fidelity weight: {self.config.fidelity_weight}")

        start_time = time.time()

        # Import CodeFormer architecture from our local copy
        # (avoids PYTHONPATH conflicts with pip-installed basicsr)
        from lipsync.archs.codeformer_arch import CodeFormer as CodeFormerArch

        # Build CodeFormer network
        self.net = CodeFormerArch(
            dim_embd=512,
            codebook_size=1024,
            n_head=8,
            n_layers=9,
            connect_list=['32', '64', '128', '256'],
        ).to(self.device)

        # Find and load weights
        model_path = Path(self.config.model_path)
        ckpt_path = None

        # Try multiple possible locations
        possible_paths = [
            model_path / "codeformer.pth",
            model_path / "CodeFormer" / "codeformer.pth",
            model_path / "weights" / "codeformer.pth",
            Path("/app/models/codeformer/codeformer.pth"),
            Path("/app/models/codeformer/CodeFormer/codeformer.pth"),
        ]

        for path in possible_paths:
            if path.exists():
                ckpt_path = path
                break

        if ckpt_path is None:
            # Try downloading if not found
            logger.warning("CodeFormer weights not found locally, downloading...")
            ckpt_path = self._download_weights(model_path)

        logger.info(f"  Loading weights from: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=self.device)

        # Handle different checkpoint formats
        if 'params_ema' in checkpoint:
            state_dict = checkpoint['params_ema']
        elif 'params' in checkpoint:
            state_dict = checkpoint['params']
        else:
            state_dict = checkpoint

        self.net.load_state_dict(state_dict, strict=True)
        self.net.eval()

        # Note: FP16 handled via autocast in enhance_image(), not model conversion
        # This avoids dtype mismatch issues between fp16 model and fp32 intermediate ops

        self._loaded = True

        elapsed = time.time() - start_time
        logger.info(f"  Load time: {elapsed:.2f}s")
        logger.info("CODEFORMER - Model loaded")

    def _download_weights(self, target_dir: Path) -> Path:
        """Download CodeFormer weights from HuggingFace."""
        from huggingface_hub import hf_hub_download

        target_dir.mkdir(parents=True, exist_ok=True)

        # Download from sczhou/CodeFormer
        ckpt_path = hf_hub_download(
            repo_id="sczhou/CodeFormer",
            filename="CodeFormer/codeformer.pth",
            local_dir=str(target_dir),
        )
        return Path(ckpt_path)

    def unload(self) -> None:
        """Unload model from GPU."""
        logger.info("CODEFORMER - Unloading model")
        if self.net is not None:
            del self.net
            self.net = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.debug("Cleared CUDA cache")

        self._loaded = False
        logger.info("CODEFORMER - Model unloaded")

    def enhance_video(
        self,
        video_path: str,
        output_path: str,
        fidelity_weight: Optional[float] = None,
        blend_ratio: Optional[float] = None,
        has_aligned: bool = True,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> str:
        """
        Enhance face quality in video.

        Args:
            video_path: Path to input video
            output_path: Path for output video
            fidelity_weight: Balance between quality and fidelity (0-1)
            blend_ratio: How much to blend enhanced with original (0-1)
            has_aligned: Whether video contains aligned faces
            start_time: Process from this time
            end_time: Process until this time

        Returns:
            Path to enhanced video
        """
        fidelity = fidelity_weight or self.config.fidelity_weight
        blend = blend_ratio or 0.7

        logger.info("=" * 50)
        logger.info("CODEFORMER - Enhancing video")
        logger.info("=" * 50)
        logger.info(f"  Input: {video_path}")
        logger.info(f"  Output: {output_path}")
        logger.info(f"  Fidelity weight: {fidelity}")
        logger.info(f"  Blend ratio: {blend}")
        logger.info(f"  Has aligned: {has_aligned}")

        start = time.time()

        if not self._loaded:
            self.load()

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(f"  Video: {width}x{height} @ {fps:.2f}fps, {frame_count} frames")

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        processed_frames = 0
        batch_size = 4  # Process multiple frames for efficiency

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Enhance the frame
            enhanced = self.enhance_image(frame, fidelity_weight=fidelity)

            # Blend with original
            result = cv2.addWeighted(
                frame.astype(np.float32),
                1 - blend,
                enhanced.astype(np.float32),
                blend,
                0
            ).astype(np.uint8)

            out.write(result)
            processed_frames += 1

            if processed_frames % 50 == 0:
                logger.debug(f"  Processed {processed_frames}/{frame_count} frames")

        cap.release()
        out.release()

        elapsed = time.time() - start
        logger.info(f"  Processed {processed_frames} frames")
        logger.info(f"  Processing time: {elapsed:.2f}s ({processed_frames/max(elapsed, 0.01):.1f} fps)")
        logger.info(f"  Output saved to: {output_path}")
        logger.info("CODEFORMER - Enhancement complete")

        return output_path

    def enhance_image(
        self,
        image: np.ndarray,
        fidelity_weight: Optional[float] = None,
    ) -> np.ndarray:
        """
        Enhance a single face image.

        Args:
            image: Input face image (BGR, any size)
            fidelity_weight: Balance between quality and fidelity (0=quality, 1=fidelity)

        Returns:
            Enhanced image (same size as input)
        """
        if not self._loaded:
            self.load()

        fidelity = fidelity_weight if fidelity_weight is not None else self.config.fidelity_weight
        original_size = image.shape[:2]

        # CodeFormer expects 512x512 input
        target_size = 512
        if image.shape[0] != target_size or image.shape[1] != target_size:
            face_input = cv2.resize(image, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
        else:
            face_input = image

        # Convert BGR to RGB and normalize to [-1, 1]
        face_input = cv2.cvtColor(face_input, cv2.COLOR_BGR2RGB)
        face_input = face_input.astype(np.float32) / 255.0
        face_input = (face_input - 0.5) / 0.5  # Normalize to [-1, 1]

        # Convert to tensor: HWC -> CHW -> NCHW
        face_tensor = torch.from_numpy(face_input.transpose(2, 0, 1)).unsqueeze(0)
        face_tensor = face_tensor.to(self.device)

        # Run CodeFormer inference with autocast for automatic mixed precision
        # This handles dtype conversions between fp16/fp32 layers properly
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=self.config.fp16 and self.device.type == 'cuda'):
                output = self.net(face_tensor, w=fidelity, adain=True)[0]

        # Convert back to numpy: NCHW -> CHW -> HWC
        output = output.squeeze(0).float().clamp(-1, 1)
        output = (output + 1) / 2  # Denormalize to [0, 1]
        output = output.cpu().numpy().transpose(1, 2, 0)  # CHW -> HWC
        output = (output * 255).astype(np.uint8)

        # Convert RGB back to BGR
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

        # Resize back to original size if needed
        if original_size[0] != target_size or original_size[1] != target_size:
            output = cv2.resize(output, (original_size[1], original_size[0]), interpolation=cv2.INTER_LINEAR)

        return output

    @staticmethod
    def compute_blend_ratio(avg_face_size: float) -> float:
        """
        Compute optimal blend ratio based on face size.

        Larger faces need less enhancement, smaller faces need more.

        Args:
            avg_face_size: Average face area as percentage of frame

        Returns:
            Recommended blend ratio
        """
        base_max = 0.75
        base_min = 0.6

        if avg_face_size <= 2:
            return base_max
        elif avg_face_size >= 10:
            return base_min
        else:
            slope = (base_min - base_max) / (10 - 2)
            return base_max + slope * (avg_face_size - 2)

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded


def download_codeformer_models(target_dir: str = "models/codeformer") -> None:
    """
    Download CodeFormer model weights.

    Args:
        target_dir: Directory to save models
    """
    from huggingface_hub import hf_hub_download

    print(f"Downloading CodeFormer models to {target_dir}...")
    os.makedirs(target_dir, exist_ok=True)

    # Download CodeFormer weights
    hf_hub_download(
        repo_id="sczhou/CodeFormer",
        filename="CodeFormer/codeformer.pth",
        local_dir=target_dir,
    )

    # Download face detection/parsing models used by facexlib
    # These are typically downloaded automatically by facexlib
    print("CodeFormer models downloaded")
