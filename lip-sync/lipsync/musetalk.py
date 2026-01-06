"""
MuseTalk model wrapper for lip-sync generation.

Uses the official TMElyralab/MuseTalk package for inference.
MuseTalk generates lip movements synchronized to audio input.
"""

import logging
import os
import sys
import time
import torch
import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import cv2
import tempfile
import subprocess

logger = logging.getLogger('lipsync.musetalk')

# Add MuseTalk to path if installed locally
MUSETALK_PATH = os.environ.get('MUSETALK_PATH', '/app/MuseTalk')
if os.path.exists(MUSETALK_PATH) and MUSETALK_PATH not in sys.path:
    sys.path.insert(0, MUSETALK_PATH)


@dataclass
class MuseTalkConfig:
    """Configuration for MuseTalk model."""
    model_path: str = "/app/models/musetalk"
    musetalk_repo_path: str = "/app/MuseTalk"
    device: str = "cuda"
    fp16: bool = True
    batch_size: int = 8
    version: str = "v15"  # v1.0 or v15


class MuseTalk:
    """
    MuseTalk lip-sync model wrapper using official package.

    Uses the official TMElyralab/MuseTalk inference pipeline.
    """

    def __init__(self, config: Optional[MuseTalkConfig] = None):
        self.config = config or MuseTalkConfig()
        self.device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")
        self._loaded = False

        # Model components from official package
        self.vae = None
        self.unet = None
        self.pe = None
        self.audio_processor = None
        self.timesteps = None

        logger.info(f"MuseTalk initialized with device={self.device}, version={self.config.version}")

    def load(self) -> None:
        """Load model weights using official MuseTalk loaders."""
        if self._loaded:
            logger.debug("MuseTalk already loaded, skipping")
            return

        logger.info("=" * 50)
        logger.info("MUSETALK - Loading model (official package)")
        logger.info("=" * 50)
        logger.info(f"  Model path: {self.config.model_path}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Version: {self.config.version}")

        start_time = time.time()

        try:
            # Import from official MuseTalk package
            from musetalk.utils.utils import load_all_model
            from musetalk.whisper.audio2feature import Audio2Feature
            from musetalk.models.unet import PositionalEncoding

            # Determine model paths based on version
            model_dir = Path(self.config.model_path)
            if self.config.version == "v15":
                unet_config = model_dir / "musetalkV15" / "musetalk.json"
                unet_model = model_dir / "musetalkV15" / "unet.pth"
            else:
                unet_config = model_dir / "musetalk" / "musetalk.json"
                unet_model = model_dir / "musetalk" / "pytorch_model.bin"

            # Load all models
            logger.info("  Loading VAE, UNet, and PE...")
            self.vae, self.unet = load_all_model(
                unet_config=str(unet_config),
                unet_model_path=str(unet_model),
                device=str(self.device),
            )

            # Positional encoding
            self.pe = PositionalEncoding(d_model=384).to(self.device)

            # Audio processor
            logger.info("  Loading audio processor (Whisper)...")
            whisper_model = model_dir / "whisper" / "tiny.pt"
            self.audio_processor = Audio2Feature(
                model_path=str(whisper_model) if whisper_model.exists() else "tiny",
                device=str(self.device),
            )

            # Timesteps for UNet (always 0 for single-step inference)
            self.timesteps = torch.tensor([0], device=self.device)

            # Apply FP16 if requested
            if self.config.fp16 and self.device.type == 'cuda':
                self.unet.model = self.unet.model.half()
                logger.info("  Using FP16 precision")

        except ImportError as e:
            logger.error(f"Failed to import MuseTalk package: {e}")
            logger.error("Make sure MuseTalk is installed: git clone https://github.com/TMElyralab/MuseTalk")
            raise

        self._loaded = True

        elapsed = time.time() - start_time
        logger.info(f"  Load time: {elapsed:.2f}s")
        logger.info("MUSETALK - Model loaded")

    def unload(self) -> None:
        """Unload model from GPU."""
        logger.info("MUSETALK - Unloading model")

        for attr in ['vae', 'unet', 'pe', 'audio_processor']:
            if hasattr(self, attr) and getattr(self, attr) is not None:
                setattr(self, attr, None)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._loaded = False
        logger.info("MUSETALK - Model unloaded")

    def process(
        self,
        aligned_video_path: str,
        audio_path: str,
        output_path: str,
        smooth: bool = False,
        bbox_shift: int = 0,
    ) -> str:
        """
        Generate lip-synced video using official MuseTalk inference.

        Args:
            aligned_video_path: Path to aligned face video (256x256)
            audio_path: Path to audio file
            output_path: Path for output video
            smooth: Apply temporal smoothing
            bbox_shift: Controls mouth openness (-9 to 9)

        Returns:
            Path to lip-synced video
        """
        logger.info("=" * 50)
        logger.info("MUSETALK - Processing lip-sync")
        logger.info("=" * 50)
        logger.info(f"  Input video: {aligned_video_path}")
        logger.info(f"  Audio: {audio_path}")
        logger.info(f"  Output: {output_path}")

        start_time = time.time()

        if not self._loaded:
            self.load()

        try:
            from musetalk.utils.utils import get_image_pred, datagen
            from musetalk.utils.preprocessing import get_landmark_and_bbox

            # Get video info and read frames
            cap = cv2.VideoCapture(aligned_video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            logger.info(f"  Video: {width}x{height} @ {fps:.2f}fps, {frame_count} frames")

            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
            cap.release()

            # Extract audio features
            logger.info("  Extracting audio features...")
            whisper_feature = self.audio_processor.get_audio_feature(audio_path)
            whisper_chunks = self.audio_processor.get_whisper_chunk(
                whisper_feature,
                fps=fps,
                audio_padding=2,
            )

            # Prepare input latents from frames
            logger.info("  Encoding frames to latent space...")
            input_latents = []
            for frame in frames:
                # Resize to 256x256 if needed (MuseTalk expectation)
                if frame.shape[0] != 256 or frame.shape[1] != 256:
                    frame = cv2.resize(frame, (256, 256))

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
                frame_tensor = frame_tensor.unsqueeze(0).to(self.device)

                if self.config.fp16:
                    frame_tensor = frame_tensor.half()

                with torch.no_grad():
                    latent = self.vae.encode_latents(frame_tensor)
                input_latents.append(latent)

            input_latents = torch.cat(input_latents, dim=0)

            # Generate data batches
            logger.info(f"  Processing {len(frames)} frames...")
            gen = datagen(
                whisper_chunks,
                input_latents,
                self.config.batch_size,
            )

            # Inference loop
            output_frames = []
            for whisper_batch, latent_batch in gen:
                # Move to device
                whisper_batch = whisper_batch.to(self.device)
                latent_batch = latent_batch.to(self.device)

                if self.config.fp16:
                    whisper_batch = whisper_batch.half()
                    latent_batch = latent_batch.half()

                # Positional encoding on audio features
                audio_feature_batch = self.pe(whisper_batch)

                # UNet inference (single step)
                with torch.no_grad():
                    pred_latents = self.unet.model(
                        latent_batch,
                        self.timesteps,
                        encoder_hidden_states=audio_feature_batch,
                    ).sample

                # Decode latents to images
                with torch.no_grad():
                    recon = self.vae.decode_latents(pred_latents)

                # Convert to numpy frames
                for i in range(recon.shape[0]):
                    frame = recon[i].cpu().numpy().transpose(1, 2, 0)
                    frame = (frame * 255).clip(0, 255).astype(np.uint8)
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    output_frames.append(frame)

            # Resize back to original size if needed
            if width != 256 or height != 256:
                output_frames = [cv2.resize(f, (width, height)) for f in output_frames]

            # Write output video
            logger.info("  Writing output video...")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            for frame in output_frames[:len(frames)]:  # Match original frame count
                out.write(frame)
            out.release()

        except ImportError as e:
            logger.error(f"MuseTalk import error: {e}")
            logger.warning("Falling back to passthrough mode")
            import shutil
            shutil.copy(aligned_video_path, output_path)

        elapsed = time.time() - start_time
        logger.info(f"  Processing time: {elapsed:.2f}s ({len(frames)/max(elapsed, 0.01):.1f} fps)")
        logger.info(f"  Output saved to: {output_path}")
        logger.info("MUSETALK - Processing complete")

        return output_path

    @property
    def is_loaded(self) -> bool:
        return self._loaded


def download_musetalk_models(target_dir: str = "/app/models/musetalk") -> None:
    """Download MuseTalk model weights."""
    from huggingface_hub import snapshot_download

    print(f"Downloading MuseTalk models to {target_dir}...")
    os.makedirs(target_dir, exist_ok=True)

    snapshot_download(
        repo_id="TMElyralab/MuseTalk",
        local_dir=target_dir,
        ignore_patterns=["*.md", "*.txt", "*.git*"],
    )
    print("MuseTalk models downloaded")
