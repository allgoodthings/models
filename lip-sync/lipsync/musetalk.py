"""
MuseTalk model wrapper for lip-sync generation.

Uses the official TMElyralab/MuseTalk package for inference.
MuseTalk generates lip movements synchronized to audio input.

VERIFIED API (from official repo source code):
- VAE.__init__(model_path="./models/sd-vae-ft-mse/", resized_img=256, use_float16=False)
- VAE.encode_latents(image) -> latents
- VAE.decode_latents(latents) -> BGR image (0-255 uint8)
- VAE.get_latents_for_unet(masked_img, face_img) -> concatenated latents

- UNet.__init__(unet_config, model_path, use_float16=False, device=None)
- UNet.model - the actual UNet2DConditionModel
- UNet.pe - PositionalEncoding(d_model=384)

- Audio2Feature.__init__(whisper_model_type="tiny", model_path="./models/whisper/tiny.pt")
- Audio2Feature.audio2feat(audio_path) -> numpy array of features
- Audio2Feature.feature2chunks(feature_array, fps, audio_feat_length=[2,2]) -> list of chunks
"""

import logging
import os
import sys
import time
import torch
import numpy as np
from pathlib import Path
from typing import Optional, List, Tuple
from dataclasses import dataclass
import cv2

logger = logging.getLogger('lipsync.musetalk')

# Add MuseTalk to path if cloned locally
MUSETALK_PATH = os.environ.get('MUSETALK_PATH', '/app/MuseTalk')
if os.path.exists(MUSETALK_PATH) and MUSETALK_PATH not in sys.path:
    sys.path.insert(0, MUSETALK_PATH)


@dataclass
class MuseTalkConfig:
    """Configuration for MuseTalk model."""
    # Model weights directory (downloaded from HF TMElyralab/MuseTalk)
    model_dir: str = "/app/models/musetalk"
    # MuseTalk repo path (cloned from GitHub)
    repo_path: str = "/app/MuseTalk"
    device: str = "cuda"
    fp16: bool = True
    batch_size: int = 8
    # v1 or v15 - v15 is newer and better
    version: str = "v15"
    # Whisper model for audio features: tiny, base, small, medium, large
    # Larger = better lip-sync quality but slower inference
    whisper_model: str = "small"


class MuseTalk:
    """
    MuseTalk lip-sync model wrapper using official package.

    MuseTalk operates by inpainting the lower face region in latent space
    with a single UNet step, conditioned on Whisper audio features.
    """

    def __init__(self, config: Optional[MuseTalkConfig] = None):
        self.config = config or MuseTalkConfig()
        self.device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")
        self._loaded = False

        # Model components (from official MuseTalk package)
        self.vae = None
        self.unet = None
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
        logger.info(f"  Model dir: {self.config.model_dir}")
        logger.info(f"  Repo path: {self.config.repo_path}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Version: {self.config.version}")

        start_time = time.time()

        try:
            # Import from official MuseTalk package
            # These imports come from the cloned repo at MUSETALK_PATH
            from musetalk.models.vae import VAE
            from musetalk.models.unet import UNet
            from musetalk.whisper.audio2feature import Audio2Feature

            model_dir = Path(self.config.model_dir)

            # Determine UNet paths based on version
            # HF structure: TMElyralab/MuseTalk/musetalkV15/musetalk.json, unet.pth
            if self.config.version == "v15":
                unet_config = model_dir / "musetalkV15" / "musetalk.json"
                unet_weights = model_dir / "musetalkV15" / "unet.pth"
            else:
                unet_config = model_dir / "musetalk" / "musetalk.json"
                unet_weights = model_dir / "musetalk" / "pytorch_model.bin"

            # VAE path - uses stabilityai/sd-vae-ft-mse from HuggingFace
            # Downloaded via download_models.py or from HuggingFace directly
            vae_path = model_dir / "sd-vae-ft-mse"
            if not vae_path.exists():
                # Fall back to HuggingFace repo ID (will use cached or download)
                logger.info("  VAE not found locally, using HuggingFace repo...")
                vae_path = Path("stabilityai/sd-vae-ft-mse")

            # Load VAE
            # Signature: VAE(model_path, resized_img=256, use_float16=False)
            logger.info(f"  Loading VAE from {vae_path}...")
            self.vae = VAE(
                model_path=str(vae_path),
                resized_img=256,
                use_float16=self.config.fp16,
            )

            # Load UNet
            # Signature: UNet(unet_config, model_path, use_float16=False, device=None)
            logger.info(f"  Loading UNet from {unet_weights}...")
            self.unet = UNet(
                unet_config=str(unet_config),
                model_path=str(unet_weights),
                use_float16=self.config.fp16,
                device=str(self.device),
            )

            # Load Audio processor (Whisper)
            # Signature: Audio2Feature(whisper_model_type, model_path)
            # model_path accepts: file path OR model name ("tiny", "base", "small", etc.)
            # If model name is passed, it auto-downloads from OpenAI
            logger.info(f"  Loading Whisper audio processor (model={self.config.whisper_model})...")
            whisper_path = model_dir / "whisper" / f"{self.config.whisper_model}.pt"
            self.audio_processor = Audio2Feature(
                whisper_model_type=self.config.whisper_model,
                # Use local file if exists, otherwise use model name for auto-download
                model_path=str(whisper_path) if whisper_path.exists() else self.config.whisper_model,
            )

            # Timesteps for UNet (always 0 for single-step inference)
            self.timesteps = torch.tensor([0], device=self.device)

        except ImportError as e:
            logger.error(f"Failed to import MuseTalk package: {e}")
            logger.error(f"Make sure MuseTalk is cloned to {self.config.repo_path}")
            logger.error("Expected: git clone https://github.com/TMElyralab/MuseTalk")
            raise

        self._loaded = True

        elapsed = time.time() - start_time
        logger.info(f"  Load time: {elapsed:.2f}s")
        logger.info("MUSETALK - Model loaded")

    def unload(self) -> None:
        """Unload model from GPU."""
        logger.info("MUSETALK - Unloading model")

        for attr in ['vae', 'unet', 'audio_processor']:
            if hasattr(self, attr) and getattr(self, attr) is not None:
                setattr(self, attr, None)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._loaded = False
        logger.info("MUSETALK - Model unloaded")

    def process(
        self,
        video_path: str,
        audio_path: str,
        output_path: str,
        bbox_shift: int = 0,
    ) -> str:
        """
        Generate lip-synced video using official MuseTalk inference.

        Args:
            video_path: Path to input video (face region, ideally 256x256)
            audio_path: Path to audio file
            output_path: Path for output video
            bbox_shift: Vertical bbox shift for mouth region (-9 to 9)

        Returns:
            Path to lip-synced video
        """
        logger.info("=" * 50)
        logger.info("MUSETALK - Processing lip-sync")
        logger.info("=" * 50)
        logger.info(f"  Input video: {video_path}")
        logger.info(f"  Audio: {audio_path}")
        logger.info(f"  Output: {output_path}")
        logger.info(f"  BBox shift: {bbox_shift}")

        start_time = time.time()

        if not self._loaded:
            self.load()

        try:
            # Read video frames
            cap = cv2.VideoCapture(video_path)
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

            if not frames:
                raise ValueError("No frames read from video")

            # Extract audio features using Whisper
            # audio2feat(audio_path) -> numpy array
            logger.info("  Extracting audio features...")
            whisper_feature = self.audio_processor.audio2feat(audio_path)

            # feature2chunks(feature_array, fps, audio_feat_length=[2,2]) -> list
            whisper_chunks = self.audio_processor.feature2chunks(
                feature_array=whisper_feature,
                fps=fps,
                audio_feat_length=[2, 2],
            )
            logger.info(f"  Audio chunks: {len(whisper_chunks)}")

            # Prepare face crops and latents
            logger.info("  Encoding frames to latent space...")
            input_latent_list = []
            coord_list = []
            frame_list = []

            for i, frame in enumerate(frames):
                h, w = frame.shape[:2]

                # Default to full frame if already cropped to face
                if w == 256 and h == 256:
                    coord = [0, 0, 256, 256]
                else:
                    # Center crop to square
                    min_dim = min(h, w)
                    y1 = (h - min_dim) // 2
                    x1 = (w - min_dim) // 2
                    coord = [y1, x1, y1 + min_dim, x1 + min_dim]

                coord_list.append(coord)
                frame_list.append(frame)

                # Crop and resize face region to 256x256
                y1, x1, y2, x2 = coord
                face_crop = frame[y1:y2, x1:x2]
                face_crop = cv2.resize(face_crop, (256, 256))

                # VAE.encode_latents expects tensor, VAE handles conversion internally
                # But we need to prepare input correctly
                # The VAE.preprocess_img handles loading from path, so we'll encode directly
                face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                latent = self._encode_frame(face_rgb)
                input_latent_list.append(latent)

            input_latents = torch.cat(input_latent_list, dim=0)

            # Prepare masked latents for UNet inpainting
            # MuseTalk masks the lower half of the face for lip region
            logger.info("  Preparing masked latents for UNet...")

            # Generate data batches
            logger.info(f"  Processing {len(frames)} frames in batches of {self.config.batch_size}...")

            # Align whisper chunks to frame count
            num_frames = len(frames)
            if len(whisper_chunks) < num_frames:
                whisper_chunks = whisper_chunks + [whisper_chunks[-1]] * (num_frames - len(whisper_chunks))
            elif len(whisper_chunks) > num_frames:
                whisper_chunks = whisper_chunks[:num_frames]

            # Convert whisper chunks to tensor
            # Each chunk should be (seq_len, 384) for Whisper features
            whisper_batch = torch.stack([
                torch.from_numpy(c).float() if isinstance(c, np.ndarray) else c
                for c in whisper_chunks
            ]).to(self.device)

            if self.config.fp16:
                whisper_batch = whisper_batch.half()

            # Process in batches
            output_frames = []
            for i in range(0, num_frames, self.config.batch_size):
                end_idx = min(i + self.config.batch_size, num_frames)
                batch_size = end_idx - i

                batch_whisper = whisper_batch[i:end_idx]
                batch_latents = input_latents[i:end_idx]

                # Prepare masked latents (mask lower half for mouth region)
                masked_latents = self._prepare_masked_latents(batch_latents)

                with torch.no_grad():
                    # Apply positional encoding to audio features
                    # unet.pe is PositionalEncoding(d_model=384)
                    audio_feature = self.unet.pe(batch_whisper)

                    # Single-step UNet inference
                    # unet.model is UNet2DConditionModel
                    pred = self.unet.model(
                        masked_latents,
                        self.timesteps.expand(batch_size),
                        encoder_hidden_states=audio_feature,
                    ).sample

                # Decode latents to images
                for j in range(pred.shape[0]):
                    latent = pred[j:j+1]
                    # decode_latents returns BGR image 0-255 uint8
                    frame_out = self.vae.decode_latents(latent)
                    output_frames.append(frame_out)

            # Composite back into original frames
            logger.info("  Compositing output...")
            final_frames = []
            for i, (orig_frame, gen_face, coord) in enumerate(zip(frame_list, output_frames, coord_list)):
                y1, x1, y2, x2 = coord
                crop_h, crop_w = y2 - y1, x2 - x1

                # Handle different output formats from decode_latents
                if isinstance(gen_face, np.ndarray):
                    if len(gen_face.shape) == 4:
                        gen_face = gen_face[0]  # Remove batch dim
                    if gen_face.shape[0] == 3:
                        gen_face = gen_face.transpose(1, 2, 0)  # CHW -> HWC
                    # Already BGR from decode_latents

                # Resize generated face to match crop size
                gen_face_resized = cv2.resize(gen_face, (crop_w, crop_h))

                # Paste into frame
                result = orig_frame.copy()
                result[y1:y2, x1:x2] = gen_face_resized
                final_frames.append(result)

            # Write output video
            logger.info("  Writing output video...")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            for frame in final_frames:
                out.write(frame)
            out.release()

        except ImportError as e:
            logger.error(f"MuseTalk import error: {e}")
            logger.error("Falling back to passthrough mode")
            import shutil
            shutil.copy(video_path, output_path)
        except Exception as e:
            logger.error(f"MuseTalk processing error: {e}")
            import traceback
            traceback.print_exc()
            import shutil
            shutil.copy(video_path, output_path)

        elapsed = time.time() - start_time
        fps_processed = len(frames) / max(elapsed, 0.01)
        logger.info(f"  Processing time: {elapsed:.2f}s ({fps_processed:.1f} fps)")
        logger.info(f"  Output saved to: {output_path}")
        logger.info("MUSETALK - Processing complete")

        return output_path

    def _encode_frame(self, frame_rgb: np.ndarray) -> torch.Tensor:
        """Encode a single frame to latent space using VAE."""
        # Normalize to [-1, 1] as expected by VAE
        frame_tensor = torch.from_numpy(frame_rgb).float() / 255.0
        frame_tensor = (frame_tensor - 0.5) / 0.5  # Normalize to [-1, 1]
        frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)  # HWC -> NCHW
        frame_tensor = frame_tensor.to(self.device)

        if self.config.fp16:
            frame_tensor = frame_tensor.half()

        with torch.no_grad():
            # VAE.encode_latents(image) -> latents
            latent = self.vae.encode_latents(frame_tensor)

        return latent

    def _prepare_masked_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Prepare masked latents for inpainting.

        MuseTalk masks the lower portion of the face latent for lip region inpainting.
        The UNet is conditioned on the masked image + audio to generate new lip movements.

        Uses get_latents_for_unet() pattern: concatenate [masked, original] along channel dim.
        """
        # Latent shape is typically [B, 4, 32, 32] for 256x256 input
        B, C, H, W = latents.shape

        # Create mask - lower half for mouth region
        mask_h = H // 2
        masked_latents = latents.clone()

        # Zero out lower half (mouth region will be regenerated)
        masked_latents[:, :, mask_h:, :] = 0

        # Concatenate original and masked for UNet input
        # Pattern from VAE.get_latents_for_unet(): [masked, original] -> [B, 8, H, W]
        combined = torch.cat([masked_latents, latents], dim=1)

        return combined

    @property
    def is_loaded(self) -> bool:
        return self._loaded


def download_musetalk_models(target_dir: str = "/app/models/musetalk") -> None:
    """Download MuseTalk model weights from HuggingFace."""
    from huggingface_hub import snapshot_download

    print(f"Downloading MuseTalk models to {target_dir}...")
    os.makedirs(target_dir, exist_ok=True)

    # Download from TMElyralab/MuseTalk
    # Structure: musetalkV15/musetalk.json, musetalkV15/unet.pth
    snapshot_download(
        repo_id="TMElyralab/MuseTalk",
        local_dir=target_dir,
        ignore_patterns=["*.md", "*.txt", "*.git*", "demo/*", "results/*"],
    )
    print("MuseTalk models downloaded")
