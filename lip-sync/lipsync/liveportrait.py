"""
LivePortrait model wrapper for face neutralization.

Uses the official KwaiVGI/LivePortrait package for inference.
LivePortrait is used to neutralize mouth expressions before lip-sync.
"""

import logging
import os
import sys
import time
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass
import cv2

logger = logging.getLogger('lipsync.liveportrait')

# Add LivePortrait to path if installed locally
LIVEPORTRAIT_PATH = os.environ.get('LIVEPORTRAIT_PATH', '/app/LivePortrait')
if os.path.exists(LIVEPORTRAIT_PATH) and LIVEPORTRAIT_PATH not in sys.path:
    sys.path.insert(0, LIVEPORTRAIT_PATH)


@dataclass
class LivePortraitConfig:
    """Configuration for LivePortrait model."""
    model_path: str = "/app/models/liveportrait"
    liveportrait_repo_path: str = "/app/LivePortrait"
    device: str = "cuda"
    fp16: bool = True
    output_size: int = 512


class LivePortrait:
    """
    LivePortrait face animation model wrapper using official package.

    Used in lip-sync pipeline to neutralize existing lip movements
    before applying new audio-driven lip sync.
    """

    def __init__(self, config: Optional[LivePortraitConfig] = None):
        self.config = config or LivePortraitConfig()
        self.device = torch.device(self.config.device if torch.cuda.is_available() else "cpu")
        self._loaded = False

        # Official LivePortrait components
        self.live_portrait_wrapper = None
        self.cropper = None

        logger.info(f"LivePortrait initialized with device={self.device}")

    def load(self) -> None:
        """Load model weights using official LivePortrait loaders."""
        if self._loaded:
            logger.debug("LivePortrait already loaded, skipping")
            return

        logger.info("=" * 50)
        logger.info("LIVEPORTRAIT - Loading model (official package)")
        logger.info("=" * 50)
        logger.info(f"  Model path: {self.config.model_path}")
        logger.info(f"  Device: {self.device}")

        start_time = time.time()

        try:
            # Import from official LivePortrait package
            from src.config.argument_config import ArgumentConfig
            from src.config.inference_config import InferenceConfig
            from src.config.crop_config import CropConfig
            from src.live_portrait_pipeline import LivePortraitPipeline

            # Create configs
            inference_cfg = InferenceConfig(
                device_id=0 if self.device.type == 'cuda' else -1,
                flag_force_cpu=self.device.type != 'cuda',
            )

            crop_cfg = CropConfig()

            # Initialize pipeline
            logger.info("  Loading LivePortrait pipeline...")
            self.pipeline = LivePortraitPipeline(
                inference_cfg=inference_cfg,
                crop_cfg=crop_cfg,
            )

            # Store wrapper for direct access to retargeting
            self.live_portrait_wrapper = self.pipeline.live_portrait_wrapper
            self.cropper = self.pipeline.cropper

        except ImportError as e:
            logger.error(f"Failed to import LivePortrait package: {e}")
            logger.error("Make sure LivePortrait is installed: git clone https://github.com/KwaiVGI/LivePortrait")
            raise

        self._loaded = True

        elapsed = time.time() - start_time
        logger.info(f"  Load time: {elapsed:.2f}s")
        logger.info("LIVEPORTRAIT - Model loaded")

    def unload(self) -> None:
        """Unload model from GPU."""
        logger.info("LIVEPORTRAIT - Unloading model")

        for attr in ['pipeline', 'live_portrait_wrapper', 'cropper']:
            if hasattr(self, attr) and getattr(self, attr) is not None:
                setattr(self, attr, None)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._loaded = False
        logger.info("LIVEPORTRAIT - Model unloaded")

    def neutralize(
        self,
        video_path: str,
        output_path: str,
        reference_image_path: Optional[str] = None,
        lip_ratio: float = 0.0,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
    ) -> str:
        """
        Neutralize lip movements in video using official LivePortrait.

        Args:
            video_path: Path to input video
            output_path: Path for output video
            reference_image_path: Optional neutral face reference
            lip_ratio: Lip retargeting ratio (0 = neutral, 1 = original)
            start_time: Process from this time (seconds)
            end_time: Process until this time (seconds)

        Returns:
            Path to neutralized video
        """
        logger.info("=" * 50)
        logger.info("LIVEPORTRAIT - Neutralizing lip movements")
        logger.info("=" * 50)
        logger.info(f"  Input: {video_path}")
        logger.info(f"  Output: {output_path}")
        logger.info(f"  Lip ratio: {lip_ratio}")

        start = time.time()

        if not self._loaded:
            self.load()

        try:
            # Open video
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            logger.info(f"  Video: {width}x{height} @ {fps:.2f}fps, {frame_count} frames")

            # Read first frame for source features
            ret, first_frame = cap.read()
            if not ret:
                raise ValueError("Could not read first frame")

            # Get source info (identity features)
            first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
            source_info = self._extract_source_info(first_frame_rgb)

            if source_info is None:
                logger.warning("No face detected, copying video unchanged")
                cap.release()
                import shutil
                shutil.copy(video_path, output_path)
                return output_path

            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # Process all frames
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            processed = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                neutralized = self._process_frame(frame_rgb, source_info, lip_ratio)
                neutralized_bgr = cv2.cvtColor(neutralized, cv2.COLOR_RGB2BGR)

                out.write(neutralized_bgr)
                processed += 1

                if processed % 50 == 0:
                    logger.debug(f"  Processed {processed}/{frame_count} frames")

            cap.release()
            out.release()

        except ImportError as e:
            logger.error(f"LivePortrait import error: {e}")
            logger.warning("Falling back to passthrough mode")
            import shutil
            shutil.copy(video_path, output_path)

        elapsed = time.time() - start
        logger.info(f"  Processing time: {elapsed:.2f}s ({frame_count/max(elapsed, 0.01):.1f} fps)")
        logger.info(f"  Output saved to: {output_path}")
        logger.info("LIVEPORTRAIT - Neutralization complete")

        return output_path

    def _extract_source_info(self, frame_rgb: np.ndarray) -> Optional[Dict[str, Any]]:
        """Extract source features using official LivePortrait."""
        try:
            # Crop face
            crop_info = self.cropper.crop_single_image(frame_rgb)
            if crop_info is None:
                return None

            img_crop_256x256 = crop_info['img_crop_256x256']

            # Convert to tensor
            img_tensor = torch.from_numpy(img_crop_256x256).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0)

            if self.device.type == 'cuda':
                img_tensor = img_tensor.cuda()

            # Extract features
            with torch.no_grad():
                # Get 3D appearance features
                feature_3d = self.live_portrait_wrapper.extract_feature_3d(img_tensor)

                # Get keypoint info (includes neutral expression)
                kp_info = self.live_portrait_wrapper.get_kp_info(img_tensor)

            return {
                'feature_3d': feature_3d,
                'kp_source': kp_info,
                'crop_info': crop_info,
            }

        except Exception as e:
            logger.error(f"Error extracting source info: {e}")
            return None

    def _process_frame(
        self,
        frame_rgb: np.ndarray,
        source_info: Dict[str, Any],
        lip_ratio: float,
    ) -> np.ndarray:
        """Process a single frame with lip neutralization."""
        try:
            # Crop driving frame
            crop_info = self.cropper.crop_single_image(frame_rgb)
            if crop_info is None:
                return frame_rgb

            img_crop_256x256 = crop_info['img_crop_256x256']

            # Convert to tensor
            img_tensor = torch.from_numpy(img_crop_256x256).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0)

            if self.device.type == 'cuda':
                img_tensor = img_tensor.cuda()

            with torch.no_grad():
                # Get driving keypoints
                kp_driving = self.live_portrait_wrapper.get_kp_info(img_tensor)

                # Apply lip retargeting
                if lip_ratio < 1.0:
                    # Get lip delta for neutralization
                    kp_source = source_info['kp_source']

                    # Interpolate lip parameters
                    # lip_ratio=0 means use source lips (neutral)
                    # lip_ratio=1 means use driving lips (original)
                    kp_neutralized = {}
                    for key in kp_driving:
                        if isinstance(kp_driving[key], torch.Tensor):
                            kp_neutralized[key] = (
                                (1 - lip_ratio) * kp_source[key] +
                                lip_ratio * kp_driving[key]
                            )
                        else:
                            kp_neutralized[key] = kp_driving[key]

                    # Apply lip retargeting module if available
                    if hasattr(self.live_portrait_wrapper, 'retarget_lip'):
                        delta = self.live_portrait_wrapper.retarget_lip(
                            kp_source['kp'],
                            lip_ratio,
                        )
                        if 'kp' in kp_neutralized:
                            kp_neutralized['kp'] = kp_neutralized['kp'] + delta
                else:
                    kp_neutralized = kp_driving

                # Generate output
                out = self.live_portrait_wrapper.warp_decode(
                    source_info['feature_3d'],
                    kp_source=source_info['kp_source'],
                    kp_driving=kp_neutralized,
                )

                # Convert output tensor to numpy
                out_np = out.squeeze(0).permute(1, 2, 0).cpu().numpy()
                out_np = (out_np * 255).clip(0, 255).astype(np.uint8)

            # Paste back into original frame
            result = self._paste_back(frame_rgb, out_np, crop_info)
            return result

        except Exception as e:
            logger.debug(f"Frame processing error: {e}")
            return frame_rgb

    def _paste_back(
        self,
        frame: np.ndarray,
        face: np.ndarray,
        crop_info: Dict,
    ) -> np.ndarray:
        """Paste processed face back into frame."""
        try:
            # Get crop coordinates
            if 'M_c2o' in crop_info:
                # Use inverse transformation matrix
                M_c2o = crop_info['M_c2o']
                h, w = frame.shape[:2]

                # Resize face to crop size
                face_resized = cv2.resize(face, (256, 256))

                # Warp face back to original coordinates
                face_warped = cv2.warpAffine(
                    face_resized,
                    M_c2o[:2],
                    (w, h),
                    borderMode=cv2.BORDER_CONSTANT,
                    flags=cv2.INTER_LINEAR,
                )

                # Create mask
                mask = np.ones((256, 256), dtype=np.float32)
                mask = cv2.warpAffine(
                    mask,
                    M_c2o[:2],
                    (w, h),
                    borderMode=cv2.BORDER_CONSTANT,
                )
                mask = cv2.GaussianBlur(mask, (21, 21), 0)
                mask = mask[:, :, np.newaxis]

                # Blend
                result = frame * (1 - mask) + face_warped * mask
                return result.astype(np.uint8)
            else:
                return frame

        except Exception as e:
            logger.debug(f"Paste back error: {e}")
            return frame

    @property
    def is_loaded(self) -> bool:
        return self._loaded


def download_liveportrait_models(target_dir: str = "/app/models/liveportrait") -> None:
    """Download LivePortrait model weights."""
    from huggingface_hub import snapshot_download

    print(f"Downloading LivePortrait models to {target_dir}...")
    os.makedirs(target_dir, exist_ok=True)

    snapshot_download(
        repo_id="KwaiVGI/LivePortrait",
        local_dir=target_dir,
        ignore_patterns=["*.md", "*.txt", "*.git*", "examples/*"],
    )
    print("LivePortrait models downloaded")
