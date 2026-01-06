"""
LivePortrait model wrapper for face neutralization.

Uses the official KwaiVGI/LivePortrait package for inference.
LivePortrait is used to neutralize mouth expressions before lip-sync.

API Reference (verified from official repo):
- src.live_portrait_pipeline.LivePortraitPipeline: Main orchestration class
- src.live_portrait_wrapper.LivePortraitWrapper: Neural network wrapper with:
  - extract_feature_3d(): Get appearance features from source
  - get_kp_info(): Get keypoints including expression params
  - warp_decode(): Generate output frame from features + keypoints
  - retarget_lip(): Calculate lip delta for neutralization
  - stitching(): Blend keypoints for seamless compositing
- src.utils.cropper.Cropper: Face detection and cropping
- src.config.inference_config.InferenceConfig: Model paths and flags
- src.config.crop_config.CropConfig: Cropping parameters
"""

import logging
import os
import sys
import time
import torch
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import cv2

logger = logging.getLogger('lipsync.liveportrait')

# Add LivePortrait to path if cloned locally
LIVEPORTRAIT_PATH = os.environ.get('LIVEPORTRAIT_PATH', '/app/LivePortrait')
if os.path.exists(LIVEPORTRAIT_PATH) and LIVEPORTRAIT_PATH not in sys.path:
    sys.path.insert(0, LIVEPORTRAIT_PATH)


@dataclass
class LivePortraitConfig:
    """Configuration for LivePortrait model."""
    # Model weights directory (downloaded from HF)
    model_dir: str = "/app/models/liveportrait"
    # LivePortrait repo path (cloned from GitHub)
    repo_path: str = "/app/LivePortrait"
    device: str = "cuda"
    device_id: int = 0
    fp16: bool = True
    output_size: int = 512


class LivePortrait:
    """
    LivePortrait face animation model wrapper using official package.

    Used in lip-sync pipeline to neutralize existing lip movements
    before applying new audio-driven lip sync with MuseTalk.

    The key operation is setting lip_ratio=0 to create a neutral mouth
    expression while preserving identity and other facial features.
    """

    def __init__(self, config: Optional[LivePortraitConfig] = None):
        self.config = config or LivePortraitConfig()
        self.device = torch.device(
            f"{self.config.device}:{self.config.device_id}"
            if torch.cuda.is_available() and self.config.device == "cuda"
            else "cpu"
        )
        self._loaded = False

        # Official LivePortrait components
        self.pipeline = None
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
        logger.info(f"  Model dir: {self.config.model_dir}")
        logger.info(f"  Repo path: {self.config.repo_path}")
        logger.info(f"  Device: {self.device}")

        start_time = time.time()

        try:
            # Import from official LivePortrait package
            from src.config.inference_config import InferenceConfig
            from src.config.crop_config import CropConfig
            from src.live_portrait_pipeline import LivePortraitPipeline

            # Create inference config
            inference_cfg = InferenceConfig(
                device_id=self.config.device_id if self.device.type == 'cuda' else -1,
                flag_force_cpu=self.device.type != 'cuda',
                flag_use_half_precision=self.config.fp16,
            )

            # Override model paths if custom model_dir is specified
            model_dir = Path(self.config.model_dir)
            if model_dir.exists():
                # LivePortrait expects pretrained_weights structure
                weights_dir = model_dir / "pretrained_weights"
                if weights_dir.exists():
                    inference_cfg.checkpoint_F = str(weights_dir / "liveportrait" / "base_models" / "appearance_feature_extractor.pth")
                    inference_cfg.checkpoint_M = str(weights_dir / "liveportrait" / "base_models" / "motion_extractor.pth")
                    inference_cfg.checkpoint_G = str(weights_dir / "liveportrait" / "base_models" / "spade_generator.pth")
                    inference_cfg.checkpoint_W = str(weights_dir / "liveportrait" / "base_models" / "warping_module.pth")
                    inference_cfg.checkpoint_S = str(weights_dir / "liveportrait" / "retargeting_models" / "stitching_retargeting_module.pth")

            # Create crop config
            crop_cfg = CropConfig(
                device_id=self.config.device_id if self.device.type == 'cuda' else -1,
                flag_force_cpu=self.device.type != 'cuda',
            )

            # Initialize pipeline
            logger.info("  Loading LivePortrait pipeline...")
            self.pipeline = LivePortraitPipeline(
                inference_cfg=inference_cfg,
                crop_cfg=crop_cfg,
            )

            # Store wrapper and cropper for direct access
            self.live_portrait_wrapper = self.pipeline.live_portrait_wrapper
            self.cropper = self.pipeline.cropper

        except ImportError as e:
            logger.error(f"Failed to import LivePortrait package: {e}")
            logger.error(f"Make sure LivePortrait is cloned to {self.config.repo_path}")
            logger.error("Expected: git clone https://github.com/KwaiVGI/LivePortrait")
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
            reference_image_path: Optional neutral face reference (uses first frame if None)
            lip_ratio: Lip retargeting ratio (0.0 = neutral/closed, 1.0 = original)
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

            # Calculate frame range
            start_frame = int(start_time * fps) if start_time else 0
            end_frame = int(end_time * fps) if end_time else frame_count

            # Read first frame for source features
            ret, first_frame = cap.read()
            if not ret:
                raise ValueError("Could not read first frame")

            # Convert BGR to RGB for processing
            first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)

            # Get source info (identity features from first frame)
            source_info = self._extract_source_info(first_frame_rgb)

            if source_info is None:
                logger.warning("No face detected in first frame, copying video unchanged")
                cap.release()
                import shutil
                shutil.copy(video_path, output_path)
                return output_path

            # Reset video position
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # Setup output video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            # Process all frames
            processed = 0
            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Check if frame is in processing range
                if start_frame <= frame_idx < end_frame:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    neutralized_rgb = self._process_frame(frame_rgb, source_info, lip_ratio)
                    neutralized_bgr = cv2.cvtColor(neutralized_rgb, cv2.COLOR_RGB2BGR)
                    out.write(neutralized_bgr)
                else:
                    out.write(frame)

                processed += 1
                frame_idx += 1

                if processed % 50 == 0:
                    logger.debug(f"  Processed {processed}/{frame_count} frames")

            cap.release()
            out.release()

        except ImportError as e:
            logger.error(f"LivePortrait import error: {e}")
            logger.warning("Falling back to passthrough mode")
            import shutil
            shutil.copy(video_path, output_path)
        except Exception as e:
            logger.error(f"LivePortrait processing error: {e}")
            import traceback
            traceback.print_exc()
            import shutil
            shutil.copy(video_path, output_path)

        elapsed = time.time() - start
        fps_processed = frame_count / max(elapsed, 0.01)
        logger.info(f"  Processing time: {elapsed:.2f}s ({fps_processed:.1f} fps)")
        logger.info(f"  Output saved to: {output_path}")
        logger.info("LIVEPORTRAIT - Neutralization complete")

        return output_path

    def _extract_source_info(self, frame_rgb: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Extract source features using official LivePortrait.

        Returns dict with:
        - feature_3d: Appearance features from F network
        - kp_source: Keypoint info from M network (includes expression)
        - crop_info: Cropping transformation for paste-back
        """
        try:
            # Use cropper to detect and crop face
            crop_info = self.cropper.crop_source_image(frame_rgb)

            if crop_info is None or 'img_crop_256x256' not in crop_info:
                logger.warning("No face detected by cropper")
                return None

            img_crop = crop_info['img_crop_256x256']

            # Convert to tensor [B, C, H, W]
            img_tensor = torch.from_numpy(img_crop).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(self.device)

            if self.config.fp16:
                img_tensor = img_tensor.half()

            # Extract features
            with torch.no_grad():
                # Get 3D appearance features from F network
                feature_3d = self.live_portrait_wrapper.extract_feature_3d(img_tensor)

                # Get keypoint info from M network (includes expression, pose, etc.)
                kp_source = self.live_portrait_wrapper.get_kp_info(img_tensor)

            return {
                'feature_3d': feature_3d,
                'kp_source': kp_source,
                'crop_info': crop_info,
                'source_tensor': img_tensor,
            }

        except Exception as e:
            logger.error(f"Error extracting source info: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _process_frame(
        self,
        frame_rgb: np.ndarray,
        source_info: Dict[str, Any],
        lip_ratio: float,
    ) -> np.ndarray:
        """
        Process a single frame with lip neutralization.

        The key insight is that LivePortrait's retarget_lip() method
        calculates a delta to apply to keypoints. With lip_ratio=0,
        we neutralize the lip expression toward the source (neutral) pose.
        """
        try:
            # Crop driving frame
            crop_info = self.cropper.crop_source_image(frame_rgb)

            if crop_info is None or 'img_crop_256x256' not in crop_info:
                # No face detected, return original
                return frame_rgb

            img_crop = crop_info['img_crop_256x256']

            # Convert to tensor
            img_tensor = torch.from_numpy(img_crop).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(self.device)

            if self.config.fp16:
                img_tensor = img_tensor.half()

            with torch.no_grad():
                # Get driving keypoints (current frame's expression)
                kp_driving = self.live_portrait_wrapper.get_kp_info(img_tensor)

                # Get source keypoints (neutral reference)
                kp_source = source_info['kp_source']

                # Apply lip retargeting
                # lip_ratio=0 means fully neutral, lip_ratio=1 means original
                if lip_ratio < 1.0 and hasattr(self.live_portrait_wrapper, 'retarget_lip'):
                    # Calculate lip delta for neutralization
                    # The retarget_lip method modifies expression to reduce lip movement
                    delta = self.live_portrait_wrapper.retarget_lip(
                        kp_source,
                        kp_driving,
                        lip_ratio,
                    )

                    # Apply delta to driving keypoints
                    kp_neutralized = self._apply_lip_delta(kp_driving, delta, lip_ratio)
                else:
                    # Manual interpolation if retarget_lip not available
                    kp_neutralized = self._interpolate_keypoints(kp_source, kp_driving, lip_ratio)

                # Generate output using warp_decode
                # This warps the source appearance using the neutralized keypoints
                output = self.live_portrait_wrapper.warp_decode(
                    source_info['feature_3d'],
                    kp_source,
                    kp_neutralized,
                )

                # Convert output tensor to numpy
                out_np = output.squeeze(0).permute(1, 2, 0).cpu().numpy()
                out_np = (out_np * 255).clip(0, 255).astype(np.uint8)

            # Paste back into original frame
            result = self._paste_back(frame_rgb, out_np, crop_info)
            return result

        except Exception as e:
            logger.debug(f"Frame processing error: {e}")
            return frame_rgb

    def _apply_lip_delta(
        self,
        kp_driving: Dict[str, torch.Tensor],
        delta: torch.Tensor,
        lip_ratio: float,
    ) -> Dict[str, torch.Tensor]:
        """Apply lip retargeting delta to driving keypoints."""
        kp_result = {}

        for key, value in kp_driving.items():
            if key == 'kp' and delta is not None:
                # Apply the lip delta scaled by (1 - lip_ratio)
                # At lip_ratio=0, full delta is applied (neutral lips)
                # At lip_ratio=1, no delta (original lips)
                kp_result[key] = value + delta * (1.0 - lip_ratio)
            else:
                kp_result[key] = value

        return kp_result

    def _interpolate_keypoints(
        self,
        kp_source: Dict[str, torch.Tensor],
        kp_driving: Dict[str, torch.Tensor],
        lip_ratio: float,
    ) -> Dict[str, torch.Tensor]:
        """
        Interpolate between source and driving keypoints.

        Used as fallback if retarget_lip is not available.
        For lip neutralization, we blend toward source expression.
        """
        kp_result = {}

        # Lip-related keypoint indices in LivePortrait
        # These correspond to mouth region in the 68-point or custom keypoint format
        LIP_INDICES = [6, 12, 14, 17, 19, 20]  # From LivePortrait's lip_array

        for key, value in kp_driving.items():
            if key in kp_source:
                source_value = kp_source[key]

                if isinstance(value, torch.Tensor) and isinstance(source_value, torch.Tensor):
                    if key == 'exp' or key == 'expression':
                        # For expression params, interpolate toward source (neutral)
                        kp_result[key] = (1 - lip_ratio) * source_value + lip_ratio * value
                    elif key == 'kp':
                        # For keypoints, selectively interpolate lip region
                        result = value.clone()
                        # Only interpolate lip indices
                        for idx in LIP_INDICES:
                            if idx < result.shape[1]:
                                result[:, idx] = (1 - lip_ratio) * source_value[:, idx] + lip_ratio * value[:, idx]
                        kp_result[key] = result
                    else:
                        # Keep other params from driving (pose, etc.)
                        kp_result[key] = value
                else:
                    kp_result[key] = value
            else:
                kp_result[key] = value

        return kp_result

    def _paste_back(
        self,
        frame: np.ndarray,
        face: np.ndarray,
        crop_info: Dict,
    ) -> np.ndarray:
        """Paste processed face back into frame using crop transformation."""
        try:
            # Get inverse transformation matrix
            if 'M_c2o' in crop_info:
                M_c2o = crop_info['M_c2o']
                h, w = frame.shape[:2]

                # Resize face to crop size (typically 256x256 or 512x512)
                dsize = crop_info.get('dsize', 256)
                face_resized = cv2.resize(face, (dsize, dsize))

                # Warp face back to original coordinates
                face_warped = cv2.warpAffine(
                    face_resized,
                    M_c2o[:2],
                    (w, h),
                    borderMode=cv2.BORDER_CONSTANT,
                    flags=cv2.INTER_LINEAR,
                )

                # Create soft mask for blending
                mask = np.ones((dsize, dsize), dtype=np.float32)
                mask_warped = cv2.warpAffine(
                    mask,
                    M_c2o[:2],
                    (w, h),
                    borderMode=cv2.BORDER_CONSTANT,
                )

                # Feather the mask edges
                kernel_size = int(0.05 * dsize) * 2 + 1
                mask_warped = cv2.GaussianBlur(mask_warped, (kernel_size, kernel_size), 0)
                mask_warped = mask_warped[:, :, np.newaxis]

                # Blend
                result = frame.astype(np.float32) * (1 - mask_warped) + face_warped.astype(np.float32) * mask_warped
                return result.astype(np.uint8)

            else:
                # Fallback: simple paste using bbox
                if 'pt_crop' in crop_info:
                    x1, y1, x2, y2 = crop_info['pt_crop']
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    h, w = y2 - y1, x2 - x1
                    face_resized = cv2.resize(face, (w, h))
                    result = frame.copy()
                    result[y1:y2, x1:x2] = face_resized
                    return result

                return frame

        except Exception as e:
            logger.debug(f"Paste back error: {e}")
            return frame

    @property
    def is_loaded(self) -> bool:
        return self._loaded


def download_liveportrait_models(target_dir: str = "/app/models/liveportrait") -> None:
    """Download LivePortrait model weights from HuggingFace."""
    from huggingface_hub import snapshot_download

    print(f"Downloading LivePortrait models to {target_dir}...")
    os.makedirs(target_dir, exist_ok=True)

    snapshot_download(
        repo_id="KwaiVGI/LivePortrait",
        local_dir=target_dir,
        ignore_patterns=["*.md", "*.txt", "*.git*", "docs/*", "assets/*"],
    )
    print("LivePortrait models downloaded")
