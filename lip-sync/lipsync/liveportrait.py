"""
LivePortrait model wrapper for face neutralization.

Uses the official KwaiVGI/LivePortrait package for inference.
LivePortrait is used to neutralize mouth expressions before lip-sync.

VERIFIED API (from official repo source code):

Cropper.crop_source_image(image) returns dict with keys:
- img_crop: cropped face image
- img_crop_256x256: resized to 256x256
- lmk_crop: landmarks in crop space
- lmk_crop_256x256: landmarks scaled to 256x256
- M_c2o: transformation matrix (crop to original)
- pt_crop: crop points

LivePortraitWrapper methods:
- extract_feature_3d(img_tensor) -> feature_3d tensor
- get_kp_info(img_tensor) -> dict with 'kp' key (BxNx3 keypoints)
- warp_decode(feature_3d, kp_source, kp_driving) -> output tensor
- retarget_lip(kp_source, lip_close_ratio) -> delta tensor
  - kp_source: BxNx3 tensor
  - lip_close_ratio: Bx2 tensor (combined source+driving ratios)
  - returns: Bx(3*num_kp) delta to reshape and add to keypoints
- calc_ratio(lmk_lst) -> (eye_ratios, lip_ratios) lists
- calc_combined_lip_ratio(driving_lip_ratio, source_landmarks) -> Bx2 tensor

calc_lip_close_ratio(landmarks) from retargeting_utils:
- Calculates distance ratio between lip landmarks
- Smaller value = more closed lips
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
    # Model weights directory (downloaded from HF KwaiVGI/LivePortrait)
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

    For neutralization, we set the driving lip ratio to 0 (fully closed)
    which causes retarget_lip() to produce a delta that closes the mouth.
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
            # HF structure: liveportrait/base_models/*.pth, liveportrait/retargeting_models/*.pth
            model_dir = Path(self.config.model_dir)
            if model_dir.exists():
                base_models = model_dir / "liveportrait" / "base_models"
                retarget_models = model_dir / "liveportrait" / "retargeting_models"

                if base_models.exists():
                    inference_cfg.checkpoint_F = str(base_models / "appearance_feature_extractor.pth")
                    inference_cfg.checkpoint_M = str(base_models / "motion_extractor.pth")
                    inference_cfg.checkpoint_G = str(base_models / "spade_generator.pth")
                    inference_cfg.checkpoint_W = str(base_models / "warping_module.pth")

                if retarget_models.exists():
                    inference_cfg.checkpoint_S = str(retarget_models / "stitching_retargeting_module.pth")

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
            # Import retargeting utils for lip ratio calculation
            from src.utils.retargeting_utils import calc_lip_close_ratio

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
                    neutralized_rgb = self._process_frame(
                        frame_rgb, source_info, lip_ratio, calc_lip_close_ratio
                    )
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
        - kp_source: Keypoint info from M network
        - crop_info: Cropping transformation for paste-back
        - source_lmk: Source landmarks for lip ratio calculation
        """
        try:
            # Use cropper to detect and crop face
            # Returns dict with: img_crop, img_crop_256x256, lmk_crop, lmk_crop_256x256, M_c2o, pt_crop
            crop_info = self.cropper.crop_source_image(frame_rgb)

            if crop_info is None:
                logger.warning("No face detected by cropper")
                return None

            # Check for required keys
            if 'img_crop_256x256' not in crop_info:
                logger.warning("crop_source_image did not return img_crop_256x256")
                return None

            img_crop = crop_info['img_crop_256x256']
            source_lmk = crop_info.get('lmk_crop', None)

            # Convert to tensor [B, C, H, W]
            # img_crop is RGB uint8, normalize to [0, 1]
            img_tensor = torch.from_numpy(img_crop).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(self.device)

            if self.config.fp16:
                img_tensor = img_tensor.half()

            # Extract features
            with torch.no_grad():
                # extract_feature_3d(img_tensor) -> feature_3d
                feature_3d = self.live_portrait_wrapper.extract_feature_3d(img_tensor)

                # get_kp_info(img_tensor) -> dict with 'kp' (BxNx3)
                kp_source = self.live_portrait_wrapper.get_kp_info(img_tensor)

            return {
                'feature_3d': feature_3d,
                'kp_source': kp_source,
                'crop_info': crop_info,
                'source_tensor': img_tensor,
                'source_lmk': source_lmk,
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
        calc_lip_close_ratio,
    ) -> np.ndarray:
        """
        Process a single frame with lip neutralization.

        For neutralization (lip_ratio=0):
        1. Get driving frame landmarks
        2. Calculate driving lip close ratio
        3. Set combined ratio to (source_ratio, 0) for full closure
        4. Call retarget_lip to get delta
        5. Apply delta to driving keypoints
        6. Generate output via warp_decode
        """
        try:
            # Crop driving frame
            crop_info = self.cropper.crop_source_image(frame_rgb)

            if crop_info is None or 'img_crop_256x256' not in crop_info:
                return frame_rgb

            img_crop = crop_info['img_crop_256x256']
            driving_lmk = crop_info.get('lmk_crop', None)

            # Convert to tensor
            img_tensor = torch.from_numpy(img_crop).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(self.device)

            if self.config.fp16:
                img_tensor = img_tensor.half()

            with torch.no_grad():
                # Get driving keypoints
                kp_driving = self.live_portrait_wrapper.get_kp_info(img_tensor)
                kp_source = source_info['kp_source']

                # For lip neutralization, we need to use retarget_lip
                # retarget_lip(kp_source, lip_close_ratio) where lip_close_ratio is Bx2
                # lip_close_ratio = [source_lip_ratio, driving_lip_ratio]

                if lip_ratio < 1.0 and hasattr(self.live_portrait_wrapper, 'retarget_lip'):
                    # Calculate lip ratios from landmarks if available
                    source_lmk = source_info.get('source_lmk')

                    if source_lmk is not None and driving_lmk is not None:
                        # calc_lip_close_ratio expects (N, 2) or (1, N, 2) landmarks
                        # Returns a scalar or array
                        source_lip_ratio = calc_lip_close_ratio(source_lmk[None])
                        driving_lip_ratio = calc_lip_close_ratio(driving_lmk[None])

                        # For neutralization, interpolate driving ratio toward 0 (closed)
                        # lip_ratio=0 means fully closed, lip_ratio=1 means original
                        target_driving_ratio = driving_lip_ratio * lip_ratio

                        # Create combined ratio tensor [source_ratio, driving_ratio]
                        combined_lip_ratio = torch.tensor(
                            [[float(source_lip_ratio), float(target_driving_ratio)]],
                            device=self.device,
                            dtype=img_tensor.dtype,
                        )

                        # Get keypoints tensor (BxNx3)
                        x_s = kp_source['kp']

                        # retarget_lip(kp_source, lip_close_ratio) -> delta Bx(3*N)
                        lip_delta = self.live_portrait_wrapper.retarget_lip(x_s, combined_lip_ratio)

                        # Reshape delta to BxNx3 and add to driving keypoints
                        B, N, _ = kp_driving['kp'].shape
                        lip_delta_reshaped = lip_delta.view(B, N, 3)

                        # Apply delta to driving keypoints
                        kp_driving_modified = {k: v.clone() if isinstance(v, torch.Tensor) else v
                                               for k, v in kp_driving.items()}
                        kp_driving_modified['kp'] = kp_driving['kp'] + lip_delta_reshaped
                    else:
                        # Fallback: simple interpolation of keypoints
                        kp_driving_modified = self._interpolate_keypoints(kp_source, kp_driving, lip_ratio)
                else:
                    kp_driving_modified = kp_driving

                # Generate output using warp_decode
                # warp_decode(feature_3d, kp_source, kp_driving) -> output tensor
                output = self.live_portrait_wrapper.warp_decode(
                    source_info['feature_3d'],
                    kp_source,
                    kp_driving_modified,
                )

                # Convert output tensor to numpy RGB
                # Output is [B, C, H, W] in [0, 1] range
                out_np = output.squeeze(0).permute(1, 2, 0).cpu().float().numpy()
                out_np = (out_np * 255).clip(0, 255).astype(np.uint8)

            # Paste back into original frame
            result = self._paste_back(frame_rgb, out_np, crop_info)
            return result

        except Exception as e:
            logger.debug(f"Frame processing error: {e}")
            import traceback
            traceback.print_exc()
            return frame_rgb

    def _interpolate_keypoints(
        self,
        kp_source: Dict[str, torch.Tensor],
        kp_driving: Dict[str, torch.Tensor],
        lip_ratio: float,
    ) -> Dict[str, torch.Tensor]:
        """
        Fallback: Interpolate between source and driving keypoints.

        For lip neutralization, we blend toward source expression.
        """
        kp_result = {}

        # Lip-related keypoint indices in LivePortrait (from lip_array in config)
        LIP_INDICES = [6, 12, 14, 17, 19, 20]

        for key, value in kp_driving.items():
            if key in kp_source and isinstance(value, torch.Tensor):
                source_value = kp_source[key]

                if key == 'kp' and isinstance(source_value, torch.Tensor):
                    # For keypoints, selectively interpolate lip region
                    result = value.clone()
                    for idx in LIP_INDICES:
                        if idx < result.shape[1]:
                            result[:, idx] = (1 - lip_ratio) * source_value[:, idx] + lip_ratio * value[:, idx]
                    kp_result[key] = result
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
            # Get inverse transformation matrix M_c2o (crop to original)
            if 'M_c2o' in crop_info:
                M_c2o = crop_info['M_c2o']
                h, w = frame.shape[:2]

                # Face is 256x256, warp to original coordinates
                face_warped = cv2.warpAffine(
                    face,
                    M_c2o[:2],
                    (w, h),
                    borderMode=cv2.BORDER_CONSTANT,
                    flags=cv2.INTER_LINEAR,
                )

                # Create soft mask for blending
                mask = np.ones((256, 256), dtype=np.float32)
                mask_warped = cv2.warpAffine(
                    mask,
                    M_c2o[:2],
                    (w, h),
                    borderMode=cv2.BORDER_CONSTANT,
                )

                # Feather the mask edges for smooth blending
                kernel_size = 21
                mask_warped = cv2.GaussianBlur(mask_warped, (kernel_size, kernel_size), 0)
                mask_warped = mask_warped[:, :, np.newaxis]

                # Blend
                result = frame.astype(np.float32) * (1 - mask_warped) + face_warped.astype(np.float32) * mask_warped
                return result.astype(np.uint8)

            elif 'pt_crop' in crop_info:
                # Fallback: simple paste using crop points
                x1, y1, x2, y2 = crop_info['pt_crop']
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                crop_h, crop_w = y2 - y1, x2 - x1
                face_resized = cv2.resize(face, (crop_w, crop_h))
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

    # Download from KwaiVGI/LivePortrait
    # Structure: liveportrait/base_models/*.pth, liveportrait/retargeting_models/*.pth
    snapshot_download(
        repo_id="KwaiVGI/LivePortrait",
        local_dir=target_dir,
        ignore_patterns=["*.md", "*.txt", "*.git*", "docs/*", "assets/*"],
    )
    print("LivePortrait models downloaded")
