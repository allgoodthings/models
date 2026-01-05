"""
Multi-face compositor for lip-sync pipeline.

Handles blending processed face regions back into the original video
with feathered masks and temporal consistency.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict
from concurrent.futures import ThreadPoolExecutor


@dataclass
class FaceRegion:
    """Processed face region to composite."""
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    frames: List[np.ndarray]  # Processed face frames
    start_frame: int
    end_frame: int
    character_id: Optional[str] = None


class FaceCompositor:
    """
    Composite processed face regions back into original video.

    Supports multiple faces with feathered elliptical masks
    and temporal blending at segment boundaries.
    """

    def __init__(
        self,
        feather_radius: int = 15,
        boundary_blend_frames: int = 5,
    ):
        """
        Initialize compositor.

        Args:
            feather_radius: Radius for mask feathering
            boundary_blend_frames: Frames to blend at segment start/end
        """
        self.feather_radius = feather_radius
        self.boundary_blend_frames = boundary_blend_frames

    def create_elliptical_mask(
        self,
        size: Tuple[int, int],
        feather: Optional[int] = None,
    ) -> np.ndarray:
        """
        Create feathered elliptical mask.

        Args:
            size: (height, width) of mask
            feather: Feather radius (defaults to self.feather_radius)

        Returns:
            Float32 mask with values 0-1
        """
        h, w = size
        feather = feather or self.feather_radius

        mask = np.zeros((h, w), dtype=np.float32)
        center = (w // 2, h // 2)

        # Ellipse axes with feather margin
        axes = (max(1, w // 2 - feather), max(1, h // 2 - feather))

        cv2.ellipse(mask, center, axes, 0, 0, 360, 1.0, -1)

        # Apply Gaussian blur for feathering
        if feather > 0:
            ksize = feather * 2 + 1
            mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)

        return mask

    def create_face_mask(
        self,
        landmarks: Optional[np.ndarray],
        shape: Tuple[int, int],
        feather: Optional[int] = None,
    ) -> np.ndarray:
        """
        Create mask from face landmarks.

        Args:
            landmarks: Face landmarks (468 points)
            shape: (height, width) of mask
            feather: Feather radius

        Returns:
            Float32 mask with values 0-1
        """
        feather = feather or self.feather_radius
        mask = np.zeros(shape[:2], dtype=np.float32)

        if landmarks is not None and len(landmarks) > 0:
            hull = cv2.convexHull(landmarks.astype(np.int32))
            cv2.fillConvexPoly(mask, hull, 1.0)

            if feather > 0:
                ksize = feather * 2 + 1
                mask = cv2.GaussianBlur(mask, (ksize, ksize), 0)

        return mask

    def composite_face(
        self,
        original: np.ndarray,
        processed: np.ndarray,
        bbox: Tuple[int, int, int, int],
        alpha: float = 1.0,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Blend processed face into original frame.

        Args:
            original: Original frame (BGR)
            processed: Processed face region (BGR)
            bbox: (x, y, w, h) position in original
            alpha: Blend strength 0-1
            mask: Optional custom mask (same size as processed)

        Returns:
            Composited frame
        """
        x, y, w, h = bbox
        oh, ow = original.shape[:2]

        # Clamp bbox to frame bounds
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(ow, x + w)
        y2 = min(oh, y + h)

        if x2 <= x1 or y2 <= y1:
            return original

        # Resize processed face to bbox size
        proc_h, proc_w = processed.shape[:2]
        if proc_w != w or proc_h != h:
            processed = cv2.resize(processed, (w, h), interpolation=cv2.INTER_LINEAR)

        # Crop processed to valid region
        proc_x1 = x1 - x
        proc_y1 = y1 - y
        proc_x2 = x2 - x
        proc_y2 = y2 - y

        processed_crop = processed[proc_y1:proc_y2, proc_x1:proc_x2]

        # Create or resize mask
        if mask is None:
            mask = self.create_elliptical_mask((h, w))

        if mask.shape[0] != h or mask.shape[1] != w:
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)

        mask_crop = mask[proc_y1:proc_y2, proc_x1:proc_x2]

        # Apply alpha
        mask_crop = mask_crop * alpha

        # Blend
        result = original.copy()
        roi = result[y1:y2, x1:x2].astype(np.float32)
        proc_float = processed_crop.astype(np.float32)
        mask_3ch = mask_crop[:, :, np.newaxis]

        blended = roi * (1 - mask_3ch) + proc_float * mask_3ch
        result[y1:y2, x1:x2] = blended.astype(np.uint8)

        return result

    def compute_boundary_alpha(
        self,
        frame_idx: int,
        start_frame: int,
        end_frame: int,
    ) -> float:
        """
        Compute alpha for temporal blending at boundaries.

        Args:
            frame_idx: Current frame index
            start_frame: First frame of processed region
            end_frame: Last frame of processed region

        Returns:
            Alpha value 0-1
        """
        blend = self.boundary_blend_frames

        if blend <= 0:
            return 1.0

        # Fade in at start
        if frame_idx < start_frame + blend:
            return (frame_idx - start_frame) / blend

        # Fade out at end
        if frame_idx > end_frame - blend:
            return (end_frame - frame_idx) / blend

        return 1.0

    def composite_video(
        self,
        original_frames: List[np.ndarray],
        face_regions: List[FaceRegion],
        parallel: bool = True,
    ) -> List[np.ndarray]:
        """
        Composite multiple face regions into video frames.

        Args:
            original_frames: List of original video frames
            face_regions: List of FaceRegion objects to composite
            parallel: Use parallel processing

        Returns:
            List of composited frames
        """
        if not face_regions:
            return original_frames

        result_frames = [f.copy() for f in original_frames]

        def process_frame(frame_idx: int) -> np.ndarray:
            frame = result_frames[frame_idx].copy()

            for region in face_regions:
                if region.start_frame <= frame_idx < region.end_frame:
                    local_idx = frame_idx - region.start_frame

                    if 0 <= local_idx < len(region.frames):
                        alpha = self.compute_boundary_alpha(
                            frame_idx, region.start_frame, region.end_frame
                        )
                        frame = self.composite_face(
                            frame,
                            region.frames[local_idx],
                            region.bbox,
                            alpha=alpha,
                        )

            return frame

        if parallel:
            with ThreadPoolExecutor() as executor:
                result_frames = list(executor.map(
                    process_frame,
                    range(len(original_frames))
                ))
        else:
            result_frames = [process_frame(i) for i in range(len(original_frames))]

        return result_frames

    def composite_video_from_paths(
        self,
        original_path: str,
        face_regions: List[FaceRegion],
        output_path: str,
    ) -> str:
        """
        Composite face regions into video file.

        Args:
            original_path: Path to original video
            face_regions: Face regions to composite
            output_path: Path for output video

        Returns:
            Path to output video
        """
        cap = cv2.VideoCapture(original_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        for frame_idx in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break

            # Composite all face regions for this frame
            for region in face_regions:
                if region.start_frame <= frame_idx < region.end_frame:
                    local_idx = frame_idx - region.start_frame

                    if 0 <= local_idx < len(region.frames):
                        alpha = self.compute_boundary_alpha(
                            frame_idx, region.start_frame, region.end_frame
                        )
                        frame = self.composite_face(
                            frame,
                            region.frames[local_idx],
                            region.bbox,
                            alpha=alpha,
                        )

            out.write(frame)

        cap.release()
        out.release()

        return output_path


def blend_histograms(
    source: np.ndarray,
    reference: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Match source histogram to reference for color consistency.

    Args:
        source: Source image (BGR)
        reference: Reference image (BGR)
        mask: Optional mask for region-based matching

    Returns:
        Color-matched source image
    """
    from skimage import exposure

    if source.shape != reference.shape:
        return source

    try:
        if mask is not None:
            # Match only in masked region
            result = source.copy()
            mask_bool = mask > 0.5

            for c in range(3):
                src_channel = source[:, :, c][mask_bool]
                ref_channel = reference[:, :, c][mask_bool]

                if len(src_channel) > 0 and len(ref_channel) > 0:
                    matched = exposure.match_histograms(src_channel, ref_channel)
                    result[:, :, c][mask_bool] = matched

            return result
        else:
            return exposure.match_histograms(source, reference, channel_axis=2)
    except Exception:
        return source
