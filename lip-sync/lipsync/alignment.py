"""
Face alignment module for lip-sync pipeline.

Based on SieveSync's alignment implementation with modifications for:
- Target bbox support for multi-face processing
- Configurable face mesh instances
- Better error handling
"""

import logging
import os
import time
import numpy as np
import cv2
import mediapipe as mp
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger('lipsync.alignment')


@dataclass
class AlignmentMetadata:
    """Metadata from face alignment process."""
    fps: float
    frame_count: int
    transforms: List[Tuple[Optional[np.ndarray], Tuple[int, int, int], int, Optional[np.ndarray]]]
    avg_face_size: float


class FaceAligner:
    """
    Face alignment using MediaPipe FaceMesh.

    Supports optional target_bbox for multi-face scenarios where
    you want to process a specific face region.
    """

    def __init__(self, max_num_faces: int = 1):
        """
        Initialize face aligner.

        Args:
            max_num_faces: Maximum faces to detect. Set > 1 for multi-face.
        """
        logger.debug(f"Initializing FaceAligner with max_num_faces={max_num_faces}")
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_num_faces,
            min_detection_confidence=0.1,
            min_tracking_confidence=0.1,
            refine_landmarks=True,
        )
        logger.debug("FaceMesh initialized successfully")

    def get_landmarks(
        self,
        image: np.ndarray,
        target_bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> Optional[np.ndarray]:
        """
        Get facial landmarks from image.

        Args:
            image: BGR image
            target_bbox: Optional (x, y, w, h) to focus detection on specific region

        Returns:
            468 facial landmarks as (x, y) coordinates, or None if no face found
        """
        ih, iw = image.shape[:2]
        logger.debug(f"get_landmarks: image size={iw}x{ih}, target_bbox={target_bbox}")

        if target_bbox is not None:
            # Crop to target region with margin
            x, y, w, h = target_bbox
            margin = 0.3
            x1 = max(0, int(x - w * margin))
            y1 = max(0, int(y - h * margin))
            x2 = min(iw, int(x + w * (1 + margin)))
            y2 = min(ih, int(y + h * (1 + margin)))

            crop = image[y1:y2, x1:x2]
            logger.debug(f"  Cropped region: ({x1},{y1}) to ({x2},{y2}), crop size={crop.shape[1]}x{crop.shape[0]}")
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(crop_rgb)

            if not results.multi_face_landmarks:
                logger.debug("  No face landmarks found in cropped region")
                return None

            logger.debug(f"  Found {len(results.multi_face_landmarks)} face(s) in cropped region")

            # Find the face closest to center of target bbox
            target_center = np.array([x + w/2 - x1, y + h/2 - y1])
            best_landmarks = None
            best_distance = float('inf')

            crop_h, crop_w = crop.shape[:2]
            for face_landmarks in results.multi_face_landmarks:
                landmarks = np.array([
                    (int(lm.x * crop_w), int(lm.y * crop_h))
                    for lm in face_landmarks.landmark
                ])
                face_center = np.mean(landmarks, axis=0)
                distance = np.linalg.norm(face_center - target_center)

                if distance < best_distance:
                    best_distance = distance
                    best_landmarks = landmarks

            if best_landmarks is not None:
                # Translate back to original image coordinates
                best_landmarks[:, 0] += x1
                best_landmarks[:, 1] += y1
                logger.debug(f"  Selected face with distance={best_distance:.2f} from target center")
                return best_landmarks
            logger.debug("  No suitable face found near target bbox")
            return None
        else:
            # Original behavior - detect in full image
            logger.debug("  Detecting face in full image (no target_bbox)")
            results = self.face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if not results.multi_face_landmarks:
                logger.debug("  No face landmarks found in full image")
                return None
            landmarks = results.multi_face_landmarks[0].landmark
            logger.debug(f"  Found face with {len(landmarks)} landmarks")
            return np.array([(int(lm.x * iw), int(lm.y * ih)) for lm in landmarks])

    def get_transform_params(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Calculate quadrilateral for face alignment.

        Args:
            landmarks: 468 facial landmarks

        Returns:
            4x2 array of quad corners for perspective transform
        """
        # Eye landmarks
        left_eye = np.mean(landmarks[[33, 133]], axis=0)
        right_eye = np.mean(landmarks[[362, 263]], axis=0)
        eye_avg = (left_eye + right_eye) * 0.5
        eye_to_eye = right_eye - left_eye

        # Mouth landmarks
        mouth_avg = (landmarks[61] + landmarks[291]) * 0.5
        eye_to_mouth = mouth_avg - eye_avg

        # Calculate alignment quad
        x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
        x /= np.hypot(*x)
        x *= max(np.hypot(*eye_to_eye) * 2.5, np.hypot(*eye_to_mouth) * 2)
        y = np.flipud(x) * [-1, 1]
        c = eye_avg + eye_to_mouth * 0.1

        return np.stack([c - x - y, c - x + y, c + x + y, c + x - y])

    def align_face(
        self,
        image: np.ndarray,
        quad: np.ndarray,
        output_size: int = 512,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Align face using perspective transform.

        Args:
            image: Source image
            quad: Quadrilateral corners
            output_size: Output image size

        Returns:
            Aligned image and transformation matrix
        """
        dst = np.array([
            (0, 0),
            (0, output_size - 1),
            (output_size - 1, output_size - 1),
            (output_size - 1, 0),
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(quad.astype(np.float32), dst)
        aligned = cv2.warpPerspective(
            image, M, (output_size, output_size),
            flags=cv2.INTER_LINEAR,
        )
        return aligned, M

    def unalign_face(
        self,
        aligned: np.ndarray,
        M: np.ndarray,
        original_shape: Tuple[int, int, int],
    ) -> np.ndarray:
        """
        Reverse the alignment transform.

        Args:
            aligned: Aligned face image
            M: Original transformation matrix
            original_shape: Shape of original image (h, w, c)

        Returns:
            Unaligned image in original coordinate space
        """
        h, w = original_shape[:2]
        M_inv = np.linalg.inv(M)
        return cv2.warpPerspective(aligned, M_inv, (w, h), flags=cv2.INTER_LINEAR)


def align_video(
    input_path: str,
    output_path: str,
    target_bbox: Optional[Tuple[int, int, int, int]] = None,
    smooth_frames: int = 5,
    output_size: int = 512,
) -> AlignmentMetadata:
    """
    Align faces in video for lip-sync processing.

    Args:
        input_path: Path to input video
        output_path: Path for aligned output video
        target_bbox: Optional (x, y, w, h) to focus on specific face
        smooth_frames: Number of frames for temporal smoothing
        output_size: Size of output frames

    Returns:
        AlignmentMetadata with transformation info
    """
    logger.info("=" * 50)
    logger.info("ALIGN VIDEO - Starting")
    logger.info("=" * 50)
    logger.info(f"  Input: {input_path}")
    logger.info(f"  Output: {output_path}")
    logger.info(f"  Target bbox: {target_bbox}")
    logger.info(f"  Smooth frames: {smooth_frames}")
    logger.info(f"  Output size: {output_size}x{output_size}")

    start_time = time.time()

    aligner = FaceAligner(max_num_faces=5 if target_bbox else 1)

    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    logger.info(f"  Video info: {width}x{height} @ {fps:.2f}fps, {total_frames} frames")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (output_size, output_size))

    transforms: List[Tuple] = []
    quad_history = deque(maxlen=smooth_frames)
    last_valid_quad = None
    last_valid_landmarks = None
    landmark_buffer = deque(maxlen=5)

    no_face_count = 0
    frames_without_landmarks = 0
    face_areas: List[float] = []
    landmark_distance_threshold = 75

    # Process initial frames for stable start
    initial_frames = []
    initial_landmarks = []

    for _ in range(10):
        ret, frame = cap.read()
        if not ret:
            break
        initial_frames.append(frame)
        lm = aligner.get_landmarks(frame, target_bbox)
        if lm is not None:
            initial_landmarks.append(lm)
            if len(initial_landmarks) >= 3:
                break

    # Initialize with average landmarks
    if initial_landmarks:
        avg_landmarks = np.mean(initial_landmarks, axis=0)
        for i, frame in enumerate(initial_frames):
            quad = aligner.get_transform_params(avg_landmarks)
            quad_history.append(quad)
            last_valid_quad = np.mean(list(quad_history), axis=0)
            aligned_frame, M = aligner.align_face(frame, last_valid_quad, output_size)
            out.write(aligned_frame)
            transforms.append((M, frame.shape, i, avg_landmarks.copy()))

            face_hull = cv2.convexHull(avg_landmarks.astype(np.int32))
            face_area = cv2.contourArea(face_hull) / (width * height)
            face_areas.append(face_area)

        last_valid_landmarks = avg_landmarks

    # Process remaining frames
    for frame_num in range(len(initial_frames), total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        lm = aligner.get_landmarks(frame, target_bbox)

        if lm is not None:
            if last_valid_landmarks is not None:
                distance = np.mean(np.linalg.norm(lm - last_valid_landmarks, axis=1))
                if distance > landmark_distance_threshold:
                    # Large movement - reset
                    quad = aligner.get_transform_params(lm)
                    last_valid_quad = quad
                    last_valid_landmarks = lm
                else:
                    # Smooth transition
                    quad = aligner.get_transform_params(lm)
                    last_valid_quad = 0.8 * last_valid_quad + 0.2 * quad
                    last_valid_landmarks = 0.8 * last_valid_landmarks + 0.2 * lm
            else:
                quad = aligner.get_transform_params(lm)
                last_valid_quad = quad
                last_valid_landmarks = lm

            quad_history.append(last_valid_quad)
            no_face_count = 0

            face_hull = cv2.convexHull(lm.astype(np.int32))
            face_area = cv2.contourArea(face_hull) / (width * height)
            face_areas.append(face_area)
            landmark_buffer.append(lm)
        else:
            no_face_count += 1
            frames_without_landmarks += 1

            if no_face_count < 5 and landmark_buffer:
                lm = landmark_buffer[-1]
                quad = aligner.get_transform_params(lm)
                quad_history.append(quad)
                last_valid_quad = np.mean(list(quad_history), axis=0)

        if no_face_count < 15 and last_valid_quad is not None:
            aligned_frame, M = aligner.align_face(frame, last_valid_quad, output_size)
            out.write(aligned_frame)
            transforms.append((M, frame.shape, frame_num, last_valid_landmarks.copy() if last_valid_landmarks is not None else None))
        else:
            # Fallback - center crop
            aspect_ratio = width / height
            if aspect_ratio > 1:
                new_width, new_height = output_size, int(output_size / aspect_ratio)
            else:
                new_height, new_width = output_size, int(output_size * aspect_ratio)

            resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_AREA)
            canvas = np.zeros((output_size, output_size, 3), dtype=np.uint8)
            pad_x, pad_y = (output_size - new_width) // 2, (output_size - new_height) // 2
            canvas[pad_y:pad_y+new_height, pad_x:pad_x+new_width] = resized

            out.write(canvas)
            transforms.append((None, frame.shape, frame_num, None))

    cap.release()
    out.release()

    elapsed = time.time() - start_time
    logger.info(f"  Face detection stats:")
    logger.info(f"    Frames without landmarks: {frames_without_landmarks}/{total_frames} ({frames_without_landmarks/total_frames*100:.1f}%)")
    logger.info(f"    Elapsed time: {elapsed:.2f}s ({total_frames/elapsed:.1f} fps)")

    # Validate detection rate
    if total_frames > 0 and frames_without_landmarks / total_frames > 0.4:
        error_msg = f"Face detection failed in {frames_without_landmarks/total_frames:.1%} of frames"
        logger.error(error_msg)
        raise ValueError(error_msg)

    avg_face_size = np.mean(face_areas) * 100 if face_areas else 0

    logger.info(f"  Average face size: {avg_face_size:.2f}% of frame")
    logger.info(f"  Output saved to: {output_path}")
    logger.info("ALIGN VIDEO - Complete")

    return AlignmentMetadata(
        fps=fps,
        frame_count=total_frames,
        transforms=transforms,
        avg_face_size=avg_face_size,
    )


def unalign_video(
    aligned_path: str,
    source_path: str,
    metadata: AlignmentMetadata,
    output_path: str,
) -> None:
    """
    Reverse alignment and composite back to original video.

    Args:
        aligned_path: Path to aligned video
        source_path: Path to original source video
        metadata: Alignment metadata from align_video
        output_path: Path for output video
    """
    logger.info("=" * 50)
    logger.info("UNALIGN VIDEO - Starting")
    logger.info("=" * 50)
    logger.info(f"  Aligned: {aligned_path}")
    logger.info(f"  Source: {source_path}")
    logger.info(f"  Output: {output_path}")
    logger.info(f"  Frames to process: {metadata.frame_count}")

    start_time = time.time()

    from skimage import exposure

    aligned_cap = cv2.VideoCapture(aligned_path)
    source_cap = cv2.VideoCapture(source_path)

    fps = metadata.fps
    total_frames = metadata.frame_count

    _, first_frame = source_cap.read()
    height, width = first_frame.shape[:2]
    source_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    aligner = FaceAligner()

    def process_frame(args):
        i, aligned_frame, source_frame, M, original_shape, landmarks = args
        if M is not None and landmarks is not None:
            unaligned = aligner.unalign_face(aligned_frame, M, original_shape)
            unaligned = cv2.resize(unaligned, (source_frame.shape[1], source_frame.shape[0]))

            # Create face mask
            mask = np.zeros(source_frame.shape[:2], dtype=np.uint8)
            if len(landmarks) > 0:
                hull = cv2.convexHull(landmarks.astype(np.int32))
                cv2.fillConvexPoly(mask, hull, 255)
                mask = cv2.GaussianBlur(mask, (15, 15), 0)

            mask_float = mask.astype(np.float32) / 255.0
            mask_3ch = np.repeat(mask_float[:, :, np.newaxis], 3, axis=2)

            # Histogram matching for color consistency
            try:
                unaligned = exposure.match_histograms(unaligned, source_frame, channel_axis=2)
            except:
                pass

            result = source_frame * (1 - mask_3ch) + unaligned * mask_3ch
            return i, result.astype(np.uint8)
        else:
            return i, source_frame

    batch_size = 100
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
        for i in range(0, total_frames, batch_size):
            batch_end = min(i + batch_size, total_frames)
            futures = []

            for j in range(i, batch_end):
                ret_aligned, aligned_frame = aligned_cap.read()
                ret_source, source_frame = source_cap.read()
                if not ret_aligned or not ret_source:
                    break

                M, original_shape, frame_num, landmarks = metadata.transforms[j]
                future = executor.submit(
                    process_frame,
                    (frame_num, aligned_frame, source_frame, M, original_shape, landmarks)
                )
                futures.append(future)

            results = [f.result() for f in as_completed(futures)]
            results.sort(key=lambda x: x[0])

            for _, frame in results:
                out.write(frame)

    aligned_cap.release()
    source_cap.release()
    out.release()

    elapsed = time.time() - start_time
    logger.info(f"  Elapsed time: {elapsed:.2f}s ({total_frames/elapsed:.1f} fps)")
    logger.info(f"  Output saved to: {output_path}")
    logger.info("UNALIGN VIDEO - Complete")


def align_image(
    image: np.ndarray,
    target_bbox: Optional[Tuple[int, int, int, int]] = None,
    output_size: int = 512,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Align a single face in an image.

    Args:
        image: BGR image
        target_bbox: Optional region to focus on
        output_size: Size of output

    Returns:
        Tuple of (aligned_image, transform_matrix, landmarks) or (None, None, None)
    """
    logger.debug(f"align_image: image size={image.shape[1]}x{image.shape[0]}, target_bbox={target_bbox}, output_size={output_size}")

    aligner = FaceAligner(max_num_faces=5 if target_bbox else 1)
    landmarks = aligner.get_landmarks(image, target_bbox)

    if landmarks is None:
        logger.debug("align_image: No landmarks found, returning None")
        return None, None, None

    quad = aligner.get_transform_params(landmarks)
    aligned, M = aligner.align_face(image, quad, output_size)

    logger.debug(f"align_image: Aligned to {aligned.shape[1]}x{aligned.shape[0]}")
    return aligned, M, landmarks
