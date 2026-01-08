"""
InsightFace-based face detection and identity matching.

Provides local face detection with:
- Bounding boxes and detection confidence
- 512-dimensional face embeddings for identity matching
- Head pose estimation (pitch, yaw, roll)
- 68-point 3D facial landmarks
- Syncability analysis for lip-sync decisions
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from insightface.app import FaceAnalysis

logger = logging.getLogger(__name__)


@dataclass
class FaceAnalysisResult:
    """Complete face analysis result from InsightFace."""

    character_id: Optional[str]  # Matched character ID or None
    bbox: Tuple[int, int, int, int]  # (x, y, width, height)
    confidence: float  # Detection confidence 0-1
    embedding: np.ndarray  # 512-dim face embedding
    head_pose: Tuple[float, float, float]  # (pitch, yaw, roll) in degrees
    landmarks_68: np.ndarray  # 68-point 3D landmarks
    mouth_landmarks: np.ndarray  # 20 mouth points (indices 48-67)
    syncable: bool  # Whether face is suitable for lip-sync
    sync_quality: float  # 0-1, recommended blend strength
    skip_reason: Optional[str]  # Reason if not syncable


class InsightFaceDetector:
    """
    Local face detection and analysis using InsightFace.

    Features:
    - Face detection with bounding boxes
    - Face embedding extraction for identity matching
    - Head pose estimation
    - 68-point landmark detection
    - Syncability analysis based on pose and quality
    """

    # Syncability thresholds
    MAX_YAW_FOR_SYNC = 45.0  # Degrees - beyond this is profile view
    REDUCED_QUALITY_YAW = 25.0  # Degrees - reduce quality beyond this
    MIN_FACE_WIDTH = 64  # Pixels - minimum face size for sync
    MIN_DETECTION_SCORE = 0.5  # Minimum confidence to consider face
    MIN_DETECTION_FOR_SYNC = 0.7  # Below this, reduce sync quality

    # Mouth landmark indices in 68-point model
    MOUTH_LANDMARK_START = 48
    MOUTH_LANDMARK_END = 68

    def __init__(
        self,
        model_name: str = "buffalo_l",
        device: str = "cuda",
    ):
        """
        Initialize InsightFace detector.

        Args:
            model_name: InsightFace model name (default: buffalo_l)
            device: Device to use ("cuda" or "cpu")
        """
        self.model_name = model_name
        self.device = device
        self.app: Optional[FaceAnalysis] = None
        self.reference_embeddings: Dict[str, np.ndarray] = {}
        self._is_loaded = False

    def load(self) -> None:
        """
        Load InsightFace models.

        Should be called at server startup.
        """
        if self._is_loaded:
            logger.info("InsightFace already loaded, skipping")
            return

        logger.info(f"Loading InsightFace model: {self.model_name}")
        logger.info(f"Device: {self.device}")

        # Set providers based on device
        if self.device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        logger.info(f"ONNX providers: {providers}")

        self.app = FaceAnalysis(
            name=self.model_name,
            providers=providers,
        )

        # ctx_id: 0 for first GPU, -1 for CPU
        ctx_id = 0 if self.device == "cuda" else -1
        logger.info(f"Preparing FaceAnalysis with ctx_id={ctx_id}")

        self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))

        self._is_loaded = True
        logger.info("InsightFace model loaded successfully")

    @property
    def is_loaded(self) -> bool:
        """Check if models are loaded."""
        return self._is_loaded

    def load_reference(
        self,
        character_id: str,
        image: np.ndarray,
    ) -> bool:
        """
        Extract and store embedding from reference image.

        Args:
            character_id: Unique identifier for the character
            image: BGR image (numpy array) containing the reference face

        Returns:
            True if reference was loaded successfully
        """
        if not self._is_loaded:
            raise RuntimeError("InsightFace not loaded. Call load() first.")

        logger.debug(f"Loading reference for character: {character_id}")

        # Detect faces in reference image
        faces = self.app.get(image)

        if not faces:
            logger.warning(f"No face found in reference image for {character_id}")
            return False

        if len(faces) > 1:
            logger.warning(
                f"Multiple faces ({len(faces)}) in reference for {character_id}, "
                "using largest face"
            )
            # Use largest face (by bbox area)
            faces = sorted(
                faces,
                key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
                reverse=True,
            )

        face = faces[0]
        self.reference_embeddings[character_id] = face.embedding

        logger.info(
            f"Loaded reference embedding for {character_id} "
            f"(detection score: {face.det_score:.3f})"
        )
        return True

    def clear_references(self) -> None:
        """Clear all stored reference embeddings."""
        self.reference_embeddings.clear()
        logger.debug("Cleared all reference embeddings")

    def detect_faces(
        self,
        frame: np.ndarray,
        similarity_threshold: float = 0.5,
    ) -> List[FaceAnalysisResult]:
        """
        Detect all faces in frame with full metadata.

        Args:
            frame: BGR image (numpy array)
            similarity_threshold: Cosine similarity threshold for matching

        Returns:
            List of FaceAnalysisResult with detection and analysis data
        """
        if not self._is_loaded:
            raise RuntimeError("InsightFace not loaded. Call load() first.")

        # Run face detection
        faces = self.app.get(frame)

        if not faces:
            return []

        results = []
        unmatched_count = 0

        for face in faces:
            # Convert bbox from (x1, y1, x2, y2) to (x, y, width, height)
            x1, y1, x2, y2 = face.bbox.astype(int)
            bbox = (int(x1), int(y1), int(x2 - x1), int(y2 - y1))

            # Extract head pose
            if hasattr(face, "pose") and face.pose is not None:
                pitch, yaw, roll = face.pose
                head_pose = (float(pitch), float(yaw), float(roll))
            else:
                head_pose = (0.0, 0.0, 0.0)

            # Extract landmarks
            if hasattr(face, "landmark_3d_68") and face.landmark_3d_68 is not None:
                landmarks_68 = face.landmark_3d_68
                mouth_landmarks = landmarks_68[
                    self.MOUTH_LANDMARK_START : self.MOUTH_LANDMARK_END
                ]
            else:
                landmarks_68 = np.array([])
                mouth_landmarks = np.array([])

            # Match to reference embeddings
            character_id = self._match_embedding(
                face.embedding, similarity_threshold
            )

            # Auto-assign ID if no match
            if character_id is None:
                unmatched_count += 1
                character_id = f"face_{unmatched_count}"

            # Analyze syncability
            syncable, sync_quality, skip_reason = self._analyze_syncability(
                bbox=bbox,
                head_pose=head_pose,
                det_score=face.det_score,
            )

            results.append(
                FaceAnalysisResult(
                    character_id=character_id,
                    bbox=bbox,
                    confidence=float(face.det_score),
                    embedding=face.embedding,
                    head_pose=head_pose,
                    landmarks_68=landmarks_68,
                    mouth_landmarks=mouth_landmarks,
                    syncable=syncable,
                    sync_quality=sync_quality,
                    skip_reason=skip_reason,
                )
            )

        # Sort by bbox x position (left to right)
        results.sort(key=lambda r: r.bbox[0])

        return results

    def _match_embedding(
        self,
        embedding: np.ndarray,
        threshold: float,
    ) -> Optional[str]:
        """
        Match embedding against stored references.

        Args:
            embedding: 512-dim face embedding
            threshold: Cosine similarity threshold

        Returns:
            Character ID of best match, or None if no match above threshold
        """
        if not self.reference_embeddings:
            return None

        best_match = None
        best_similarity = threshold

        for char_id, ref_embedding in self.reference_embeddings.items():
            # Cosine similarity
            similarity = np.dot(embedding, ref_embedding) / (
                np.linalg.norm(embedding) * np.linalg.norm(ref_embedding)
            )

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = char_id

        if best_match:
            logger.debug(
                f"Matched face to {best_match} (similarity: {best_similarity:.3f})"
            )

        return best_match

    def _analyze_syncability(
        self,
        bbox: Tuple[int, int, int, int],
        head_pose: Tuple[float, float, float],
        det_score: float,
    ) -> Tuple[bool, float, Optional[str]]:
        """
        Determine if face is suitable for lip-sync.

        Args:
            bbox: Face bounding box (x, y, width, height)
            head_pose: (pitch, yaw, roll) in degrees
            det_score: Detection confidence score

        Returns:
            Tuple of (syncable, sync_quality, skip_reason)
        """
        pitch, yaw, roll = head_pose
        _, _, width, _ = bbox

        # Hard failures - face cannot be synced
        if abs(yaw) > self.MAX_YAW_FOR_SYNC:
            return (False, 0.0, "profile_view")

        if width < self.MIN_FACE_WIDTH:
            return (False, 0.0, "face_too_small")

        if det_score < self.MIN_DETECTION_SCORE:
            return (False, 0.0, "low_detection_quality")

        # Quality adjustments
        quality = 1.0

        # Reduce quality for angled faces
        if abs(yaw) > self.REDUCED_QUALITY_YAW:
            quality *= 0.7

        # Reduce quality for lower confidence
        if det_score < self.MIN_DETECTION_FOR_SYNC:
            quality *= 0.8

        # Reduce quality for extreme pitch (looking up/down)
        if abs(pitch) > 30:
            quality *= 0.8

        return (True, quality, None)

    def compute_similarity(
        self,
        embedding1: np.ndarray,
        embedding2: np.ndarray,
    ) -> float:
        """
        Compute cosine similarity between two embeddings.

        Args:
            embedding1: First 512-dim embedding
            embedding2: Second 512-dim embedding

        Returns:
            Cosine similarity (-1 to 1, higher is more similar)
        """
        return float(
            np.dot(embedding1, embedding2)
            / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        )
