"""
Face reference caching for optimized lip-sync pipeline.

When processing multiple audio segments for the same face, we can cache:
1. VAE latent of reference face (once per face identity)
2. Face mask template (once per face)
3. Whisper audio features (once per audio clip)

This dramatically reduces preprocessing time since InsightFace already
provides consistent face crops - MuseTalk receives the same face identity
for all frames, making caching highly effective.
"""

import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Any, Tuple
import numpy as np
import torch

logger = logging.getLogger('lipsync.cache')


@dataclass
class FaceReference:
    """Cached reference data for a face identity."""
    # Face identity hash (from embedding or image hash)
    face_id: str
    # VAE latent of reference frame [1, 4, 32, 32]
    reference_latent: torch.Tensor
    # Face mask for inpainting [1, 1, 32, 32]
    face_mask: torch.Tensor
    # Original reference image for compositing
    reference_image: Optional[np.ndarray] = None
    # Face landmarks (68-point)
    landmarks: Optional[np.ndarray] = None
    # Creation timestamp
    created_at: float = field(default_factory=time.time)
    # Usage count
    use_count: int = 0


@dataclass
class AudioFeatures:
    """Cached audio features for an audio clip."""
    # Audio file hash
    audio_id: str
    # Whisper encoder features
    features: torch.Tensor
    # Audio duration in seconds
    duration: float
    # Target FPS for chunking
    fps: float
    # Pre-chunked features aligned to frames
    chunks: Optional[torch.Tensor] = None
    # Creation timestamp
    created_at: float = field(default_factory=time.time)


class FaceCache:
    """
    Cache for face references and audio features.

    Enables significant speedup when:
    - Same face appears across multiple segments
    - Same audio is lip-synced to multiple faces
    - Repeated processing of same face+audio combination

    Cache Keys:
    - Face: MD5 of flattened face image bytes
    - Audio: MD5 of audio file path + mtime
    """

    def __init__(self, max_faces: int = 32, max_audio: int = 16, device: str = "cuda"):
        """
        Initialize face cache.

        Args:
            max_faces: Maximum number of faces to cache
            max_audio: Maximum number of audio clips to cache
            device: Device for cached tensors
        """
        self.max_faces = max_faces
        self.max_audio = max_audio
        self.device = device

        self._faces: Dict[str, FaceReference] = {}
        self._audio: Dict[str, AudioFeatures] = {}

        # Stats
        self.face_hits = 0
        self.face_misses = 0
        self.audio_hits = 0
        self.audio_misses = 0

        logger.info(f"FaceCache initialized (max_faces={max_faces}, max_audio={max_audio})")

    def get_face_id(self, image: np.ndarray) -> str:
        """Generate unique ID for a face image."""
        # Use MD5 of downsampled image for fast hashing
        # Downsample to 64x64 for consistent hashing regardless of input size
        import cv2
        small = cv2.resize(image, (64, 64))
        return hashlib.md5(small.tobytes()).hexdigest()[:16]

    def get_audio_id(self, audio_path: str) -> str:
        """Generate unique ID for an audio file."""
        path = Path(audio_path)
        if path.exists():
            mtime = path.stat().st_mtime
            return hashlib.md5(f"{audio_path}:{mtime}".encode()).hexdigest()[:16]
        return hashlib.md5(audio_path.encode()).hexdigest()[:16]

    def get_face(self, image: np.ndarray) -> Optional[FaceReference]:
        """
        Get cached face reference if exists.

        Args:
            image: Face image (BGR, 256x256)

        Returns:
            FaceReference if cached, None otherwise
        """
        face_id = self.get_face_id(image)

        if face_id in self._faces:
            self.face_hits += 1
            ref = self._faces[face_id]
            ref.use_count += 1
            logger.debug(f"Face cache HIT: {face_id} (uses: {ref.use_count})")
            return ref

        self.face_misses += 1
        logger.debug(f"Face cache MISS: {face_id}")
        return None

    def put_face(
        self,
        image: np.ndarray,
        reference_latent: torch.Tensor,
        face_mask: torch.Tensor,
        landmarks: Optional[np.ndarray] = None,
    ) -> FaceReference:
        """
        Cache a face reference.

        Args:
            image: Face image (BGR, 256x256)
            reference_latent: VAE-encoded latent [1, 4, 32, 32]
            face_mask: Inpainting mask [1, 1, 32, 32]
            landmarks: 68-point landmarks

        Returns:
            Cached FaceReference
        """
        face_id = self.get_face_id(image)

        # Evict oldest if at capacity
        if len(self._faces) >= self.max_faces:
            self._evict_oldest_face()

        ref = FaceReference(
            face_id=face_id,
            reference_latent=reference_latent.to(self.device),
            face_mask=face_mask.to(self.device),
            reference_image=image.copy(),
            landmarks=landmarks.copy() if landmarks is not None else None,
        )

        self._faces[face_id] = ref
        logger.debug(f"Face cached: {face_id}")

        return ref

    def get_audio(self, audio_path: str) -> Optional[AudioFeatures]:
        """
        Get cached audio features if exists.

        Args:
            audio_path: Path to audio file

        Returns:
            AudioFeatures if cached, None otherwise
        """
        audio_id = self.get_audio_id(audio_path)

        if audio_id in self._audio:
            self.audio_hits += 1
            logger.debug(f"Audio cache HIT: {audio_id}")
            return self._audio[audio_id]

        self.audio_misses += 1
        logger.debug(f"Audio cache MISS: {audio_id}")
        return None

    def put_audio(
        self,
        audio_path: str,
        features: torch.Tensor,
        duration: float,
        fps: float,
        chunks: Optional[torch.Tensor] = None,
    ) -> AudioFeatures:
        """
        Cache audio features.

        Args:
            audio_path: Path to audio file
            features: Whisper encoder features
            duration: Audio duration in seconds
            fps: Target FPS for frame alignment
            chunks: Pre-chunked features aligned to frames

        Returns:
            Cached AudioFeatures
        """
        audio_id = self.get_audio_id(audio_path)

        # Evict oldest if at capacity
        if len(self._audio) >= self.max_audio:
            self._evict_oldest_audio()

        af = AudioFeatures(
            audio_id=audio_id,
            features=features.to(self.device),
            duration=duration,
            fps=fps,
            chunks=chunks.to(self.device) if chunks is not None else None,
        )

        self._audio[audio_id] = af
        logger.debug(f"Audio cached: {audio_id}")

        return af

    def _evict_oldest_face(self) -> None:
        """Evict least recently used face."""
        if not self._faces:
            return

        # Find oldest by created_at (could also use LRU based on use_count)
        oldest_id = min(self._faces, key=lambda k: self._faces[k].created_at)
        del self._faces[oldest_id]
        logger.debug(f"Evicted face: {oldest_id}")

    def _evict_oldest_audio(self) -> None:
        """Evict least recently used audio."""
        if not self._audio:
            return

        oldest_id = min(self._audio, key=lambda k: self._audio[k].created_at)
        del self._audio[oldest_id]
        logger.debug(f"Evicted audio: {oldest_id}")

    def clear(self) -> None:
        """Clear all cached data."""
        self._faces.clear()
        self._audio.clear()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("Cache cleared")

    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        face_total = self.face_hits + self.face_misses
        audio_total = self.audio_hits + self.audio_misses

        return {
            "faces_cached": len(self._faces),
            "audio_cached": len(self._audio),
            "face_hit_rate": self.face_hits / face_total if face_total > 0 else 0,
            "audio_hit_rate": self.audio_hits / audio_total if audio_total > 0 else 0,
            "face_hits": self.face_hits,
            "face_misses": self.face_misses,
            "audio_hits": self.audio_hits,
            "audio_misses": self.audio_misses,
        }

    def __repr__(self) -> str:
        stats = self.stats()
        return (
            f"FaceCache(faces={stats['faces_cached']}, audio={stats['audio_cached']}, "
            f"face_hit_rate={stats['face_hit_rate']:.1%}, audio_hit_rate={stats['audio_hit_rate']:.1%})"
        )
