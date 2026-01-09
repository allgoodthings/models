"""
In-process Wav2Lip model for efficient inference.

Loads model once at startup and keeps it on GPU for fast inference.
Eliminates subprocess overhead from calling inference.py.
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

logger = logging.getLogger("lipsync.wav2lip_model")

# Add Wav2Lip to path for imports
WAV2LIP_DIR = Path(__file__).parent.parent / "vendors" / "wav2lip-hd" / "Wav2Lip-master"


class Wav2LipModel:
    """
    In-process Wav2Lip model wrapper.

    Loads model once and keeps it on GPU for efficient inference.
    """

    def __init__(self, checkpoint_path: str, device: str = "cuda"):
        """
        Initialize Wav2Lip model.

        Args:
            checkpoint_path: Path to wav2lip_gan.pth
            device: Device to run on ("cuda" or "cpu")
        """
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = None
        self.img_size = 96
        self.mel_step_size = 16
        self._loaded = False

        # Add Wav2Lip to path for model imports
        wav2lip_path = str(WAV2LIP_DIR)
        if wav2lip_path not in sys.path:
            sys.path.insert(0, wav2lip_path)

    def load(self):
        """Load Wav2Lip model."""
        if self._loaded:
            return

        logger.info(f"Loading Wav2Lip model from {self.checkpoint_path}...")
        start = time.time()

        # Import Wav2Lip model architecture
        from models import Wav2Lip

        # Load checkpoint
        if self.device.type == "cuda":
            checkpoint = torch.load(self.checkpoint_path)
        else:
            checkpoint = torch.load(
                self.checkpoint_path,
                map_location=lambda storage, loc: storage
            )

        # Create model and load weights
        self.model = Wav2Lip()
        state_dict = checkpoint["state_dict"]

        # Remove 'module.' prefix if present (from DataParallel)
        new_state_dict = {}
        for k, v in state_dict.items():
            new_state_dict[k.replace("module.", "")] = v

        self.model.load_state_dict(new_state_dict)
        self.model = self.model.to(self.device)
        self.model.eval()

        self._loaded = True
        logger.info(f"Wav2Lip model loaded in {time.time() - start:.2f}s on {self.device}")

    def process_audio(self, audio_path: str, fps: float) -> List[np.ndarray]:
        """
        Process audio file into mel spectrogram chunks.

        Args:
            audio_path: Path to audio file
            fps: Video frame rate

        Returns:
            List of mel chunks for each frame
        """
        import subprocess
        import tempfile

        # Import audio processing from Wav2Lip
        import audio as wav2lip_audio

        # Convert to WAV if needed
        if not audio_path.endswith('.wav'):
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
                temp_wav = f.name

            cmd = ['ffmpeg', '-y', '-i', audio_path, '-ar', '16000', '-ac', '1', temp_wav]
            subprocess.run(cmd, capture_output=True, check=True)
            audio_path = temp_wav

        # Load and process audio
        wav = wav2lip_audio.load_wav(audio_path, 16000)
        mel = wav2lip_audio.melspectrogram(wav)

        if np.isnan(mel.reshape(-1)).sum() > 0:
            raise ValueError("Mel spectrogram contains NaN values")

        # Split into chunks
        mel_chunks = []
        mel_idx_multiplier = 80.0 / fps
        i = 0

        while True:
            start_idx = int(i * mel_idx_multiplier)
            if start_idx + self.mel_step_size > len(mel[0]):
                mel_chunks.append(mel[:, len(mel[0]) - self.mel_step_size:])
                break
            mel_chunks.append(mel[:, start_idx:start_idx + self.mel_step_size])
            i += 1

        # Clean up temp file
        if 'temp_wav' in locals():
            os.unlink(temp_wav)

        return mel_chunks

    def infer_batch(
        self,
        face_crops: List[np.ndarray],
        mel_chunks: List[np.ndarray],
    ) -> List[np.ndarray]:
        """
        Run Wav2Lip inference on a batch of faces.

        Args:
            face_crops: List of face crops (BGR, any size)
            mel_chunks: List of mel spectrogram chunks

        Returns:
            List of lip-synced face crops (same size as input)
        """
        if not self._loaded:
            self.load()

        results = []
        batch_size = 128

        # Process in batches
        for batch_start in range(0, len(face_crops), batch_size):
            batch_end = min(batch_start + batch_size, len(face_crops))

            batch_faces = face_crops[batch_start:batch_end]
            batch_mels = mel_chunks[batch_start:batch_end]

            # Prepare face batch
            img_batch = []
            original_sizes = []

            for face in batch_faces:
                original_sizes.append((face.shape[1], face.shape[0]))  # (w, h)
                resized = cv2.resize(face, (self.img_size, self.img_size))
                img_batch.append(resized)

            img_batch = np.array(img_batch)
            mel_batch = np.array(batch_mels)

            # Mask lower half of face
            img_masked = img_batch.copy()
            img_masked[:, self.img_size // 2:] = 0

            # Concatenate masked and original
            img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.0
            mel_batch = mel_batch.reshape(len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1)

            # Convert to tensors
            img_tensor = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(self.device)
            mel_tensor = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(self.device)

            # Inference
            with torch.no_grad():
                pred = self.model(mel_tensor, img_tensor)

            # Convert back to numpy
            pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.0

            # Resize back to original sizes
            for p, orig_size in zip(pred, original_sizes):
                p = cv2.resize(p.astype(np.uint8), orig_size)
                results.append(p)

        return results

    def process_video(
        self,
        frames: List[np.ndarray],
        face_coords: List[Optional[Tuple[int, int, int, int]]],
        mel_chunks: List[np.ndarray],
        loop_mode: str = "crossfade",
        crossfade_frames: int = 10,
    ) -> List[np.ndarray]:
        """
        Process full video with lip-sync.

        Args:
            frames: List of video frames (BGR)
            face_coords: Per-frame face coordinates [(y1, y2, x1, x2), ...]
            mel_chunks: Mel spectrogram chunks
            loop_mode: How to handle audio > video
            crossfade_frames: Frames to blend at loop boundary

        Returns:
            List of processed frames
        """
        if not self._loaded:
            self.load()

        num_frames = len(frames)
        num_mels = len(mel_chunks)

        logger.info(f"Processing {num_frames} frames with {num_mels} mel chunks")

        # Collect face crops and coordinates for each output frame
        face_crops = []
        coords_list = []
        frame_indices = []

        for i in range(num_mels):
            # Get frame index based on loop mode
            if loop_mode == "none":
                idx = min(i, num_frames - 1)
            elif loop_mode == "repeat":
                idx = i % num_frames
            elif loop_mode == "pingpong":
                cycle = (num_frames - 1) * 2 if num_frames > 1 else 1
                pos = i % cycle
                idx = pos if pos < num_frames else cycle - pos
            else:  # crossfade
                idx = i % num_frames

            frame_indices.append(idx)
            coords = face_coords[idx] if idx < len(face_coords) else None

            if coords is None:
                # No face detected, skip
                face_crops.append(None)
                coords_list.append(None)
            else:
                y1, y2, x1, x2 = coords
                face_crop = frames[idx][y1:y2, x1:x2].copy()
                face_crops.append(face_crop)
                coords_list.append(coords)

        # Filter out None entries
        valid_indices = [i for i, f in enumerate(face_crops) if f is not None]
        valid_faces = [face_crops[i] for i in valid_indices]
        valid_mels = [mel_chunks[i] for i in valid_indices]

        if not valid_faces:
            logger.warning("No valid face crops found")
            return frames[:num_mels]

        # Run inference
        logger.info(f"Running Wav2Lip inference on {len(valid_faces)} faces...")
        start = time.time()

        synced_faces = self.infer_batch(valid_faces, valid_mels)

        logger.info(f"Inference completed in {time.time() - start:.2f}s")

        # Compose output frames
        output_frames = []
        synced_idx = 0

        for i in range(num_mels):
            frame_idx = frame_indices[i]
            frame = frames[frame_idx].copy()

            if i in valid_indices:
                coords = coords_list[i]
                y1, y2, x1, x2 = coords
                synced_face = synced_faces[synced_idx]
                synced_idx += 1

                # Paste synced face back with Poisson blending
                frame = self._blend_face(synced_face, frame, coords)

            output_frames.append(frame)

        # Apply crossfade at loop boundaries if needed
        if loop_mode == "crossfade" and num_mels > num_frames:
            output_frames = self._apply_crossfade(output_frames, num_frames, crossfade_frames)

        return output_frames

    def _blend_face(
        self,
        face: np.ndarray,
        frame: np.ndarray,
        coords: Tuple[int, int, int, int],
    ) -> np.ndarray:
        """Blend face into frame using Poisson blending."""
        y1, y2, x1, x2 = coords
        h, w = face.shape[:2]

        try:
            # Create elliptical mask
            mask = np.zeros((h, w), dtype=np.uint8)
            center = (w // 2, h // 2)
            axes = (w // 2 - 2, h // 2 - 2)
            cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
            mask = cv2.GaussianBlur(mask, (7, 7), 3)

            # Poisson blend
            dst_center = ((x1 + x2) // 2, (y1 + y2) // 2)
            result = cv2.seamlessClone(face, frame, mask, dst_center, cv2.NORMAL_CLONE)
            return result

        except cv2.error:
            # Fallback to simple paste
            result = frame.copy()
            result[y1:y2, x1:x2] = face
            return result

    def _apply_crossfade(
        self,
        frames: List[np.ndarray],
        loop_length: int,
        crossfade_frames: int,
    ) -> List[np.ndarray]:
        """Apply crossfade at loop boundaries."""
        result = frames.copy()

        for loop_start in range(0, len(frames), loop_length):
            if loop_start == 0:
                continue

            for i in range(crossfade_frames):
                if loop_start + i >= len(result):
                    break

                alpha = i / crossfade_frames
                idx_current = loop_start + i
                idx_blend = i

                if idx_blend < len(result):
                    result[idx_current] = cv2.addWeighted(
                        result[idx_current], alpha,
                        result[idx_blend], 1 - alpha,
                        0
                    )

        return result

    def unload(self):
        """Unload model from GPU."""
        if self.model is not None:
            del self.model
            self.model = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self._loaded = False
        logger.info("Wav2Lip model unloaded")

    @property
    def is_loaded(self) -> bool:
        return self._loaded
