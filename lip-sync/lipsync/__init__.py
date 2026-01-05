"""
Multi-face lip-sync pipeline.

This module provides a complete pipeline for lip-syncing multiple faces
in a video using MuseTalk, LivePortrait, and CodeFormer.
"""

from .pipeline import LipSyncPipeline, PipelineConfig, FaceJob, process_lipsync
from .compositor import FaceCompositor, FaceRegion
from .alignment import FaceAligner, AlignmentMetadata, align_video, unalign_video
from .musetalk import MuseTalk, MuseTalkConfig
from .liveportrait import LivePortrait, LivePortraitConfig
from .codeformer import CodeFormer, CodeFormerConfig

__version__ = "0.1.0"

__all__ = [
    # Main pipeline
    "LipSyncPipeline",
    "PipelineConfig",
    "FaceJob",
    "process_lipsync",
    # Compositor
    "FaceCompositor",
    "FaceRegion",
    # Alignment
    "FaceAligner",
    "AlignmentMetadata",
    "align_video",
    "unalign_video",
    # Models
    "MuseTalk",
    "MuseTalkConfig",
    "LivePortrait",
    "LivePortraitConfig",
    "CodeFormer",
    "CodeFormerConfig",
]