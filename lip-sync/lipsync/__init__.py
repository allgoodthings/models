"""
Multi-face lip-sync pipeline.

This module provides a complete pipeline for lip-syncing multiple faces
in a video using MuseTalk, LivePortrait, and CodeFormer.
"""

# =============================================================================
# TORCHVISION COMPATIBILITY FIX - Must run before any basicsr import
# =============================================================================
# basicsr 1.4.2 imports from torchvision.transforms.functional_tensor, which was
# removed in torchvision 0.17+. This fix creates a dummy module so basicsr works.
import sys
import types

try:
    from torchvision.transforms.functional_tensor import rgb_to_grayscale
except ImportError:
    from torchvision.transforms.functional import rgb_to_grayscale
    _functional_tensor = types.ModuleType("torchvision.transforms.functional_tensor")
    _functional_tensor.rgb_to_grayscale = rgb_to_grayscale
    sys.modules["torchvision.transforms.functional_tensor"] = _functional_tensor
# =============================================================================

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