"""
Lip-Sync V2 - Multi-face lip-sync pipeline using Wav2Lip-HD.

This module provides a simplified pipeline for lip-syncing multiple faces
in a video using Wav2Lip-HD with per-frame bounding box tracking.
"""

# =============================================================================
# TORCHVISION COMPATIBILITY FIX - Must run before any basicsr import
# =============================================================================
# basicsr 1.4.2 imports from torchvision.transforms.functional_tensor, which was
# removed in torchvision 0.17+. This fix creates a dummy module so basicsr works.
# Only runs if torchvision is installed (skipped in unit test environments).
import sys
import types

try:
    from torchvision.transforms.functional_tensor import rgb_to_grayscale
except ImportError:
    try:
        from torchvision.transforms.functional import rgb_to_grayscale
        _functional_tensor = types.ModuleType("torchvision.transforms.functional_tensor")
        _functional_tensor.rgb_to_grayscale = rgb_to_grayscale
        sys.modules["torchvision.transforms.functional_tensor"] = _functional_tensor
    except ImportError:
        # torchvision not installed (unit test environment) - skip fix
        pass
# =============================================================================

__version__ = "0.1.0"

# Imports will be added as modules are implemented:
# from .tracker import FaceTracker
# from .wav2lip import Wav2LipHD

__all__ = [
    "__version__",
]
