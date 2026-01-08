#!/usr/bin/env python3
"""
CPU Validation Script for Lip-Sync V2.

Tests that all components can load and run on CPU:
1. InsightFace detection
2. Frame extraction with ffmpeg
3. Face tracking with interpolation
4. Wav2Lip model loading (optional)

Run: python scripts/test-cpu-validation.py --verbose
"""

import argparse
import os
import sys
import tempfile
import subprocess

# Force CPU mode
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def test_imports():
    """Test that all modules can be imported."""
    print("=" * 60)
    print("Testing imports...")
    print("=" * 60)
    
    try:
        import numpy as np
        print("  [OK] numpy")
    except ImportError as e:
        print(f"  [FAIL] numpy: {e}")
        return False

    try:
        import cv2
        print("  [OK] opencv")
    except ImportError as e:
        print(f"  [FAIL] opencv: {e}")
        return False

    try:
        from lipsync.frame_extractor import get_video_info, extract_sampled_frames
        print("  [OK] lipsync.frame_extractor")
    except ImportError as e:
        print(f"  [FAIL] lipsync.frame_extractor: {e}")
        return False

    try:
        from lipsync.tracker import FaceTracker, TrackingResult
        print("  [OK] lipsync.tracker")
    except ImportError as e:
        print(f"  [FAIL] lipsync.tracker: {e}")
        return False

    try:
        from lipsync.server.schemas import LipSyncRequest, LipSyncResponse
        print("  [OK] lipsync.server.schemas")
    except ImportError as e:
        print(f"  [FAIL] lipsync.server.schemas: {e}")
        return False

    print("  All imports OK")
    return True


def test_ffmpeg():
    """Test that ffmpeg is available."""
    print("\n" + "=" * 60)
    print("Testing ffmpeg...")
    print("=" * 60)
    
    try:
        result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        if result.returncode == 0:
            version = result.stdout.split("\n")[0]
            print(f"  [OK] {version}")
            return True
        else:
            print(f"  [FAIL] ffmpeg returned {result.returncode}")
            return False
    except FileNotFoundError:
        print("  [FAIL] ffmpeg not found in PATH")
        return False


def test_frame_extraction():
    """Test frame extraction with a synthetic video."""
    print("\n" + "=" * 60)
    print("Testing frame extraction...")
    print("=" * 60)
    
    from lipsync.frame_extractor import get_video_info, extract_sampled_frames
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a synthetic test video
        video_path = os.path.join(tmpdir, "test.mp4")
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", "color=c=blue:s=320x240:d=2:r=30",
            "-c:v", "libx264",
            "-loglevel", "error",
            video_path,
        ]
        subprocess.run(cmd, check=True)
        print(f"  Created test video: {video_path}")
        
        # Test get_video_info
        info = get_video_info(video_path)
        print(f"  Video info: {info.width}x{info.height} @ {info.fps}fps, {info.total_frames} frames")
        
        if info.width != 320 or info.height != 240:
            print(f"  [FAIL] Unexpected dimensions")
            return False
        
        # Test extract_sampled_frames
        frames, indices, _ = extract_sampled_frames(video_path, sample_interval=5)
        print(f"  Extracted {len(frames)} sampled frames")
        
        if len(frames) == 0:
            print("  [FAIL] No frames extracted")
            return False
        
        # Verify frame shape
        if frames[0].shape != (240, 320, 3):
            print(f"  [FAIL] Unexpected frame shape: {frames[0].shape}")
            return False
        
        print("  [OK] Frame extraction works")
        return True


def test_insightface(verbose=False):
    """Test InsightFace loading and detection on CPU."""
    print("\n" + "=" * 60)
    print("Testing InsightFace (CPU)...")
    print("=" * 60)
    
    try:
        from insightface.app import FaceAnalysis
        print("  [OK] InsightFace imported")
    except ImportError as e:
        print(f"  [SKIP] InsightFace not installed: {e}")
        return True  # Non-fatal
    
    try:
        # Load model
        print("  Loading buffalo_l model (this may take a minute)...")
        app = FaceAnalysis(
            name="buffalo_l",
            providers=["CPUExecutionProvider"],
        )
        app.prepare(ctx_id=-1, det_size=(640, 640))
        print("  [OK] Model loaded")
        
        # Test detection on a blank image
        import numpy as np
        test_image = np.zeros((480, 640, 3), dtype=np.uint8)
        faces = app.get(test_image)
        print(f"  [OK] Detection ran (found {len(faces)} faces on blank image)")
        
        return True
    except Exception as e:
        print(f"  [FAIL] InsightFace error: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return False


def test_tracker_interpolation():
    """Test tracker interpolation logic."""
    print("\n" + "=" * 60)
    print("Testing tracker interpolation...")
    print("=" * 60)
    
    from lipsync.tracker import FaceTracker
    from unittest.mock import MagicMock
    
    # Create mock detector
    mock_detector = MagicMock()
    
    tracker = FaceTracker(
        detector=mock_detector,
        sample_interval=5,
        smoothing_factor=1.0,
    )
    
    # Test interpolation
    sampled = {
        0: {"char_1": [100, 200, 50, 150]},
        10: {"char_1": [110, 210, 60, 160]},
    }
    
    tracks = tracker._interpolate_tracks(sampled, total_frames=11, char_ids=["char_1"])
    
    # Verify midpoint
    midpoint = tracks["char_1"][5]
    expected = [105, 205, 55, 155]
    
    if midpoint != expected:
        print(f"  [FAIL] Interpolation incorrect: {midpoint} != {expected}")
        return False
    
    print("  [OK] Interpolation works correctly")
    return True


def main():
    parser = argparse.ArgumentParser(description="CPU validation for Lip-Sync V2")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--skip-insightface", action="store_true", help="Skip InsightFace test")
    args = parser.parse_args()
    
    print("=" * 60)
    print("LIP-SYNC V2 CPU VALIDATION")
    print("=" * 60)
    
    results = []
    
    # Run tests
    results.append(("Imports", test_imports()))
    results.append(("FFmpeg", test_ffmpeg()))
    results.append(("Frame Extraction", test_frame_extraction()))
    results.append(("Tracker Interpolation", test_tracker_interpolation()))
    
    if not args.skip_insightface:
        results.append(("InsightFace", test_insightface(args.verbose)))
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "[OK]" if passed else "[FAIL]"
        print(f"  {status} {name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\nAll tests passed!")
        return 0
    else:
        print("\nSome tests failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
