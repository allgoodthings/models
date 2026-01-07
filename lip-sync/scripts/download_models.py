#!/usr/bin/env python3
"""
Download all model weights for the lip-sync pipeline.

This script downloads models from HuggingFace:
- MuseTalk (~4GB) from TMElyralab/MuseTalk
  - musetalkV15/musetalk.json
  - musetalkV15/unet.pth
- LivePortrait (~700MB) from KwaiVGI/LivePortrait
  - liveportrait/base_models/*.pth
  - liveportrait/retargeting_models/*.pth
- CodeFormer (~350MB) from GitHub releases
  - codeformer.pth
  - facelib/*.pth
- SD VAE (~335MB) from stabilityai/sd-vae-ft-mse
  - config.json
  - diffusion_pytorch_model.safetensors

Usage:
    python scripts/download_models.py
    python scripts/download_models.py --models-dir /custom/path
"""

import argparse
import os
import sys
from pathlib import Path


def download_musetalk(target_dir: str) -> None:
    """
    Download MuseTalk model weights from HuggingFace.

    Downloads to: {target_dir}/musetalk/
    Structure:
    - musetalkV15/musetalk.json (config)
    - musetalkV15/unet.pth (3.4GB weights)
    """
    from huggingface_hub import snapshot_download

    print(f"Downloading MuseTalk to {target_dir}/musetalk...")

    snapshot_download(
        repo_id="TMElyralab/MuseTalk",
        local_dir=os.path.join(target_dir, "musetalk"),
        ignore_patterns=["*.md", "*.txt", "*.git*", "demo/*", "results/*"],
    )

    print("MuseTalk downloaded successfully")


def download_liveportrait(target_dir: str) -> None:
    """
    Download LivePortrait model weights from HuggingFace.

    Downloads to: {target_dir}/liveportrait/
    Structure:
    - liveportrait/base_models/appearance_feature_extractor.pth (3.39MB)
    - liveportrait/base_models/motion_extractor.pth (113MB)
    - liveportrait/base_models/spade_generator.pth (222MB)
    - liveportrait/base_models/warping_module.pth (182MB)
    - liveportrait/retargeting_models/stitching_retargeting_module.pth (2.39MB)
    - liveportrait/landmark.onnx (115MB)
    - insightface/models/buffalo_l/* (face detection)
    """
    from huggingface_hub import snapshot_download

    print(f"Downloading LivePortrait to {target_dir}/liveportrait...")

    # Note: HuggingFace repo moved from KwaiVGI to KlingTeam
    snapshot_download(
        repo_id="KlingTeam/LivePortrait",
        local_dir=os.path.join(target_dir, "liveportrait"),
        ignore_patterns=["*.md", "*.txt", "*.git*", "docs/*", "assets/*"],
    )

    print("LivePortrait downloaded successfully")


def download_sd_vae(target_dir: str) -> None:
    """
    Download Stable Diffusion VAE from HuggingFace.

    Downloads to: {target_dir}/musetalk/sd-vae-ft-mse/
    Required by MuseTalk for encoding/decoding face images.
    Structure:
    - config.json
    - diffusion_pytorch_model.safetensors (335MB)
    """
    from huggingface_hub import snapshot_download

    print(f"Downloading SD VAE to {target_dir}/musetalk/sd-vae-ft-mse...")

    vae_dir = os.path.join(target_dir, "musetalk", "sd-vae-ft-mse")
    os.makedirs(vae_dir, exist_ok=True)

    snapshot_download(
        repo_id="stabilityai/sd-vae-ft-mse",
        local_dir=vae_dir,
        ignore_patterns=["*.md", "*.txt", "*.git*"],
    )

    print("SD VAE downloaded successfully")


def download_codeformer(target_dir: str) -> None:
    """
    Download CodeFormer model weights from GitHub releases.

    Downloads to: {target_dir}/codeformer/
    Structure:
    - codeformer.pth (~350MB)
    - facelib/detection_Resnet50_Final.pth
    - facelib/parsing_parsenet.pth
    """
    import urllib.request

    print(f"Downloading CodeFormer to {target_dir}/codeformer...")

    codeformer_dir = os.path.join(target_dir, "codeformer")
    os.makedirs(codeformer_dir, exist_ok=True)

    # Download main CodeFormer model from GitHub releases (official source)
    codeformer_path = os.path.join(codeformer_dir, "codeformer.pth")
    if not os.path.exists(codeformer_path):
        print("  Downloading codeformer.pth from GitHub releases...")
        urllib.request.urlretrieve(
            "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth",
            codeformer_path,
        )

    # Download facelib models from GitHub releases
    facelib_dir = os.path.join(codeformer_dir, "facelib")
    os.makedirs(facelib_dir, exist_ok=True)

    detection_path = os.path.join(facelib_dir, "detection_Resnet50_Final.pth")
    if not os.path.exists(detection_path):
        print("  Downloading detection_Resnet50_Final.pth...")
        urllib.request.urlretrieve(
            "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/detection_Resnet50_Final.pth",
            detection_path,
        )

    parsing_path = os.path.join(facelib_dir, "parsing_parsenet.pth")
    if not os.path.exists(parsing_path):
        print("  Downloading parsing_parsenet.pth...")
        urllib.request.urlretrieve(
            "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/parsing_parsenet.pth",
            parsing_path,
        )

    print("CodeFormer downloaded successfully")


def download_face_models(target_dir: str) -> None:
    """Download additional face detection models."""
    print(f"Note: InsightFace buffalo_l model will be auto-downloaded on first use")
    print("Face models setup complete")


def verify_downloads(target_dir: str) -> bool:
    """Verify that required models are present."""
    required_paths = [
        # MuseTalk
        ("musetalk/musetalkV15/musetalk.json", "MuseTalk config"),
        ("musetalk/musetalkV15/unet.pth", "MuseTalk UNet weights"),
        ("musetalk/sd-vae-ft-mse/config.json", "SD VAE config"),
        # LivePortrait
        ("liveportrait/liveportrait/base_models/appearance_feature_extractor.pth", "LivePortrait F"),
        ("liveportrait/liveportrait/base_models/motion_extractor.pth", "LivePortrait M"),
        ("liveportrait/liveportrait/base_models/spade_generator.pth", "LivePortrait G"),
        ("liveportrait/liveportrait/base_models/warping_module.pth", "LivePortrait W"),
        ("liveportrait/liveportrait/retargeting_models/stitching_retargeting_module.pth", "LivePortrait S"),
        ("liveportrait/liveportrait/landmark.onnx", "LivePortrait landmark ONNX"),
        # CodeFormer
        ("codeformer/codeformer.pth", "CodeFormer weights"),
    ]

    missing = []
    for path, name in required_paths:
        full_path = os.path.join(target_dir, path)
        if not os.path.exists(full_path):
            missing.append(f"{name}: {path}")
        else:
            size_mb = os.path.getsize(full_path) / (1024 * 1024)
            print(f"  ✓ {name}: {size_mb:.1f} MB")

    if missing:
        print(f"\nWarning: Missing files:")
        for m in missing:
            print(f"  ✗ {m}")
        return False

    print("\nAll required models verified!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Download lip-sync model weights")
    parser.add_argument(
        "--models-dir",
        type=str,
        default="/app/models",
        help="Directory to save models (default: /app/models)",
    )
    parser.add_argument(
        "--skip-musetalk",
        action="store_true",
        help="Skip MuseTalk download",
    )
    parser.add_argument(
        "--skip-liveportrait",
        action="store_true",
        help="Skip LivePortrait download",
    )
    parser.add_argument(
        "--skip-codeformer",
        action="store_true",
        help="Skip CodeFormer download",
    )
    parser.add_argument(
        "--skip-vae",
        action="store_true",
        help="Skip SD VAE download",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing downloads",
    )

    args = parser.parse_args()

    target_dir = os.path.abspath(args.models_dir)
    os.makedirs(target_dir, exist_ok=True)

    print(f"Models directory: {target_dir}")
    print()

    if args.verify_only:
        success = verify_downloads(target_dir)
        sys.exit(0 if success else 1)

    try:
        if not args.skip_musetalk:
            download_musetalk(target_dir)
            print()

        if not args.skip_vae:
            download_sd_vae(target_dir)
            print()

        if not args.skip_liveportrait:
            download_liveportrait(target_dir)
            print()

        if not args.skip_codeformer:
            download_codeformer(target_dir)
            print()

        # Download additional face models
        download_face_models(target_dir)
        print()

        print("=" * 50)
        print("All models downloaded!")
        print("=" * 50)
        print()

        if not verify_downloads(target_dir):
            print("\nERROR: Model verification failed!")
            print("Some required files are missing. Check download logs above.")
            sys.exit(1)

    except Exception as e:
        print(f"Error downloading models: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
