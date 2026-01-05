#!/usr/bin/env python3
"""
Download all model weights for the lip-sync pipeline.

This script downloads models from HuggingFace:
- MuseTalk (~4GB)
- LivePortrait (~2GB)
- CodeFormer (~1GB)

Usage:
    python scripts/download_models.py
    python scripts/download_models.py --models-dir /custom/path
"""

import argparse
import os
import sys
from pathlib import Path


def download_musetalk(target_dir: str) -> None:
    """Download MuseTalk model weights."""
    from huggingface_hub import snapshot_download

    print(f"Downloading MuseTalk to {target_dir}/musetalk...")

    snapshot_download(
        repo_id="TMElyralab/MuseTalk",
        local_dir=os.path.join(target_dir, "musetalk"),
        ignore_patterns=["*.md", "*.txt", "*.git*", "demo/*", "results/*"],
    )

    print("MuseTalk downloaded successfully")


def download_liveportrait(target_dir: str) -> None:
    """Download LivePortrait model weights."""
    from huggingface_hub import snapshot_download

    print(f"Downloading LivePortrait to {target_dir}/liveportrait...")

    snapshot_download(
        repo_id="KwaiVGI/LivePortrait",
        local_dir=os.path.join(target_dir, "liveportrait"),
        ignore_patterns=["*.md", "*.txt", "*.git*", "docs/*"],
    )

    print("LivePortrait downloaded successfully")


def download_codeformer(target_dir: str) -> None:
    """Download CodeFormer model weights."""
    from huggingface_hub import hf_hub_download

    print(f"Downloading CodeFormer to {target_dir}/codeformer...")

    codeformer_dir = os.path.join(target_dir, "codeformer")
    os.makedirs(codeformer_dir, exist_ok=True)

    # Download main CodeFormer model
    hf_hub_download(
        repo_id="sczhou/CodeFormer",
        filename="CodeFormer/codeformer.pth",
        local_dir=codeformer_dir,
    )

    # Download face detection model (RetinaFace)
    hf_hub_download(
        repo_id="sczhou/CodeFormer",
        filename="facelib/detection_Resnet50_Final.pth",
        local_dir=codeformer_dir,
    )

    # Download face parsing model
    hf_hub_download(
        repo_id="sczhou/CodeFormer",
        filename="facelib/parsing_parsenet.pth",
        local_dir=codeformer_dir,
    )

    print("CodeFormer downloaded successfully")


def download_face_models(target_dir: str) -> None:
    """Download additional face detection models."""
    from huggingface_hub import hf_hub_download

    print(f"Downloading face models to {target_dir}/face...")

    face_dir = os.path.join(target_dir, "face")
    os.makedirs(face_dir, exist_ok=True)

    # Download InsightFace models (used by some pipelines)
    try:
        hf_hub_download(
            repo_id="deepinsight/insightface",
            subfolder="models/buffalo_l",
            filename="det_10g.onnx",
            local_dir=face_dir,
        )
        print("InsightFace detection model downloaded")
    except Exception as e:
        print(f"Warning: Could not download InsightFace: {e}")

    print("Face models downloaded successfully")


def verify_downloads(target_dir: str) -> bool:
    """Verify that required models are present."""
    required_files = [
        "musetalk",
        "liveportrait",
        "codeformer/CodeFormer/codeformer.pth",
    ]

    missing = []
    for path in required_files:
        full_path = os.path.join(target_dir, path)
        if not os.path.exists(full_path):
            missing.append(path)

    if missing:
        print(f"Warning: Missing files: {missing}")
        return False

    print("All required models verified")
    return True


def main():
    parser = argparse.ArgumentParser(description="Download lip-sync model weights")
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory to save models (default: models)",
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
        print("All models downloaded successfully!")
        print(f"Total size: Check {target_dir}")
        print("=" * 50)

        verify_downloads(target_dir)

    except Exception as e:
        print(f"Error downloading models: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
