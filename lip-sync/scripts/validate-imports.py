#!/usr/bin/env python3
"""
Fast local validation script - catches import/syntax errors in seconds.

Run this BEFORE pushing to catch errors early instead of waiting 20+ min for CI.

Usage:
    python scripts/validate-imports.py

This script validates:
1. All lipsync modules can be imported
2. Config classes can be instantiated
3. Model wrapper classes can be created (without loading weights)
4. Server app can be imported

It does NOT:
- Load actual model weights
- Require GPU
- Download anything
- Run inference
"""

import sys
import os

# Add the lip-sync directory to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
sys.path.insert(0, project_dir)

def validate_imports():
    """Validate all imports work."""
    print("=" * 60)
    print("VALIDATING IMPORTS")
    print("=" * 60)

    errors = []

    # Core modules
    modules = [
        ("lipsync", "Main package"),
        ("lipsync.pipeline", "Pipeline orchestration"),
        ("lipsync.musetalk", "MuseTalk wrapper"),
        ("lipsync.liveportrait", "LivePortrait wrapper"),
        ("lipsync.codeformer", "CodeFormer wrapper"),
        ("lipsync.alignment", "Face alignment"),
        ("lipsync.compositor", "Face compositing"),
        ("lipsync.utils", "Utilities"),
    ]

    for module_name, description in modules:
        try:
            __import__(module_name)
            print(f"  [OK] {module_name} - {description}")
        except Exception as e:
            print(f"  [FAIL] {module_name} - {e}")
            errors.append((module_name, str(e)))

    return errors


def validate_configs():
    """Validate config classes can be instantiated."""
    print("\n" + "=" * 60)
    print("VALIDATING CONFIGS")
    print("=" * 60)

    errors = []

    try:
        from lipsync.musetalk import MuseTalkConfig
        config = MuseTalkConfig(device="cpu", fp16=False)
        print(f"  [OK] MuseTalkConfig - model_dir={config.model_dir}, whisper={config.whisper_model}")
    except Exception as e:
        print(f"  [FAIL] MuseTalkConfig - {e}")
        errors.append(("MuseTalkConfig", str(e)))

    try:
        from lipsync.liveportrait import LivePortraitConfig
        config = LivePortraitConfig(device="cpu", fp16=False)
        print(f"  [OK] LivePortraitConfig - model_dir={config.model_dir}")
    except Exception as e:
        print(f"  [FAIL] LivePortraitConfig - {e}")
        errors.append(("LivePortraitConfig", str(e)))

    try:
        from lipsync.codeformer import CodeFormerConfig
        config = CodeFormerConfig(device="cpu", fp16=False)
        print(f"  [OK] CodeFormerConfig - model_path={config.model_path}")
    except Exception as e:
        print(f"  [FAIL] CodeFormerConfig - {e}")
        errors.append(("CodeFormerConfig", str(e)))

    try:
        from lipsync.pipeline import PipelineConfig
        config = PipelineConfig(device="cpu", fp16=False)
        print(f"  [OK] PipelineConfig - device={config.device}")
    except Exception as e:
        print(f"  [FAIL] PipelineConfig - {e}")
        errors.append(("PipelineConfig", str(e)))

    return errors


def validate_model_wrappers():
    """Validate model wrapper classes can be created (no weight loading)."""
    print("\n" + "=" * 60)
    print("VALIDATING MODEL WRAPPERS (no weight loading)")
    print("=" * 60)

    errors = []

    try:
        from lipsync.musetalk import MuseTalk, MuseTalkConfig
        model = MuseTalk(MuseTalkConfig(device="cpu", fp16=False))
        print(f"  [OK] MuseTalk wrapper created (not loaded)")
    except Exception as e:
        print(f"  [FAIL] MuseTalk - {e}")
        errors.append(("MuseTalk", str(e)))

    try:
        from lipsync.liveportrait import LivePortrait, LivePortraitConfig
        model = LivePortrait(LivePortraitConfig(device="cpu", fp16=False))
        print(f"  [OK] LivePortrait wrapper created (not loaded)")
    except Exception as e:
        print(f"  [FAIL] LivePortrait - {e}")
        errors.append(("LivePortrait", str(e)))

    try:
        from lipsync.codeformer import CodeFormer, CodeFormerConfig
        model = CodeFormer(CodeFormerConfig(device="cpu", fp16=False))
        print(f"  [OK] CodeFormer wrapper created (not loaded)")
    except Exception as e:
        print(f"  [FAIL] CodeFormer - {e}")
        errors.append(("CodeFormer", str(e)))

    return errors


def validate_server():
    """Validate server module can be imported."""
    print("\n" + "=" * 60)
    print("VALIDATING SERVER")
    print("=" * 60)

    errors = []

    try:
        from lipsync.server import schemas
        print(f"  [OK] Server schemas imported")
    except Exception as e:
        print(f"  [FAIL] Server schemas - {e}")
        errors.append(("server.schemas", str(e)))

    # Note: We don't import server.main as it triggers app creation
    # which might have side effects

    return errors


def main():
    print("\n" + "=" * 60)
    print("FAST LOCAL VALIDATION")
    print("Run this before pushing to catch errors early!")
    print("=" * 60 + "\n")

    all_errors = []

    all_errors.extend(validate_imports())
    all_errors.extend(validate_configs())
    all_errors.extend(validate_model_wrappers())
    all_errors.extend(validate_server())

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if all_errors:
        print(f"\nFAILED with {len(all_errors)} errors:\n")
        for name, error in all_errors:
            print(f"  - {name}: {error}")
        print("\nFix these errors before pushing!")
        return 1
    else:
        print("\nAll validations passed!")
        print("Safe to push (CI will do full model loading test)")
        return 0


if __name__ == "__main__":
    sys.exit(main())
