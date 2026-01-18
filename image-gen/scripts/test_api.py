#!/usr/bin/env python3
"""
API Test Script for Image-Gen Service

Tests the server endpoints locally.

Usage:
    python scripts/test_api.py [--base-url http://localhost:7000]
"""

import argparse
import json
import sys
import time

import httpx


def test_health(client: httpx.Client) -> bool:
    """Test health endpoint."""
    print("\n" + "=" * 60)
    print("TEST: GET /health")
    print("=" * 60)

    response = client.get("/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    if response.status_code != 200:
        print("FAIL: Health check returned non-200")
        return False

    data = response.json()
    if data["status"] != "healthy":
        print(f"FAIL: Status is {data['status']}, expected 'healthy'")
        return False

    if not data["flux_loaded"]:
        print("FAIL: FLUX model not loaded")
        return False

    print("PASS: Health check OK")
    return True


def test_root(client: httpx.Client) -> bool:
    """Test root endpoint."""
    print("\n" + "=" * 60)
    print("TEST: GET /")
    print("=" * 60)

    response = client.get("/")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    if response.status_code != 200:
        print("FAIL: Root endpoint returned non-200")
        return False

    print("PASS: Root endpoint OK")
    return True


def test_generate(client: httpx.Client) -> bool:
    """Test generate endpoint."""
    print("\n" + "=" * 60)
    print("TEST: POST /generate")
    print("=" * 60)

    # Use httpbin for testing (it accepts PUT requests)
    request_data = {
        "prompt": "A cute robot cat sitting on a windowsill, digital art",
        "upload_url": "https://httpbin.org/put",
        "width": 512,
        "height": 512,
        "num_steps": 4,
        "guidance_scale": 0.0,
        "output_format": "png",
    }

    print(f"Request: {json.dumps(request_data, indent=2)}")

    start_time = time.time()
    response = client.post("/generate", json=request_data, timeout=120.0)
    elapsed = time.time() - start_time

    print(f"\nStatus: {response.status_code}")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    if response.status_code != 200:
        print("FAIL: Generate returned non-200")
        return False

    data = response.json()
    if not data.get("success"):
        print(f"FAIL: Generation failed: {data.get('error_message')}")
        return False

    print("PASS: Generate OK")
    return True


def test_generate_large(client: httpx.Client) -> bool:
    """Test generate with larger resolution."""
    print("\n" + "=" * 60)
    print("TEST: POST /generate (1024x1024)")
    print("=" * 60)

    request_data = {
        "prompt": "A majestic mountain landscape at sunset, photorealistic, 8k",
        "upload_url": "https://httpbin.org/put",
        "width": 1024,
        "height": 1024,
        "num_steps": 4,
        "guidance_scale": 0.0,
        "seed": 42,
        "output_format": "png",
    }

    print(f"Request: {json.dumps(request_data, indent=2)}")

    start_time = time.time()
    response = client.post("/generate", json=request_data, timeout=180.0)
    elapsed = time.time() - start_time

    print(f"\nStatus: {response.status_code}")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    if response.status_code != 200:
        print("FAIL: Generate returned non-200")
        return False

    data = response.json()
    if not data.get("success"):
        print(f"FAIL: Generation failed: {data.get('error_message')}")
        return False

    # Check seed was used
    if data.get("seed") != 42:
        print(f"WARN: Seed mismatch: got {data.get('seed')}, expected 42")

    print("PASS: Generate (1024x1024) OK")
    return True


def main():
    parser = argparse.ArgumentParser(description="Test Image-Gen API")
    parser.add_argument(
        "--base-url",
        default="http://localhost:7000",
        help="Base URL of the server",
    )
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Skip generation tests (faster)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("IMAGE-GEN API TESTS")
    print("=" * 60)
    print(f"Base URL: {args.base_url}")

    results = []

    with httpx.Client(base_url=args.base_url, timeout=30.0) as client:
        # Test health first
        results.append(("health", test_health(client)))
        results.append(("root", test_root(client)))

        # Only run generation tests if health passes
        if results[0][1] and not args.skip_generate:
            results.append(("generate", test_generate(client)))
            results.append(("generate_large", test_generate_large(client)))

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {name}: {status}")

    print(f"\nTotal: {passed}/{total} passed")

    if passed != total:
        sys.exit(1)


if __name__ == "__main__":
    main()
