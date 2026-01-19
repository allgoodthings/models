#!/usr/bin/env python3
"""
API Test Script for Image-Gen Service.

Usage:
    python scripts/test_api.py [--base-url http://localhost:7000]
"""

import argparse
import json
import sys

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
        return False

    data = response.json()
    if data["status"] != "healthy" or not data["flux_loaded"]:
        print(f"FAIL: status={data['status']}, flux_loaded={data['flux_loaded']}")
        return False

    print("PASS")
    return True


def test_generate_single(client: httpx.Client) -> bool:
    """Test single image generation."""
    print("\n" + "=" * 60)
    print("TEST: POST /generate (single)")
    print("=" * 60)

    request_data = {
        "prompts": ["A cute robot cat sitting on a windowsill, digital art"],
        "upload_urls": ["https://httpbin.org/put"],
        "width": 512,
        "height": 512,
        "num_steps": 4,
        "seed": 42,
    }

    print(f"Request: {json.dumps(request_data, indent=2)}")

    response = client.post("/generate", json=request_data, timeout=120.0)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    if response.status_code != 200:
        return False

    data = response.json()
    if not data.get("success") or len(data.get("results", [])) != 1:
        print(f"FAIL: success={data.get('success')}, results={len(data.get('results', []))}")
        return False

    result = data["results"][0]
    if not result.get("success") or result.get("seed") != 42:
        print(f"FAIL: result success={result.get('success')}, seed={result.get('seed')}")
        return False

    print("PASS")
    return True


def test_generate_batch(client: httpx.Client) -> bool:
    """Test batch image generation."""
    print("\n" + "=" * 60)
    print("TEST: POST /generate (batch)")
    print("=" * 60)

    request_data = {
        "prompts": [
            "A red apple on a wooden table",
            "A green apple on a wooden table",
            "A yellow banana on a wooden table",
        ],
        "upload_urls": [
            "https://httpbin.org/put",
            "https://httpbin.org/put",
            "https://httpbin.org/put",
        ],
        "width": 512,
        "height": 512,
        "num_steps": 4,
        "seed": 100,
    }

    print(f"Generating {len(request_data['prompts'])} images...")

    response = client.post("/generate", json=request_data, timeout=300.0)
    print(f"Status: {response.status_code}")

    if response.status_code != 200:
        return False

    data = response.json()
    print(f"Response: success={data.get('success')}, total_ms={data.get('timing_total_ms')}")

    if not data.get("success") or len(data.get("results", [])) != 3:
        return False

    # Check sequential seeds
    for i, result in enumerate(data["results"]):
        expected_seed = 100 + i
        if result.get("seed") != expected_seed:
            print(f"FAIL: result[{i}] seed={result.get('seed')}, expected={expected_seed}")
            return False
        print(f"  [{i}] seed={result['seed']}, inference={result['timing_inference_ms']}ms")

    print("PASS")
    return True


def test_generate_1024(client: httpx.Client) -> bool:
    """Test 1024x1024 generation."""
    print("\n" + "=" * 60)
    print("TEST: POST /generate (1024x1024)")
    print("=" * 60)

    request_data = {
        "prompts": ["A majestic mountain landscape at sunset, photorealistic"],
        "upload_urls": ["https://httpbin.org/put"],
        "width": 1024,
        "height": 1024,
        "num_steps": 4,
        "seed": 42,
    }

    response = client.post("/generate", json=request_data, timeout=180.0)
    print(f"Status: {response.status_code}")

    if response.status_code != 200:
        return False

    data = response.json()
    result = data["results"][0] if data.get("results") else {}
    print(f"Response: success={result.get('success')}, inference={result.get('timing_inference_ms')}ms")

    if not data.get("success"):
        print(f"FAIL: {result.get('error')}")
        return False

    print("PASS")
    return True


def main():
    parser = argparse.ArgumentParser(description="Test Image-Gen API")
    parser.add_argument("--base-url", default="http://localhost:7000")
    parser.add_argument("--skip-generate", action="store_true", help="Skip generation tests")
    args = parser.parse_args()

    print("=" * 60)
    print("IMAGE-GEN API TESTS")
    print("=" * 60)
    print(f"Base URL: {args.base_url}")

    results = []

    with httpx.Client(base_url=args.base_url, timeout=30.0) as client:
        results.append(("health", test_health(client)))

        if results[0][1] and not args.skip_generate:
            results.append(("generate_single", test_generate_single(client)))
            results.append(("generate_batch", test_generate_batch(client)))
            results.append(("generate_1024", test_generate_1024(client)))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    passed = sum(1 for _, r in results if r)
    for name, result in results:
        print(f"  {name}: {'PASS' if result else 'FAIL'}")
    print(f"\nTotal: {passed}/{len(results)}")

    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
