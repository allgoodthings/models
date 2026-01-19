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
    print("TEST: POST /generate (single, non-sequential)")
    print("=" * 60)

    request_data = {
        "images": [
            {
                "prompt": "A cute robot cat sitting on a windowsill, digital art",
                "referenceImageUrls": [],
            }
        ],
        "uploadUrls": ["https://httpbin.org/put"],
        "config": {
            "width": 512,
            "height": 512,
            "seed": 42,
            "steps": 4,
            "guidanceScale": 1.0,
        },
        "sequential": False,
    }

    print(f"Request: {json.dumps(request_data, indent=2)}")

    response = client.post("/generate", json=request_data, timeout=120.0)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    if response.status_code != 200:
        return False

    data = response.json()
    if len(data.get("results", [])) != 1:
        print(f"FAIL: results={len(data.get('results', []))}")
        return False

    result = data["results"][0]
    if not result.get("success"):
        print(f"FAIL: result success={result.get('success')}, error={result.get('error')}")
        return False

    print("PASS")
    return True


def test_generate_sequential(client: httpx.Client) -> bool:
    """Test sequential keyframe generation."""
    print("\n" + "=" * 60)
    print("TEST: POST /generate (sequential keyframes)")
    print("=" * 60)

    request_data = {
        "images": [
            {
                "prompt": "A warrior stands in a misty forest, cinematic style",
                "referenceImageUrls": [],
            },
            {
                "prompt": "The warrior draws a sword, previous scene visible, cinematic style",
                "referenceImageUrls": [],
            },
            {
                "prompt": "The warrior charges forward, previous scene visible, cinematic style",
                "referenceImageUrls": [],
            },
        ],
        "uploadUrls": [
            "https://httpbin.org/put",
            "https://httpbin.org/put",
            "https://httpbin.org/put",
        ],
        "config": {
            "width": 512,
            "height": 512,
            "seed": 100,
            "steps": 4,
            "guidanceScale": 1.0,
        },
        "sequential": True,
    }

    print(f"Generating {len(request_data['images'])} sequential keyframes...")

    response = client.post("/generate", json=request_data, timeout=300.0)
    print(f"Status: {response.status_code}")

    if response.status_code != 200:
        return False

    data = response.json()
    print(f"Response: {len(data.get('results', []))} results")

    if len(data.get("results", [])) != 3:
        return False

    for i, result in enumerate(data["results"]):
        status = "OK" if result.get("success") else f"FAIL: {result.get('error')}"
        print(f"  [{i}] {status}")
        if not result.get("success"):
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
            results.append(("generate_sequential", test_generate_sequential(client)))

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
