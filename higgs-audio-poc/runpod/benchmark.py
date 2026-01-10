#!/usr/bin/env python3
"""
Higgs Audio v2 Benchmark for RunPod vLLM

Tests concurrent request handling and measures throughput.

Usage:
    python benchmark.py                           # Default: 10 requests, 5 concurrent
    python benchmark.py --requests 50 --concurrent 10
    python benchmark.py --url https://<pod-id>-8000.proxy.runpod.net
"""

import asyncio
import argparse
import time
import statistics
from typing import List, Tuple

import httpx

# Test sentences of varying lengths
TEST_SENTENCES = [
    "Hello, this is a test.",
    "The quick brown fox jumps over the lazy dog.",
    "In a world where technology advances rapidly, we must adapt.",
    "To be or not to be, that is the question that haunts us all.",
    "The rain in Spain falls mainly on the plain, or so they say in the old tales.",
    "Artificial intelligence is transforming how we work, live, and communicate with each other.",
    "Once upon a time, in a land far away, there lived a wise old wizard who knew many secrets.",
    "The future belongs to those who believe in the beauty of their dreams and work hard to achieve them.",
]


async def make_request(
    client: httpx.AsyncClient,
    url: str,
    text: str,
    request_id: int,
) -> Tuple[int, float, bool, str]:
    """Make a single generation request."""
    start = time.time()
    try:
        resp = await client.post(
            f"{url}/v1/audio/speech",
            json={
                "model": "higgs-audio-v2",
                "input": text,
                "voice": "alloy",
                "response_format": "wav",
            },
        )
        elapsed = time.time() - start
        success = resp.status_code == 200
        error = "" if success else resp.text[:100]
        return request_id, elapsed, success, error
    except Exception as e:
        elapsed = time.time() - start
        return request_id, elapsed, False, str(e)[:100]


async def run_benchmark(
    url: str,
    num_requests: int,
    max_concurrent: int,
) -> dict:
    """Run the benchmark with specified concurrency."""
    print(f"\n{'='*60}")
    print(f"Benchmark: {num_requests} requests, {max_concurrent} concurrent")
    print(f"URL: {url}")
    print(f"{'='*60}\n")

    # Create semaphore for concurrency control
    semaphore = asyncio.Semaphore(max_concurrent)

    async def bounded_request(client, url, text, req_id):
        async with semaphore:
            return await make_request(client, url, text, req_id)

    # Prepare requests
    texts = [TEST_SENTENCES[i % len(TEST_SENTENCES)] for i in range(num_requests)]

    results = []
    start_time = time.time()

    async with httpx.AsyncClient(timeout=120.0) as client:
        # Check health first
        try:
            health = await client.get(f"{url}/health")
            print(f"Health check: {health.status_code}")
        except Exception as e:
            print(f"Health check failed: {e}")
            print("Make sure vLLM is running!")
            return {}

        # Run all requests
        tasks = [
            bounded_request(client, url, text, i)
            for i, text in enumerate(texts)
        ]

        print(f"Starting {num_requests} requests...\n")

        for coro in asyncio.as_completed(tasks):
            req_id, elapsed, success, error = await coro
            status = "OK" if success else f"FAIL: {error}"
            print(f"  Request {req_id:3d}: {elapsed:6.2f}s - {status}")
            results.append({
                "id": req_id,
                "elapsed": elapsed,
                "success": success,
                "error": error,
            })

    total_time = time.time() - start_time

    # Calculate statistics
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    times = [r["elapsed"] for r in successful]

    stats = {
        "total_requests": num_requests,
        "successful": len(successful),
        "failed": len(failed),
        "total_time": round(total_time, 2),
        "requests_per_second": round(len(successful) / total_time, 2) if total_time > 0 else 0,
    }

    if times:
        stats.update({
            "avg_latency": round(statistics.mean(times), 2),
            "min_latency": round(min(times), 2),
            "max_latency": round(max(times), 2),
            "p50_latency": round(statistics.median(times), 2),
            "p95_latency": round(sorted(times)[int(len(times) * 0.95)] if len(times) > 1 else times[0], 2),
        })

    # Print summary
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Total requests:     {stats['total_requests']}")
    print(f"Successful:         {stats['successful']}")
    print(f"Failed:             {stats['failed']}")
    print(f"Total time:         {stats['total_time']}s")
    print(f"Throughput:         {stats['requests_per_second']} req/s")
    print()
    if times:
        print(f"Latency (avg):      {stats['avg_latency']}s")
        print(f"Latency (min):      {stats['min_latency']}s")
        print(f"Latency (max):      {stats['max_latency']}s")
        print(f"Latency (p50):      {stats['p50_latency']}s")
        print(f"Latency (p95):      {stats['p95_latency']}s")
    print(f"{'='*60}\n")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Benchmark Higgs Audio vLLM")
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="vLLM server URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--requests", "-n",
        type=int,
        default=10,
        help="Number of requests (default: 10)",
    )
    parser.add_argument(
        "--concurrent", "-c",
        type=int,
        default=5,
        help="Max concurrent requests (default: 5)",
    )
    args = parser.parse_args()

    asyncio.run(run_benchmark(
        url=args.url,
        num_requests=args.requests,
        max_concurrent=args.concurrent,
    ))


if __name__ == "__main__":
    main()
