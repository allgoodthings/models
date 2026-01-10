#!/usr/bin/env python3
"""
Higgs Audio v2 vLLM Benchmark Script

Tests concurrent request handling and measures throughput.

Usage:
    python benchmark.py                    # Default: 10 concurrent requests
    python benchmark.py --concurrent 20   # 20 concurrent requests
    python benchmark.py --requests 50     # 50 total requests
"""

import argparse
import asyncio
import time
import statistics
from datetime import datetime

import httpx

# Test sentences of varying lengths
TEST_TEXTS = [
    "Hello, this is a test.",
    "The quick brown fox jumps over the lazy dog.",
    "To be, or not to be, that is the question.",
    "In the beginning, there was light, and it was good.",
    "The sun rises in the east and sets in the west. This simple fact has been observed by humans for thousands of years.",
    "I can't believe you did that without even asking me first! You always make decisions without considering my opinion.",
    "Welcome to the future of artificial intelligence. Today we explore the boundaries of what machines can achieve.",
    "Once upon a time, in a land far away, there lived a young princess who dreamed of adventure beyond the castle walls.",
]

# Emotional prompts to test variety
SYSTEM_PROMPTS = [
    None,  # Default
    "Generate audio following instruction.\n\n<|scene_desc_start|>\nThe speaker sounds happy and excited.\n<|scene_desc_end|>",
    "Generate audio following instruction.\n\n<|scene_desc_start|>\nThe speaker sounds sad and melancholic.\n<|scene_desc_end|>",
    "Generate audio following instruction.\n\n<|scene_desc_start|>\nA dramatic theatrical performance.\n<|scene_desc_end|>",
]


async def make_request(client: httpx.AsyncClient, url: str, text: str, system_prompt: str = None, request_id: int = 0):
    """Make a single TTS request and return timing info."""
    payload = {
        "text": text,
        "temperature": 0.5,
    }
    if system_prompt:
        payload["system_prompt"] = system_prompt

    start = time.time()
    try:
        resp = await client.post(f"{url}/generate", json=payload)
        elapsed = time.time() - start

        if resp.status_code == 200:
            audio_size = len(resp.content)
            gen_time = float(resp.headers.get("X-Generation-Time", elapsed))
            return {
                "request_id": request_id,
                "success": True,
                "elapsed": elapsed,
                "generation_time": gen_time,
                "audio_size": audio_size,
                "text_length": len(text),
            }
        else:
            return {
                "request_id": request_id,
                "success": False,
                "elapsed": elapsed,
                "error": resp.text[:100],
            }
    except Exception as e:
        return {
            "request_id": request_id,
            "success": False,
            "elapsed": time.time() - start,
            "error": str(e)[:100],
        }


async def run_benchmark(url: str, num_requests: int, concurrency: int):
    """Run the benchmark with specified concurrency."""
    print(f"\n{'='*60}")
    print(f"Higgs Audio v2 vLLM Benchmark")
    print(f"{'='*60}")
    print(f"URL: {url}")
    print(f"Total Requests: {num_requests}")
    print(f"Concurrency: {concurrency}")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"{'='*60}\n")

    # Check health first
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(f"{url}/health")
            health = resp.json()
            print(f"Backend Status: {health.get('status', 'unknown')}")
            print(f"vLLM Healthy: {health.get('vllm_healthy', False)}")
            if not health.get('vllm_healthy'):
                print("ERROR: vLLM backend is not healthy. Start it with ./start-vllm.sh")
                return
        except Exception as e:
            print(f"ERROR: Cannot connect to server: {e}")
            return

    # Prepare requests
    requests_to_make = []
    for i in range(num_requests):
        text = TEST_TEXTS[i % len(TEST_TEXTS)]
        prompt = SYSTEM_PROMPTS[i % len(SYSTEM_PROMPTS)]
        requests_to_make.append((text, prompt, i))

    # Run with concurrency limit
    semaphore = asyncio.Semaphore(concurrency)
    results = []

    async def bounded_request(client, text, prompt, req_id):
        async with semaphore:
            return await make_request(client, url, text, prompt, req_id)

    print(f"Starting {num_requests} requests with concurrency {concurrency}...\n")
    overall_start = time.time()

    async with httpx.AsyncClient(timeout=120.0) as client:
        tasks = [bounded_request(client, text, prompt, req_id)
                 for text, prompt, req_id in requests_to_make]
        results = await asyncio.gather(*tasks)

    overall_elapsed = time.time() - overall_start

    # Analyze results
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"Total Requests: {num_requests}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Success Rate: {len(successful)/num_requests*100:.1f}%")
    print(f"Total Time: {overall_elapsed:.2f}s")
    print(f"Throughput: {num_requests/overall_elapsed:.2f} requests/sec")

    if successful:
        gen_times = [r["generation_time"] for r in successful]
        elapsed_times = [r["elapsed"] for r in successful]
        audio_sizes = [r["audio_size"] for r in successful]

        print(f"\nGeneration Time (server-side):")
        print(f"  Min: {min(gen_times):.2f}s")
        print(f"  Max: {max(gen_times):.2f}s")
        print(f"  Mean: {statistics.mean(gen_times):.2f}s")
        print(f"  Median: {statistics.median(gen_times):.2f}s")
        if len(gen_times) > 1:
            print(f"  Std Dev: {statistics.stdev(gen_times):.2f}s")

        print(f"\nRound-Trip Time (including network):")
        print(f"  Min: {min(elapsed_times):.2f}s")
        print(f"  Max: {max(elapsed_times):.2f}s")
        print(f"  Mean: {statistics.mean(elapsed_times):.2f}s")

        print(f"\nAudio Output:")
        print(f"  Total: {sum(audio_sizes)/1024/1024:.2f} MB")
        print(f"  Avg per request: {statistics.mean(audio_sizes)/1024:.1f} KB")

    if failed:
        print(f"\nFailed Requests:")
        for r in failed[:5]:  # Show first 5 failures
            print(f"  Request {r['request_id']}: {r.get('error', 'Unknown error')}")
        if len(failed) > 5:
            print(f"  ... and {len(failed)-5} more failures")

    # Get final stats from server
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            resp = await client.get(f"{url}/stats")
            stats = resp.json()
            print(f"\nServer Stats:")
            print(f"  Peak Concurrent: {stats.get('peak_concurrent', 'N/A')}")
            print(f"  Avg Generation Time: {stats.get('avg_generation_time', 'N/A')}s")
        except:
            pass

    print(f"\n{'='*60}")
    print(f"Benchmark Complete")
    print(f"{'='*60}\n")

    return results


def main():
    parser = argparse.ArgumentParser(description="Benchmark Higgs Audio vLLM")
    parser.add_argument("--url", default="http://localhost:8888", help="Server URL")
    parser.add_argument("--requests", "-n", type=int, default=10, help="Total requests")
    parser.add_argument("--concurrent", "-c", type=int, default=5, help="Concurrent requests")
    args = parser.parse_args()

    asyncio.run(run_benchmark(args.url, args.requests, args.concurrent))


if __name__ == "__main__":
    main()
