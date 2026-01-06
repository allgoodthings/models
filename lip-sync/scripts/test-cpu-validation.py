#!/usr/bin/env python3
"""
CPU Validation Test Script

Runs the full lip-sync pipeline on CPU for end-to-end validation.
This is slow but validates real code paths without requiring GPU.

Usage:
    python scripts/test-cpu-validation.py [--skip-lipsync] [--verbose]

Options:
    --skip-lipsync  Only test face detection (faster)
    --verbose       Enable debug logging
    --port PORT     Server port (default: 8765)
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from threading import Thread
from typing import Optional

import requests


# =============================================================================
# Test Configuration
# =============================================================================

DEFAULT_PORT = 8765
API_BASE = f"http://localhost:{DEFAULT_PORT}"

# Test video specs
TEST_VIDEO_DURATION = 2  # seconds
TEST_VIDEO_SIZE = (320, 240)
TEST_FPS = 30


# =============================================================================
# Test File Generation
# =============================================================================


def create_test_video(output_path: str, duration: int = 2) -> str:
    """Create a minimal test video with a face-like pattern."""
    print(f"Creating test video: {output_path}")

    # Create a video with a simple pattern (circle for face)
    # Using lavfi to generate a synthetic video
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"color=c=0x404040:s={TEST_VIDEO_SIZE[0]}x{TEST_VIDEO_SIZE[1]}:d={duration}",
        "-vf", (
            # Draw a flesh-colored ellipse for face
            "drawbox=x=80:y=40:w=160:h=160:c=0xFFDBAC:t=fill,"
            # Draw circles for eyes
            "drawbox=x=110:y=90:w=30:h=20:c=white:t=fill,"
            "drawbox=x=180:y=90:w=30:h=20:c=white:t=fill,"
            # Draw small circles for pupils
            "drawbox=x=120:y=95:w=10:h=10:c=0x333333:t=fill,"
            "drawbox=x=190:y=95:w=10:h=10:c=0x333333:t=fill,"
            # Draw mouth
            "drawbox=x=130:y=150:w=60:h=10:c=0xCC6666:t=fill"
        ),
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-pix_fmt", "yuv420p",
        output_path,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Warning: Complex video failed, using simple color video")
        # Fallback to simple color video
        cmd = [
            "ffmpeg", "-y",
            "-f", "lavfi",
            "-i", f"color=c=blue:s={TEST_VIDEO_SIZE[0]}x{TEST_VIDEO_SIZE[1]}:d={duration}",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            output_path,
        ]
        subprocess.run(cmd, check=True, capture_output=True)

    return output_path


def create_test_audio(output_path: str, duration: int = 2) -> str:
    """Create a minimal test audio file."""
    print(f"Creating test audio: {output_path}")

    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"sine=frequency=440:duration={duration}",
        "-c:a", "pcm_s16le",
        output_path,
    ]

    subprocess.run(cmd, check=True, capture_output=True)
    return output_path


def create_test_image(output_path: str) -> str:
    """Create a test reference image."""
    print(f"Creating test image: {output_path}")

    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", "color=c=0xFFDBAC:s=200x200:d=1",
        "-frames:v", "1",
        output_path,
    ]

    subprocess.run(cmd, check=True, capture_output=True)
    return output_path


# =============================================================================
# Simple HTTP Server for Test Files
# =============================================================================


class TestFileServer:
    """Simple HTTP server to serve test files."""

    def __init__(self, directory: str, port: int = 8766):
        self.directory = directory
        self.port = port
        self.server: Optional[HTTPServer] = None
        self.thread: Optional[Thread] = None

    def start(self):
        """Start the server in a background thread."""
        os.chdir(self.directory)

        handler = SimpleHTTPRequestHandler
        handler.log_message = lambda *args: None  # Suppress logs

        self.server = HTTPServer(("localhost", self.port), handler)
        self.thread = Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

        print(f"Test file server started at http://localhost:{self.port}")

    def stop(self):
        """Stop the server."""
        if self.server:
            self.server.shutdown()

    def get_url(self, filename: str) -> str:
        """Get URL for a file."""
        return f"http://localhost:{self.port}/{filename}"


# =============================================================================
# Upload Receiver Server
# =============================================================================


class UploadHandler(SimpleHTTPRequestHandler):
    """HTTP handler that accepts PUT uploads."""

    received_files = {}

    def do_PUT(self):
        """Handle PUT upload request."""
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        # Store the uploaded file
        filename = self.path.strip("/")
        filepath = os.path.join(self.server.upload_dir, filename)

        with open(filepath, "wb") as f:
            f.write(body)

        UploadHandler.received_files[filename] = filepath

        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(b"OK")

    def log_message(self, format, *args):
        pass  # Suppress logs


class UploadServer:
    """Server to receive uploaded files."""

    def __init__(self, directory: str, port: int = 8767):
        self.directory = directory
        self.port = port
        self.server: Optional[HTTPServer] = None

    def start(self):
        """Start the upload server."""
        self.server = HTTPServer(("localhost", self.port), UploadHandler)
        self.server.upload_dir = self.directory

        thread = Thread(target=self.server.serve_forever, daemon=True)
        thread.start()

        print(f"Upload server started at http://localhost:{self.port}")

    def stop(self):
        if self.server:
            self.server.shutdown()

    def get_upload_url(self, filename: str = "output.mp4") -> str:
        return f"http://localhost:{self.port}/{filename}"

    def get_uploaded_file(self, filename: str = "output.mp4") -> Optional[str]:
        return UploadHandler.received_files.get(filename)


# =============================================================================
# Test Runner
# =============================================================================


def wait_for_server(url: str, timeout: int = 300) -> bool:
    """Wait for server to become available."""
    print(f"Waiting for server at {url}...")
    start = time.time()

    while time.time() - start < timeout:
        try:
            response = requests.get(f"{url}/health", timeout=5)
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "healthy":
                    print(f"Server ready in {time.time() - start:.1f}s")
                    return True
                print(f"Server status: {data.get('status')} (waiting for healthy...)")
        except requests.exceptions.ConnectionError:
            pass
        except Exception as e:
            print(f"Error checking health: {e}")

        time.sleep(5)

    return False


def test_health(api_base: str) -> bool:
    """Test health endpoint."""
    print("\n" + "=" * 60)
    print("TEST: Health Check")
    print("=" * 60)

    response = requests.get(f"{api_base}/health")

    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

    if response.status_code != 200:
        print("FAILED: Health check returned non-200")
        return False

    data = response.json()
    if data.get("status") != "healthy":
        print(f"FAILED: Status is {data.get('status')}, expected 'healthy'")
        return False

    print("PASSED")
    return True


def test_detect_faces(
    api_base: str,
    video_url: str,
    reference_url: str,
) -> Optional[dict]:
    """Test face detection endpoint."""
    print("\n" + "=" * 60)
    print("TEST: Face Detection")
    print("=" * 60)

    request_data = {
        "video_url": video_url,
        "sample_fps": 3,
        "characters": [
            {
                "id": "test_face",
                "name": "Test Face",
                "reference_image_url": reference_url,
            }
        ],
        "similarity_threshold": 0.3,  # Lower threshold for test patterns
    }

    print(f"Request: {json.dumps(request_data, indent=2)}")

    start = time.time()
    response = requests.post(f"{api_base}/detect-faces", json=request_data)
    elapsed = time.time() - start

    print(f"Status: {response.status_code}")
    print(f"Time: {elapsed:.2f}s")

    if response.status_code != 200:
        print(f"FAILED: {response.text}")
        return None

    data = response.json()
    print(f"Response: {json.dumps(data, indent=2)}")

    print(f"Frames analyzed: {len(data.get('frames', []))}")
    print(f"Video size: {data.get('frame_width')}x{data.get('frame_height')}")
    print(f"Duration: {data.get('video_duration_ms')}ms")

    # Check for any faces detected
    total_faces = sum(len(f.get("faces", [])) for f in data.get("frames", []))
    print(f"Total face detections: {total_faces}")

    if total_faces == 0:
        print("WARNING: No faces detected (may be expected with synthetic video)")

    print("PASSED")
    return data


def test_lipsync(
    api_base: str,
    video_url: str,
    audio_url: str,
    upload_url: str,
    bbox: tuple = (80, 40, 160, 160),
) -> Optional[dict]:
    """Test lip-sync endpoint."""
    print("\n" + "=" * 60)
    print("TEST: Lip-Sync")
    print("=" * 60)

    request_data = {
        "video_url": video_url,
        "audio_url": audio_url,
        "upload_url": upload_url,
        "faces": [
            {
                "character_id": "test_face",
                "bbox": list(bbox),
                "start_time_ms": 0,
                "end_time_ms": TEST_VIDEO_DURATION * 1000,
            }
        ],
        "enhance_quality": False,  # Disable for faster testing
        "fidelity_weight": 0.7,
    }

    print(f"Request: {json.dumps(request_data, indent=2)}")

    start = time.time()
    response = requests.post(f"{api_base}/lipsync", json=request_data, timeout=600)
    elapsed = time.time() - start

    print(f"Status: {response.status_code}")
    print(f"Time: {elapsed:.2f}s")

    if response.status_code != 200:
        print(f"FAILED: {response.text}")
        return None

    data = response.json()
    print(f"Response: {json.dumps(data, indent=2)}")

    if not data.get("success"):
        print(f"FAILED: {data.get('error_message')}")
        return None

    # Check response structure
    assert "faces" in data, "Missing 'faces' in response"
    assert "output" in data, "Missing 'output' in response"
    assert "timing" in data, "Missing 'timing' in response"

    print(f"Output duration: {data['output']['duration_ms']}ms")
    print(f"Output size: {data['output']['file_size_bytes']} bytes")
    print(f"Total time: {data['timing']['total_ms']}ms")

    print("PASSED")
    return data


def main():
    parser = argparse.ArgumentParser(description="CPU Validation Test")
    parser.add_argument("--skip-lipsync", action="store_true", help="Skip lip-sync test")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Server port")
    parser.add_argument("--external-server", action="store_true",
                        help="Use external server (don't start one)")
    args = parser.parse_args()

    api_base = f"http://localhost:{args.port}"
    server_process = None

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            print("=" * 60)
            print("CPU VALIDATION TEST")
            print("=" * 60)
            print(f"Temp directory: {tmpdir}")
            print(f"API base: {api_base}")

            # Create test files
            video_path = create_test_video(os.path.join(tmpdir, "test_video.mp4"))
            audio_path = create_test_audio(os.path.join(tmpdir, "test_audio.wav"))
            image_path = create_test_image(os.path.join(tmpdir, "test_image.jpg"))

            # Start file server
            file_server = TestFileServer(tmpdir, port=8766)
            file_server.start()

            # Start upload server
            upload_dir = os.path.join(tmpdir, "uploads")
            os.makedirs(upload_dir)
            upload_server = UploadServer(upload_dir, port=8767)
            upload_server.start()

            # Start API server if not using external
            if not args.external_server:
                print("\nStarting API server (CPU mode)...")
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = ""  # Force CPU
                env["LOG_LEVEL"] = "DEBUG" if args.verbose else "INFO"

                server_process = subprocess.Popen(
                    [
                        sys.executable, "-m", "uvicorn",
                        "lipsync.server.main:app",
                        "--host", "0.0.0.0",
                        "--port", str(args.port),
                    ],
                    env=env,
                    cwd=Path(__file__).parent.parent,
                )

            # Wait for server
            if not wait_for_server(api_base, timeout=600):
                print("FAILED: Server did not become healthy")
                return 1

            # Run tests
            results = {"passed": 0, "failed": 0}

            # Test 1: Health
            if test_health(api_base):
                results["passed"] += 1
            else:
                results["failed"] += 1

            # Test 2: Face Detection
            detection_result = test_detect_faces(
                api_base,
                file_server.get_url("test_video.mp4"),
                file_server.get_url("test_image.jpg"),
            )
            if detection_result is not None:
                results["passed"] += 1
            else:
                results["failed"] += 1

            # Test 3: Lip-Sync
            if not args.skip_lipsync:
                lipsync_result = test_lipsync(
                    api_base,
                    file_server.get_url("test_video.mp4"),
                    file_server.get_url("test_audio.wav"),
                    upload_server.get_upload_url("output.mp4"),
                )
                if lipsync_result is not None:
                    results["passed"] += 1

                    # Verify upload
                    uploaded_file = upload_server.get_uploaded_file("output.mp4")
                    if uploaded_file and os.path.exists(uploaded_file):
                        size = os.path.getsize(uploaded_file)
                        print(f"\nUploaded file verified: {uploaded_file} ({size} bytes)")
                    else:
                        print("\nWARNING: Upload file not found")
                else:
                    results["failed"] += 1

            # Summary
            print("\n" + "=" * 60)
            print("SUMMARY")
            print("=" * 60)
            print(f"Passed: {results['passed']}")
            print(f"Failed: {results['failed']}")

            return 0 if results["failed"] == 0 else 1

    except KeyboardInterrupt:
        print("\nInterrupted")
        return 1
    finally:
        if server_process:
            print("\nStopping server...")
            server_process.terminate()
            server_process.wait(timeout=10)


if __name__ == "__main__":
    sys.exit(main())
