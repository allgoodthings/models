#!/bin/bash
# Build custom Higgs Audio vLLM Docker image
#
# This builds from the public boson-ai/higgs-audio-vllm fork
# to avoid the broken official image.
#
# Usage:
#   ./build-vllm.sh              # Build locally
#   ./build-vllm.sh --push       # Build and push to registry

set -e

# Configuration
IMAGE_NAME="${IMAGE_NAME:-higgs-audio-vllm}"
IMAGE_TAG="${IMAGE_TAG:-custom}"
REGISTRY="${REGISTRY:-}"  # e.g., "ghcr.io/your-org" or "your-dockerhub-username"

FULL_IMAGE="${REGISTRY:+$REGISTRY/}${IMAGE_NAME}:${IMAGE_TAG}"

echo "=== Building Higgs Audio vLLM Docker Image ==="
echo "Image: $FULL_IMAGE"
echo ""

# Clone the vLLM fork if not present
VLLM_DIR="higgs-audio-vllm"
if [ ! -d "$VLLM_DIR" ]; then
    echo "Cloning boson-ai/higgs-audio-vllm..."
    git clone https://github.com/boson-ai/higgs-audio-vllm.git "$VLLM_DIR"
else
    echo "Updating existing clone..."
    cd "$VLLM_DIR"
    git pull
    cd ..
fi

cd "$VLLM_DIR"

echo ""
echo "Building Docker image (this will take a while)..."
echo ""

# Build the vllm-bosonai target (audio-enabled version)
DOCKER_BUILDKIT=1 docker build . \
    --target vllm-bosonai \
    --tag "$FULL_IMAGE" \
    --file docker/Dockerfile \
    --build-arg CUDA_VERSION=12.4.1 \
    --build-arg PYTHON_VERSION=3.12

echo ""
echo "=== Build Complete ==="
echo "Image: $FULL_IMAGE"
echo ""

# Push if requested
if [ "$1" = "--push" ]; then
    if [ -z "$REGISTRY" ]; then
        echo "ERROR: Set REGISTRY environment variable to push"
        echo "  Example: REGISTRY=ghcr.io/your-org ./build-vllm.sh --push"
        exit 1
    fi
    echo "Pushing to registry..."
    docker push "$FULL_IMAGE"
    echo "Pushed: $FULL_IMAGE"
fi

echo ""
echo "To run locally:"
echo "  docker run --gpus all --ipc=host --shm-size=20gb -p 8000:8000 $FULL_IMAGE \\"
echo "    --model bosonai/higgs-audio-v2-generation-3B-base \\"
echo "    --audio-tokenizer-type bosonai/higgs-audio-v2-tokenizer \\"
echo "    --served-model-name higgs-audio-v2 \\"
echo "    --port 8000"
echo ""
