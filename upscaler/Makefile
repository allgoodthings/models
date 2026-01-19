# Video Upscaler Makefile
# =======================

.PHONY: install test lint format clean dev docker-build docker-run help

# Install dependencies
install:
	pip install -e ".[dev]"

# Run all tests
test:
	pytest tests/ -v --tb=short

# Run unit tests only
test-unit:
	pytest tests/test_schemas.py -v

# Lint code
lint:
	ruff check upscaler/ tests/
	black --check upscaler/ tests/

# Format code
format:
	ruff check --fix upscaler/ tests/
	black upscaler/ tests/

# Clean build artifacts
clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache
	rm -rf upscaler/__pycache__ tests/__pycache__
	rm -rf upscaler/server/__pycache__
	rm -rf *.egg-info dist build
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -delete

# Start development server (CPU mode - for testing)
dev-cpu:
	CUDA_VISIBLE_DEVICES="" PRELOAD_MODELS=false uvicorn upscaler.server.main:app --reload --port 8000

# Start development server (GPU mode)
dev:
	uvicorn upscaler.server.main:app --reload --port 8000

# Build Docker image
docker-build:
	docker build -t video-upscaler:latest .

# Build Docker image with deflicker support
docker-build-full:
	docker build -t video-upscaler:latest --build-arg INCLUDE_DEFLICKER=true .

# Build Docker image (skip model download for faster builds)
docker-build-fast:
	docker build -t video-upscaler:latest --build-arg SKIP_MODEL_DOWNLOAD=true .

# Run Docker container
docker-run:
	docker run --gpus all -p 8000:8000 video-upscaler:latest

# Run with docker-compose
docker-up:
	docker-compose up -d

# Stop docker-compose
docker-down:
	docker-compose down

# View logs
docker-logs:
	docker-compose logs -f

# Download models manually (useful for development)
download-models:
	mkdir -p models
	wget -q "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth" -O models/realesr-animevideov3.pth
	wget -q "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth" -O models/realesrgan-x4plus.pth
	@echo "Models downloaded to ./models/"

# Show help
help:
	@echo "Video Upscaler Commands:"
	@echo ""
	@echo "  make install          Install dependencies"
	@echo "  make test             Run all tests"
	@echo "  make test-unit        Run unit tests only"
	@echo "  make lint             Check code style"
	@echo "  make format           Auto-format code"
	@echo "  make clean            Remove build artifacts"
	@echo ""
	@echo "  make dev              Start dev server (GPU)"
	@echo "  make dev-cpu          Start dev server (CPU)"
	@echo ""
	@echo "  make docker-build     Build Docker image"
	@echo "  make docker-build-full  Build with deflicker"
	@echo "  make docker-build-fast  Build without model download"
	@echo "  make docker-run       Run Docker container"
	@echo "  make docker-up        Run with docker-compose"
	@echo "  make docker-down      Stop docker-compose"
	@echo "  make docker-logs      View logs"
	@echo ""
	@echo "  make download-models  Download models locally"
