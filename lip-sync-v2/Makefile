# Lip-Sync Pipeline Makefile
# ===========================

.PHONY: install test test-unit test-integration test-cpu lint format clean

# Install dependencies
install:
	pip install -e ".[dev]"

# Run all tests (fast - no GPU required)
test:
	pytest tests/ -v --tb=short

# Run unit tests only (fastest - no heavy deps needed)
test-unit:
	pytest tests/test_segment_builder.py tests/test_schemas.py -v

# Run integration tests with mocked models
test-integration:
	pytest tests/test_api.py -v

# Run full CPU validation (slow - full pipeline on CPU)
test-cpu:
	python scripts/test-cpu-validation.py

# Run CPU validation skipping lip-sync (faster)
test-cpu-quick:
	python scripts/test-cpu-validation.py --skip-lipsync

# Lint code
lint:
	ruff check lipsync/ tests/
	black --check lipsync/ tests/

# Format code
format:
	ruff check --fix lipsync/ tests/
	black lipsync/ tests/

# Clean build artifacts
clean:
	rm -rf __pycache__ .pytest_cache .ruff_cache
	rm -rf lipsync/__pycache__ tests/__pycache__
	rm -rf *.egg-info dist build
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -delete

# Start development server (CPU mode)
dev-cpu:
	CUDA_VISIBLE_DEVICES="" uvicorn lipsync.server.main:app --reload --port 8000

# Start development server (GPU mode)
dev:
	uvicorn lipsync.server.main:app --reload --port 8000

# Build Docker image
docker-build:
	docker build -t lip-sync:latest .

# Run Docker container
docker-run:
	docker run --gpus all -p 8000:8000 lip-sync:latest

# Show help
help:
	@echo "Lip-Sync Pipeline Commands:"
	@echo ""
	@echo "  make install          Install dependencies"
	@echo "  make test             Run all tests (fast)"
	@echo "  make test-unit        Run unit tests only"
	@echo "  make test-integration Run API tests with mocks"
	@echo "  make test-cpu         Run full CPU validation (slow)"
	@echo "  make test-cpu-quick   Run CPU validation without lip-sync"
	@echo "  make lint             Check code style"
	@echo "  make format           Auto-format code"
	@echo "  make clean            Remove build artifacts"
	@echo "  make dev-cpu          Start dev server (CPU)"
	@echo "  make dev              Start dev server (GPU)"
	@echo "  make docker-build     Build Docker image"
	@echo "  make docker-run       Run Docker container"
