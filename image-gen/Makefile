# Image-Gen Service Makefile
# FLUX.2 klein image generation

.PHONY: install install-dev test lint format clean run benchmark

# Python and pip
PYTHON := python3
PIP := pip3
UVICORN := uvicorn

# Default target
all: install

# Install production dependencies
install:
	$(PIP) install -e .

# Install with dev dependencies
install-dev:
	$(PIP) install -e ".[dev]"

# Install with uv (faster)
install-uv:
	uv pip install -e .

install-uv-dev:
	uv pip install -e ".[dev]"

# Run the server
run:
	$(UVICORN) imagegen.server.main:app --host 0.0.0.0 --port 7000

run-reload:
	$(UVICORN) imagegen.server.main:app --host 0.0.0.0 --port 7000 --reload

# Run VRAM benchmark (for Vast AI testing)
benchmark:
	$(PYTHON) scripts/benchmark_vram.py

# Run tests
test:
	pytest tests/ -v

test-cov:
	pytest tests/ -v --cov=imagegen --cov-report=term-missing

# Lint with ruff
lint:
	ruff check imagegen/ scripts/ tests/

lint-fix:
	ruff check --fix imagegen/ scripts/ tests/

# Format with black
format:
	black imagegen/ scripts/ tests/

format-check:
	black --check imagegen/ scripts/ tests/

# Clean build artifacts
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .ruff_cache/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

# Health check
health:
	curl -s http://localhost:7000/health | python3 -m json.tool

# Test API manually
test-api:
	$(PYTHON) scripts/test_api.py

# Help
help:
	@echo "Image-Gen Service - FLUX.2 klein"
	@echo ""
	@echo "Commands:"
	@echo "  make install      - Install production dependencies"
	@echo "  make install-dev  - Install with dev dependencies"
	@echo "  make run          - Run the server"
	@echo "  make run-reload   - Run with auto-reload"
	@echo "  make benchmark    - Run VRAM benchmark script"
	@echo "  make test         - Run tests"
	@echo "  make lint         - Run linter"
	@echo "  make format       - Format code"
	@echo "  make clean        - Clean build artifacts"
	@echo "  make health       - Check server health"
	@echo "  make test-api     - Run API test script"
