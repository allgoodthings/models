# Scenema AI Models

Collection of AI model pipelines for video generation and processing.

## Available Models

| Model | Description | Docker Image |
|-------|-------------|--------------|
| [lip-sync](./lip-sync/) | Multi-face lip-sync using MuseTalk + LivePortrait + CodeFormer | `ghcr.io/allgoodthings/lip-sync` |

## Usage

Each model is self-contained in its own directory with:
- Dockerfile for containerized deployment
- FastAPI server for HTTP API
- Model download scripts
- Documentation

See individual model READMEs for specific usage instructions.

## Docker Images

All images are published to GitHub Container Registry (GHCR):

```bash
# Pull a specific model
docker pull ghcr.io/allgoodthings/lip-sync:latest

# Run with GPU
docker run --gpus all -p 8000:8000 ghcr.io/allgoodthings/lip-sync:latest
```

## Development

```bash
# Clone the repo
git clone https://github.com/allgoodthings/models.git
cd models

# Work on a specific model
cd lip-sync
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## License

MIT
