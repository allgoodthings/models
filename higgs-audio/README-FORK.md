# Higgs Audio vLLM (Custom Build)

Fork of [boson-ai/higgs-audio-vllm](https://github.com/boson-ai/higgs-audio-vllm) with automated GHCR builds.

## Why This Fork?

The official `bosonai/higgs-audio-vllm:latest` image has known issues:
- [Issue #123](https://github.com/boson-ai/higgs-audio/issues/123): References non-existent `xcodec_tps25_0215` model
- [Issue #156](https://github.com/boson-ai/higgs-audio/issues/156): Audio truncation bugs

This fork builds from source to get a working image.

## Image

```
ghcr.io/bizyb/higgs-audio-vllm:latest
```

## Usage on RunPod

1. Create a GPU Pod (RTX 4090 / A100 recommended)
2. Set container image: `ghcr.io/bizyb/higgs-audio-vllm:latest`
3. Set container command:
   ```
   --model bosonai/higgs-audio-v2-generation-3B-base
   --audio-tokenizer-type bosonai/higgs-audio-v2-tokenizer
   --served-model-name higgs-audio-v2
   --port 8000
   --gpu-memory-utilization 0.85
   --max-model-len 8192
   ```
4. Expose port 8000

## API

Once running, use the OpenAI-compatible endpoint:

```bash
curl -X POST https://<pod-url>/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "higgs-audio-v2",
    "input": "Hello, this is a test.",
    "voice": "alloy"
  }' --output test.wav
```

## Building

GitHub Actions automatically builds and pushes on every push to `main`.

To build locally:
```bash
DOCKER_BUILDKIT=1 docker build . \
  --target vllm-bosonai \
  --tag higgs-audio-vllm:local \
  --file docker/Dockerfile
```

## Syncing with Upstream

```bash
git fetch upstream
git merge upstream/main
git push origin main
```

## GitHub Actions Requirements

The workflow uses `ubuntu-latest-16-cores` runner (paid). Options:
1. **Larger runners** (recommended): Enable in repo settings → Actions → Runners
2. **Self-hosted**: Change `runs-on: self-hosted` in workflow
3. **Manual build**: Build on a VM and push manually
