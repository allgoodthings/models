# Higgs Audio vLLM with surgical fixes:
# 1. is_audio_out_model fix (tokenizer bug)
# 2. Single-pass audio generation (eliminates clicking + WAV header bugs)
FROM bosonai/higgs-audio-vllm:latest

# Clear Python bytecode cache for files we're patching
# (base image may have stale .pyc files that would be used instead of our source)
RUN find /usr/local/lib/python3.12/dist-packages/vllm -name "*.pyc" -delete && \
    find /usr/local/lib/python3.12/dist-packages/vllm -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Fix 1: get_processor() now passes is_audio_out_model=True
# when using an audio output tokenizer (non-whisper)
COPY higgs_audio.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/higgs_audio.py

# Fix 2: Single-pass audio generation (no streaming/chunking)
# - Eliminates clicking noise (no chunk boundaries)
# - Fixes WAV headers (single file with correct duration)
COPY serving_audio_patched.py /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/bosonai/serving_audio.py

# Add custom voice presets with system prompts for style control
COPY custom_voices.json /usr/local/lib/python3.12/dist-packages/vllm/entrypoints/bosonai/voice_presets/custom_voices.json

ENTRYPOINT ["python3", "-m", "vllm.entrypoints.bosonai.api_server"]
