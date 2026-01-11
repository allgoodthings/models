# Higgs Audio v3 - Integrated vLLM Server
# Uses our v2 image (with is_audio_out_model fix) as base
# Adds our clean audio decoding server

FROM ghcr.io/allgoodthings/higgs-audio-vllm:latest

# Copy our server (replaces their buggy audio endpoint with proper decoding)
COPY server_integrated.py /app/server.py

# Expose our port
EXPOSE 8080

# Run our server instead of theirs
CMD ["python3", "/app/server.py"]
