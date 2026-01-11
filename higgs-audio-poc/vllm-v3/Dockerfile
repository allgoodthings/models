# Higgs Audio v3 - Integrated vLLM Server
# Layers our clean audio decoding on top of bosonai's vLLM image

FROM bosonai/higgs-audio-vllm:latest

# Copy our server (replaces their buggy audio endpoint)
COPY server_integrated.py /app/server.py

# Expose our port
EXPOSE 8080

# Run our server instead of theirs
CMD ["python3", "/app/server.py"]
