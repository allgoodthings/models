# Higgs Audio vLLM with is_audio_out_model fix
# This patches the official bosonai/higgs-audio-vllm image to fix the tokenizer bug
FROM bosonai/higgs-audio-vllm:latest

# Apply the fix: get_processor() now passes is_audio_out_model=True
# when using an audio output tokenizer (non-whisper)
COPY higgs_audio.py /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/higgs_audio.py

ENTRYPOINT ["python3", "-m", "vllm.entrypoints.bosonai.api_server"]
