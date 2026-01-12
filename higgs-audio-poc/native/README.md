# Higgs Audio Native POC

Native implementation using `boson_multimodal` library directly (not vLLM).

## Purpose

Test whether `system_prompt` (emotional expression) works together with voice cloning in the native Higgs Audio implementation.

## API

### POST /generate

```json
{
  "text": "Text to synthesize",
  "system_prompt": "Generate audio following instruction.\n\n<|scene_desc_start|>\nEmotional description here.\n<|scene_desc_end|>",
  "voice_id": "unique-voice-id",
  "voice_url": "https://cdn.example.com/voice.wav",
  "reference_text": "Transcript of the reference audio"
}
```

**Priority:**
- `system_prompt`: Always used when provided (controls emotion/style)
- `voice_id` + `voice_url`: Voice cloning with disk caching
- `reference_audio_base64`: Legacy direct audio (no caching)

## Docker

```bash
# Build
docker build -t higgs-audio-native .

# Run (requires NVIDIA GPU)
docker run --gpus all -p 8000:8000 higgs-audio-native
```

## Test Matrix

| Test | system_prompt | voice_clone | Expected Result |
|------|--------------|-------------|-----------------|
| 1 | Yes | No | Emotional expression works |
| 2 | No | Yes | Voice cloning works |
| 3 | Yes | Yes | **Critical test** - both work together? |
| 4 | Same voice, different emotions | Yes | Different emotional expressions |

See `test-requests.http` for test cases.
