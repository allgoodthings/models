# TTS Research Analysis

## Summary

This document captures our research into Text-to-Speech solutions for Scenema, focusing on:
- Voice cloning (per-character voices)
- Emotional expression (scared, angry, excited, sad, etc.)
- Cost efficiency
- Quality

---

## Key Findings

### The Core Problem

**No open-source TTS in 2025 has reliable, granular emotion control.**

| Model | Emotion Control | Reality |
|-------|-----------------|---------|
| Higgs Audio v2 | "Emergent from text semantics" | Undocumented, unreliable |
| Chatterbox | `exaggeration` slider (0-2) | Single intensity dial, no emotion types |
| Fish Audio S1 | `(angry)` `(whispered)` tags | Best, but NC license (API only) |

---

## Model Comparison

### Cost Analysis

| Option | Type | Cost Model | $/Hr Audio |
|--------|------|------------|------------|
| **Fish Audio** | API | $15/1M UTF-8 bytes | $0.75 |
| **Google Chirp 3** | API | $30/1M chars | $1.50 |
| **Chatterbox** | Self-hosted | $0.25/hr GPU | $0.08 |
| **Higgs + vLLM** | Self-hosted | $0.25/hr GPU | $0.03-0.05 |

**Note:** Fish Audio charges per UTF-8 byte, making it 3x more expensive for CJK languages.

### Self-Hosted vs API Break-Even

At $0.25/hr GPU (~$180/mo always-on):

| Daily Audio | GPU (on-demand) | Fish Audio | Google Chirp |
|-------------|-----------------|------------|--------------|
| 1 hr/day | $3-5/mo | $22/mo | $45/mo |
| 5 hrs/day | $15-25/mo | $112/mo | $225/mo |
| 10 hrs/day | $30-50/mo | $225/mo | $450/mo |

**Self-hosted wins at any volume** for quality APIs.

---

## Higgs Audio v2

### Strengths
- Excellent voice cloning (5-10s reference)
- Best naturalness/expressiveness when it works
- Multi-speaker dialogue native support
- Apache 2.0 license
- 75.7% win rate on EmergentTTS-Eval emotions benchmark

### Critical Limitation: Emotion Control is Broken

**Official position:** Emotions are "emergent, not directly controllable" ([GitHub Issue #40](https://github.com/boson-ai/higgs-audio/issues/40))

**What we tested:**

| Prompt Style | Example | Result |
|--------------|---------|--------|
| Emotion names | `angry`, `scared` | No effect |
| Scene descriptions | `SPEAKER0: whispered;breathy;trembling` | Inconsistent |
| Natural language | "sounds excited and energetic" | Minimal effect |

**Root cause:** The base model (`higgs-audio-v2-generation-3B-base`) has NOT been post-trained. Emotions are inferred from text content semantics, not explicit tags.

**Maintainer response:** "We are working on a post-trained model" - no timeline.

### Scene Description Format (Undocumented)

From source code analysis, the only confirmed tags are sound effects:
- `[laugh]` → `<SE>[Laughter]</SE>`
- `[humming start]` / `[humming end]`
- `[applause]`, `[cheering]`, `[cough]`

**NO confirmed emotion tags like `[angry]` or `[whisper]`.**

### How Emotions Actually Work

The model relies on **text semantics** - the emotional content of the words themselves:
- "I can't believe you did this!" → infers anger from content
- "Please, someone help me" → infers fear from content

This means:
- ✅ Works for obviously emotional text
- ❌ Cannot make neutral text sound angry
- ❌ Cannot control intensity or specific emotion type

### Future Research Needed

1. Test if emotional reference audio affects output emotion
2. Explore fine-tuning on emotional speech datasets (ESD)
3. Monitor for official post-trained model release
4. Test temperature/sampling parameters for emotion variation

---

## Chatterbox

### Strengths
- MIT license (fully permissive)
- Production-tested, stable
- `exaggeration` parameter works
- `[laugh]`, `[cough]`, `[chuckle]` tags (Turbo)
- Lower VRAM (8-16GB vs 16-18GB for Higgs)

### Critical Limitation: Single Emotion Dial

**Only two parameters:**
- `exaggeration` (0.0 - 2.0): Intensity of expression
- `cfg_weight` (0.0 - 1.0): Pacing control

**Cannot specify emotion type** - only intensity.

### Best Practices

| Desired Effect | Exaggeration | CFG Weight |
|----------------|--------------|------------|
| Neutral/Calm | 0.3 - 0.5 | 0.5 |
| Sad/Subdued | 0.25 - 0.4 | 0.5 - 0.6 |
| Angry/Intense | 0.7 - 1.0 | 0.3 - 0.4 |
| Excited/Energetic | 0.7 - 0.9 | 0.4 - 0.5 |
| Whispered/Scared | 0.3 - 0.5 | 0.3 |

**Key insight:** Higher exaggeration speeds up speech. Counter with lower cfg_weight.

### Voice Cloning Requirements
- 10+ seconds of audio
- WAV format, 24kHz+ sample rate
- Single speaker, no background noise
- Match reference emotion to desired output

### Limitations
- 40 second max generation
- >350 chars can cause hallucinations
- Non-English quality varies
- Cannot do "whispered but fast" or "loud but slow"

---

## Fish Audio API (Selected Solution)

> **Decision:** Fish Audio API is our primary TTS solution due to best-in-class emotion control, reasonable cost, and high concurrency support.

### Strengths
- **Best emotion control** with 64+ explicit tags
- #1 on TTS-Arena benchmark
- Voice cloning from 15-30s audio (instant, no fine-tuning)
- 70+ languages with emotion support for 13 languages
- High concurrency / low latency
- Streaming support via WebSocket

### Pricing
- $15 / 1M UTF-8 bytes
- English: 1 byte per character (~$0.75/hr audio)
- CJK: 3 bytes per character (~$2.25/hr audio)

### Models
| Model | Parameters | Use Case |
|-------|------------|----------|
| OpenAudio S1 | 4B | Full features, highest quality |
| OpenAudio S1-mini | 0.5B | Faster inference, good quality |

---

## Fish Audio API Implementation Guide

### Installation

```bash
pip install fish-audio-sdk
# Or with audio playback utilities
pip install fish-audio-sdk[utils]
```

### Basic Usage

```python
from fish_audio_sdk import Session, TTSRequest

session = Session(apikey="YOUR_API_KEY")

# Basic TTS
request = TTSRequest(text="Hello, world!")
for chunk in session.tts(request):
    # Process audio chunk (bytes)
    pass
```

### Voice Cloning

**Option 1: One-off cloning with reference audio**
```python
from fish_audio_sdk import Session, TTSRequest, ReferenceAudio

session = Session(apikey="YOUR_API_KEY")

# Load reference audio (15-30 seconds recommended)
with open("character_voice.wav", "rb") as f:
    voice_sample = f.read()

request = TTSRequest(
    text="(excited) This is amazing!",
    references=[ReferenceAudio(
        audio=voice_sample,
        text="The exact transcript of the reference audio"
    )]
)

audio_chunks = list(session.tts(request))
```

**Option 2: Persistent voice model**
```python
# Create voice model once
voice = session.voices.create(
    title="Character Name",
    voices=[ReferenceAudio(audio=voice_sample, text="transcript")],
    description="Male protagonist, 30s"
)

# Use voice_id in subsequent requests
request = TTSRequest(
    text="(angry) You betrayed me!",
    reference_id=voice.id
)
```

### TTSRequest Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | str | required | Text to synthesize (with emotion tags) |
| `reference_id` | str | None | Persistent voice model ID |
| `references` | list | None | One-off ReferenceAudio objects |
| `format` | str | "mp3" | Output: "wav", "mp3", "pcm" |
| `sample_rate` | int | None | Custom sample rate |
| `chunk_length` | int | 200 | 100-300 |
| `top_p` | float | 0.7 | 0.0-1.0 |
| `temperature` | float | 0.7 | 0.0-1.0 |
| `latency` | str | "balanced" | "normal" or "balanced" |

### Prosody Control

```python
from fish_audio_sdk import Prosody

request = TTSRequest(
    text="(calm) Take your time.",
    prosody=Prosody(
        speed=0.9,    # 0.5-2.0 (1.0 = normal)
        volume=-3.0   # dB adjustment
    )
)
```

---

## Fish Audio Emotion Reference

### Basic Emotions (24)
| Tag | Description |
|-----|-------------|
| `(happy)` | General happiness |
| `(sad)` | Sadness, sorrow |
| `(angry)` | Anger, frustration |
| `(excited)` | High energy enthusiasm |
| `(calm)` | Peaceful, relaxed |
| `(nervous)` | Anxiety, unease |
| `(confident)` | Self-assured |
| `(surprised)` | Shock, amazement |
| `(satisfied)` | Contentment |
| `(delighted)` | Joy, pleasure |
| `(scared)` | Fear |
| `(worried)` | Concern, anxiety |
| `(upset)` | Distress |
| `(frustrated)` | Annoyance |
| `(depressed)` | Deep sadness |
| `(empathetic)` | Understanding |
| `(embarrassed)` | Shame |
| `(disgusted)` | Revulsion |
| `(moved)` | Touched emotionally |
| `(proud)` | Pride |
| `(relaxed)` | At ease |
| `(grateful)` | Thankfulness |
| `(curious)` | Interest |
| `(sarcastic)` | Ironic tone |

### Advanced Emotions (25)
| Tag | Description |
|-----|-------------|
| `(disdainful)` | Contempt |
| `(unhappy)` | Discontent |
| `(anxious)` | Worry, nervousness |
| `(hysterical)` | Extreme emotion |
| `(indifferent)` | Apathy |
| `(uncertain)` | Doubt |
| `(confused)` | Bewilderment |
| `(disappointed)` | Letdown |
| `(regretful)` | Remorse |
| `(guilty)` | Self-blame |
| `(ashamed)` | Disgrace |
| `(jealous)` | Envy |
| `(hopeful)` | Optimism |
| `(optimistic)` | Positive outlook |
| `(pessimistic)` | Negative outlook |
| `(nostalgic)` | Longing for past |
| `(lonely)` | Isolation |
| `(bored)` | Disinterest |
| `(contemptuous)` | Scorn |
| `(sympathetic)` | Understanding |
| `(compassionate)` | Caring |
| `(determined)` | Resolve |
| `(resigned)` | Acceptance |
| `(panicked)` | Extreme fear |
| `(furious)` | Intense anger |

### Tone Markers (5)
| Tag | Use Case |
|-----|----------|
| `(in a hurry tone)` | Rushed speech |
| `(shouting)` | Loud, urgent |
| `(screaming)` | Extreme volume |
| `(whispering)` | Quiet, secretive |
| `(soft tone)` | Gentle delivery |

### Audio Effects (10)
| Tag | Description |
|-----|-------------|
| `(laughing)` | Laughter in speech |
| `(chuckling)` | Light laugh |
| `(sobbing)` | Crying |
| `(crying loudly)` | Intense crying |
| `(sighing)` | Exhale |
| `(groaning)` | Pain/frustration sound |
| `(panting)` | Heavy breathing |
| `(gasping)` | Sharp breath |
| `(yawning)` | Tired exhale |

### Special Effects
| Tag | Description |
|-----|-------------|
| `(audience laughing)` | Background laughter |
| `(crowd laughing)` | Multiple people |
| `(break)` | Brief pause |
| `(long-break)` | Extended pause |

---

## Fish Audio Best Practices

### 1. Emotion Tag Placement
```python
# ✅ CORRECT - Tag at start of sentence
"(angry) You think you can just walk in here?"

# ❌ WRONG - Tag mid-sentence
"You think you can (angry) just walk in here?"
```

### 2. Combining Emotions + Tones
```python
# Layer emotions with tones
"(scared)(whispering) Did you hear that?"
"(angry)(shouting) Get out of here!"
"(sad)(soft tone) I miss you so much."
```

### 3. Audio Effects Placement
```python
# Effects can go anywhere (unlike emotions)
"That's (laughing) absolutely hilarious!"
"I can't believe (sighing) this happened again."
```

### 4. Gradual Transitions
```python
# Build emotional arc across sentences
dialogue = """
(curious) What's that sound?
(nervous) It's getting closer.
(scared)(whispering) Oh no, it's here.
(panicked)(screaming) Run!
"""
```

### 5. Character Consistency
```python
# Define character voice profiles
SARAH_VOICE = {
    "reference_id": "sarah_voice_id",
    "default_emotion": "friendly",
    "speaking_style": "energetic"
}

# Apply consistently
def generate_sarah_line(text, emotion="friendly"):
    return TTSRequest(
        text=f"({emotion}) {text}",
        reference_id=SARAH_VOICE["reference_id"]
    )
```

### 6. Emotion-to-Tag Mapping for Scenes
```python
SCENE_EMOTION_MAP = {
    "action": ["excited", "determined", "shouting"],
    "horror": ["scared", "nervous", "whispering", "panicked"],
    "romance": ["happy", "moved", "soft tone"],
    "drama": ["sad", "angry", "frustrated", "crying loudly"],
    "comedy": ["happy", "sarcastic", "laughing", "chuckling"],
    "thriller": ["nervous", "worried", "in a hurry tone"]
}
```

### 7. Avoid Common Mistakes
```python
# ❌ DON'T overuse in short text
"(excited)(happy)(delighted) Hi!"  # Too many tags

# ✅ DO use one primary emotion
"(excited) Hi there!"

# ❌ DON'T mix conflicting emotions
"(happy)(sad) I'm feeling something."

# ✅ DO transition naturally
"(happy) We won! (sad) But at what cost?"

# ❌ DON'T use custom tags
"(super-angry) This won't work"

# ✅ DO use official tags
"(furious) This will work"
```

---

## Fish Audio Integration for Scenema

### Scene-to-TTS Pipeline
```python
from fish_audio_sdk import Session, TTSRequest, ReferenceAudio

async def generate_scene_audio(scene, character_voices):
    session = Session(apikey=API_KEY)

    for line in scene.dialogue:
        character = line.character
        emotion = map_scene_emotion(line.emotion)

        request = TTSRequest(
            text=f"({emotion}) {line.text}",
            reference_id=character_voices[character.id],
            format="wav"
        )

        audio_chunks = []
        for chunk in session.tts(request):
            audio_chunks.append(chunk)

        yield character.id, b"".join(audio_chunks)

def map_scene_emotion(scene_emotion: str) -> str:
    """Map Scenema scene emotions to Fish Audio tags."""
    mapping = {
        "neutral": "calm",
        "happy": "happy",
        "sad": "sad",
        "angry": "angry",
        "scared": "scared",
        "excited": "excited",
        "whisper": "whispering",
        "shout": "shouting",
        # Add more as needed
    }
    return mapping.get(scene_emotion, "calm")
```

### Voice Cloning Workflow
```python
async def clone_character_voice(character_id: str, audio_url: str, transcript: str):
    """Create persistent voice model for a character."""
    session = Session(apikey=API_KEY)

    # Download reference audio
    audio_bytes = await download_audio(audio_url)

    # Create voice model
    voice = session.voices.create(
        title=f"character_{character_id}",
        voices=[ReferenceAudio(audio=audio_bytes, text=transcript)],
        description=f"Voice for character {character_id}"
    )

    # Store voice.id in database for future use
    return voice.id
```

### Supported Languages (with Emotion Control)
English, Chinese, Japanese, German, French, Spanish, Korean, Arabic, Russian, Dutch, Italian, Polish, Portuguese

---

## Google Cloud TTS (Chirp 3)

### Strengths
- Instant voice cloning (10s audio)
- Natural language emotion prompts
- 30+ languages
- Enterprise reliability

### Pricing
- $30 / 1M chars (Chirp 3: HD)
- 1M chars/month free tier

### Emotion Control
Uses natural language prompts:
```
"Speak this angrily"
"Use a whispered, scared tone"
"Sound excited and energetic"
```

**Less predictable** than explicit tags (Fish Audio).

---

## Cloud Provider Comparison (Excluding AWS Polly - Robotic Quality)

| Provider | Price/1M | Voice Clone | Emotion Control |
|----------|----------|-------------|-----------------|
| Fish Audio | $15 | ✅ 15s | ✅ 50+ tags |
| Google Chirp 3 | $30 | ✅ 10s | ⚠️ Natural prompts |
| Azure Neural | $16-24 | ⚠️ Apply for access | ✅ SSML styles |

### Azure SSML Emotions (If Approved)
```xml
<mstts:express-as style="angry" styledegree="1.5">
  You think you can just walk in here?
</mstts:express-as>
```

Styles: `cheerful`, `angry`, `sad`, `excited`, `hopeful`, `friendly`, `unfriendly`, `terrified`, `shouting`, `whispering`

---

## Recommendations

### For Our Use Case (Voice Cloning + Emotions)

| Priority | Recommendation |
|----------|----------------|
| **Best emotion control** | **Fish Audio API ($15/1M)** ← Selected |
| Cheapest self-hosted | Higgs (if emotions not critical) |
| Most predictable self-hosted | Chatterbox (intensity only) |
| Enterprise/Zero-ops | Google Chirp 3 ($30/1M) |

### Current Strategy (Updated 2026-01-13)

> **Decision: Fish Audio API is our primary TTS solution.**

1. **Primary: Fish Audio API**
   - 64+ explicit emotion tags for precise control
   - Instant voice cloning (15-30s reference audio)
   - High concurrency support for parallel generation
   - $15/1M UTF-8 bytes (~$0.75/hr audio for English)

2. **Implementation Priority:**
   - Create Fish Audio worker in media-generators
   - Map Scenema scene emotions to Fish Audio tags
   - Build voice cloning workflow for characters
   - Add emotion parameters to speech job schema

3. **Fallback: Higgs Audio (Self-hosted)**
   - Keep for potential cost optimization at scale
   - Monitor for post-trained emotion model release
   - Use if Fish Audio costs become prohibitive

### Why Fish Audio Over Self-Hosted

| Factor | Fish Audio API | Self-Hosted (Higgs/Chatterbox) |
|--------|----------------|--------------------------------|
| Emotion Control | 64+ explicit tags | Broken (Higgs) / Intensity-only (Chatterbox) |
| Setup Time | Immediate | GPU provisioning, deployment |
| Scalability | Auto-scales | Manual GPU management |
| Cost @ 5 hrs/day | ~$112/mo | ~$15-25/mo (but emotion broken) |
| Maintenance | Zero | Updates, monitoring, debugging |

**Verdict:** The emotion control gap makes self-hosted unsuitable for our use case. Fish Audio's explicit emotion tags are worth the premium.

---

## Open Questions (Resolved)

| Question | Status |
|----------|--------|
| Can Higgs emotion be influenced by emotional reference audio? | ⏸️ Deprioritized - using Fish Audio |
| Will Boson release a post-trained emotion model? | 👀 Monitoring - no timeline |
| Is fine-tuning Higgs on ESD dataset feasible? | ⏸️ Deprioritized |
| Can we hybrid: Chatterbox for neutral, Fish Audio for emotional? | ❌ Rejected - complexity not worth it |

## Next Steps

1. [ ] Add `fish-audio-sdk` to media-generators dependencies
2. [ ] Create Fish Audio speech worker
3. [ ] Define voice cloning API endpoints
4. [ ] Map Scenema emotions to Fish Audio tags
5. [ ] Test emotion control with sample scenes

---

## Sources

### Fish Audio (Primary)
- [Fish Audio](https://fish.audio/)
- [Fish Audio TTS Docs](https://docs.fish.audio/developer-guide/core-features/text-to-speech)
- [Fish Audio Emotion Reference](https://docs.fish.audio/api-reference/emotion-reference)
- [Fish Audio Emotion Best Practices](https://docs.fish.audio/developer-guide/best-practices/emotion-control)
- [Fish Audio Python SDK](https://github.com/fishaudio/fish-audio-python)
- [OpenAudio S1 Blog](https://fish.audio/blog/introducing-s1/)

### Alternatives (For Reference)
- [Higgs Audio GitHub](https://github.com/boson-ai/higgs-audio)
- [Higgs Audio v2 Blog](https://www.boson.ai/blog/higgs-audio-v2)
- [Higgs Emotion Issue #40](https://github.com/boson-ai/higgs-audio/issues/40)
- [Higgs Tags Issue #120](https://github.com/boson-ai/higgs-audio/issues/120)
- [Chatterbox GitHub](https://github.com/resemble-ai/chatterbox)
- [Google Cloud TTS Pricing](https://cloud.google.com/text-to-speech/pricing)
- [Azure Speech Pricing](https://azure.microsoft.com/en-us/pricing/details/cognitive-services/speech-services/)

---

*Last Updated: 2026-01-13*
