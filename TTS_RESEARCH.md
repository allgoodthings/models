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

## Fish Audio (Commercial API)

### Strengths
- **Best emotion control** with explicit tags
- 50+ emotions: `(angry)`, `(whispered)`, `(excited)`, `(laughing)`, `(sarcastic)`, etc.
- #1 on TTS-Arena benchmark
- Voice cloning from 15s audio
- 70+ languages

### Pricing
- $15 / 1M UTF-8 bytes (English)
- ~$45 / 1M chars (CJK - 3x more expensive)

### Emotion Tags (Confirmed Working)
```
(angry)You think you can just walk in here?
(whispered)I don't think so.
(excited)Oh my god, we did it!
(sad)I never thought it would end this way.
(laughing)That's hilarious!
(sarcastic)Oh, great, another meeting.
```

### Limitation
- CC-BY-NC-SA license for open-source model
- Must use API for commercial ($15/1M bytes)
- 3x cost for non-English languages

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
| Best emotion control | Fish Audio API ($15/1M) |
| Cheapest self-hosted | Higgs (if emotions not critical) |
| Most predictable self-hosted | Chatterbox (intensity only) |
| Enterprise/Zero-ops | Google Chirp 3 ($30/1M) |

### Current Strategy

1. **Primary:** Continue with Chatterbox for now
   - Map scene emotions to exaggeration/cfg_weight presets
   - Accept limitation of intensity-only control

2. **Research:** Deep dive into Higgs emotion control
   - Monitor for post-trained model release
   - Test emotional reference audio approach
   - Explore fine-tuning options

3. **Fallback:** Fish Audio API for critical emotional scenes
   - Use sparingly due to cost
   - Best for hero content requiring specific emotions

---

## Open Questions

1. Can Higgs emotion be influenced by emotional reference audio?
2. Will Boson release a post-trained emotion model?
3. Is fine-tuning Higgs on ESD dataset feasible?
4. Can we hybrid: Chatterbox for neutral, Fish Audio for emotional?

---

## Sources

- [Higgs Audio GitHub](https://github.com/boson-ai/higgs-audio)
- [Higgs Audio v2 Blog](https://www.boson.ai/blog/higgs-audio-v2)
- [Higgs Emotion Issue #40](https://github.com/boson-ai/higgs-audio/issues/40)
- [Higgs Tags Issue #120](https://github.com/boson-ai/higgs-audio/issues/120)
- [Chatterbox GitHub](https://github.com/resemble-ai/chatterbox)
- [Fish Audio](https://fish.audio/)
- [Google Cloud TTS Pricing](https://cloud.google.com/text-to-speech/pricing)
- [Azure Speech Pricing](https://azure.microsoft.com/en-us/pricing/details/cognitive-services/speech-services/)

---

*Last Updated: 2026-01-11*
