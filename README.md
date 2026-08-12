# TARZ

Local-first multilingual voice assistant built in Python.

## Highlights

- Voice-first with automatic speech endpoint detection (no button to press)
- Faster-Whisper + optional IndicConformer for accurate multilingual STT
- Multilingual: English, Hindi, Nepali, Telugu, Malayalam, Arabic
- Roman-script input supported: Hinglish, Nepali Roman, Telglish, Manglish, Arabizi
- Streaming pipeline: every LLM token flows to TTS immediately — no end-to-end wait
- Dual TTS backends: SuperTonic (English, Hindi) and Piper ONNX (Nepali, Telugu, Malayalam, Arabic)
- Context-aware filler phrases mask LLM latency so there is no silent gap
- Intent routing: Chat, Vision, OCR, System commands
- Task tracking for multi-turn follow-up questions
- Barge-in: press Enter while Tarz is speaking to interrupt and ask the next question
- Automatic microphone amplification for quiet devices

## Defaults

| Component    | Default                                                                 |
|-------------|-------------------------------------------------------------------------|
| STT          | Whisper small · CPU · int8                                              |
| STT (Indic)  | IndicConformer for Hindi, Telugu, Malayalam — Nepali uses Whisper path  |
| LLM          | Gemma 3 4B via Ollama                                                   |
| TTS (en/hi)  | SuperTonic                                                              |
| TTS (others) | Piper ONNX                                                              |
| Languages    | English, Hindi, Nepali, Telugu, Malayalam, Arabic                       |
| Sample Rate  | 16 kHz mono                                                             |

## Pipeline

```
┌─────────────────────────────────────────────────────────┐
│                      INPUT LAYER                        │
│                                                         │
│   ┌──────────────────┐      ┌──────────────────┐        │
│   │    Microphone    │      │   Text Console   │        │
│   └────────┬─────────┘      └────────┬─────────┘        │
│            │                         │                  │
│   ┌────────▼─────────┐               │                  │
│   │  Voice Activity  │               │                  │
│   │   Detection      │               │                  │
│   └────────┬─────────┘               │                  │
│            │                         │                  │
│   ┌────────▼─────────┐               │                  │
│   │       STT        │               │                  │
│   │ Whisper +        │               │                  │
│   │ IndicConformer   │               │                  │
│   └────────┬─────────┘               │                  │
│            └────────────┬────────────┘                  │
└─────────────────────────┼───────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────┐
│                   UNDERSTANDING LAYER                   │
│                                                         │
│            ┌────────────────────────┐                   │
│            │   Language Detection   │                   │
│            │  Script + Vocabulary   │                   │
│            └────────────┬───────────┘                   │
│                         │                               │
│            ┌────────────▼───────────┐                   │
│            │     Intent Router      │                   │
│            │  Chat / Vision / OCR / │                   │
│            │       System           │                   │
│            └────────────┬───────────┘                   │
│                         │                               │
│            ┌────────────▼───────────┐                   │
│            │  Conversation Memory   │                   │
│            │  (last N turns)        │                   │
│            └────────────┬───────────┘                   │
└─────────────────────────┼───────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────┐
│                   GENERATION LAYER                      │
│                                                         │
│            ┌────────────────────────┐                   │
│            │     LLM Generator      │                   │
│            │   Ollama streaming     │                   │
│            └────────────┬───────────┘                   │
│                         │  tokens stream out             │
│            ┌────────────▼───────────┐                   │
│            │   Sentence Buffer      │                   │
│            │  groups tokens into    │                   │
│            │  speakable chunks      │                   │
│            └────────────┬───────────┘                   │
└─────────────────────────┼───────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────┐
│                     SPEECH LAYER                        │
│                                                         │
│            ┌────────────────────────┐                   │
│            │       TTS Router       │                   │
│            └──────┬─────────────────┘                   │
│                   │                                     │
│       ┌───────────┴────────────┐                        │
│       │                        │                        │
│  ┌────▼──────────┐   ┌─────────▼──────────┐            │
│  │  SuperTonic   │   │     Piper ONNX      │            │
│  │  English      │   │  Nepali / Telugu /  │            │
│  │  Hindi        │   │  Malayalam / Arabic │            │
│  └────┬──────────┘   └─────────┬──────────┘            │
│       └───────────┬────────────┘                        │
│                   │                                     │
│            ┌──────▼─────────────────┐                   │
│            │    Audio Playback      │                   │
│            │  device auto-select    │                   │
│            └──────┬─────────────────┘                   │
│                   │                                     │
│            ┌──────▼─────────────────┐                   │
│            │   Barge-in Listener    │                   │
│            │  press ENTER to stop   │                   │
│            └────────────────────────┘                   │
└─────────────────────────────────────────────────────────┘
```

## Installation

```bash
pip install -r requirements.txt
```

### Optional: IndicConformer (Hindi / Telugu / Malayalam)

IndicConformer gives higher-accuracy transcription for Indic languages. It requires additional packages:

```bash
pip install onnxruntime torch indic-asr-onnx
```

`STT_INDIC_ASR_ENABLED = True` is the default in `config/stt_config.py`. Falls back to Whisper automatically if the package is missing.

Nepali uses the Whisper multilingual path (better compatibility with the current model set).

## Run

```bash
python app.py
```

## Smoke Test

Quick end-to-end configuration and language-routing sanity check (no microphone needed):

```bash
python tests/smoke_pipeline.py
```

## Multilingual Behavior

- Language detected every turn using script detection, vocabulary banks, and STT confidence
- Native scripts (Devanagari, Telugu, Malayalam, Arabic) detected via Unicode ranges
- Roman-script (Hinglish, Nepali Roman, Telglish, Manglish, Arabizi) identified by language-specific word banks
- Whisper language confidence takes priority — prevents false detection when scripts mix
- Explicit language-switch commands override auto-detection
- LLM reply is locked to the detected language for the turn

## Filler Phrases

- A short contextual phrase plays immediately ("Sure, let me check...") while the LLM generates
- Phrase is chosen by intent category and language — works across all six languages
- LLM generation runs in parallel so playback begins with zero silent gap
- LLM response streams into TTS sentence-by-sentence as tokens arrive

## Voice Mode Notes

- No key press needed to stop recording — VAD endpoint detection is automatic
- Press **Enter** while Tarz is speaking to interrupt and ask the next question
- Live partial transcripts print to console during recording (`ENABLE_PARTIAL_TRANSCRIPTS` in `config/audio_config.py`)
- Quiet mics (amplitude < 0.01) are auto-amplified ~4× for reliable STT

## TTS Backends

| Language  | Backend       |
|-----------|---------------|
| English   | SuperTonic    |
| Hindi     | SuperTonic    |
| Nepali    | Piper ONNX    |
| Telugu    | Piper ONNX    |
| Malayalam | Piper ONNX    |
| Arabic    | Piper ONNX    |

If a Piper model file is missing, Tarz falls back to SuperTonic so the conversation continues. Add the matching `.onnx` file to `models/piper/` for native-language voice.

Piper model paths (configured in `config/tts_config.py`):

```
models/piper/ne_NP-chitwan-medium.onnx
models/piper/te_IN-maya-medium.onnx
models/piper/ml_IN-arjun-medium.onnx
models/piper/ar_JO-kareem-medium.onnx
```

Hindi/Nepali note: Devanagari ambiguity defaults to Hindi unless Nepali-specific vocabulary is detected. Explicit language-switch commands always override.

## Project Structure

```
Avatar_base/
├── app.py                      # Entry point — main orchestration loop and barge-in
├── config/
│   ├── __init__.py             # Re-exports all names (backward-compatible)
│   ├── app_config.py           # Debug, camera, memory size, router threshold
│   ├── audio_config.py         # Sample rate, VAD thresholds, live transcript
│   ├── language_config.py      # Supported languages, vocabulary banks
│   ├── llm_config.py           # Model selection, token limits, sampling, history
│   ├── stt_config.py           # Whisper settings, hotwords, quality filters
│   └── tts_config.py           # Chunk sizes, filler, backends, pronunciation map
├── core/
│   ├── language.py             # Script detection and dominant-language classification
│   ├── memory.py               # Per-session conversation history
│   ├── router.py               # Weighted semantic intent scoring (Chat/Vision/OCR/System)
│   └── tracing.py              # Optional latency tracing instrumentation
├── llm/
│   └── llm.py                  # Ollama streaming, prompt building, sentence chunking
├── stt/
│   ├── stt.py                  # Faster-Whisper decode with hallucination filter
│   ├── indic_stt.py            # IndicConformer ONNX transcriber
│   └── stt_livedecoding.py     # Live partial-transcript rolling decoder
├── tts/
│   ├── tts.py                  # SuperTonic synthesis with threaded playback workers
│   ├── piper_tts.py            # Piper ONNX synthesis bridge
│   └── tts_router.py           # Per-language backend selection with lazy Piper loading
├── audio/
│   ├── microphone.py           # Audio capture with energy-based VAD
│   └── audio_utils.py          # Audio helper utilities
├── camera/
│   └── camera.py               # OpenCV capture for Vision and OCR turns
├── tests/                      # Benchmarks and unit tests
├── models/piper/               # Piper ONNX voice model files
└── captures/                   # Saved camera frames (auto-pruned)
```

## Configuration

All settings live in `config/` as [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) classes. Override any value via environment variable or `.env` file:

```
LLM_MODEL=qwen2.5:3b
WHISPER_SIZE=medium
VOICE=M1
TTS_SPEED=0.95
DEBUG=true
```

Each config file maps to one pipeline layer — edit only the file relevant to what you are tuning.
