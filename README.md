# TARZ

Local-first multilingual voice assistant built in Python.

## Highlights

- Voice-first interaction with automatic energy-based VAD endpoint detection
- Faster-Whisper STT with script/language verification, temperature fallback, and selective retry
- Optional IndicConformer (onnxruntime) routing for Hindi, Telugu, Malayalam
- Multilingual conversation: English, Hindi, Telugu, Malayalam, Arabic
- Roman-script input handling: Hinglish, Telglish, Manglish, Arabizi
- Per-language Whisper initial prompts, script prefixes, and hotwords for improved native-script accuracy
- Real-time streaming pipeline: Microphone → VAD → STT → Language → Router → Memory → LLM → TTS
- TTS router with lazy backend loading:
  - SuperTonic for English and Hindi
  - Piper for Telugu, Malayalam, and Arabic
- Context-aware latency-masking filler sentence spoken in parallel with LLM streaming
- Intent routing with weighted semantic scoring across Chat, Vision, OCR, and System intents
- Per-turn task tracking (story, joke, poem, travel, weather, math, coding, translation, camera)
- Terse follow-up detection with task-lock continuation for natural multi-turn conversations
- Per-turn language lock with continuity-focused LLM prompting
- Barge-in interrupt: press Enter while Tarz is speaking to stop and re-prompt
- Langfuse tracing for end-to-end turn and span telemetry

## Current Defaults (config.py)

| Component | Setting | Default |
| --- | --- | --- |
| STT engine | `STT_MODEL` | `whisper` |
| Whisper model size | `WHISPER_SIZE` | `small` (`cpu`, `int8`) |
| Whisper beam size | `WHISPER_BEAM_SIZE` | `1` |
| Whisper temperature fallback | `WHISPER_TEMPERATURES` | `(0.0, 0.2)` |
| IndicConformer routing | `STT_INDIC_ASR_ENABLED` | `True` (hi, te, ml) |
| LLM model | `LLM_MODEL` | `gemma3:4b` |
| LLM max tokens | `LLM_MAX_TOKENS` | `512` |
| Social-turn max tokens | `LLM_SOCIAL_MAX_TOKENS` | `96` |
| LLM warmup | `LLM_WARMUP_ON_STARTUP` | Enabled |
| LLM history mode | `LLM_HISTORY_MODE` | `full` |
| LLM history turns | `LLM_HISTORY_TURNS` | `6` |
| Max stored history messages | `MAX_HISTORY` | `10` turns |
| Voice | `VOICE` | `F2` |
| Sample rate | `SAMPLE_RATE` | `16000` Hz |
| VAD silence threshold | `VAD_SILENCE_THRESHOLD` | `0.015` RMS |
| VAD silence duration | `VAD_SILENCE_DURATION` | `0.5 s` |
| VAD max record | `VAD_MAX_RECORD_SECONDS` | `30 s` |
| Router confidence threshold | `ROUTER_CONFIDENCE_THRESHOLD` | `0.60` |
| Supported languages | `SUPPORTED_LANGUAGES` | `en`, `hi`, `te`, `ml`, `ar` |

## Installation

```bash
pip install -r requirements.txt
```

### Optional: IndicConformer (Hindi / Telugu / Malayalam)

IndicConformer provides higher-accuracy transcription for Indic languages when a language hint is available. It requires additional packages:

```bash
pip install onnxruntime torch indic-asr-onnx
```

Set `STT_INDIC_ASR_ENABLED = True` in `config.py` (already the default). Falls back to Whisper automatically if the package is not installed.

## Run

```bash
python app.py
```

## Runtime Pipeline

```mermaid
flowchart TD
    subgraph INPUT["Input"]
        MIC["🎤 Microphone"]
        TXT["⌨️ Text input"]
    end

    subgraph VAD_BLOCK["Voice Activity Detection"]
        VAD_ENERGY["Energy-based VAD\n(RMS threshold)"]
        VAD_SILERO["Silero VAD filter\n(audio cleaning inside Whisper)"]
    end

    subgraph STT_BLOCK["Speech-to-Text"]
        WHISPER["Faster-Whisper\n(small, cpu, int8)"]
        HALLUC["Hallucination filter\n(known bad phrases)"]
        RETRY_CHECK{"Retry needed?\n(lang mismatch /\nscript error /\nsuspicious text)"}
        INDIC_CHECK{"IndicConformer\nenabled &\nlang hint known?"}
        INDIC["IndicConformer\n(hi / te / ml)"]
        WHISPER2["Whisper re-decode\n(forced lang + hotwords\n+ script prefix)"]
        TRANSCRIPT["Transcript"]
    end

    subgraph LANG_BLOCK["Language & Script Detection"]
        SCRIPT["detect_script()\nDevanagari / Telugu /\nMalayalam / Arabic / Latin"]
        LANGDET["detect_dominant_language()\nCore word banks + STT hint\n+ previous turn language"]
        NORM["normalize_text()\nHinglish token map +\nphrase map →  Devanagari\nTelglish / Manglish / Arabizi\n(no native normalisation)"]
    end

    subgraph ROUTER_BLOCK["Intent Router"]
        SCORE["Weighted semantic scoring\nCHAT · VISION · OCR · SYSTEM"]
        INTENT{"Intent"}
    end

    subgraph CONTEXT_BLOCK["Context & Memory"]
        TASK["Task detection\nstory / joke / poem / travel\nweather / math / coding\ntranslation / camera"]
        FOLLOWUP["Terse follow-up check\n+ task-lock prompt"]
        MEMORY["Conversation memory\n(full or strict history mode)"]
        FILLER["Context preface / filler\n(intent + language aware\nrandom selection)"]
    end

    subgraph LLM_BLOCK["LLM (Ollama)"]
        CAMERA["📷 Camera capture\n+ frame encode"]
        PROMPT["Prompt builder\nsocial wrap / task lock\n+ history injection"]
        STREAM["Streaming generation\ngemma3:4b\n(sentence-chunked)"]
    end

    subgraph TTS_BLOCK["TTS Router"]
        TTS_ROUTE{"Language?"}
        SUPERTONIC["SuperTonic\n(en / hi)"]
        PIPER["Piper ONNX\n(te / ml / ar)\nlazy-loaded per language"]
        PRONOUNCE["Pronunciation map\n(custom G2P corrections)"]
    end

    subgraph OUTPUT["Output"]
        PLAY["🔊 Audio playback"]
        BARGEIN["⏎ Barge-in interrupt\n(Enter key stops speech)"]
    end

    LANGFUSE["📊 Langfuse tracing\nVAD · STT · Language · Router\nMemory · LLM · TTS · Camera"]

    MIC --> VAD_ENERGY
    VAD_ENERGY -->|"Speech ended\n(silence ≥ 0.5 s)"| VAD_SILERO
    VAD_SILERO --> WHISPER
    TXT --> SCRIPT

    WHISPER --> HALLUC
    HALLUC --> RETRY_CHECK
    RETRY_CHECK -->|"No retry"| INDIC_CHECK
    RETRY_CHECK -->|"Retry"| INDIC_CHECK
    INDIC_CHECK -->|"Yes"| INDIC
    INDIC_CHECK -->|"No"| WHISPER2
    INDIC --> TRANSCRIPT
    WHISPER2 --> TRANSCRIPT
    TRANSCRIPT --> SCRIPT

    SCRIPT --> LANGDET
    LANGDET --> NORM
    NORM --> SCORE

    SCORE --> INTENT
    INTENT -->|"CHAT"| TASK
    INTENT -->|"VISION"| CAMERA
    INTENT -->|"OCR"| CAMERA

    TASK --> FOLLOWUP
    FOLLOWUP --> MEMORY
    MEMORY --> FILLER
    FILLER --> PROMPT
    FILLER -.->|"Spoken in parallel\nwhile LLM generates"| SUPERTONIC

    CAMERA --> PROMPT
    PROMPT --> STREAM

    STREAM --> TTS_ROUTE
    TTS_ROUTE -->|"en / hi"| SUPERTONIC
    TTS_ROUTE -->|"te / ml / ar"| PIPER
    SUPERTONIC --> PRONOUNCE
    PIPER --> PRONOUNCE
    PRONOUNCE --> PLAY
    PLAY --> BARGEIN

    STREAM -.->|"per-turn span"| LANGFUSE
    WHISPER -.->|"STT span"| LANGFUSE
    SCORE -.->|"Router span"| LANGFUSE
    MEMORY -.->|"Memory span"| LANGFUSE
    PLAY -.->|"TTS + Playback span"| LANGFUSE
```

## Multilingual Behavior

- Language and script are resolved independently on every turn.
- Native-script input (Devanagari, Telugu, Malayalam, Arabic) is detected directly via Unicode range checks.
- Roman-script input is classified using per-language core-word banks (Hinglish, Telglish, Manglish, Arabizi) before falling back to English.
- Technical borrowed words (wifi, laptop, etc.) are excluded from language voting.
- Ambiguous short tokens (ok, yeah, hmm) inherit the previous turn's language.
- Explicit language-switch requests ("switch to Telugu") override in-turn detection.
- STT retry runs only on strong failure signals: unsupported language ID, script mismatch, hallucination match, or suspicious transcript patterns. Low-confidence-only retry is disabled by default (`STT_RETRY_ON_LOW_CONFIDENCE = False`).
- LLM output is hard-locked to the detected language for each turn.

## Intro/Filler + Continuation Policy

Tarz speaks a short contextual filler sentence before the LLM continuation to mask first-token latency.

- Filler is selected per intent category (greeting, wellbeing, story, joke, search, etc.) and per language.
- Filler is non-semantic: LLM is instructed to answer from the user message and history, not from what the filler said.
- LLM generation starts immediately while the filler is being spoken in parallel.
- Post-filler continuation is sentence-chunked for natural speech pacing.

## Voice Mode Notes

- No key press is required to stop recording — endpoint is declared automatically after configured silence.
- Press **Enter** while Tarz is speaking to interrupt playback and ask the next question.
- Live partial transcripts are printed to console during recording (configurable via `ENABLE_PARTIAL_TRANSCRIPTS`).

## Langfuse Tracing

Set `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` to enable one trace per turn.

Traced spans:

| Span | Contents |
| --- | --- |
| `VAD` | Microphone listen time, speech/silence durations |
| `STT` | Transcript, language hint, confidence, retry count |
| `Language` | Detected language, script, normalized message |
| `Router` | Selected intent, confidence score, routing reason |
| `Memory` | History message count |
| `LLM` | Prompt, model, streaming chunks, TTFT |
| `TTS` | Backend, language, synthesis latency |
| `Playback` | Audio duration |
| `Camera` | Frame capture latency (vision/OCR turns only) |

```bash
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

## TTS Backends

| Language | Backend | Model |
| --- | --- | --- |
| English (`en`) | SuperTonic | voice `F2` (configurable M1–M2, F1–F6) |
| Hindi (`hi`) | SuperTonic | voice `F2` |
| Telugu (`te`) | Piper | `models/piper/te_IN-maya-medium.onnx` |
| Malayalam (`ml`) | Piper | `models/piper/ml_IN-arjun-medium.onnx` |
| Arabic (`ar`) | Piper | `models/piper/ar_JO-kareem-medium.onnx` |

Piper backends are lazy-loaded on first use. `TTS_PRONUNCIATION_MAP` in `config.py` applies G2P corrections before synthesis (assistant name, AI terms, Telugu cluster fixes).

## Project Files

| File | Purpose |
| --- | --- |
| `app.py` | Main orchestration loop, turn processing, barge-in |
| `config.py` | All runtime configuration (models, VAD, language, TTS, LLM) |
| `microphone.py` | Audio capture with energy-based VAD endpoint detection |
| `stt.py` | Faster-Whisper decode, hallucination filter, retry strategy, IndicConformer routing |
| `indic_stt.py` | IndicConformer ONNX streaming transcriber |
| `language.py` | Script detection, dominant-language classification, Hinglish normalization |
| `router.py` | Weighted semantic intent scoring (CHAT / VISION / OCR / SYSTEM) |
| `memory.py` | Per-session conversation history with configurable turn window |
| `llm.py` | Ollama prompt building, multimodal support, streaming sentence chunking |
| `tts.py` | SuperTonic synthesis with threaded playback workers |
| `piper_tts.py` | Piper ONNX synthesis bridge |
| `tts_router.py` | Per-language TTS backend selection with lazy Piper loading |
| `tracing.py` | Langfuse span/turn instrumentation |
| `camera.py` | OpenCV camera capture for vision and OCR turns |
| `audio_utils.py` | Audio helper utilities |
