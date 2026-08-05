# TARZ

Local-first multilingual voice assistant built in Python.

## Highlights

- Voice-first interaction with automatic endpoint detection (VAD)
- Faster-Whisper STT with script/language verification and selective retry
- Multilingual conversation: English, Hindi, Telugu, Malayalam, Arabic
- Roman-script input handling: Hinglish, Telglish, Manglish, Arabizi
- Real-time streaming pipeline: STT -> language -> router -> LLM -> TTS
- TTS router:
  - SuperTonic for English/Hindi
  - Piper for Telugu/Malayalam/Arabic
- Context-aware latency-masking intro sentence (filler) before LLM continuation
- Per-turn language lock and continuity-focused LLM prompting
- Langfuse tracing for end-to-end turn telemetry

## Current Defaults (config.py)

| Component | Default |
| --- | --- |
| STT engine | `whisper` |
| Whisper model | `base` (`cpu`, `int8`) |
| Whisper beam size | `1` |
| LLM model | `gemma3:4b` |
| LLM max tokens | `512` |
| Social-turn max tokens | `96` |
| LLM warmup | Enabled |
| LLM history mode | `strict` |
| LLM history turns | `2` |
| Voice | `F2` |
| Sample rate | `16000` |
| TTS speed | `0.92` |
| Supported languages | `en`, `hi`, `te`, `ml`, `ar` |
| STT previous-language hint | Disabled (`False`) |

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
python app.py
```

## Runtime Pipeline

```mermaid
flowchart TD
    INPUT{Voice or text input}
    INPUT -->|Voice| VAD[Voice endpoint detection]
    INPUT -->|Text| TEXT[Text message]
    VAD --> STT[Speech to text]
    STT --> LANG[Language + script detection]
    TEXT --> LANG
    LANG --> ROUTER[Intent router]
    ROUTER --> ACTION{Chat | Vision | OCR}
    ACTION --> LLM[LLM streaming]
    LLM --> TTSR[TTS router]
    TTSR -->|en / hi| SUPERTONIC[SuperTonic]
    TTSR -->|te / ml / ar| PIPER[Piper]
    SUPERTONIC --> OUT[Playback]
    PIPER --> OUT
```

## Multilingual Behavior

- Language and script are resolved per turn.
- Native-script input is preferred when available.
- Roman-script input is classified via core vocabulary sets.
- STT retry runs only for strong failure signals (unsupported language, script mismatch, suspicious transcript, low-confidence failure conditions).
- LLM output is hard-locked to the active language for each turn.

## Intro/Filler + Continuation Policy

Tarz may speak a short intro sentence before the LLM continuation to mask latency.

Current behavior:

- Intro is selected contextually (intent and language aware).
- Intro is non-semantic for LLM reasoning.
- LLM is instructed to ignore intro semantics and answer from user message + actual history.
- LLM starts immediately while filler can be spoken in parallel.
- Post-filler continuation is tuned for natural chunking (less choppy starts).

## Voice Mode Notes

- No key press required to stop recording.
- Endpoint is declared after configured silence duration.
- Press ENTER while Tarz is speaking to interrupt and ask the next question.

## Langfuse Tracing

If `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are set, Tarz records one trace per turn.

Common spans include:

- `VAD`
- `STT`
- `Router`
- `Memory`
- `LLM`
- `TTS`
- `Playback`
- `Camera` (when vision/OCR is used)

Example env values:

```bash
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=https://cloud.langfuse.com
```

## Piper Models

Configured in `TTS_LANGUAGE_BACKENDS` with model paths under `models/piper/`.

Current mappings:

- Telugu: `models/piper/te_IN-maya-medium.onnx`
- Malayalam: `models/piper/ml_IN-arjun-medium.onnx`
- Arabic: `models/piper/ar_JO-kareem-medium.onnx`

## Project Files

- `app.py` - Main orchestration loop
- `config.py` - Runtime configuration
- `microphone.py` - Capture + VAD endpointing
- `stt.py` - Faster-Whisper decode + retry strategy
- `language.py` - Language/script detection + normalization
- `router.py` - Intent routing
- `llm.py` - Prompting + streaming response chunking
- `tts.py` - SuperTonic synthesis + playback workers
- `piper_tts.py` - Piper synthesis bridge
- `tts_router.py` - Per-language TTS backend routing
- `tracing.py` - Langfuse integration
- `camera.py` - Camera capture for vision/OCR
