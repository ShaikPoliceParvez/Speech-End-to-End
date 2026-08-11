"""
IndicConformer voice-to-text test.

Records from microphone using energy VAD (auto-stops on silence),
then transcribes the full utterance via IndicTranscriber.
No temp files. No fixed timer.

Usage:
    python indic_test.py [--lang te] [--threshold 0.01]
"""

import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import queue
import time

import numpy as np
import sounddevice as sd

from indic_asr_onnx import IndicTranscriber
from stt.indic_stt import StreamingIndicTranscriber

# ── CLI ───────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--lang",      default="te",   help="Language code (te, hi, ml, ...)")
parser.add_argument("--threshold", default=0.015,  type=float, help="Energy threshold for VAD")
args = parser.parse_args()

LANGUAGE    = args.lang
THRESHOLD   = args.threshold
SAMPLE_RATE = 16_000
BLOCK_MS    = 30
BLOCK_SIZE  = int(SAMPLE_RATE * BLOCK_MS / 1000)
SILENCE_S   = 0.8    # seconds of silence to declare end-of-speech
MIN_SPEECH  = 0.3    # minimum speech before endpoint is considered

# ── Load ──────────────────────────────────────────────────────────────
print(f"Loading IndicConformer (lang={LANGUAGE})...")
t0 = time.perf_counter()
base = IndicTranscriber()
print(f"  Loaded in {time.perf_counter() - t0:.1f}s")
print(f"  Building streamer...")
streamer = StreamingIndicTranscriber(base, languages=[LANGUAGE], chunk_ms=9999)
print(f"  Ready\n")

# ── VAD record loop ───────────────────────────────────────────────────
audio_q: queue.Queue = queue.Queue()

def _callback(indata, frames, time_info, status):
    audio_q.put(indata.copy().flatten().astype(np.float32))

print(f"🎤  Say something in {LANGUAGE.upper()} — stops automatically when you pause")
print(f"    Threshold: {THRESHOLD}  (use --threshold to adjust)\n")

recorded: list[np.ndarray] = []
speech_started = False
speech_time    = 0.0
silence_time   = 0.0
t_speech_start: float | None = None

def _energy_bar(energy: float, threshold: float, width: int = 30) -> str:
    filled = min(width, int(energy / max(threshold * 3, 0.001) * width))
    bar = "█" * filled + "░" * (width - filled)
    marker = min(width - 1, int(threshold / max(threshold * 3, 0.001) * width))
    bar = bar[:marker] + "|" + bar[marker + 1:]
    return bar

with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                    blocksize=BLOCK_SIZE, callback=_callback):
    while True:
        try:
            chunk = audio_q.get(timeout=2.0)
        except queue.Empty:
            break

        energy   = float(np.sqrt(np.mean(chunk ** 2)))
        dur      = BLOCK_SIZE / SAMPLE_RATE
        is_voice = energy > THRESHOLD

        if not speech_started:
            bar = _energy_bar(energy, THRESHOLD)
            status = "VOICE" if is_voice else "     "
            print(f"  [{status}] {bar} {energy:.4f}", end="\r")

        if is_voice:
            if not speech_started:
                speech_started = True
                t_speech_start = time.perf_counter()
                print(f"\n  Speech detected (energy {energy:.4f})")
            speech_time  += dur
            silence_time  = 0.0
            recorded.append(chunk)

        elif speech_started:
            recorded.append(chunk)
            silence_time += dur
            if speech_time >= MIN_SPEECH and silence_time >= SILENCE_S:
                print(f"  Endpoint detected  ({speech_time:.2f}s speech)\n")
                break

        if speech_started and (speech_time + silence_time) > 30:
            print("  Max recording time reached\n")
            break

# ── Transcribe ────────────────────────────────────────────────────────
if not recorded:
    print("Nothing recorded — check microphone or lower --threshold")
    raise SystemExit(1)

audio = np.concatenate(recorded)
audio_ms = len(audio) / SAMPLE_RATE * 1000

print(f"Transcribing {audio_ms:.0f} ms of audio...")

# Feed the whole utterance as one chunk via the numpy path (no WAV file)
streamer.reset()
streamer._chunk_samples = len(audio)   # override chunk size for full-utterance mode
t1 = time.perf_counter()
result = streamer.feed(audio)
transcription_ms = time.perf_counter() - t1

total_ms = (time.perf_counter() - t_speech_start) * 1000 if t_speech_start else 0

print(f"\n{'═'*52}")
print(f"  {result['partial'] or '(nothing recognised)'}")
print(f"{'─'*52}")
print(f"  Audio duration  : {audio_ms:.0f} ms")
print(f"  Encoder         : {result['encoder_ms']:.0f} ms")
print(f"  Decoder         : {result['decoder_ms']:.0f} ms")
print(f"  Transcription   : {transcription_ms*1000:.0f} ms")
print(f"  Time from speech start to result : {total_ms:.0f} ms")
print(f"  RTF             : {result['rtf']:.3f}  "
      f"{'✓ real-time' if result['rtf'] < 1.0 else '✗ too slow'}")
print(f"{'═'*52}")

