"""
Standalone STT benchmark (accurate + streamed input).

Speak into the microphone; the audio is consumed as a *stream* (growing
windows, like live partial transcripts), then a high-accuracy final pass is
run. A warm-up pass loads the model first so timings reflect steady state.

Accuracy focus:
  - Final pass uses beam search + temperature fallback + quality gates.
  - If Hindi is detected, a second pass re-decodes with the Devanagari
    initial prompt for better Hindi script accuracy.

Run:
    python time_test_stt.py

This file only *reads* the existing modules/config. It does not modify or
affect the main app in any way.
"""

import time

import numpy as np

from microphone import Microphone
from stt import STT
from config import (
    SAMPLE_RATE,
    WHISPER_BEAM_SIZE,
    WHISPER_TEMPERATURES,
    WHISPER_COMPRESSION_RATIO_THRESHOLD,
    WHISPER_LOG_PROB_THRESHOLD,
    WHISPER_NO_SPEECH_THRESHOLD,
    WHISPER_HINDI_PROMPT,
    STT_MIN_PARTIAL_SECONDS,
    STT_PARTIAL_INTERVAL,
)


def warm_up(stt):
    """Run a tiny decode so the model is resident before real timing."""
    t0 = time.perf_counter()
    try:
        segs, _ = stt.model.transcribe(
            np.zeros(SAMPLE_RATE, dtype=np.float32),
            beam_size=1,
            without_timestamps=True,
        )
        list(segs)
    except Exception as e:
        print(f"(warm-up skipped: {e})")
        return
    print(f"Warm-up: {time.perf_counter() - t0:.2f}s")


def partial_decode(stt, window):
    """Fast greedy pass for a streamed partial hypothesis."""
    segs, _ = stt.model.transcribe(
        window,
        beam_size=1,
        task="transcribe",
        vad_filter=False,
        without_timestamps=True,
        condition_on_previous_text=False,
    )
    return "".join(s.text for s in segs).strip()


def final_decode(stt, audio, language=None):
    """High-accuracy pass: beam search + temperature fallback + gates."""
    initial_prompt = WHISPER_HINDI_PROMPT if language == "hi" else None
    segments, info = stt.model.transcribe(
        audio,
        beam_size=WHISPER_BEAM_SIZE,
        language=language,
        task="transcribe",
        vad_filter=True,
        without_timestamps=True,
        condition_on_previous_text=False,
        initial_prompt=initial_prompt,
        temperature=WHISPER_TEMPERATURES,
        compression_ratio_threshold=WHISPER_COMPRESSION_RATIO_THRESHOLD,
        log_prob_threshold=WHISPER_LOG_PROB_THRESHOLD,
        no_speech_threshold=WHISPER_NO_SPEECH_THRESHOLD,
    )
    text = "".join(s.text for s in segments).strip()
    return text, info


def run():
    print("=== STT BENCHMARK (accurate, streamed input) ===")

    load_start = time.perf_counter()
    mic = Microphone()
    stt = STT()
    print(f"Model load: {time.perf_counter() - load_start:.2f}s")

    warm_up(stt)
    print()

    while True:
        try:
            input("Press ENTER, then speak (Ctrl+C to quit)... ")
        except EOFError:
            break

        # ---- 1. Record (VAD auto-stops on silence) ----
        rec_start = time.perf_counter()
        audio = mic.listen()
        rec_time = time.perf_counter() - rec_start
        audio_seconds = (len(audio) / SAMPLE_RATE) if len(audio) else 0.0

        if len(audio) == 0:
            print("No audio captured.\n")
            continue

        # ---- 2. Consume audio as a stream (growing partial windows) ----
        print("\n--- Streaming partials ---")
        first_partial_time = None
        stream_start = time.perf_counter()

        t = STT_MIN_PARTIAL_SECONDS
        while t < audio_seconds:
            window = audio[: int(t * SAMPLE_RATE)]
            partial = partial_decode(stt, window)
            now = time.perf_counter()
            if first_partial_time is None:
                first_partial_time = now - stream_start
            print(f"  [{now - stream_start:6.2f}s @ {t:4.1f}s audio] {partial}")
            t += STT_PARTIAL_INTERVAL

        # ---- 3. High-accuracy final pass (two-pass for Hindi) ----
        print("\n--- Final (accurate) ---")
        fin_start = time.perf_counter()
        text, info = final_decode(stt, audio)
        language = getattr(info, "language", None)

        second_pass_time = None
        if language == "hi":
            sp_start = time.perf_counter()
            hi_text, hi_info = final_decode(stt, audio, language="hi")
            second_pass_time = time.perf_counter() - sp_start
            if hi_text:
                text, info = hi_text, hi_info

        fin_time = time.perf_counter() - fin_start

        # ---- 4. Report ----
        print(f"Transcript : {text}")
        print(
            f"Language   : {getattr(info, 'language', None)} "
            f"({getattr(info, 'language_probability', 0.0):.2f})"
        )

        print("\n--- Timings ---")
        print(f"Recording          : {rec_time:6.2f}s  (audio {audio_seconds:.2f}s)")
        print(f"First partial       : {(first_partial_time or 0.0):6.2f}s")
        print(f"Final decode        : {fin_time:6.2f}s")
        if second_pass_time is not None:
            print(f"  (Hindi 2nd pass)  : {second_pass_time:6.2f}s")
        if audio_seconds:
            print(f"Real-time factor    : {fin_time / audio_seconds:6.2f}x")
        print()


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("\nBye.")
