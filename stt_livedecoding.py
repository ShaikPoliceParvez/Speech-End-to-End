"""
Fast STT benchmark (matches Tarz production pipeline) — LIVE DECODING VERSION.

Key change from the original: faster-whisper's `model.transcribe()` returns a
LAZY generator of segments. The original code did:

    text = "".join(s.text for s in segments).strip()

which silently blocks until every segment is decoded before printing anything.
This version iterates the generator directly and prints each segment the
moment it's ready, so the transcript appears "live" while decoding is still
in progress — same total decode time, much lower *perceived* latency.

Measures:
- Model load
- Warm-up
- Recording time
- Time-to-first-segment (TTFS) -- real latency-to-first-output
- Per-segment decode timing
- Final transcription latency (total)
- Real-time factor
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
)

# If you want max speed over max accuracy, try beam_size=1 (greedy decoding)
# instead of WHISPER_BEAM_SIZE. Worth A/B testing on your real traffic —
# greedy is meaningfully faster but can be less accurate on noisy audio.
LIVE_BEAM_SIZE = WHISPER_BEAM_SIZE

# Only re-run the Hindi second pass if the first pass's confidence in "hi"
# is below this threshold. Previously the code ALWAYS re-decoded when
# language == "hi", even at high confidence — that's a wasted full decode.
HINDI_CONFIDENCE_REDECODE_THRESHOLD = 0.80


def warm_up(stt):
    t0 = time.perf_counter()

    try:
        segments, _ = stt.model.transcribe(
            np.zeros(SAMPLE_RATE, dtype=np.float32),
            beam_size=1,
            without_timestamps=True,
        )
        list(segments)

    except Exception as e:
        print(f"Warm-up skipped: {e}")
        return

    print(f"Warm-up           : {time.perf_counter()-t0:.2f}s")


def live_decode(stt, audio, language=None, label="Pass"):
    """
    Streams segments as faster-whisper decodes them, printing each one
    the instant it's ready instead of waiting for the whole utterance.

    Returns: (full_text, info, timings_dict)
    """

    prompt = WHISPER_HINDI_PROMPT if language == "hi" else None

    decode_start = time.perf_counter()

    segments, info = stt.model.transcribe(
        audio,
        beam_size=LIVE_BEAM_SIZE,
        language=language,
        task="transcribe",
        vad_filter=True,
        without_timestamps=True,
        condition_on_previous_text=False,
        initial_prompt=prompt,
        temperature=WHISPER_TEMPERATURES,
        compression_ratio_threshold=WHISPER_COMPRESSION_RATIO_THRESHOLD,
        log_prob_threshold=WHISPER_LOG_PROB_THRESHOLD,
        no_speech_threshold=WHISPER_NO_SPEECH_THRESHOLD,
    )

    # NOTE: `info` (language, language_probability) is available immediately —
    # faster-whisper runs language ID before returning, the generator only
    # lazily defers the actual segment decoding.

    chunks = []
    first_segment_time = None
    segment_times = []
    prev_t = decode_start

    print(f"\n[{label}] streaming transcript (lang={info.language}, "
          f"conf={info.language_probability:.2f}):")
    print("  ", end="", flush=True)

    for seg in segments:
        now = time.perf_counter()

        if first_segment_time is None:
            first_segment_time = now - decode_start

        segment_times.append(now - prev_t)
        prev_t = now

        chunks.append(seg.text)
        print(seg.text, end="", flush=True)

    print()  # newline after streamed transcript

    total_decode_time = time.perf_counter() - decode_start
    text = "".join(chunks).strip()

    timings = {
        "ttfs": first_segment_time if first_segment_time is not None else total_decode_time,
        "segment_times": segment_times,
        "total_decode_time": total_decode_time,
    }

    return text, info, timings


def run():

    print("\n========== STT BENCHMARK (LIVE DECODING) ==========\n")

    load_start = time.perf_counter()

    mic = Microphone()
    stt = STT()

    print(f"Model load        : {time.perf_counter()-load_start:.2f}s")

    warm_up(stt)

    while True:

        try:
            input("\nPress ENTER and speak (Ctrl+C to quit)... ")

        except EOFError:
            break

        print("Listening...\n")

        record_start = time.perf_counter()

        audio = mic.listen()

        record_time = time.perf_counter() - record_start

        if len(audio) == 0:
            print("No audio captured.")
            continue

        audio_seconds = len(audio) / SAMPLE_RATE

        text, info, timings = live_decode(stt, audio, label="Pass 1")

        second_pass = 0.0
        second_pass_ttfs = 0.0

        needs_hindi_redecode = (
            getattr(info, "language", None) == "hi"
            and info.language_probability < HINDI_CONFIDENCE_REDECODE_THRESHOLD
        )

        if needs_hindi_redecode:

            hi_text, hi_info, hi_timings = live_decode(
                stt,
                audio,
                language="hi",
                label="Pass 2 (Hindi re-decode)",
            )

            second_pass = hi_timings["total_decode_time"]
            second_pass_ttfs = hi_timings["ttfs"]

            if hi_text:
                text = hi_text
                info = hi_info

        decode_time = timings["total_decode_time"] + second_pass

        print("----------------------------------")
        print(f"Transcript        : {text}")
        print(f"Language          : {info.language}")
        print(f"Confidence        : {info.language_probability:.2f}")
        print("----------------------------------")
        print(f"Recording                    : {record_time:.2f}s")
        print(f"Time to first segment (TTFS) : {timings['ttfs']:.2f}s")
        print(f"Pass 1 decode time           : {timings['total_decode_time']:.2f}s")
        print(f"  segments in pass 1         : {len(timings['segment_times'])}")

        if second_pass:
            print(f"Hindi 2nd pass TTFS               : {second_pass_ttfs:.2f}s")
            print(f"Hindi 2nd pass decode time        : {second_pass:.2f}s")

        print(f"Total time to generate transcript     : {decode_time:.2f}s")
        print(f"Audio length                  : {audio_seconds:.2f}s")
        print(f"Real-time Factor              : {decode_time/audio_seconds:.2f}x")

        if audio_seconds:
            speed = audio_seconds / decode_time
            print(f"Speed                         : {speed:.2f}x realtime")

        print("----------------------------------")


if __name__ == "__main__":

    try:
        run()

    except KeyboardInterrupt:
        print("\nBye.")