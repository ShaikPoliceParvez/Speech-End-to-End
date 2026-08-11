"""
Fast STT benchmark (matches Tarz production pipeline).

Measures:
- Model load
- Warm-up
- Recording time
- Final transcription latency
- Real-time factor
"""

import sys, os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import numpy as np

from audio.microphone import Microphone
from stt.stt import STT
from config import (
    SAMPLE_RATE,
    WHISPER_BEAM_SIZE,
    WHISPER_TEMPERATURES,
    WHISPER_COMPRESSION_RATIO_THRESHOLD,
    WHISPER_LOG_PROB_THRESHOLD,
    WHISPER_NO_SPEECH_THRESHOLD,
    WHISPER_HINDI_PROMPT,
)


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


def final_decode(stt, audio, language=None):

    prompt = WHISPER_HINDI_PROMPT if language == "hi" else None

    segments, info = stt.model.transcribe(
        audio,
        beam_size=WHISPER_BEAM_SIZE,
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

    text = "".join(s.text for s in segments).strip()

    return text, info


def run():

    print("\n========== STT BENCHMARK ==========\n")

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

        decode_start = time.perf_counter()

        text, info = final_decode(stt, audio)

        second_pass = 0.0

        if getattr(info, "language", None) == "hi":

            t = time.perf_counter()

            hi_text, hi_info = final_decode(
                stt,
                audio,
                language="hi",
            )

            second_pass = time.perf_counter() - t

            if hi_text:
                text = hi_text
                info = hi_info

        decode_time = time.perf_counter() - decode_start

        print("----------------------------------")
        print(f"Transcript        : {text}")
        print(f"Language          : {info.language}")
        print(f"Confidence        : {info.language_probability:.2f}")
        print("----------------------------------")
        print(f"Recording         : {record_time:.2f}s")
        print(f"Time took to generate the transcript : {decode_time:.2f}s")

        if second_pass:
            print(f"Hindi 2nd Pass    : {second_pass:.2f}s")

        print(f"Audio Length      : {audio_seconds:.2f}s")
        print(f"Real-time Factor  : {decode_time/audio_seconds:.2f}x")

        if audio_seconds:
            speed = audio_seconds / decode_time
            print(f"Speed             : {speed:.2f}x realtime")

        print("----------------------------------")


if __name__ == "__main__":

    try:
        run()

    except KeyboardInterrupt:
        print("\nBye.")

