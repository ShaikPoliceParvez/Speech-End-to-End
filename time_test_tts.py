"""
Standalone TTS benchmark (accurate + streamed input).

Type text; it is consumed as a *stream* of sentences (like LLM output
arriving token by token) and each sentence is synthesized and streamed to
your speakers. A warm-up pass loads the voice model first.

Accuracy focus:
    - Language is detected per sentence (so English, Hindi, and Telugu text is
        sent to the appropriate backend).
  - The existing Speaker text preparation (number expansion, pronunciation
    map) is used unchanged.

Run:
    python time_test_tts.py

This file only *reads* the existing modules. It does not modify or affect
the main app in any way.
"""

import re
import time
import threading

from language import detect_dominant_language
from tts_router import TTSRouter


class _EventTimer:
    """Timestamps Speaker events to derive per-sentence synth/playback times."""

    def __init__(self):
        self.t0 = time.perf_counter()
        self.queued = {}
        self.speak_start = {}
        self.first_audio = None
        self.lock = threading.Lock()

    def on_event(self, name, data):
        now = time.perf_counter()
        sentence = data.get("sentence")

        with self.lock:
            if name == "TTS_QUEUE":
                self.queued[sentence] = now

            elif name in ("TTS_SPEAKING", "AUDIO_PLAYING"):
                if sentence not in self.speak_start:
                    self.speak_start[sentence] = now
                    if self.first_audio is None:
                        self.first_audio = now - self.t0
                        print(f"  [first audio out @ {self.first_audio:.2f}s]")

            elif name == "TTS_COMPLETED":
                q = self.queued.get(sentence)
                s = self.speak_start.get(sentence, now)
                synth = (s - q) if q else 0.0
                play = now - s
                short = (sentence[:45] + "...") if len(sentence) > 45 else sentence
                print(f"  [{synth:5.2f}s synth | {play:5.2f}s play] {short}")

            elif name == "TTS_ERROR":
                print(f"  [error] {data.get('error')}")


def _split_sentences(text):
    parts = re.split(r"(?<=[.!?।])\s+", text.strip())
    return [p for p in parts if p.strip()]


def _guess_language(text):
    return detect_dominant_language(text)


def warm_up(speaker):
    """Synthesize a short phrase so the voice model is resident."""
    t0 = time.perf_counter()
    try:
        speaker.supertonic.tts.synthesize(
            text="ready",
            voice_style=speaker.supertonic.style,
            lang="en",
        )
    except Exception as e:
        print(f"(warm-up skipped: {e})")
        return
    print(f"Warm-up: {time.perf_counter() - t0:.2f}s")


def stream_sentences(sentences, speaker, delay=0.15):
    """
    Simulate sentences arriving as a stream (like LLM output). Language is set
    per sentence right before it is queued, so mixed-language text is spoken
    accurately.
    """
    for s in sentences:
        time.sleep(delay)
        speaker.set_language(_guess_language(s))
        short = (s[:45] + "...") if len(s) > 45 else s
        print(f"  (sentence in) {short}")
        yield s


def run():
    print("=== TTS BENCHMARK (accurate, streamed input) ===")

    load_start = time.perf_counter()
    speaker = TTSRouter(on_event=None)
    print(f"Model load: {time.perf_counter() - load_start:.2f}s")

    warm_up(speaker)
    print()

    while True:
        try:
            text = input("Text to speak (blank to quit): ").strip()
        except EOFError:
            break

        if not text:
            break

        sentences = _split_sentences(text)

        timer = _EventTimer()
        speaker.set_event_handler(timer.on_event)
        speaker.start_turn()

        print(f"\n--- Streaming audio ({len(sentences)} sentence(s)) ---")

        overall = time.perf_counter()
        speaker.speak_stream(stream_sentences(sentences, speaker))
        speaker.wait_until_idle()
        total = time.perf_counter() - overall

        print("\n--- Timings ---")
        print(f"Sentences           : {len(sentences)}")
        print(f"Time to first audio : {(timer.first_audio or 0.0):6.2f}s")
        print(f"Total (synth+play)  : {total:6.2f}s")
        print()


if __name__ == "__main__":
    try:
        run()
    except KeyboardInterrupt:
        print("\nBye.")
