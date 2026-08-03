from pathlib import Path
import time

import numpy as np


class PiperSpeaker:
    """Cached Piper synthesis backend that uses the shared Speaker playback queue."""

    def __init__(self, model_path, playback, on_event=None):
        from piper import PiperVoice

        self.playback = playback
        self.on_event = on_event
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Piper model not found: {self.model_path}")

        model_start = time.perf_counter()
        self.voice = PiperVoice.load(self.model_path)
        self.model_startup_metrics = {
            "model": self.model_path.name,
            "model_startup_ms": round((time.perf_counter() - model_start) * 1000, 2),
        }
        self.started = False

    def speak_stream(self, sentences, language):
        for sentence in sentences:
            if self.playback.is_interrupted():
                break

            sentence = sentence.strip()
            if not sentence:
                continue

            if not self.started and self.on_event is not None:
                self.on_event("TTS_STARTED", {})
                self.started = True
            if self.on_event is not None:
                self.on_event("TTS_QUEUE", {"sentence": sentence, "language": language})

            self.playback.begin_synthesis()
            for chunk in self.voice.synthesize(sentence):
                if self.playback.is_interrupted():
                    break
                audio = np.asarray(chunk.audio_float_array, dtype=np.float32)
                self.playback.enqueue_audio(
                    audio,
                    chunk.sample_rate,
                    sentence,
                    language,
                )