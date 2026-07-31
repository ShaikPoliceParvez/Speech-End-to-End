import queue
import time

import numpy as np
import sounddevice as sd

from config import (
    SAMPLE_RATE,
    VAD_SILENCE_THRESHOLD,
    VAD_SILENCE_DURATION,
    VAD_MIN_SPEECH_DURATION,
    VAD_GRACE_PERIOD,
    VAD_MAX_RECORD_SECONDS,
)


class Microphone:
    """
    Energy-based VAD microphone with automatic endpoint detection.

    Note: this is a lightweight RMS-energy VAD used only to decide *when to
    stop recording*. Silero VAD (via faster-whisper's vad_filter=True) is
    applied separately inside stt.py to clean the recorded audio before
    transcription -- the two VAD passes serve different purposes and are
    intentionally not the same one.
    """

    def __init__(
        self,
        sample_rate=SAMPLE_RATE,
        block_duration=0.03,
        silence_threshold=VAD_SILENCE_THRESHOLD,
        silence_duration=VAD_SILENCE_DURATION,
        min_speech_duration=VAD_MIN_SPEECH_DURATION,
        grace_period=VAD_GRACE_PERIOD,
        max_record_seconds=VAD_MAX_RECORD_SECONDS,
    ):
        self.sample_rate = sample_rate
        self.block_size = max(1, int(sample_rate * block_duration))

        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        self.min_speech_duration = min_speech_duration
        self.grace_period = grace_period
        self.max_record_seconds = max_record_seconds

        self.queue = queue.Queue()

    def _callback(self, indata, frames, time_info, status):
        if status:
            print(status)

        self.queue.put(indata.copy())

    @staticmethod
    def _energy(chunk: np.ndarray) -> float:
        if chunk.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(chunk))))

    def listen(self, return_metrics=False):
        """
        Block until the user speaks, then record until VAD detects the end
        of the utterance (sustained silence after speech) or the hard cap
        is reached. Returns a float32 mono numpy array at self.sample_rate.
        """

        print("\n🎤 Listening... (Auto-stop on silence)")

        frames = []
        microphone_start = time.perf_counter()
        speech_start = None
        speech_end = None
        endpoint_time = None

        speech_started = False
        speech_time = 0.0
        silence_time = 0.0
        total_time = 0.0

        with sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.block_size,
            callback=self._callback,
        ):
            while True:

                chunk = self.queue.get().flatten()
                chunk_duration = len(chunk) / self.sample_rate
                energy = self._energy(chunk)
                total_time += chunk_duration

                is_speech = energy > self.silence_threshold

                if is_speech:
                    if not speech_started:
                        speech_start = time.perf_counter()
                    speech_started = True
                    speech_time += chunk_duration
                    silence_time = 0.0
                    speech_end = time.perf_counter()
                    frames.append(chunk)
                    print(f"  🔊 Speaking (energy: {energy:.2f})", end="\r")

                elif speech_started:
                    frames.append(chunk)
                    silence_time += chunk_duration

                    # Grace period protects short mid-sentence pauses from
                    # being counted as the end of the utterance.
                    effective_silence = max(0.0, silence_time - self.grace_period)

                    if (
                        speech_time >= self.min_speech_duration
                        and effective_silence >= self.silence_duration
                    ):
                        endpoint_time = time.perf_counter()
                        print("\n  ✓ Endpoint detected")
                        break

                if total_time >= self.max_record_seconds:
                    print("\n  ⏱️  Max recording time reached")
                    break

        microphone_end = time.perf_counter()
        metrics = {
            "vad_start": microphone_start,
            "speech_start": speech_start,
            "speech_end": speech_end,
            "endpoint_time": endpoint_time,
            "duration_ms": round((microphone_end - microphone_start) * 1000, 2),
            "speech_duration_ms": round(speech_time * 1000, 2),
            "silence_duration_ms": round(silence_time * 1000, 2),
            "endpoint_delay_ms": round(silence_time * 1000, 2) if endpoint_time is not None else None,
        }

        if not frames:
            audio = np.array([], dtype=np.float32)
            return (audio, metrics) if return_metrics else audio

        audio = np.concatenate(frames).astype(np.float32)

        print(f"✓ Recorded: {len(audio) / self.sample_rate:.2f}s")

        return (audio, metrics) if return_metrics else audio
