from faster_whisper import WhisperModel
import numpy as np
import time

from config import (
    WHISPER_SIZE,
    WHISPER_DEVICE,
    WHISPER_COMPUTE,
    WHISPER_BEAM_SIZE,
    WHISPER_TEMPERATURES,
    WHISPER_COMPRESSION_RATIO_THRESHOLD,
    WHISPER_LOG_PROB_THRESHOLD,
    WHISPER_NO_SPEECH_THRESHOLD,
    WHISPER_HINDI_PROMPT,
)


class STT:
    def __init__(self, verbose=False):
        self.verbose = verbose

        print("Loading Whisper...")
        model_start = time.perf_counter()

        self.model = WhisperModel(
            WHISPER_SIZE,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE,
        )
        self.model_startup_metrics = {
            "model": WHISPER_SIZE,
            "device": WHISPER_DEVICE,
            "compute_type": WHISPER_COMPUTE,
            "model_startup_ms": round((time.perf_counter() - model_start) * 1000, 2),
        }

        print("Whisper Ready")

    def transcribe(
        self,
        audio: np.ndarray,
        language=None,
        final=True,
    ):
        """
        Parameters
        ----------
        audio : float32 numpy array
            Mono PCM at SAMPLE_RATE.

        Returns
        -------
        dict
        {
            text,
            language,
            confidence,
            script
        }
        """

        if len(audio) == 0:
            return self.empty_result()

        # Final transcription uses beam search + temperature fallback for
        # accuracy; partials stay greedy (beam_size=1) so they stay fast.
        # A Devanagari initial prompt biases the decoder toward correct Hindi
        # script when we already know the turn is Hindi.
        initial_prompt = WHISPER_HINDI_PROMPT if language == "hi" else None

        transcription_start = time.perf_counter()
        segments, info = self.model.transcribe(
            audio,
            beam_size=WHISPER_BEAM_SIZE if final else 1,
            language=language,
            task="transcribe",
            vad_filter=final,
            without_timestamps=True,
            condition_on_previous_text=False,
            initial_prompt=initial_prompt,
            temperature=WHISPER_TEMPERATURES if final else 0.0,
            compression_ratio_threshold=WHISPER_COMPRESSION_RATIO_THRESHOLD,
            log_prob_threshold=WHISPER_LOG_PROB_THRESHOLD,
            no_speech_threshold=WHISPER_NO_SPEECH_THRESHOLD,
        )

        parts = []
        first_segment_ms = None
        for segment in segments:
            if first_segment_ms is None:
                first_segment_ms = round((time.perf_counter() - transcription_start) * 1000, 2)
            parts.append(segment.text)

        text = "".join(parts).strip()
        transcription_latency_ms = round((time.perf_counter() - transcription_start) * 1000, 2)

        detected_language = getattr(info, "language", None)
        probability = getattr(info, "language_probability", 0.0)

        result = {
            "text": text,
            "language": detected_language,
            "confidence": probability,
            "script": self.detect_script(text),
            "first_segment_ms": first_segment_ms,
            "first_partial_transcript": parts[0].strip() if parts else None,
            "latency_ms": transcription_latency_ms,
        }

        if self.verbose:
            print(result)

        return result

    @staticmethod
    def detect_script(text):

        has_dev = any("\u0900" <= c <= "\u097F" for c in text)
        has_lat = any(c.isascii() and c.isalpha() for c in text)

        if has_dev and has_lat:
            return "mixed"

        if has_dev:
            return "devanagari"

        if has_lat:
            return "latin"

        return "unknown"

    @staticmethod
    def empty_result():
        return {
            "text": "",
            "language": None,
            "confidence": 0.0,
            "script": "unknown",
            "first_segment_ms": None,
            "first_partial_transcript": None,
            "latency_ms": 0.0,
        }