import threading
import queue
import re
import importlib
import time
import numpy as np
import sounddevice as sd
from numbers import Number

from supertonic import TTS
from config import (
    VOICE,
    TTS_SPEED,
    TTS_PRONUNCIATION_MAP,
    TTS_PREFETCH_TEXT,
    TTS_PREFETCH_AUDIO,
    SUPPORTED_LANGUAGES,
    DEFAULT_LANGUAGE,
)

try:
    _num2words_module = importlib.import_module("num2words")
    num2words = getattr(_num2words_module, "num2words", None)
except Exception:
    num2words = None


class Speaker:

    def __init__(self, on_event=None):

        model_start = time.perf_counter()
        self.tts = TTS(auto_download=True)
        self.style = None
        for candidate in [VOICE, "F1", "F2", "M1"]:
            try:
                self.style = self.tts.get_voice_style(candidate)
                break
            except Exception:
                continue

        if self.style is None:
            raise RuntimeError("Could not load any SuperTonic voice style")
        self.model_startup_metrics = {
            "voice": VOICE,
            "model_startup_ms": round((time.perf_counter() - model_start) * 1000, 2),
        }
        self.on_event = on_event
        self.config_languages = SUPPORTED_LANGUAGES.copy()
        self.engine_languages = self._discover_engine_languages()
        self.active_languages = self._resolve_active_languages()

        # Keep queues unbounded so LLM streaming is never blocked by TTS throughput.
        self.text_queue = queue.Queue()
        self.audio_queue = queue.Queue()
        self.started = False
        self.current_language = DEFAULT_LANGUAGE
        self._metrics_lock = threading.Lock()
        self._turn_metrics = {}
        self._stream = None
        self._stream_rate = None
        self.num2words_lang_map = {
            "en": "en",
            "hi": "hi",
            "te": "te",
            "ta": "ta",
            "kn": "kn",
            "ml": "ml",
            "bn": "bn",
            "mr": "mr",
            "gu": "gu",
            "pa": "pa",
            "ur": "ur",
        }

        threading.Thread(
            target=self._synth_worker,
            daemon=True
        ).start()

        threading.Thread(
            target=self._play_worker,
            daemon=True
        ).start()

    def start_turn(self):
        self.started = False
        with self._metrics_lock:
            self._turn_metrics = {
                "turn_start": time.perf_counter(),
                "synthesis_start": None,
                "first_audio_ready": None,
                "synthesis_end": None,
                "playback_start": None,
                "playback_end": None,
                "audio_duration_ms": 0.0,
                "chunk_count": 0,
                "queue_delay_ms": None,
                "sample_rate": None,
            }

    def get_turn_metrics(self):
        with self._metrics_lock:
            metrics = self._turn_metrics.copy()

        start = metrics.get("turn_start")
        synthesis_start = metrics.get("synthesis_start")
        first_audio_ready = metrics.get("first_audio_ready")
        synthesis_end = metrics.get("synthesis_end")
        playback_start = metrics.get("playback_start")
        playback_end = metrics.get("playback_end")

        metrics.update({
            "first_audio_latency_ms": round((first_audio_ready - start) * 1000, 2)
            if first_audio_ready is not None and start is not None else None,
            "synthesis_duration_ms": round((synthesis_end - synthesis_start) * 1000, 2)
            if synthesis_end is not None and synthesis_start is not None else None,
            "playback_duration_ms": round((playback_end - playback_start) * 1000, 2)
            if playback_end is not None and playback_start is not None else None,
        })
        return metrics

    def speak_stream(self, sentence_stream):

        for sentence in sentence_stream:

            if sentence.strip():
                if not self.started and self.on_event is not None:
                    self.on_event("TTS_STARTED", {})
                    self.started = True
                if self.on_event is not None:
                    self.on_event("TTS_QUEUE", {"sentence": sentence, "language": self.current_language})
                self.text_queue.put((sentence, self.current_language))

    def set_language(self, language):
        lang = (language or DEFAULT_LANGUAGE).lower()
        if lang not in self.active_languages:
            lang = DEFAULT_LANGUAGE
        self.current_language = lang

    def get_supported_languages(self):
        return self.active_languages

    def wait_until_idle(self):
        self.text_queue.join()
        self.audio_queue.join()

    def _discover_engine_languages(self):
        discovered = {}

        for method_name in ["get_supported_languages", "list_languages"]:
            method = getattr(self.tts, method_name, None)
            if callable(method):
                try:
                    value = method()
                    if isinstance(value, dict):
                        for code, name in value.items():
                            discovered[str(code).lower()] = str(name)
                    elif isinstance(value, (list, tuple, set)):
                        for code in value:
                            code = str(code).lower()
                            discovered[code] = self.config_languages.get(code, code)
                except Exception:
                    continue

        for attr_name in ["supported_languages", "languages", "language_codes"]:
            value = getattr(self.tts, attr_name, None)
            if isinstance(value, dict):
                for code, name in value.items():
                    discovered[str(code).lower()] = str(name)
            elif isinstance(value, (list, tuple, set)):
                for code in value:
                    code = str(code).lower()
                    discovered[code] = self.config_languages.get(code, code)

        return discovered

    def _resolve_active_languages(self):
        if not self.engine_languages:
            return self.config_languages

        resolved = {}
        for code, name in self.config_languages.items():
            if code in self.engine_languages:
                resolved[code] = name

        if not resolved:
            return self.config_languages

        return resolved

    def _prepare_text(self, sentence, language):
        text = sentence

        # Remove sentence-ending punctuation after sentence buffer split.
        text = re.sub(r"[.!?]+\s*$", "", text)

        # Expand numbers in current conversation language for natural speech.
        text = self._expand_numbers(text, language)

        # Punctuation normalization for smooth pacing in TTS.
        text = text.replace("...", "  ")
        text = re.sub(r"[,:;]+", "  ", text)
        text = re.sub(r"[\(\)\[\]\{\}]", " ", text)
        text = re.sub(r"[\"'`]", "", text)
        text = re.sub(r"[-–—]+", " ", text)
        text = re.sub(r"[!?]+", "  ", text)
        text = re.sub(r"\.{2,}", "  ", text)

        for name, spoken in TTS_PRONUNCIATION_MAP.items():
            text = re.sub(rf"\b{re.escape(name)}\b", spoken, text, flags=re.IGNORECASE)

        # Collapse spacing but keep at most two spaces for pacing.
        text = re.sub(r"\s{3,}", "  ", text)
        text = re.sub(r"\s*\n\s*", " ", text).strip()

        return text

    def _expand_numbers(self, text, language):
        lang = (language or DEFAULT_LANGUAGE).lower()
        num_lang = self.num2words_lang_map.get(lang, "en")

        def convert(match):
            value = match.group(0)
            try:
                if num2words is not None:
                    return num2words(int(value), lang=num_lang)
            except Exception:
                pass

            # Fallback: keep original if conversion backend is unavailable.
            return value

        return re.sub(r"\b\d+\b", convert, text)

    def _trim_silence(self, audio, sample_rate, threshold=0.008, pad_ms=60):
        if audio is None:
            return np.array([], dtype=np.float32)

        arr = np.asarray(np.squeeze(audio), dtype=np.float32)
        if arr.size == 0:
            return arr

        mask = np.abs(arr) > threshold
        if not np.any(mask):
            return np.array([], dtype=np.float32)

        start = int(np.argmax(mask))
        end = int(len(mask) - np.argmax(mask[::-1]))
        pad = int((sample_rate or 24000) * (pad_ms / 1000.0))

        start = max(0, start - pad)
        end = min(len(arr), end + pad)

        return arr[start:end]

    def _synth_worker(self):

        while True:

            sentence, language = self.text_queue.get()

            try:
                with self._metrics_lock:
                    if self._turn_metrics.get("synthesis_start") is None:
                        self._turn_metrics["synthesis_start"] = time.perf_counter()
                tts_text = self._prepare_text(sentence, language)

                if not tts_text:
                    continue

                wav, info = self.tts.synthesize(
                    text=tts_text,
                    voice_style=self.style,
                    lang=language
                )

                sample_rate = getattr(self.tts, "sample_rate", None)
                if isinstance(info, dict) and isinstance(info.get("sample_rate"), Number):
                    sample_rate = int(info["sample_rate"])
                elif isinstance(info, Number):
                    sample_rate = int(info)

                output_rate = int((sample_rate or 24000) * TTS_SPEED)
                trimmed_audio = self._trim_silence(wav, output_rate)

                if trimmed_audio.size:
                    ready_at = time.perf_counter()
                    with self._metrics_lock:
                        if self._turn_metrics.get("first_audio_ready") is None:
                            self._turn_metrics["first_audio_ready"] = ready_at
                        self._turn_metrics["synthesis_end"] = ready_at
                        self._turn_metrics["audio_duration_ms"] += len(trimmed_audio) / output_rate * 1000
                        self._turn_metrics["chunk_count"] += 1
                        self._turn_metrics["sample_rate"] = output_rate
                    self.audio_queue.put((trimmed_audio, output_rate, sentence, language, ready_at))

            except Exception as e:
                if self.on_event is not None:
                    self.on_event("TTS_ERROR", {"error": str(e), "sentence": sentence, "language": language})
                print(e)

            finally:
                self.text_queue.task_done()

    def _play_worker(self):

        while True:
            audio, output_rate, sentence, language, ready_at = self.audio_queue.get()

            try:
                playback_start = time.perf_counter()
                with self._metrics_lock:
                    if self._turn_metrics.get("playback_start") is None:
                        self._turn_metrics["playback_start"] = playback_start
                        self._turn_metrics["queue_delay_ms"] = round(
                            (playback_start - ready_at) * 1000,
                            2,
                        )
                if self.on_event is not None:
                    self.on_event("TTS_SPEAKING", {"sentence": sentence, "language": language})
                    self.on_event("AUDIO_PLAYING", {"sentence": sentence, "language": language})

                if self._stream is None or self._stream_rate != output_rate:
                    if self._stream is not None:
                        self._stream.stop()
                        self._stream.close()
                    self._stream = sd.OutputStream(
                        samplerate=output_rate,
                        channels=1,
                        dtype="float32",
                    )
                    self._stream.start()
                    self._stream_rate = output_rate

                self._stream.write(np.asarray(audio, dtype=np.float32).reshape(-1, 1))

                with self._metrics_lock:
                    self._turn_metrics["playback_end"] = time.perf_counter()

                if self.on_event is not None:
                    self.on_event("TTS_COMPLETED", {"sentence": sentence, "language": language})

            except Exception as e:
                if self.on_event is not None:
                    self.on_event("TTS_ERROR", {"error": str(e), "sentence": sentence, "language": language})
                print(e)

            finally:
                self.audio_queue.task_done()