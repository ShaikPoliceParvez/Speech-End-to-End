from pathlib import Path

from config import DEFAULT_LANGUAGE, TTS_LANGUAGE_BACKENDS
from tts.piper_tts import PiperSpeaker
from tts.tts import Speaker

# Languages routed to SuperTonic; all others use Piper.
_SUPERTONIC_LANGUAGES = {"en", "hi"}


class TTSRouter:
    """Selects a TTS backend by language while preserving Speaker's API."""

    def __init__(self, on_event=None):
        self.on_event = on_event
        self.supertonic = Speaker(on_event=on_event)
        self._piper_backends = {}
        self._piper_unavailable = set()
        self._language = DEFAULT_LANGUAGE
        self._backend_name = None
        self.set_language(DEFAULT_LANGUAGE)

    @property
    def model_startup_metrics(self):
        metrics = self.supertonic.model_startup_metrics.copy()
        metrics["router"] = "tts-router"
        if self._piper_backends:
            metrics["piper"] = {
                language: backend.model_startup_metrics
                for language, backend in self._piper_backends.items()
            }
        return metrics

    @property
    def backend_name(self):
        return self._backend_name

    def _route(self, language):
        language = (language or DEFAULT_LANGUAGE).lower()
        route = TTS_LANGUAGE_BACKENDS.get(language)
        if route is None:
            raise ValueError(
                "Unsupported language. Currently supported languages are English, Hindi, Nepali, Telugu, Malayalam, and Arabic."
            )
        return language, route

    def _get_piper(self, language, route):
        if language in self._piper_unavailable:
            return None
        backend = self._piper_backends.get(language)
        if backend is None:
            project_root = Path(__file__).resolve().parent.parent
            model_path = project_root / route["model"]
            try:
                backend = PiperSpeaker(model_path, self.supertonic, on_event=self.on_event)
            except FileNotFoundError as error:
                self._piper_unavailable.add(language)
                if self.on_event is not None:
                    self.on_event(
                        "TTS_ERROR",
                        {
                            "error": str(error),
                            "language": language,
                            "fallback": "supertonic",
                        },
                    )
                return None
            self._piper_backends[language] = backend
        return backend

    @staticmethod
    def _supertonic_fallback_language(language):
        # Nepali is closest to Hindi script/phonetics among current SuperTonic voices.
        if language in {"hi", "ne"}:
            return "hi"
        return "en"

    def start_turn(self):
        self.supertonic.start_turn()
        for backend in self._piper_backends.values():
            backend.started = False

    def set_language(self, language):
        language, route = self._route(language)
        self._language = language
        self._backend_name = route["backend"]
        if language in _SUPERTONIC_LANGUAGES:
            self.supertonic.set_language(language)
        # te / ml / ar go to Piper — SuperTonic is not called for these

    def set_event_handler(self, on_event):
        """Update event handling for the active and lazily loaded backends."""
        self.on_event = on_event
        self.supertonic.on_event = on_event
        for backend in self._piper_backends.values():
            backend.on_event = on_event

    def speak_stream(self, sentences):
        _, route = self._route(self._language)
        if route["backend"] == "supertonic":
            self.supertonic.speak_stream(sentences)
            return
        if route["backend"] == "piper":
            backend = self._get_piper(self._language, route)
            if backend is not None:
                backend.speak_stream(sentences, self._language)
                return
            # Graceful degradation: keep the response audible even if a Piper model is missing.
            self.supertonic.set_language(self._supertonic_fallback_language(self._language))
            self.supertonic.speak_stream(sentences)
            return
        raise RuntimeError(f"Unknown TTS backend: {route['backend']}")

    def speak(self, text, language):
        self.start_turn()
        self.set_language(language)
        self.speak_stream([text])
        self.wait_until_idle()

    def wait_until_idle(self):
        self.supertonic.wait_until_idle()

    def stop(self):
        self.supertonic.stop()

    def is_interrupted(self):
        return self.supertonic.is_interrupted()

    def is_speaking(self):
        return self.supertonic.is_speaking()

    def get_turn_metrics(self):
        return self.supertonic.get_turn_metrics()

    def get_supported_languages(self):
        return TTS_LANGUAGE_BACKENDS.copy()