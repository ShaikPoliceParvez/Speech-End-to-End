from pathlib import Path

from config import DEFAULT_LANGUAGE, TTS_LANGUAGE_BACKENDS
from piper_tts import PiperSpeaker
from tts import Speaker

# Languages routed to SuperTonic; all others use Piper.
_SUPERTONIC_LANGUAGES = {"en", "hi"}


class TTSRouter:
    """Selects a TTS backend by language while preserving Speaker's API."""

    def __init__(self, on_event=None):
        self.on_event = on_event
        self.supertonic = Speaker(on_event=on_event)
        self._piper_backends = {}
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
                "Unsupported language. Currently supported languages are English, Hindi, Telugu, Malayalam, and Arabic."
            )
        return language, route

    def _get_piper(self, language, route):
        backend = self._piper_backends.get(language)
        if backend is None:
            model_path = Path(__file__).resolve().parent / route["model"]
            backend = PiperSpeaker(model_path, self.supertonic, on_event=self.on_event)
            self._piper_backends[language] = backend
        return backend

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
            self._get_piper(self._language, route).speak_stream(sentences, self._language)
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