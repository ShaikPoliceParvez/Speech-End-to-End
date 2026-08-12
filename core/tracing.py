import os
import re
import time
import uuid
from contextlib import nullcontext

from dotenv import load_dotenv

load_dotenv()

from langfuse import Langfuse, propagate_attributes
from config import STT_MODEL, WHISPER_SIZE, LLM_MODEL, VOICE

try:
    import psutil
except ImportError:
    psutil = None


_EMAIL_PATTERN = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_PHONE_PATTERN = re.compile(r"\b(?:\+?\d[\d .()-]{7,}\d)\b")


def redact_sensitive_text(value):
    if isinstance(value, dict):
        return {key: redact_sensitive_text(item) for key, item in value.items()}

    if isinstance(value, list):
        return [redact_sensitive_text(item) for item in value]

    if not isinstance(value, str):
        return value

    value = _EMAIL_PATTERN.sub("[REDACTED_EMAIL]", value)
    return _PHONE_PATTERN.sub("[REDACTED_PHONE]", value)


class LangfuseTracer:
    def __init__(self):
        public_key = (os.getenv("LANGFUSE_PUBLIC_KEY") or "").strip().strip("\"'")
        secret_key = (os.getenv("LANGFUSE_SECRET_KEY") or "").strip().strip("\"'")
        self.enabled = bool(public_key and secret_key)
        self.client = None
        self.session_id = str(uuid.uuid4())
        self.conversation_id = str(uuid.uuid4())
        self.model_startup_metrics = {}

        if self.enabled:
            base_url = os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")
            self.client = Langfuse(
                public_key=public_key,
                secret_key=secret_key,
                base_url=base_url.strip().strip("\"'"),
                environment=os.getenv("LANGFUSE_TRACING_ENVIRONMENT", "development"),
            )

    def start_turn(self, text, language, intent, input_mode):
        if not self.enabled:
            return nullcontext(None)

        request_id = str(uuid.uuid4())
        return self.client.start_as_current_observation(
            as_type="span",
            name="chat-turn",
            input={"message": redact_sensitive_text(text)},
            metadata={
                "language": language,
                "intent": intent,
                "input_mode": input_mode,
                "app": "tarz",
                "stt_model": f"{STT_MODEL}:{WHISPER_SIZE}",
                "llm_model": LLM_MODEL,
                "tts_voice": VOICE,
                "session_id": self.session_id,
                "conversation_id": self.conversation_id,
                "request_id": request_id,
                "timestamp": time.time(),
                "cpu_percent": self._cpu_percent(),
                "ram_percent": self._ram_percent(),
                "model_startup": self.model_startup_metrics,
            },
        )

    def set_model_startup_metrics(self, **metrics):
        self.model_startup_metrics = metrics

    @staticmethod
    def now():
        return time.perf_counter()

    @staticmethod
    def elapsed_ms(start):
        return round((time.perf_counter() - start) * 1000, 2)

    @staticmethod
    def _cpu_percent():
        return psutil.cpu_percent(interval=None) if psutil is not None else None

    @staticmethod
    def _ram_percent():
        return psutil.virtual_memory().percent if psutil is not None else None

    def update_turn_metrics(self, turn, metrics):
        if turn is None:
            return

        # Surface the most actionable latencies directly in output so they appear
        # at the top level in Langfuse without drilling into metadata.
        output_summary = {}
        for key in (
            "total_task_ms",
            "time_to_first_audio_ms",
            "pipeline_before_tts_ms",
            "language_latency_ms",
            "router_latency_ms",
            "memory_latency_ms",
            "stt_latency_ms",
        ):
            if key in metrics:
                output_summary[key] = metrics[key]

        llm = metrics.get("llm") or {}
        if llm.get("first_token_ms") is not None:
            output_summary["llm_first_token_ms"] = llm["first_token_ms"]
        if llm.get("total_latency_ms") is not None:
            output_summary["llm_total_ms"] = llm["total_latency_ms"]
        if llm.get("tokens_per_second") is not None:
            output_summary["llm_tokens_per_second"] = llm["tokens_per_second"]

        tts = metrics.get("tts") or {}
        if tts.get("first_audio_latency_ms") is not None:
            output_summary["tts_first_audio_ms"] = tts["first_audio_latency_ms"]
        if tts.get("playback_duration_ms") is not None:
            output_summary["playback_ms"] = tts["playback_duration_ms"]

        turn.update(
            output=output_summary,
            metadata={
                "timing": metrics,
                "cpu_percent_end": self._cpu_percent(),
                "ram_percent_end": self._ram_percent(),
            },
        )

    def update_turn(self, turn, text, language, intent):
        if turn is None:
            return

        turn.update(
            input={"message": redact_sensitive_text(text)},
            metadata={"language": language, "intent": intent},
        )

    def turn_attributes(self):
        if not self.enabled:
            return nullcontext()

        return propagate_attributes(
            session_id=self.session_id,
            tags=["voice-assistant"],
            trace_name="chat-turn",
        )

    def start_generation(self, model, messages, model_parameters=None):
        if not self.enabled:
            return nullcontext(None)

        sanitized_messages = [
            {
                "role": message.get("role"),
                "content": redact_sensitive_text(message.get("content", "")),
            }
            for message in messages
        ]

        return self.client.start_as_current_observation(
            as_type="generation",
            name="LLM",
            model=model,
            model_parameters=model_parameters,
            input=sanitized_messages,
            metadata={"provider": "ollama", "streaming": True},
        )

    def start_step(self, name, input=None, metadata=None, as_type="span"):
        if not self.enabled:
            return nullcontext(None)

        return self.client.start_as_current_observation(
            as_type=as_type,
            name=name,
            input=redact_sensitive_text(input),
            metadata=redact_sensitive_text(metadata),
        )

    def start_manual_step(self, parent, name, input=None, metadata=None, as_type="span"):
        if parent is None or not self.enabled:
            return None

        return parent.start_observation(
            as_type=as_type,
            name=name,
            input=redact_sensitive_text(input),
            metadata=redact_sensitive_text(metadata),
        )

    def update_step(self, observation, output=None, metadata=None):
        if observation is None:
            return

        observation.update(
            output=redact_sensitive_text(output),
            metadata=redact_sensitive_text(metadata),
        )

    def record_event(self, parent, name, metadata=None):
        if parent is None or not self.enabled:
            return

        event = parent.start_observation(
            as_type="event",
            name=name,
            metadata=redact_sensitive_text(metadata),
        )

    def flush(self):
        if self.enabled:
            self.client.flush()