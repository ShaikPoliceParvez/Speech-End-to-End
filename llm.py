import ollama
from contextlib import nullcontext
from io import BytesIO
from pathlib import Path
import time

from PIL import Image

from memory import Memory
from config import SUPPORTED_LANGUAGES, DEFAULT_LANGUAGE, LLM_MODEL, LLM_MAX_TOKENS


class LLM:

    def __init__(self, model=None, tracer=None):

        self.model = model or LLM_MODEL
        self.memory = Memory()
        self.tracer = tracer
        self._vision_supported = None
        self.last_metrics = {}
        self.model_startup_metrics = None
        self._first_request = True

    def measure_model_startup(self):
        """Measure Ollama model readiness without generating a response."""
        start = time.perf_counter()
        try:
            model_info = ollama.show(self.model)
            capabilities = getattr(model_info, "capabilities", None)
            if capabilities is None:
                capabilities = model_info.get("capabilities", [])
            self._vision_supported = "vision" in capabilities
            self.model_startup_metrics = {
                "model": self.model,
                "model_readiness_ms": round((time.perf_counter() - start) * 1000, 2),
                "ready": True,
                "vision_supported": self._vision_supported,
            }
        except Exception as error:
            self.model_startup_metrics = {
                "model": self.model,
                "model_readiness_ms": round((time.perf_counter() - start) * 1000, 2),
                "ready": False,
                "error": str(error),
            }

        return self.model_startup_metrics

    def supports_vision(self):
        if self._vision_supported is None:
            try:
                model_info = ollama.show(self.model)
                capabilities = getattr(model_info, "capabilities", None)
                if capabilities is None:
                    capabilities = model_info.get("capabilities", [])
                self._vision_supported = "vision" in capabilities
            except Exception:
                self._vision_supported = False

        return self._vision_supported

    def stream(
        self,
        prompt,
        image=None,
        vision_requested=False,
        on_event=None,
        language=None,
        allow_roman_output=False,
    ):

        lang = (language or self.memory.get_language() or DEFAULT_LANGUAGE).lower()
        if lang not in SUPPORTED_LANGUAGES:
            lang = DEFAULT_LANGUAGE

        self.memory.set_language(lang)

        self.memory.add_user(prompt)

        if vision_requested and not self.supports_vision():
            if lang == "hi":
                assistant = "माफ़ कीजिए, वर्तमान मॉडल कैमरा या इमेज नहीं देख सकता।"
            else:
                assistant = "Sorry, the current model cannot see camera images."

            self.memory.add_assistant(assistant)
            self.last_metrics = {
                "first_token_ms": 0.0,
                "total_latency_ms": 0.0,
                "input_tokens": 0,
                "output_tokens": 0,
                "tokens_per_second": None,
            }

            if on_event is not None:
                on_event("LLM_FIRST_TOKEN", {"token": assistant})
                on_event("LLM_TOKEN", {"token": assistant, "text": assistant})
                on_event("LLM_STREAMING", {"token": assistant})
                on_event("LLM_COMPLETED", {"text": assistant})

            yield assistant
            return

        language_name = SUPPORTED_LANGUAGES.get(lang, "English")
        if lang == "hi":
            if allow_roman_output:
                system_instruction = (
                    "The user's conversation language is Hindi. "
                    "The user explicitly requested Hinglish output. "
                    "Respond in Hindi using Roman script (Hinglish). "
                    "Never translate unless the user explicitly asks for translation."
                )
            else:
                system_instruction = (
                    "The user's conversation language is Hindi. "
                    "The user may type in Roman Hindi, but input is normalized before you receive it. "
                    "Always respond ONLY in proper Hindi using Devanagari script. "
                    "Never respond in Hinglish unless the user explicitly requests Hinglish. "
                    "Never translate unless the user explicitly asks for translation."
                )
        else:
            system_instruction = (
                f"Current conversation language is {language_name}. "
                f"Always answer ONLY in {language_name}. "
                "Never translate unless the user explicitly asks for translation."
            )

        # Adapt response length to the matter at hand: keep simple questions
        # short, but give full, detailed answers when the topic needs depth
        # (stories, explanations, how-to steps, comparisons).
        system_instruction += (
            " Match the length of your answer to what the question needs: "
            "be concise for simple or factual questions, and give a thorough, "
            "well-developed answer when the topic calls for detail, such as "
            "stories, explanations, instructions, or comparisons. "
            "You are an AI assistant model, you should listen to the user's input carefully and respond appropriately. "
            "do not give lengthy responses, keep your responses short and concise. "
        )

        messages = [{
            "role": "system",
            "content": system_instruction,
        }]
        messages.extend(self.memory.messages().copy())

        # If vision is requested
        if image is not None:

            if isinstance(image, Image.Image):
                # Ollama accepts bytes/path; convert in-memory PIL image to PNG bytes.
                buffer = BytesIO()
                image.save(buffer, format="PNG")
                image = buffer.getvalue()
            elif isinstance(image, Path):
                image = str(image)

            messages[-1] = {
                "role": "user",
                "content": prompt,
                "images": [image],      # image path or bytes
            }

        assistant = ""
        saw_first_token = False
        usage_details = {}
        request_start = time.perf_counter()
        first_token_ms = None
        is_first_request = self._first_request

        generation = (
            self.tracer.start_generation(
                self.model,
                messages,
                model_parameters={"max_tokens": LLM_MAX_TOKENS},
            )
            if self.tracer is not None
            else None
        )

        with generation if generation is not None else nullcontext(None) as observation:
            if self.tracer is not None:
                self.tracer.record_event(observation, "LLM Streaming Started")
            response = ollama.chat(
                model=self.model,
                messages=messages,
                stream=True,
                options={"num_predict": LLM_MAX_TOKENS},
            )

            for chunk in response:

                token = chunk["message"]["content"]

                assistant += token

                if not saw_first_token:
                    first_token_ms = round((time.perf_counter() - request_start) * 1000, 2)
                    if self.tracer is not None:
                        self.tracer.record_event(
                            observation,
                            "LLM First Token Received",
                            {"ttft_ms": first_token_ms},
                        )

                if on_event is not None:
                    if not saw_first_token:
                        on_event("LLM_FIRST_TOKEN", {"token": token})
                        saw_first_token = True

                    on_event("LLM_TOKEN", {
                        "token": token,
                        "text": assistant,
                    })
                    on_event("LLM_STREAMING", {
                        "token": token,
                    })

                if "prompt_eval_count" in chunk:
                    usage_details["input_tokens"] = chunk["prompt_eval_count"]
                if "eval_count" in chunk:
                    usage_details["output_tokens"] = chunk["eval_count"]

                yield token

            total_latency_ms = round((time.perf_counter() - request_start) * 1000, 2)
            output_tokens = usage_details.get("output_tokens", 0)
            self.last_metrics = {
                "first_token_ms": first_token_ms,
                "total_latency_ms": total_latency_ms,
                "input_tokens": usage_details.get("input_tokens", 0),
                "output_tokens": output_tokens,
                "tokens_per_second": round(
                    output_tokens / max(total_latency_ms / 1000, 0.001),
                    2,
                ) if output_tokens else None,
                "first_request": is_first_request,
                "cold_start_ttft_ms": first_token_ms if is_first_request else None,
            }
            self._first_request = False
            if observation is not None:
                observation.update(
                    output=assistant,
                    usage_details=usage_details or None,
                    metadata={
                        "request_sent": request_start,
                        **self.last_metrics,
                        "prompt_characters": sum(
                            len(message.get("content", "")) for message in messages
                        ),
                    },
                )

        self.memory.add_assistant(assistant)

        if on_event is not None:
            on_event("LLM_COMPLETED", {"text": assistant})


def sentence_stream(
    token_stream,
    on_event=None,
    min_chars=72,
    min_words=12,
    max_chars=220,
    max_words=36,
):

    buffer = ""
    pending = ""

    endings = [".", "!", "?"]

    def push_piece(piece):
        nonlocal pending
        piece = piece.strip()
        if not piece:
            return None

        if pending:
            pending = f"{pending} {piece}".strip()
        else:
            pending = piece

        chars = len(pending)
        words = len(pending.split())

        if chars >= min_chars and words >= min_words:
            out = pending
            pending = ""
            return out

        if chars >= max_chars or words >= max_words:
            out = pending
            pending = ""
            return out

        return None

    for token in token_stream:

        buffer += token

        while True:
            index = -1

            for end_char in endings:
                pos = buffer.find(end_char)
                if pos != -1 and (index == -1 or pos < index):
                    index = pos

            if index == -1:
                break

            sentence = buffer[:index + 1].strip()
            buffer = buffer[index + 1:].lstrip()

            if sentence:
                out = push_piece(sentence)
                if out:
                    if on_event is not None:
                        on_event("SENTENCE_READY", {"sentence": out})
                        on_event("LLM_SENTENCE_READY", {"sentence": out})
                    yield out

        cleaned = buffer.strip()
        if cleaned:
            words = len(cleaned.split())

            # Safety flush for long run-on output without punctuation.
            if len(cleaned) >= max_chars or words >= max_words:
                sentence = cleaned
                buffer = ""
                out = push_piece(sentence)
                if out:
                    if on_event is not None:
                        on_event("SENTENCE_READY", {"sentence": out})
                        on_event("LLM_SENTENCE_READY", {"sentence": out})
                    yield out

    if buffer.strip():
        sentence = buffer.strip()
        out = push_piece(sentence)
        if out:
            if on_event is not None:
                on_event("SENTENCE_READY", {"sentence": out})
                on_event("LLM_SENTENCE_READY", {"sentence": out})
            yield out

    if pending.strip():
        sentence = pending.strip()
        if on_event is not None:
            on_event("SENTENCE_READY", {"sentence": sentence})
            on_event("LLM_SENTENCE_READY", {"sentence": sentence})
        yield sentence