import ollama
from contextlib import nullcontext
from io import BytesIO
from pathlib import Path
import re
import time

from PIL import Image

from memory import Memory
from config import SUPPORTED_LANGUAGES, DEFAULT_LANGUAGE, LLM_MODEL, LLM_MAX_TOKENS


def _is_telugu_character(character):
    return "\u0C00" <= character <= "\u0C7F"


def _filter_telugu_token(token, parenthetical_depth):
    """Remove non-Telugu translations while keeping streamed Telugu readable."""
    filtered = []
    for character in token:
        if character in "([{":
            parenthetical_depth += 1
            continue
        if character in ")]}":
            parenthetical_depth = max(0, parenthetical_depth - 1)
            continue
        if parenthetical_depth:
            continue
        if (
            _is_telugu_character(character)
            or character.isspace()
            or character.isdigit()
            or character in ".,!?;:।…-–—"
            or ord(character) >= 0x1F000
        ):
            filtered.append(character)

    result = "".join(filtered)
    result = re.sub(r"\s*[-–—]\s*", " ", result)
    result = re.sub(r"\s+([.,!?;:।])", r"\1", result)
    result = re.sub(r"([.,!?;:।])\s*(?=[.,!?;:।])", r"\1", result)
    return result, parenthetical_depth


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
            elif lang == "te":
                assistant = "క్షమించండి, ప్రస్తుత మోడల్ కెమెరా లేదా చిత్రాలను చూడలేను."
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
                    "Do NOT respond in Arabic, Urdu, Persian, or any other script under any circumstances. "
                    "Even if the user's name sounds Arabic or Persian, respond exclusively in Hindi Devanagari. "
                    "Never respond in Hinglish unless the user explicitly requests Hinglish. "
                    "Never translate unless the user explicitly asks for translation."
                )
        elif lang == "te":
            system_instruction = (
                "You are Tarz, a multilingual AI assistant.\n\n"
                "The detected language is Telugu.\n\n"
                "Rules:\n"
                "- Reply only in Telugu.\n"
                "- Use only Telugu Unicode script. Do not include English, Latin transliteration, "
                "translations, or parenthetical explanations.\n"
                "- Understand common Roman Telugu vocabulary, including niku, neeku, telusa, "
                "atanu, and chanipoyadu; reply in Telugu script.\n"
                "- If the user specifies a length (for example, 50 words or 100 words), follow it closely.\n"
                "- Write natural, grammatically correct Telugu.\n"
                "- You can communicate in Telugu. Never claim that you cannot speak or understand Telugu.\n"
                "- Fulfill ordinary harmless requests, including fictional stories and jokes. "
                "Do not say that you cannot tell a story.\n"
                "- Be engaging and conversational."
            )
        elif lang == "ml":
            system_instruction = (
                "You are Tarz, a multilingual AI assistant.\n\n"
                "The detected language is Malayalam.\n\n"
                "Rules:\n"
                "- Reply ONLY in Malayalam using Malayalam Unicode script (കേരളം, not Kerala).\n"
                "- Do not include English, Latin transliteration, Arabic, Hindi, or any other script.\n"
                "- The user may write in Roman Malayalam (Manglish); always reply in proper Malayalam script.\n"
                "- Write natural, grammatically correct Malayalam.\n"
                "- You can communicate in Malayalam. Never claim otherwise.\n"
                "- Fulfill ordinary harmless requests including stories and jokes.\n"
                "- Be engaging and conversational."
            )
        elif lang == "ar":
            system_instruction = (
                "You are Tarz, a multilingual AI assistant.\n\n"
                "The detected language is Arabic.\n\n"
                "Rules:\n"
                "- Reply ONLY in Arabic using Arabic Unicode script.\n"
                "- Do not include English, Latin transliteration, or any other script.\n"
                "- The user may write in Arabizi (Roman Arabic); always reply in proper Arabic script.\n"
                "- Write natural, grammatically correct Modern Standard Arabic or the detected dialect.\n"
                "- You can communicate in Arabic. Never claim otherwise.\n"
                "- Fulfill ordinary harmless requests including stories and jokes.\n"
                "- Be engaging and conversational."
            )
        else:
            system_instruction = (
                f"Current conversation language is {language_name}. "
                f"Always answer ONLY in {language_name}. "
                "Never translate unless the user explicitly asks for translation."
            )

        # Keep simple answers concise, but let creative requests have enough
        # room to feel complete rather than ending after an acknowledgement.
        system_instruction += (
            " Keep simple factual answers short, clear, and complete. "
            "When the user requests a story, begin the story immediately instead of only confirming that you can tell one. "
            "Write a complete short story with a beginning, development, and ending in several short paragraphs. "
            "For numbered plans, use consecutive numbers exactly once and put "
            "each item on its own line. For an N-day itinerary, provide Day 1 "
            "through Day N without skipping or repeating days. Do not output a "
            "number by itself, and do not invent uncertain place names. "
        )

        messages = [{
            "role": "system",
            "content": system_instruction,
        }]
        history = self.memory.messages().copy()
        current_message = history.pop()
        messages.extend(history)
        # Keep the active turn's language closest to the current request so
        # previous Hindi or Telugu replies cannot pull a new English turn into
        # the wrong output language.
        messages.append({
            "role": "system",
            "content": f"FINAL LANGUAGE LOCK: Respond only in {language_name} for this user message.",
        })
        messages.append(current_message)

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
        telugu_parenthetical_depth = 0

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

                raw_token = chunk["message"]["content"]

                if "prompt_eval_count" in chunk:
                    usage_details["input_tokens"] = chunk["prompt_eval_count"]
                if "eval_count" in chunk:
                    usage_details["output_tokens"] = chunk["eval_count"]

                token = raw_token
                if lang == "te":
                    token, telugu_parenthetical_depth = _filter_telugu_token(
                        raw_token,
                        telugu_parenthetical_depth,
                    )
                    if not token:
                        continue
                    if (
                        not any(_is_telugu_character(character) or character.isdigit() for character in token)
                        and not any(_is_telugu_character(character) for character in assistant)
                    ):
                        continue

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

                yield token

            if lang == "te" and not any(_is_telugu_character(character) for character in assistant):
                assistant = "క్షమించండి, దయచేసి మీ ప్రశ్నను మళ్లీ అడగండి."
                if first_token_ms is None:
                    first_token_ms = round((time.perf_counter() - request_start) * 1000, 2)
                if on_event is not None:
                    on_event("LLM_FIRST_TOKEN", {"token": assistant})
                    on_event("LLM_TOKEN", {"token": assistant, "text": assistant})
                    on_event("LLM_STREAMING", {"token": assistant})
                yield assistant

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
    first_sentence_immediately=False,
    should_stop=None,
):

    buffer = ""
    pending = ""
    has_emitted = False
    list_marker = ""

    endings = [".", "!", "?", "।"]

    def push_piece(piece):
        nonlocal pending, has_emitted, list_marker
        piece = piece.strip()
        if not piece:
            return None

        # Models sometimes stream a list number (for example, "1.") before
        # its text. Keep it until the following sentence so TTS never speaks a
        # bare number as a separate response.
        if re.fullmatch(r"\d+[.)]", piece):
            list_marker = piece
            return None
        if list_marker:
            piece = f"{list_marker} {piece}"
            list_marker = ""

        if pending:
            pending = f"{pending} {piece}".strip()
        else:
            pending = piece

        chars = len(pending)
        words = len(pending.split())

        if first_sentence_immediately and not has_emitted:
            out = pending
            pending = ""
            has_emitted = True
            return out

        if chars >= min_chars and words >= min_words:
            out = pending
            pending = ""
            has_emitted = True
            return out

        if chars >= max_chars or words >= max_words:
            out = pending
            pending = ""
            has_emitted = True
            return out

        return None

    for token in token_stream:

        if should_stop is not None and should_stop():
            return

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

    if list_marker:
        pending = f"{pending} {list_marker}".strip()

    if pending.strip():
        sentence = pending.strip()
        if on_event is not None:
            on_event("SENTENCE_READY", {"sentence": sentence})
            on_event("LLM_SENTENCE_READY", {"sentence": sentence})
        yield sentence