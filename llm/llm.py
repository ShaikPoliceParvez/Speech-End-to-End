import ollama
from contextlib import nullcontext
from io import BytesIO
from pathlib import Path
import re
import time

from PIL import Image

from core.memory import Memory
from config import (
    SUPPORTED_LANGUAGES,
    DEFAULT_LANGUAGE,
    LLM_MODEL,
    LLM_MAX_TOKENS,
    LLM_SOCIAL_MAX_TOKENS,
    LLM_HISTORY_MODE,
    LLM_HISTORY_TURNS,
)


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


def _is_hindi_character(character):
    return "\u0900" <= character <= "\u097F"


def _is_devanagari_character(character):
    return "\u0900" <= character <= "\u097F"


def _filter_hindi_token(token, parenthetical_depth):
    """Remove non-Hindi additions while keeping streamed Hindi readable."""
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
            _is_hindi_character(character)
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

    def warmup(self):
        """Issue a tiny request so the first user turn starts faster."""
        try:
            ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": "hi"}],
                stream=False,
                options={"num_predict": 1},
            )
        except Exception:
            # Warmup is best-effort and should never block startup.
            pass

    @staticmethod
    def _history_explicitly_requested(prompt):
        text = (prompt or "").strip().lower()
        if not text:
            return False

        if "continue the previous" in text or "user follow-up:" in text:
            return True

        memory_keywords = {
            "remember", "recall", "previous", "earlier", "as i said", "as mentioned",
            "before", "last message", "our last", "continue from",
            "याद", "पहले", "जैसा मैंने कहा",
            "గత", "ముందు", "గుర్తు", "ముందు చెప్పిన",
            "മുമ്പ്", "ഓർമ്മ", "മുന്‍പ്",
            "تذكر", "السابق", "قبل", "كما قلت",
        }
        return any(keyword in text for keyword in memory_keywords)

    @staticmethod
    def _is_social_or_compliment_turn(prompt):
        text = (prompt or "").strip().lower()
        if not text:
            return False

        if text.startswith("this is a greeting/small-talk turn."):
            return True

        # Do not classify explicit task requests as social turns.
        task_markers = {"story", "poem", "plan", "trip", "flight", "code", "math", "translate"}
        if any(marker in text for marker in task_markers):
            return False

        keywords = {
            "hi", "hello", "hey", "how are you", "what about you", "i am good", "i'm good", "im good",
            "ela unnavu", "ela unnaru", "meeru ela unnaru", "nuvvu ela unnavu",
            "nenu bagunnanu", "nenu kuda bagunnanu",
            "wow", "great", "awesome", "nice", "cool", "thanks", "thank you", "good job",
            "bahut badiya", "bahut badhiya", "bahut badia", "बहुत बढ़िया", "बहुत बढिया",
            "chaala bagundi", "chala bagundi", "చాలా బాగుంది", "బాగుంది",
            "valare nannayi", "വളരെ നല്ലത്",
            "namaste", "नमस्ते", "धन्यवाद", "sanchai", "sanchai cha", "ma thik chu",
            "mumtaz", "ممتاز", "رائع", "شكرا",
        }
        return any(keyword in text for keyword in keywords)

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

        social_turn = self._is_social_or_compliment_turn(prompt)

        lang = (language or self.memory.get_language() or DEFAULT_LANGUAGE).lower()
        if lang not in SUPPORTED_LANGUAGES:
            lang = DEFAULT_LANGUAGE

        self.memory.set_language(lang)

        self.memory.add_user(prompt)

        if vision_requested and not self.supports_vision():
            if lang == "hi":
                assistant = "माफ़ कीजिए, वर्तमान मॉडल कैमरा या इमेज नहीं देख सकता।"
            elif lang == "ne":
                assistant = "माफ गर्नुहोस्, हालको मोडेलले क्यामेरा वा तस्बिर हेर्न सक्दैन।"
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
                    "The user's conversation language is Hindi.\n"
                    "The user may type in Roman Hindi, but input is normalized before you receive it.\n\n"
                    "STRICT OUTPUT RULES — no exceptions:\n"
                    "1. Always respond ONLY in standard Hindi (Khari Boli / Modern Standard Hindi).\n"
                    "2. Use Devanagari script. Every word must be Hindi, not Marathi, Bhojpuri, or any other Devanagari language.\n"
                    "3. Do NOT respond in Marathi under any circumstances. "
                    "Marathi words like तुमच्या, केलेल्या, आहे, होतो are NOT Hindi — avoid them completely.\n"
                    "4. Do NOT respond in Arabic, Urdu, Persian, or any other script.\n"
                    "5. Do NOT respond in Hinglish unless the user explicitly requests it.\n"
                    "6. Never translate unless the user explicitly asks for translation.\n"
                    "7. Write natural conversational Hindi as spoken in Delhi / North India.\n"
                    "8. If user input is mixed, unclear, or Marathi-influenced Devanagari, interpret the meaning and answer in clean Hindi.\n"
                    "9. Never mirror Marathi grammar endings or function words such as मध्ये, आणि, आहे, होते, पुढे, कडे, वरून.\n"
                    "10. Keep sentence structure simple and idiomatic Hindi; prefer phrases like 'में', 'और', 'है', 'था/थे', 'अगला'."
                )
        elif lang == "ne":
            system_instruction = (
                "You are Tarz, a multilingual AI assistant. The user is speaking Nepali.\n\n"
                "STRICT OUTPUT RULES — no exceptions:\n"
                "1. Always answer in standard Nepali language.\n"
                "2. Use only Devanagari script (Unicode U+0900–U+097F).\n"
                "3. Do not use Hindi, English, or Roman transliteration unless the user explicitly asks for mixed output.\n"
                "4. Write natural, grammatically correct Nepali as spoken by native speakers.\n"
                "5. The user may type Roman Nepali; still reply in Devanagari Nepali.\n"
                "6. Never translate unless the user explicitly asks for translation.\n"
                "7. Fulfill ordinary harmless requests like stories, jokes, explanations, and planning in Nepali.\n"
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
                "- For greetings and small talk, use natural Telugu conversational grammar.\n"
                "  Example style: 'నేను బాగున్నాను, మీరు ఎలా ఉన్నారు?'\n"
                "- Keep 'బాగున్నాను' as one word; do not split it as 'బాగ ఉన్నాను'.\n"
                "- Do not use wellbeing lines like 'నేను బాగున్నాను' unless the user actually asked a wellbeing question.\n"
                "- If the user requests a story, start the story content immediately; do not ask 'ఒక కథ చెప్పనా' or add unrelated small-talk lines first.\n"
                "- Do not repeat the same meaning in two adjacent clauses or sentences.\n"
                "- Avoid malformed or literal constructions such as 'నీకు ఎలా ఉన్నాలో?'.\n"
                "- You can communicate in Telugu. Never claim that you cannot speak or understand Telugu.\n"
                "- Fulfill ordinary harmless requests, including fictional stories and jokes. "
                "Do not say that you cannot tell a story.\n"
                "- Be engaging and conversational."
            )
        elif lang == "ml":
            system_instruction = (
                "You are Tarz, a multilingual AI assistant. The user is speaking Malayalam.\n\n"
                "STRICT OUTPUT RULES — no exceptions:\n"
                "1. Always answer in native Malayalam script (Unicode U+0D00–U+0D7F). "
                "Every word of your reply must be written in Malayalam script.\n"
                "2. Do not use English words or Roman transliterations unless they are proper nouns "
                "(names of people, places, brands). Even then, write them in Malayalam phonetic script.\n"
                "3. Do NOT use Tamil script (U+0B80–U+0BFF). Malayalam and Tamil look similar — "
                "verify you are using Malayalam characters: ക, ന, ഞ, ൻ, ്, ാ.\n"
                "4. Write natural Malayalam exactly as a native speaker would — "
                "correct grammar, natural flow, conversational tone.\n"
                "5. The user may type in Manglish (Roman Malayalam) like 'njan', 'ente', 'vanakkam'; "
                "always reply in Malayalam script regardless.\n"
                "6. You can speak Malayalam fluently. Never claim you cannot.\n"
                "7. Fulfill all ordinary requests — stories, jokes, explanations — in Malayalam."
            )
        elif lang == "ar":
            system_instruction = (
                "You are Tarz, a multilingual AI assistant. The user is speaking Arabic.\n\n"
                "STRICT OUTPUT RULES — no exceptions:\n"
                "1. Always answer in native Arabic script (Unicode U+0600–U+06FF). "
                "Every word of your reply must be written in Arabic script.\n"
                "2. Do not use English words or Roman transliterations unless they are proper nouns. "
                "Even then, write them in Arabic phonetic script.\n"
                "3. Write natural Arabic exactly as a native speaker would — "
                "correct grammar, natural flow, conversational tone.\n"
                "4. The user may type in Arabizi (Roman Arabic) like 'marhaban', 'ana', 'habibi'; "
                "always reply in Arabic script regardless.\n"
                "5. You can speak Arabic fluently. Never claim you cannot.\n"
                "6. Fulfill all ordinary requests — stories, jokes, explanations — in Arabic."
            )
        else:
            system_instruction = (
                f"Current conversation language is {language_name}. "
                f"Always answer ONLY in {language_name}. "
                "Never translate unless the user explicitly asks for translation."
            )

        # The TTS bridge/preface is a latency-hiding spoken transition.
        # The model must ignore it semantically.
        system_instruction += (
            "A short conversational bridge may already be spoken before your generated response is heard. "
            "That bridge exists only to hide latency and is not the actual answer. "
            "The bridge is not the user message, not your response, not conversation history, not an instruction, and not additional context. "
            "It must never influence your reasoning or intent understanding. "
            "Ignore the bridge semantically: do not repeat it, paraphrase it, react to it, or answer it. "
            "Do not acknowledge, continue, or generate another introductory sentence. "
            "Do not infer intent from the bridge. Infer intent only from the current user message and conversation history. "
            "Always respond to the MOST RECENT user message first. "
            "Treat the conversation as continuous. Do not restart the conversation unless the user clearly starts a completely new topic. "
            "The latest user intent has highest priority. "
            "If previous conversational patterns conflict with the latest request, follow the latest request. "
            "For direct actionable requests (for example: tell a story, translate this, solve this), start fulfilling the request immediately instead of delaying with unnecessary follow-up questions. "
            "Never reuse a previous-turn response as the current reply. "
            "Never answer an old question again when the user has already moved forward. "
            "Always treat the latest user message as part of the ongoing conversation, not as a fresh start. "
            "If the user is answering your previous question, acknowledge that answer naturally and move the conversation forward. "
            "Do not repeat your previous question after the user has already answered it. "
            "Do not mirror the user's answer as if it were your own state. "
            "Do not repeat the same meaning twice in the same reply. "
            "Do not add another bridge-like opener such as 'sure', 'certainly', 'okay', or 'let me help' unless essential. "
            "Start with substantive continuation as early as possible. "
            "For factual or task requests, continue directly with the answer, steps, story content, translation, or solution instead of re-announcing the action. "
            "For greetings, compliments, or acknowledgements, continue naturally in a concise conversational way without repeating what was already acknowledged. "
            "The user should feel an immediate and seamless response, while the bridge itself remains non-semantic filler. "
        )

        if social_turn:
            # Keep social-turn instructions compact to reduce TTFT for greetings.
            system_instruction += (
                "You are Tarz. Never claim your name is anything else. "
                "Reply naturally in the selected language as one short flowing conversational sentence when possible. "
                "When combining an acknowledgement with a follow-up question, keep it in one sentence using natural connector punctuation (usually a comma) instead of splitting into two sentences with a period. "
                "Do not repeat acknowledgement content more than once. "
                "If the user message is a greeting, avoid repeating another greeting and continue directly with a helpful next-question or continuation. "
                "Do not continue any previous story or task context unless explicitly asked. "
                "Keep it under about 35 words and avoid literal translation artifacts. "
            )
        else:
            # Keep simple answers concise, but let creative requests have enough
            # room to feel complete rather than ending after an acknowledgement.
            system_instruction += (
                "You are Tarz. Never claim your name is anything else. "
                " Keep simple factual answers short, clear, and complete. "
                "Use grammatically correct, natural native phrasing for the selected language in every reply. "
                "For greetings and small-talk (for example: hi, hello, how are you), respond in 1-2 short natural conversational sentences. "
                "Do not use literal translated grammar or broken forms. "
                "Examples of desired tone: Telugu -> 'నేను బాగున్నాను. మీరు ఎలా ఉన్నారు?'; Hindi -> 'मैं ठीक हूँ। आप कैसे हैं?'. "
                "For the ENTIRE response, keep every sentence fluent and native in grammar, word order, and idiom. "
                "Do not mix languages unless the user explicitly asks for mixed output. "
                "Do not mirror user typos or malformed grammar; correct them and answer naturally. "
                "Before finishing, self-check that the whole reply reads like a native speaker wrote it. "
                "For practical tasks (planning, recommendations, translation, search-style help), provide concrete and useful details, not just a short acknowledgement. "
                "If the request is underspecified, ask exactly one concise clarifying question. "
                "Use the conversation history silently to maintain context and continuity — for example, remembering a destination, topic, or preference the user already stated. "
                "Do NOT proactively mention, quote, or summarise previous responses unless the current message is a direct follow-up to them or the user explicitly asks about earlier content. "
                "If the current message is only a greeting, compliment, or acknowledgement, reply briefly and naturally, and do not continue prior stories or tasks. "
                "Always prioritize the user's current message over prior creative context. "
                "If the current user message is a follow-up (for example: also, add this, include flights, for the trip), continue the same task directly. "
                "Do not switch to stories, poems, or fictional content unless the CURRENT user message explicitly asks for a story or poem. "
                "When the user requests a story, begin the story immediately instead of only confirming that you can tell one. "
                "Do not insert unrelated wellbeing/small-talk lines (for example, 'I am fine, how are you') before starting the requested story. "
                "Write a complete short story with a beginning, development, and ending in several short paragraphs. "
                "For numbered plans, use consecutive numbers exactly once and put "
                "each item on its own line. For an N-day itinerary, provide Day 1 "
                "through Day N without skipping or repeating days. Do not output a "
                "number by itself, and do not invent uncertain place names. "
                "When user says hi,hello or greets, do not greet back and go directly to the next question or continuation."
            )

        messages = [{
            "role": "system",
            "content": system_instruction,
        }]
        history = self.memory.messages().copy()
        current_message = history.pop()
        include_history = LLM_HISTORY_MODE == "full"
        if LLM_HISTORY_MODE == "strict":
            include_history = self._history_explicitly_requested(prompt)

        if include_history:
            window = max(0, int(LLM_HISTORY_TURNS)) * 2
            if window > 0:
                messages.extend(history[-window:])

        if self._is_social_or_compliment_turn(prompt):
            messages.append({
                "role": "system",
                "content": (
                    "This user message is a greeting/compliment/acknowledgement. "
                    "Reply as one short natural sentence when possible. "
                    "Prefer comma-linked continuation before a follow-up question instead of a full-stop split when both parts are tightly related. "
                    "Do not continue previous story/task context unless user explicitly asks to continue."
                ),
            })
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
        hindi_parenthetical_depth = 0

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
                options={
                    "num_predict": min(LLM_SOCIAL_MAX_TOKENS, LLM_MAX_TOKENS) if social_turn else LLM_MAX_TOKENS,
                },
            )

            for chunk in response:

                raw_token = chunk["message"]["content"]

                if "prompt_eval_count" in chunk:
                    usage_details["input_tokens"] = chunk["prompt_eval_count"]
                if "eval_count" in chunk:
                    usage_details["output_tokens"] = chunk["eval_count"]

                token = raw_token
                if lang in {"hi", "ne"}:
                    token, hindi_parenthetical_depth = _filter_hindi_token(
                        raw_token,
                        hindi_parenthetical_depth,
                    )
                    if not token:
                        continue
                    if (
                        not any(_is_devanagari_character(character) or character.isdigit() for character in token)
                        and not any(_is_devanagari_character(character) for character in assistant)
                    ):
                        continue
                elif lang == "te":
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
            elif lang in {"hi", "ne"} and not any(_is_devanagari_character(character) for character in assistant):
                assistant = "क्षमा करें, कृपया अपना प्रश्न फिर से पूछें।" if lang == "hi" else "माफ गर्नुहोस्, कृपया आफ्नो प्रश्न फेरि सोध्नुहोस्।"
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
    first_chunk_min_chars=24,
    first_chunk_min_words=4,
    first_word_immediately=False,
    first_sentence_wordwise=False,
    first_sentence_word_chunk_size=1,
    chunk_on_minor_punctuation=False,
    lead_words_immediate=False,
    lead_words_count=2,
    should_stop=None,
    # Short clauses below these thresholds are held in pending instead of
    # being spoken alone, so adjacent short sentences play as one breath.
    min_force_chars=20,
    min_force_words=4,
):

    buffer = ""
    pending = ""
    has_emitted = False
    list_marker = ""
    first_sentence_done = False
    first_sentence_word_buffer = []
    lead_words_sent = False

    major_endings = [".", "!", "?", "।"]
    minor_endings = [",", ";", ":"] if chunk_on_minor_punctuation else []
    endings = major_endings + minor_endings

    def emit_chunk(chunk):
        nonlocal has_emitted
        chunk = chunk.strip()
        if not chunk:
            return None
        has_emitted = True
        if on_event is not None:
            on_event("SENTENCE_READY", {"sentence": chunk})
            on_event("LLM_SENTENCE_READY", {"sentence": chunk})
        return chunk

    def emit_word_groups(words, flush=False):
        nonlocal first_sentence_word_buffer
        chunk_size = max(1, int(first_sentence_word_chunk_size or 1))
        for word in words:
            first_sentence_word_buffer.append(word)
            if len(first_sentence_word_buffer) >= chunk_size:
                out = emit_chunk(" ".join(first_sentence_word_buffer))
                first_sentence_word_buffer = []
                if out:
                    yield out
        if flush and first_sentence_word_buffer:
            out = emit_chunk(" ".join(first_sentence_word_buffer))
            first_sentence_word_buffer = []
            if out:
                yield out
    def push_piece(piece, force_emit=False):
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

        # Punctuation boundaries should flush immediately for natural pacing.
        if force_emit:
            # Hold short clauses so they join the next sentence and play
            # as one unbroken breath (e.g. "मैं ठीक हूँ। आपका दिन कैसा है?").
            if len(pending) < min_force_chars and len(pending.split()) < min_force_words:
                return None
            out = pending
            pending = ""
            has_emitted = True
            return out

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

        # Fast startup: emit a tiny initial phrase (e.g., first 2 words) as
        # soon as it is stable, then continue with punctuation-based chunks.
        if lead_words_immediate and not lead_words_sent and not has_emitted:
            count = max(1, int(lead_words_count or 1))
            words = buffer.strip().split()
            if len(words) >= count:
                lead = " ".join(words[:count]).strip()
                remainder = " ".join(words[count:]).strip()
                out = emit_chunk(lead)
                if out:
                    yield out
                lead_words_sent = True
                buffer = remainder

        # Ultra-low-latency start: emit only the first sentence word-by-word.
        # After the first sentence ends, switch back to normal sentence chunks.
        if first_sentence_wordwise and not first_sentence_done:
            while True:
                end_index = -1
                for end_char in major_endings:
                    pos = buffer.find(end_char)
                    if pos != -1 and (end_index == -1 or pos < end_index):
                        end_index = pos

                if end_index != -1:
                    sentence_slice = buffer[:end_index + 1].strip()
                    buffer = buffer[end_index + 1:].lstrip()
                    for out in emit_word_groups(sentence_slice.split(), flush=True):
                        yield out
                    first_sentence_done = True
                    break

                match = re.match(r"\s*([^\s]+)\s+(.*)", buffer, flags=re.DOTALL)
                if not match:
                    break

                word = match.group(1)
                buffer = match.group(2)
                for out in emit_word_groups([word]):
                    yield out

            if not first_sentence_done:
                continue

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
                out = push_piece(sentence, force_emit=True)
                if out:
                    if on_event is not None:
                        on_event("SENTENCE_READY", {"sentence": out})
                        on_event("LLM_SENTENCE_READY", {"sentence": out})
                    yield out

        cleaned = buffer.strip()
        if cleaned:
            words = len(cleaned.split())

            # For low-latency voice replies, do not wait for punctuation before
            # emitting the very first chunk when enough text has arrived.
            if (
                first_sentence_immediately
                and not has_emitted
                and (len(cleaned) >= first_chunk_min_chars or words >= first_chunk_min_words)
                and buffer[-1].isspace()
            ):
                sentence = cleaned
                buffer = ""
                out = push_piece(sentence)
                if out:
                    if on_event is not None:
                        on_event("SENTENCE_READY", {"sentence": out})
                        on_event("LLM_SENTENCE_READY", {"sentence": out})
                    yield out
                continue

            # Lowest-latency mode: begin playback after the first stable word.
            if first_word_immediately and not has_emitted:
                parts = cleaned.split()
                stable_boundary = (" " in cleaned) or any(end_char in cleaned for end_char in endings)
                if len(parts) >= 1 and stable_boundary:
                    sentence = parts[0]
                    remainder = cleaned[len(sentence):].lstrip()
                    buffer = remainder
                    out = push_piece(sentence)
                    if out:
                        if on_event is not None:
                            on_event("SENTENCE_READY", {"sentence": out})
                            on_event("LLM_SENTENCE_READY", {"sentence": out})
                        yield out
                    continue

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