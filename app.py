import config
import msvcrt
import threading
import queue
import random
import re
from microphone import Microphone
from stt import STT
from router import Router
from llm import LLM, sentence_stream
from tts_router import TTSRouter
from camera import Camera
from language import detect_dominant_language, normalize_text, detect_script
from tracing import LangfuseTracer


class Tarz:

    FOLLOWUP_TOKENS = {
        "also", "add", "include", "more", "continue", "next",
        "too", "and", "plus", "then", "after", "another",
        "details", "options", "examples", "price", "cost", "budget",
    }

    TASK_KEYWORDS = {
        "story": {"story", "katha", "kahani", "कहानी", "कथा", "కథ", "حكاية", "قصة"},
        "poem": {"poem", "poetry", "shayari", "कविता", "शायरी", "కవిత", "قصيدة"},
        "travel": {
            "trip", "travel", "itinerary", "flight", "flights", "ticket", "tickets",
            "airfare", "hotel", "stay", "bombay", "mumbai", "tour",
        },
        "weather": {"weather", "forecast", "temperature", "मौसम", "వాతావరణం", "الطقس"},
        "news": {"news", "headlines", "समाचार", "వార్తలు", "أخبار"},
        "coding": {"code", "coding", "program", "python", "bug", "debug", "fix", "script"},
        "math": {"math", "calculate", "equation", "sum", "multiply", "divide", "गणना", "లెక్క"},
        "camera": {"camera", "photo", "image", "picture", "ocr", "read this"},
        "translation": {"translate", "translation", "अनुवाद", "అనువాదం", "ترجمة"},
    }

    SOCIAL_TOKENS = {
        "hi", "hello", "hey", "namaste", "नमस्ते", "नमस्कार",
        "hii", "heyy", "yo", "ok", "okay", "thanks", "thank", "wow", "great",
        "thik", "theek", "ठीक",
        "నమస్కారం", "హాయ్", "ధన్యవాదాలు", "బాగుంది",
        "നമസ്കാരം", "ഹലോ", "നന്ദി",
        "مرحبا", "أهلا", "شكرا",
    }

    def __init__(self):

        print("Starting Tarz...\n")

        self.tracing = LangfuseTracer()
        self.mic = Microphone()
        self.stt = STT()
        self.router = Router()
        self.camera = Camera()
        self.llm = LLM(model=config.LLM_MODEL, tracer=self.tracing)
        if config.LLM_WARMUP_ON_STARTUP:
            self.llm.warmup()
        self.tts = TTSRouter(on_event=self._on_event)
        self.current_task = None
        self.tracing.set_model_startup_metrics(
            stt=self.stt.model_startup_metrics,
            llm=self.llm.measure_model_startup(),
            tts=self.tts.model_startup_metrics,
        )

    def _on_event(self, name, data):
        if config.DEBUG:
            print(f"[EVENT] {name}: {data}")

    @staticmethod
    def _return_to_menu_requested(text):
        command = text.strip().lower().strip(".,!?;:。")
        return command in {
            "/menu", "0", "menu", "back", "go back", "back to menu",
            "menu par jao", "menu par wapas jao", "मेनू पर जाओ", "मेनू पर वापस जाओ",
            "మెనూకి వెళ్ళు", "మెనూకి వెళ్లు", "మెనూకు వెళ్ళు",
        }

    def _watch_for_barge_in(self, stop_event):
        # Ignore Enter left in the console input buffer before speech begins.
        while msvcrt.kbhit():
            msvcrt.getwch()

        while not stop_event.wait(0.05):
            if not msvcrt.kbhit():
                continue

            key = msvcrt.getwch()
            if self.tts.is_speaking() and key in ("\r", "\n"):
                self.tts.stop()
                print("\nTarz: (stopped - listening for your next question)")
                return

    @staticmethod
    def _build_context_preface(prompt, language, current_task=None):
        text = (prompt or "").strip().lower()
        if not text:
            return None

        tokens = set(re.findall(r"[a-zA-Z]+|[\u0900-\u097f]+|[\u0c00-\u0c7f]+|[\u0d00-\u0d7f]+|[\u0600-\u06ff]+", text))

        def has_any(words):
            for word in words:
                w = word.lower()
                if " " in w:
                    if w in text:
                        return True
                elif w in tokens:
                    return True
            return False

        category_keywords = {
            "greeting": {
                "hello", "hi", "hey", "heymate", "mate", "wassup", "wassupp", "whatsup", "sup",
                "namaste", "नमस्ते", "నమస్కారం", "നമസ്കാരം", "مرحبا",
            },
            "smalltalk": {
                "how are you", "what about you", "i am good", "i'm good", "im good", "fine",
                "कैसे हो", "मैं ठीक हूँ", "आप कैसे हैं", "నేను బాగున్నా", "మీరు ఎలా ఉన్నారు",
                "ഞാൻ സുഖമാണ്", "എങ്ങനെയാണ്", "كيف حالك", "انا بخير",
            },
            "appreciation": {
                "wow", "great", "awesome", "nice", "super", "cool", "excellent",
                "बहुत बढ़िया", "शानदार", "वाह",
                "చాలా బాగుంది", "బాగుంది", "సూపర్", "అద్భుతం",
                "വളരെ നല്ലത്", "അടിപൊളി", "സൂപർ",
                "رائع", "ممتاز", "مذهل",
            },
            "story": {"story", "katha", "kahani", "कहानी", "कथा", "కథ", "kadha", "حكاية", "قصة"},
            "poem": {"poem", "poetry", "shayari", "कविता", "शायरी", "పద్య", "కవిత", "قصيدة"},
            "weather": {"weather", "temperature", "forecast", "मौसम", "వాతావరణం", "കാലാവസ്ഥ", "الطقس"},
            "news": {"news", "headlines", "समाचार", "వార్తలు", "വാർത്ത", "أخبار"},
            "camera": {"camera", "photo", "image", "picture", "कैमरा", "కెమెరా", "ക്യാമറ", "كاميرا"},
            "translation": {"translate", "translation", "अनुवाद", "అనువాదం", "വിവർത്തനം", "ترجمة"},
            "math": {"calculate", "math", "equation", "sum", "गणना", "లెక్క", "കണക്കു", "حساب"},
            "coding": {"code", "coding", "program", "python", "bug", "कोड", "కోడ్", "കോഡ്", "كود"},
            "search": {
                "search", "find", "look up", "ढूंढ", "వెతుకు", "തിരയ", "ابحث",
                "flight", "flights", "airfare", "ticket", "tickets", "trip", "travel", "itinerary",
            },
            "thanks": {"thanks", "thank you", "धन्यवाद", "శుక్రియా", "ధన్యవాదాలు", "നന്ദി", "شكرا"},
            "goodbye": {"bye", "goodbye", "see you", "फिर मिलेंगे", "వెళ్తాను", "പോയി വരാം", "مع السلامة"},
            "apology": {"sorry", "apologize", "माफ", "క్షమించ", "ക്ഷമിക്ക", "آسف"},
            "confirmation": {"done", "completed", "ok done", "हो गया", "అయింది", "കഴിഞ്ഞു", "تم"},
            "clarification": {"clarify", "clear", "स्पष्ट", "స్పష్టం", "വ്യക്തമാക്ക", "توضيح"},
        }

        category = "generic"
        if Tarz._is_social_turn(text):
            if has_any({
                "hello", "hi", "hey", "heymate", "mate", "wassup", "wassupp", "whatsup", "sup",
                "namaste", "नमस्ते", "నమస్కారం", "നമസ്കാരം", "مرحبا", "أهلا",
            }):
                category = "greeting"
            elif has_any({"thanks", "thank you", "धन्यवाद", "ధన్యవాదాలు", "നന്ദി", "شكرا"}):
                category = "thanks"
            elif has_any({
                "wow", "great", "awesome", "nice", "super", "cool", "excellent",
                "बहुत बढ़िया", "शानदार", "वाह",
                "చాలా బాగుంది", "బాగుంది", "సూపర్", "అద్భుతం",
                "വളരെ നല്ലത്", "അടിപൊളി",
                "رائع", "ممتاز", "مذهل",
            }):
                category = "appreciation"
            else:
                category = "smalltalk"
        else:
            for key, words in category_keywords.items():
                if has_any(words):
                    category = key
                    break

        task_to_category = {
            "story": "story",
            "poem": "poem",
            "travel": "search",
            "weather": "weather",
            "news": "news",
            "coding": "coding",
            "math": "math",
            "camera": "camera",
            "translation": "translation",
        }
        if category == "generic" and current_task in task_to_category:
            category = task_to_category[current_task]

        # Very short task follow-ups should get a longer practical opener, but
        # smalltalk/appreciation should stay natural and concise.
        if category == "generic" and len(text.split()) <= 2 and current_task in {"travel", "coding", "math", "news", "weather", "camera", "translation"}:
            category = "answer"

        if category == "generic" and ("?" in text or has_any({"how", "what", "why", "when", "where", "can you", "help"})):
            category = "answer"

        tables = getattr(config, "LANGUAGE_PREFACES", {})
        table = tables.get(language) or tables.get("en", {})
        candidates = table.get(category) or table.get("generic") or table.get("fallback") or []
        if not candidates:
            return None
        multiword_candidates = [candidate for candidate in candidates if len(candidate.strip().split()) >= 2]
        source = multiword_candidates
        if not source:
            generic_multi = [candidate for candidate in table.get("generic", []) if len(candidate.strip().split()) >= 2]
            fallback_multi = [candidate for candidate in table.get("fallback", []) if len(candidate.strip().split()) >= 2]
            source = generic_multi or fallback_multi

        defaults = {
            "en": "Okay, I can help with that.",
            "hi": "ठीक है, मैं मदद करता हूँ।",
            "te": "సరే, నేను సహాయం చేస్తాను.",
            "ml": "ശരി, ഞാൻ സഹായിക്കാം.",
            "ar": "حسنًا، سأساعدك في ذلك.",
        }
        if not source:
            return defaults.get(language, defaults["en"])

        min_words = max(2, int(getattr(config, "TTS_PREFACE_MIN_WORDS", 2)))
        pacing = str(getattr(config, "TTS_PREFACE_PACING", "normal")).lower()
        if pacing == "slow":
            paced = [candidate for candidate in source if len(candidate.strip().split()) >= min_words]
            if paced:
                source = paced

        if config.TTS_CONTEXT_PREFACE_RANDOM and len(candidates) > 1:
            selected = random.choice(source)
        else:
            selected = source[0]

        return selected

    @staticmethod
    def _detect_task(normalized_text, intent):
        text = (normalized_text or "").strip().lower()
        if not text:
            return None

        if intent in ("VISION", "OCR"):
            return "camera"

        tokens = set(re.findall(r"[a-zA-Z]+|[\u0900-\u097f]+|[\u0c00-\u0c7f]+|[\u0d00-\u0d7f]+|[\u0600-\u06ff]+", text))

        for task, words in Tarz.TASK_KEYWORDS.items():
            for word in words:
                token = word.lower()
                if " " in token:
                    if token in text:
                        return task
                elif token in tokens:
                    return task

        return None

    @staticmethod
    def _is_terse_followup(text):
        lowered = (text or "").strip().lower()
        if not lowered:
            return False
        words = lowered.split()
        followup_signals = {"also", "add", "include", "more", "next", "continue", "too", "and"}
        return len(words) <= 4 or any(signal in lowered for signal in followup_signals)

    @staticmethod
    def _is_social_turn(text):
        lowered = (text or "").strip().lower()
        if not lowered:
            return False
        tokens = set(re.findall(r"[a-zA-Z]+|[\u0900-\u097f]+|[\u0c00-\u0c7f]+|[\u0d00-\u0d7f]+|[\u0600-\u06ff]+", lowered))
        if not tokens:
            return False
        if tokens.intersection(Tarz.SOCIAL_TOKENS):
            return True
        social_phrases = {
            "how are you", "what about you", "i am good", "i'm good", "im good",
            "me bhi", "mein bhi", "main bhi", "mai bhi", "me too", "same here",
            "me bhi thik", "mein bhi thik", "main bhi thik", "main bhi theek", "i am fine too",
            "में भी", "में भी ठीक", "मैं भी", "मैं भी ठीक", "मैं भी ठीक हूँ",
            "bahut badiya", "bahut badhiya", "bahut badia", "बहुत बढ़िया", "बहुत बढिया",
            "chaala bagundi", "chala bagundi", "చాలా బాగుంది",
            "valare nannayi", "വളരെ നല്ലത്",
            "mumtaz", "ممتاز", "رائع",
            "heymate", "wassup", "wassupp", "whatsup",
            "ok", "okay", "alright",
        }
        return any(phrase in lowered for phrase in social_phrases)

    def _build_effective_prompt(self, normalized_text, previous_task=None, detected_task=None, language=None):
        """Add a small task lock for terse follow-ups to avoid topic drift."""
        text = (normalized_text or "").strip()
        lowered = text.lower()
        if not lowered:
            return text

        if self._is_social_turn(text):
            language_name = config.SUPPORTED_LANGUAGES.get(language or "", "the selected language")
            return (
                "This is a greeting/small-talk turn. "
                f"Reply naturally in {language_name} in 1-2 short conversational sentences. "
                "Avoid literal translation artifacts. "
                f"User message: {text}"
            )

        # Any explicit current intent should override prior task lock.
        if detected_task is not None:
            return text

        has_followup_signal = any(token in lowered for token in self.FOLLOWUP_TOKENS)

        if previous_task and detected_task is None and (self._is_terse_followup(text) or has_followup_signal):
            return (
                f"Continue the previous {previous_task} task. "
                f"User follow-up: {text}. "
                "Keep the same context and intent. "
                "Do not switch to stories or poems unless explicitly requested now."
            )

        return text

    def process(
        self,
        text,
        stt_language_hint=None,
        stt_language_confidence=None,
        turn=None,
        input_mode=None,
        request_start=None,
        pipeline_metrics=None,
    ):

        if not text.strip():
            return

        input_mode = input_mode or ("voice" if stt_language_hint else "text")
        request_start = request_start or self.tracing.now()
        owns_turn = turn is None
        if owns_turn:
            with self.tracing.turn_attributes():
                with self.tracing.start_turn(text, None, "CHAT", input_mode) as root_turn:
                    self._process(
                        text,
                        stt_language_hint,
                        stt_language_confidence,
                        root_turn,
                        input_mode,
                        request_start,
                        pipeline_metrics,
                    )
        else:
            self._process(
                text,
                stt_language_hint,
                stt_language_confidence,
                turn,
                input_mode,
                request_start,
                pipeline_metrics,
            )

        if owns_turn:
            self.tracing.flush()

    def _process(self, text, stt_language_hint, stt_language_confidence, turn, input_mode, request_start, pipeline_metrics):

        print(f"\nYou: {text}")

        # ---- Language + script detection, Hinglish normalization ----
        with self.tracing.start_step(
            "Language",
            input={
                "message": text,
                "stt_language_hint": stt_language_hint,
                "stt_language_confidence": stt_language_confidence,
            },
        ) as classification:
            script = detect_script(text)
            language = detect_dominant_language(
                text,
                stt_hint=stt_language_hint,
                stt_confidence=stt_language_confidence,
                previous_language=self.llm.memory.get_language(),
            )
            normalized_text = normalize_text(text, language)
            self.llm.memory.set_input_script(script)
            self.tracing.update_step(
                classification,
                output={
                    "language": language,
                    "script": script,
                    "normalized_message": normalized_text,
                },
            )

        if normalized_text != text:
            print(f"Normalized: {normalized_text}")

        language_name = config.SUPPORTED_LANGUAGES.get(language, language)
        print(f"Language: {language_name} | Script: {script}")

        # ---- Intent routing ----
        router_start = self.tracing.now()
        with self.tracing.start_step(
            "Router",
            input={"message": normalized_text},
        ) as routing:
            route = self.router.route(normalized_text)
            intent = route["intent"]
            self.tracing.update_step(
                routing,
                output={"selected_tool": intent, "confidence": route["confidence"]},
                metadata={
                    "reason": route["reason"],
                    "latency_ms": self.tracing.elapsed_ms(router_start),
                },
            )

        previous_task = self.current_task
        detected_task = self._detect_task(normalized_text, intent)
        if detected_task is not None:
            self.current_task = detected_task
        elif self._is_social_turn(normalized_text):
            self.current_task = None
        elif self._is_terse_followup(normalized_text):
            # Keep prior task lock for terse follow-ups like "flights".
            pass
        else:
            self.current_task = None

        self.tracing.update_turn(turn, normalized_text, language, intent)

        with self.tracing.start_step(
            "Memory",
            metadata={"type": "conversation-history", "rag_enabled": False},
        ) as memory:
            history = self.llm.memory.messages()
            self.tracing.update_step(
                memory,
                output={"message_count": len(history), "retrieved_chunks": 0},
            )

        image = None
        vision_requested = intent in ("VISION", "OCR")
        if vision_requested and self.llm.supports_vision():
            with self.tracing.start_step(
                "Camera",
                metadata={"image_included_in_trace": False},
                as_type="tool",
            ) as capture:
                print("Opening camera...")
                image, image_size = self.camera.capture()
                self.tracing.update_step(
                    capture,
                    output={"captured": True, "image_size": image_size},
                )
        elif vision_requested:
            print("Camera analysis is unavailable with the configured text-only model.")

        # ---- LLM streaming -> sentence buffering -> TTS streaming ----
        stream_state = {"printed": False}
        context_preface = None
        if config.TTS_CONTEXT_PREFACE_ENABLED:
            context_preface = self._build_context_preface(normalized_text, language, self.current_task)
        effective_prompt = self._build_effective_prompt(
            normalized_text,
            previous_task=previous_task,
            detected_task=detected_task,
            language=language,
        )
        # Keep lead-word streaming enabled even with a spoken preface so
        # generation can overlap and avoid a noticeable gap after filler.
        use_lead_words = config.TTS_LEAD_WORDS_IMMEDIATE

        token_queue = queue.Queue()
        stream_done = object()

        def produce_tokens():
            try:
                stream = self.llm.stream(
                    prompt=effective_prompt,
                    image=image,
                    vision_requested=vision_requested,
                    language=language,
                )
                for token in stream:
                    if self.tts.is_interrupted():
                        break
                    token_queue.put(token)
            finally:
                token_queue.put(stream_done)

        producer_thread = threading.Thread(target=produce_tokens, daemon=True)
        producer_thread.start()

        self.tts.start_turn()
        self.tts.set_language(language)
        barge_in_stop = threading.Event()
        barge_in_listener = None
        print("Press ENTER to interrupt Tarz and ask your next question.")
        barge_in_listener = threading.Thread(
            target=self._watch_for_barge_in,
            args=(barge_in_stop,),
            daemon=True,
        )
        barge_in_listener.start()

        def queued_tokens():
            while True:
                token = token_queue.get()
                if token is stream_done:
                    return
                if not stream_state["printed"]:
                    print("Tarz: ", end="", flush=True)
                    stream_state["printed"] = True
                print(token, end="", flush=True)
                yield token

        sentences = sentence_stream(
            queued_tokens(),
            min_chars=config.TTS_MIN_CHARS,
            min_words=config.TTS_MIN_WORDS,
            max_chars=config.TTS_MAX_CHARS,
            max_words=config.TTS_MAX_WORDS,
            first_sentence_immediately=config.TTS_FIRST_SENTENCE_IMMEDIATELY,
            first_chunk_min_chars=config.TTS_FIRST_CHUNK_MIN_CHARS,
            first_chunk_min_words=config.TTS_FIRST_CHUNK_MIN_WORDS,
            first_word_immediately=config.TTS_FIRST_WORD_IMMEDIATELY,
            first_sentence_wordwise=config.TTS_FIRST_SENTENCE_WORDWISE,
            first_sentence_word_chunk_size=config.TTS_FIRST_SENTENCE_WORD_CHUNK_SIZE,
            chunk_on_minor_punctuation=config.TTS_CHUNK_ON_MINOR_PUNCTUATION,
            lead_words_immediate=use_lead_words,
            lead_words_count=config.TTS_LEAD_WORDS_COUNT,
            should_stop=self.tts.is_interrupted,
        )

        full_response = []

        def with_preface(stream):
            if context_preface and not self.tts.is_interrupted():
                if not stream_state["printed"]:
                    print("Tarz: ", end="", flush=True)
                    stream_state["printed"] = True
                print(f"{context_preface} ", end="", flush=True)
                yield context_preface
            for item in stream:
                yield item

        def relay(stream):
            for sentence in stream:
                full_response.append(sentence)
                yield sentence

        tts = self.tracing.start_manual_step(
            turn,
            "TTS",
            metadata={"language": language, "engine": self.tts.backend_name, "voice": config.VOICE},
        )
        playback = self.tracing.start_manual_step(
            turn,
            "Playback",
            metadata={"engine": "sounddevice"},
        )
        try:
            self.tts.speak_stream(relay(with_preface(sentences)))
            self.tts.wait_until_idle()
            if stream_state["printed"]:
                print()
        finally:
            barge_in_stop.set()
            if barge_in_listener is not None:
                barge_in_listener.join(timeout=0.1)
            if producer_thread.is_alive():
                producer_thread.join(timeout=0.2)
        tts_metrics = self.tts.get_turn_metrics()
        self.tracing.update_step(
            tts,
            output={
                "first_audio_latency_ms": tts_metrics["first_audio_latency_ms"],
                "synthesis_duration_ms": tts_metrics["synthesis_duration_ms"],
                "audio_duration_ms": round(tts_metrics["audio_duration_ms"], 2),
                "chunk_count": tts_metrics["chunk_count"],
            },
            metadata={"sample_rate": tts_metrics["sample_rate"], "speed": config.TTS_SPEED},
        )
        self.tracing.update_step(
            playback,
            output={
                "playback_duration_ms": tts_metrics["playback_duration_ms"],
                "queue_delay_ms": tts_metrics["queue_delay_ms"],
                "audio_duration_ms": round(tts_metrics["audio_duration_ms"], 2),
            },
        )
        if tts is not None:
            tts.end()
        if playback is not None:
            playback.end()

        if not full_response:
            print("Tarz: (no response)")

        if turn is not None:
            turn.update(output={"response": " ".join(full_response)})
            total_latency_ms = self.tracing.elapsed_ms(request_start)
            timing = {
                "total_interaction_latency_ms": total_latency_ms,
                "router_latency_ms": self.tracing.elapsed_ms(router_start),
                "llm": self.llm.last_metrics,
                "tts": {
                    "synthesis_duration_ms": tts_metrics["synthesis_duration_ms"],
                    "playback_duration_ms": tts_metrics["playback_duration_ms"],
                },
            }
            timing.update(pipeline_metrics or {})
            for name, duration in {
                "router_percent": timing["router_latency_ms"],
                "llm_percent": self.llm.last_metrics.get("total_latency_ms", 0),
                "tts_percent": tts_metrics["synthesis_duration_ms"] or 0,
                "playback_percent": tts_metrics["playback_duration_ms"] or 0,
                "stt_percent": timing.get("stt_latency_ms", 0),
            }.items():
                timing[name] = round(duration / max(total_latency_ms, 0.001) * 100, 2)
            self.tracing.update_turn_metrics(turn, timing)

    def run_voice(self):

        print("Say 'back to menu' to choose a different mode.")

        while True:

            try:

                request_start = self.tracing.now()
                with self.tracing.turn_attributes():
                    with self.tracing.start_turn("", None, "VOICE", "voice") as turn:
                        with self.tracing.start_step(
                            "VAD",
                            metadata={"audio_included_in_trace": False},
                        ) as recording:
                            audio, vad_metrics = self.mic.listen(return_metrics=True)
                            audio_seconds = len(audio) / config.SAMPLE_RATE
                            self.tracing.update_step(
                                recording,
                                output={
                                    "audio_duration_seconds": round(audio_seconds, 3),
                                    **vad_metrics,
                                },
                            )

                        with self.tracing.start_step(
                            "STT",
                            metadata={
                                "engine": "faster-whisper",
                                "model": config.WHISPER_SIZE,
                                "beam_size": config.WHISPER_BEAM_SIZE,
                                "compute_type": config.WHISPER_COMPUTE,
                                "audio_duration_seconds": round(audio_seconds, 3),
                                "audio_included_in_trace": False,
                                "partial_transcript_supported": False,
                            },
                        ) as transcription:
                            decode_hint = None
                            if config.STT_PREFER_PREVIOUS_LANGUAGE_HINT:
                                previous_language = self.llm.memory.get_language()
                                if previous_language in config.STT_ALLOWED_LANGUAGES and previous_language != "en":
                                    decode_hint = previous_language

                            result = self.stt.transcribe(audio, language=decode_hint)
                            self.tracing.update_step(
                                transcription,
                                output={
                                    "text": result["text"],
                                    "language": result["language"],
                                    "confidence": result["confidence"],
                                    "first_segment_ms": result["first_segment_ms"],
                                    "transcript_length": len(result["text"]),
                                    "latency_ms": result["latency_ms"],
                                },
                            )

                        if self._return_to_menu_requested(result["text"]):
                            print("Returning to mode selection...")
                            self.tracing.record_event(turn, "Conversation Ended", {"returned_to_menu": True})
                            self.tracing.flush()
                            return

                        self.process(
                            result["text"],
                            stt_language_hint=result["language"],
                            stt_language_confidence=result["confidence"],
                            turn=turn,
                            input_mode="voice",
                            request_start=request_start,
                            pipeline_metrics={
                                "vad": vad_metrics,
                                "stt_latency_ms": result["latency_ms"],
                            },
                        )

                        self.tracing.record_event(
                            turn,
                            "Conversation Ended",
                            {"response_completed": True},
                        )

                self.tracing.flush()

            except KeyboardInterrupt:
                print("\nVoice mode interrupted. Returning to mode selection...")
                self.tracing.flush()
                return

    def run_text(self):

        while True:

            text = input("\nYou (/menu to change mode): ")

            if self._return_to_menu_requested(text):
                print("Returning to mode selection...")
                return

            self.process(text)


if __name__ == "__main__":

    tarz = Tarz()

    while True:
        mode = input(
            "\nChoose mode\n"
            "1. Voice\n"
            "2. Text\n"
            "0. Exit\n\n"
            "Choice: "
        ).strip()

        if mode == "1":
            tarz.run_voice()
        elif mode == "2":
            tarz.run_text()
        elif mode == "0":
            break
        else:
            print("Please choose 1, 2, or 0.")


