# ═══════════════════════════════════════════════════════════════════════════════
# TARZ: Multilingual Voice Assistant with Streaming Pipeline
# ═══════════════════════════════════════════════════════════════════════════════
#
# ARCHITECTURE OVERVIEW:
# ├─ Input Layer: Microphone (VAD-gated audio) or Text Console
# ├─ STT (Speech-to-Text): Whisper (English) + IndicConformer (Hindi/Telugu/Malayalam)
# ├─ Intent Router: Classifies intent (GENERAL, SEARCH, VISION, OCR)
# ├─ Task Detection: Maps user input to task context (story, joke, travel, weather, etc)
# ├─ Conversation Memory: Manages turn history and language preference
# ├─ LLM (Language Model): Streams response tokens for low-latency generation
# ├─ Sentence Buffer: Groups tokens into speech-friendly chunks (respects punctuation)
# ├─ TTS Router: Multi-backend synthesis (SuperTonic: en/hi, Piper: te/ml/ar)
# └─ Playback: Intelligent device selection + resampling + stream priming
#
# KEY FEATURES:
# • Streaming output: LLM tokens printed to console as they are generated (token-by-token)
# • Filler phrases: Contextual \"Let me find that\" plays while LLM generates (low latency)
# • Streaming pipeline: LLM tokens flow to TTS immediately (not end-to-end latency)
# • Multilingual: English, Hindi, Telugu, Malayalam, Arabic with native vocabulary
# • Task-lock mode: Follow-ups like \"flights\" stay in travel context automatically
# • Barge-in support: User can press ENTER during playback to interrupt and ask new question
# • Low-latency audio: Device probing, resampling, DMA buffer priming
# • Comprehensive tracing: Langfuse integration for latency debugging
#
# PROCESSING PIPELINE (7 stages):
# 1. Language Detection: Identify script (Latin/Devanagari/Telugu/Malayalam/Arabic)
#                        and language using heuristics + STT hints
# 2. Intent Routing: Route to GENERAL, SEARCH (web), VISION (photo), OCR (text recognition)
# 3. Task Detection: Match vocabulary in TASK_KEYWORDS dict to identify task context
# 4. Memory Retrieval: Load conversation history (persona + prior turns)
# 5. Optional Vision: Process uploaded image/PDF media
# 6. LLM → TTS Pipeline: 
#    - Producer: Stream LLM tokens into queue (background thread)
#    - Sentence buffer: Group tokens by punctuation/word count
#    - Preface: Inject contextual filler phrase before response
#    - TTS: Synthesize buffered sentences to audio
#    - Playback: Stream audio with device selection + resampling + barge-in interrupt
# 7. Metrics: Log latencies (language detection, routing, LLM, TTS, playback) to Langfuse
#
# MULTILINGUAL SUPPORT:
# English:   Direct Whisper ASR, SuperTonic TTS, full feature set
# Hindi:     Whisper + IndicConformer ASR, SuperTonic TTS, full feature set
# Telugu:    Whisper + IndicConformer ASR, Piper TTS, full feature set
# Malayalam: Whisper + IndicConformer ASR, Piper TTS, full feature set
# Arabic:    Whisper ASR, Piper TTS, full feature set
#
# VOCABULARY MATCHING SYSTEM:
# • FOLLOWUP_TOKENS: Words indicating continuation (also, add, more, भी, కూడా, കൂടി)
# • TASK_KEYWORDS: Maps 10 task types to multilingual vocabulary (story, joke, travel, etc)
# • SOCIAL_TOKENS: Greetings/acknowledgments (hi, hello, namaste, thanks, ધન્યવાદ, etc)
# • Intent categories: 17+ categories (greeting, joke, weather, coding, translation, etc)
#   Each category has multilingual filler phrases in config.LANGUAGE_PREFACES
#
# CONFIGURATION:
# See config.py for tunable parameters:
# • TTS_MIN/MAX_CHARS/WORDS: Chunk size constraints
# • TTS_FIRST_SENTENCE_IMMEDIATELY: Start TTS ASAP (don't wait for 2nd sentence)
# • TTS_FIRST_WORD_IMMEDIATELY: Send individual words instead of waiting for sentence
# • TTS_CONTEXT_PREFACE_ENABLED: Enable/disable filler phrases
# • TTS_CONTEXT_PREFACE_RANDOM: Randomize vs deterministic filler selection
# • STT_INDIC_LANGUAGES: List of Indic languages for IndicConformer routing
# • VOICE: TTS voice identifier (language + speaker)
# • LLM_MODEL: Model identifier (gpt-4, gpt-3.5, etc)
#
# ═══════════════════════════════════════════════════════════════════════════════

import config
import msvcrt
import time
import threading
import queue
import random
import re
import json
from urllib import request
from datetime import datetime
from audio.microphone import Microphone
from stt.stt import STT
from core.router import Router
from llm.llm import LLM, sentence_stream
from tts.tts_router import TTSRouter
from core.language import detect_dominant_language, normalize_text, detect_script
from core.tracing import LangfuseTracer
from services.vision import VisionService
from ui.media_chooser import MediaChooser
from pathlib import Path


class Tarz:
    """
    Voice/text conversational AI assistant implementing a complete end-to-end pipeline:
    Microphone (voice input) → STT (speech-to-text) → Router (intent detection) → 
    LLM (language model) → TTS (text-to-speech) → Speaker (audio playback).
    
    Supports multilingual interaction (English, Hindi, Telugu, Malayalam, Arabic) with 
    intelligent task detection, conversation memory, and real-time interrupt handling.
    Features ChatGPT-style response printing, automatic device selection, and latency optimization.
    """

    # FOLLOWUP_TOKENS: Words/phrases that indicate the user is continuing the previous task
    # rather than starting a new one. Enables task-lock mode for terse follow-ups like "also show flights".
    # Covers English continuation (also, add, more), Hindi (भी, और), Telugu (కూడా, ఇంకా),
    # and Malayalam (കൂടി, കൂടെ) to maintain context across multiple turns.
    FOLLOWUP_TOKENS = {
        # English continuation signals
        "also", "add", "include", "more", "continue", "next",
        "too", "and", "plus", "then", "after", "another",
        "details", "options", "examples", "price", "cost", "budget",
        # Telugu continuation signals
        "కూడా", "ఇంకా", "అలాగే", "అదనంగా",
        # Hindi continuation signals
        "भी", "और", "के अलावा", "इसके",
        # Nepali continuation signals
        "पनि", "अझै", "थप", "र", "pani", "ajhai", "thap", "ra",
        # Malayalam continuation signals
        "കൂടി", "കൂടെ", "ഇനിയും",
    }

    # TASK_KEYWORDS: Maps task names to multilingual vocabulary that triggers them.
    # Used by _detect_task() to identify user intent (story, joke, poem, travel booking, etc).
    # Each task is identified by a set of English and native-language keywords. Task detection
    # enables task-lock mode where follow-ups remain in context (e.g., "flights" stays in travel mode).
    # Supports all 5 languages: English, Hindi, Telugu, Malayalam, Arabic.
    TASK_KEYWORDS = {
        "story": {"story", "katha", "kahani", "कहानी", "कथा", "కథ", "कथा", "nepali katha", "कहानी सुनाउ", "حكاية", "قصة"},
        "joke": {"joke", "funny", "मजाक", "चुटकुला", "జోక్", "ठट्टा", "thatta", "തമാശ", "نكتة"},
        "poem": {"poem", "poetry", "shayari", "कविता", "शायरी", "కవిత", "कविता", "kabita", "قصيدة"},
        "travel": {
            "trip", "travel", "itinerary", "flight", "flights", "ticket", "tickets",
            "airfare", "hotel", "stay", "bombay", "mumbai", "tour",
        },
        "weather": {"weather", "forecast", "temperature", "मौसम", "वातावरण", "వాతావరణం", "mausam", "الطقس"},
        "news": {"news", "headlines", "समाचार", "వార్తలు", "समाचार", "samachar", "أخبار"},
        "coding": {"code", "coding", "program", "python", "bug", "debug", "fix", "script"},
        "math": {"math", "calculate", "equation", "sum", "multiply", "divide", "गणना", "हिसाब", "hisab", "లెక్క"},
        "translation": {"translate", "translation", "अनुवाद", "అనువాదం", "अनुवाद", "anuwad", "ترجمة"},
    }

    # SOCIAL_TOKENS: Vocabulary identifying small-talk/greetings (hi, hello, thanks, etc).
    # Social turns bypass task-based routing and receive conversational replies instead.
    # Includes English greetings (hi, hello, hey), Hindi (नमस्ते), Telugu (నమస్కారం), 
    # Malayalam (നമസ്കാരം), and Arabic (مرحبا) equivalents. Used by _is_social_turn() and 
    # _build_context_preface() to handle non-transactional conversation naturally.
    SOCIAL_TOKENS = {
        "hi", "hello", "hey", "namaste", "namaskaram", "नमस्ते", "नमस्कार",
        "alaga", "oho",
        "hii", "heyy", "yo", "ok", "okay", "thanks", "thank", "wow", "great",
        "thik", "theek", "ठीक",
        "धन्यवाद", "सन्चै", "सञ्चै", "dhanyabad", "sanchai", "sanchai cha", "thikcha", "thik chha",
        "నమస్కారం", "హాయ్", "ధన్యవాదాలు", "బాగుంది",
        "നമസ്കാരം", "ഹലോ", "നന്ദി",
        "مرحبا", "أهلا", "شكرا",
    }

    def __init__(self):
        """
        Initialize the Tarz assistant by loading all models and services:
        - Tracing infrastructure (Langfuse) for debugging and latency tracking
        - Microphone (audio input with VAD noise gate)
        - STT engine (Whisper + IndicConformer for Indic languages)
        - Router (intent classifier: GENERAL, SEARCH, VISION, OCR)
        - Upload media (optional image/PDF input for analysis)
        - LLM (language model with conversation memory)
        - TTS router (multi-backend: SuperTonic for en/hi, Piper for te/ml/ar)
        Optionally runs LLM warmup on startup if configured.
        Sets up tracing reporters and initializes per-turn state (current_task=None).
        """
        print("Starting Tarz...\n")

        self.tracing = LangfuseTracer()
        self.mic = Microphone()
        self.stt = STT()
        self.router = Router()
        self.vision = VisionService()
        self.media_chooser = MediaChooser()
        print("Loading LLM...", flush=True)
        self.llm = LLM(model=config.LLM_MODEL, tracer=self.tracing)
        if config.LLM_WARMUP_ON_STARTUP:
            self.llm.warmup()
        print("✓ LLM Loaded", flush=True)
        print("Loading OCR...", flush=True)
        self.vision.ocr.initialize()
        print("✓ OCR Loaded", flush=True)
        print("Loading TTS...", flush=True)
        self.tts = TTSRouter(on_event=self._on_event)
        print("✓ TTS Loaded", flush=True)
        self.current_task = None
        self.tracing.set_model_startup_metrics(
            stt=self.stt.model_startup_metrics,
            llm=self.llm.measure_model_startup(),
            tts=self.tts.model_startup_metrics,
        )
        print("\nTarz Ready\n", flush=True)

    def _on_event(self, name, data):
        if config.DEBUG:
            print(f"[EVENT] {name}: {data}")

    def speak_intro(self):
        """Speak a welcome greeting in the configured intro language."""
        if not config.INTRO_ENABLED:
            return
        greetings = {
            "en": "Hi, this is Tarz, your personal assistant. How can I help you today?",
            "hi": "नमस्ते, मैं Tarz हूँ, आपका पर्सनल असिस्टेंट। आज आपकी कैसे मدد करूँ?",
            "ne": "नमस्ते, म तपाईंको पर्सनल असिस्टेन्ट Tarz हुँ। आज म तपाईंलाई कसरी सहयोग गर्न सक्छु?",
            "te": "నమస్కారం, నేను Tarz, మీ పర్సనల్ అసిస్టెంట్। నేను మీకు ఎలా సహాయం చేయగలను?",
            "ml": "നമസ്കാരം, ഞാൻ Tarz, നിങ്ങളുടെ പേഴ്സണൽ അസിസ്റ്റന്റ്। ഇന്ന് നിങ്ങള്ക്ക് എന്ത് സഹായം വേണം?",
            "ar": "مرحبا، أنا Tarz، مساعدك الشخصي। كيف يمكنني مساعدتك اليوم؟",
        }
        lang = config.INTRO_LANGUAGE
        text = greetings.get(lang, greetings["en"])
        print(f"\nTarz: {text}")
        self.tts.speak(text, lang)

    @staticmethod
    def _return_to_menu_requested(text):
        """
        Check if the user is requesting to exit the current mode (voice/text) and return
        to the main menu. Recognizes commands like '/menu', 'back', 'menu par jao' (Hindi),
        'మెనూకి వెళ్ళు' (Telugu) across all supported languages.
        Strips punctuation before comparison to handle "back!" → "back" equivalence.
        Returns True if menu-exit intent detected, False otherwise.
        """
        command = text.strip().lower().strip(".,!?;:。")
        return command in {
            "/menu", "0", "menu", "back", "go back", "back to menu",
            "menu par jao", "menu par wapas jao", "मेनू पर जाओ", "मेनू पर वापस जाओ",
            "మెనూకి వెళ్ళు", "మెనూకి వెళ్లు", "మెనూకు వెళ్ళు",
        }

    def _watch_for_barge_in(self, stop_event):
        """
        Watch for user keyboard interrupt (ENTER key) during TTS playback to enable barge-in.
        Barge-in allows users to interrupt Tarz's speech and ask a new question immediately
        without waiting for full playback to complete. This improves conversational responsiveness.
        
        Algorithm:
        1. Drain any leftover Enter key presses from console input buffer (from STT prompt)
        2. Poll keyboard every 50ms while TTS is speaking
        3. If Enter pressed during active playback, call tts.stop() to halt audio
        4. Print friendly message and return to listening mode
        5. If stop_event is set (end of turn), exit loop gracefully
        
        Runs on background thread during _process() to avoid blocking TTS streaming.
        Uses Windows-specific msvcrt.kbhit() and msvcrt.getwch() for non-blocking keyboard input.
        """
        # Ignore Enter key leftovers in console buffer from previous input() call.
        while msvcrt.kbhit():
            msvcrt.getwch()

        while not stop_event.wait(0.05):
            # Poll keyboard at 20Hz (50ms sleep between checks) to minimize CPU usage.
            if not msvcrt.kbhit():
                continue

            key = msvcrt.getwch()
            # Only interrupt if TTS is currently playing audio (is_speaking() = True).
            if self.tts.is_speaking() and key in ("\r", "\n"):
                self.tts.stop()  # Signal TTS to halt playback and audio production.
                print("\nTarz: (stopped - listening for your next question)")
                return

    @staticmethod
    def _build_context_preface(prompt, language, current_task=None):
        """
        Build a "filler phrase" that Tarz speaks before responding to the user.
        Fillers improve perceived responsiveness by starting speech immediately
        (e.g., "Sure, I can help with that") while the LLM generates the real response.
        
        Intent Detection Strategy:
        1. Extract tokens (words) from user input using Unicode regex for multilingual support
        2. Check if input is social/greeting using _is_social_turn() → pick social category
        3. For non-social turns, match vocabulary against 17+ intent categories (greeting, 
           joke, travel, coding, translation, etc) using has_any() helper
        4. If no category matches and current_task is known, inherit previous task category
        5. Fall back to "generic" if still unmatched
        6. If still generic and no prior task, return None (no filler)
        
        Phrase Selection:
        - Look up filler table for (language, category) pair from config.LANGUAGE_PREFACES
        - Prefer multi-word phrases (2+ words) over single words for natural speech rhythm
        - Fallback to generic/fallback category if primary category has no phrases
        - If config.TTS_CONTEXT_PREFACE_RANDOM enabled, randomly select from candidates
        - Else use first phrase from sorted list (deterministic)
        
        Returns: Filler phrase string (e.g., "Okay, let me help.") or None if no filler needed.
        """
        # Normalize input: lowercase for case-insensitive matching, strip whitespace.
        text = (prompt or "").strip().lower()
        if not text:
            return None

        # Extract tokens (whole words) using Unicode regex supporting English/Hindi/Telugu/Malayalam/Arabic.
        # This allows matching "hello" in "hello there" without splitting Tamil/Hindi compound words.
        tokens = set(re.findall(r"[a-zA-Z]+|[\u0900-\u097f]+|[\u0c00-\u0c7f]+|[\u0d00-\u0d7f]+|[\u0600-\u06ff]+", text))

        # Helper: Returns True if any keyword from words set matches the input text.
        # Handles both phrase matching ("how are you" as substring) and word matching.
        def has_any(words):
            for word in words:
                w = word.lower()
                if " " in w:
                    if w in text:
                        return True
                elif w in tokens:
                    return True
            return False

        # Intent category keywords: Maps 17+ intent categories to multilingual vocabulary.
        # Each category represents a type of conversation (greeting, joke, coding, weather, etc).
        # Used to select appropriate filler phrases from config.LANGUAGE_PREFACES.
        category_keywords = {
            "greeting": {
                "hello", "hi", "hey", "heymate", "mate", "wassup", "wassupp", "whatsup", "sup",
                "namaste", "नमस्ते", "నమస్కారం", "നമസ്കാരം", "مرحبا",
            },
            "wellbeing_query": {
                "how are you", "what about you", "and you", "how about you",
                "कैसे हो", "कैसे हैं", "आप कैसे हैं", "तुम कैसे हो",
                "మీరు ఎలా ఉన్నారు", "నువ్వు ఎలా ఉన్నావు", "ఎలా ఉన్నారు",
                "ela unnavu", "ela unnaru", "meeru ela unnaru", "nuvvu ela unnavu",
                "സുഖമാണോ", "എങ്ങനെയുണ്ട്", "നിങ്ങൾ എങ്ങനെയുണ്ട്",
                "كيف حالك", "كيف حالكم", "كيفك",
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
                "chala manchi katha", "chaala manchi katha", "అది చాలా మంచి కథ",
                "വളരെ നല്ലത്", "അടിപൊളി", "സൂപർ",
                "رائع", "ممتاز", "مذهل",
            },
            "story": {"story", "katha", "kahani", "कहानी", "कथा", "కథ", "kadha", "حكاية", "قصة"},
            "joke": {"joke", "funny", "मजाक", "चुटकुला", "జోక్", "തമാശ", "نكتة"},
            "poem": {"poem", "poetry", "shayari", "कविता", "शायरी", "పద్య", "కవిత", "قصيدة"},
            "weather": {"weather", "temperature", "forecast", "मौसम", "వాతావరణం", "കാലാവസ്ഥ", "الطقس"},
            "news": {"news", "headlines", "समाचार", "వార్తలు", "വാർത്ത", "أخبار"},
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

        # Category resolution: Start with "generic" (no specific intent).
        # Then check: (1) Is it social/greeting? (2) Match vocabulary keywords? (3) Inherit from prior task?
        # This cascading logic ensures "okay" with no prior task stays silent, but "more flights" reuses travel context.
        category = "generic"
        if Tarz._is_social_turn(text):  # Greetings/small-talk get conversational categories.
            if has_any({
                "hello", "hi", "hey", "heymate", "mate", "wassup", "wassupp", "whatsup", "sup",
                "namaste", "नमस्ते", "నమస్కారం", "നമസ്കാരം", "مرحبا", "أهلا",
            }):
                category = "greeting"
            elif has_any({
                "how are you", "what about you", "and you", "how about you",
                "कैसे हो", "कैसे हैं", "आप कैसे हैं", "तुम कैसे हो",
                "మీరు ఎలా ఉన్నారు", "నువ్వు ఎలా ఉన్నావు", "ఎలా ఉన్నారు",
                "ela unnavu", "ela unnaru", "meeru ela unnaru", "nuvvu ela unnavu",
                "സുഖമാണോ", "എങ്ങനെയുണ്ട്", "നിങ്ങൾ എങ്ങനെയുണ്ട്",
                "كيف حالك", "كيف حالكم", "كيفك",
            }):
                category = "wellbeing_query"
            elif has_any({"thanks", "thank you", "धन्यवाद", "ధన్యవాదాలు", "നന്ദി", "شكرا"}):
                category = "thanks"
            elif has_any({
                "wow", "great", "awesome", "nice", "super", "cool", "excellent",
                "बहुत बढ़िया", "शानदार", "वाह",
                "చాలా బాగుంది", "బాగుంది", "సూపర్", "అద్భుతం",
                "chala manchi katha", "chaala manchi katha", "అది చాలా మంచి కథ",
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

        # Task-to-category mapping: If no explicit category matched but we're in a task context,
        # reuse the task's category for coherent filler. E.g., prior task="story" → category="story".
        task_to_category = {
            "story": "story",
            "joke": "joke",
            "poem": "poem",
            "travel": "search",
            "weather": "weather",
            "news": "news",
            "coding": "coding",
            "math": "math",
            "translation": "translation",
        }
        # Inherit task category if in task context and no explicit category detected.
        if category == "generic" and current_task in task_to_category:
            category = task_to_category[current_task]

        # Terse follow-ups (≤2 words) on known tasks are better labeled as "answer" than "generic".
        # "Flights" alone should get "I'll search for flights" (answer) not "I can help" (generic).
        if category == "generic" and len(text.split()) <= 2 and current_task in {"travel", "coding", "math", "news", "weather", "translation"}:
            category = "answer"

        # Questions with question marks or query words (how, what, where) → "answer" category.
        # Triggers response-oriented filler like "Let me find that for you" instead of generic acknowledgment.
        if category == "generic" and ("?" in text or has_any({"how", "what", "why", "when", "where", "can you", "help"})):
            category = "answer"

        # Final safety check: If still generic with no task context, return None to avoid generic filler.
        # Prevents awkward "I can help" responses to bare "okay" or untasked follow-ups.
        if category == "generic" and current_task is None:
            return None

        # Phrase lookup from config.LANGUAGE_PREFACES[language][category].
        # Cascade: Try target language → fallback to English → empty list.
        tables = getattr(config, "LANGUAGE_PREFACES", {})
        table = tables.get(language) or tables.get("en", {})
        
        # Get candidate phrases for the detected category (or fallback to generic/fallback).
        candidates = table.get(category) or table.get("generic") or table.get("fallback") or []
        if not candidates:
            return None  # No phrases configured for this language/category combination.
        
        # Multi-word phrases (≥2 words) sound more natural as spoken fillers than single words.
        # "Sure, let me help" flows better than just "Help" during speech playback.
        multiword_candidates = [candidate for candidate in candidates if len(candidate.strip().split()) >= 2]
        source = multiword_candidates
        
        # Fallback: If no multi-word phrases in category, try generic or fallback categories.
        if not source:
            generic_multi = [candidate for candidate in table.get("generic", []) if len(candidate.strip().split()) >= 2]
            fallback_multi = [candidate for candidate in table.get("fallback", []) if len(candidate.strip().split()) >= 2]
            source = generic_multi or fallback_multi

        # Last-resort hardcoded defaults in all supported languages.
        # Used if config.LANGUAGE_PREFACES is missing or incompletely configured.
        defaults = {
            "en": "Okay, I can help with that.",
            "hi": "ठीक है, मैं मदद करता हूँ।",
            "ne": "ठिक छ, म सहयोग गर्छु।",
            "te": "సరే, నేను సహాయం చేస్తాను.",
            "ml": "ശരി, ഞാൻ സഹായിക്കാം.",
            "ar": "حسنًا، سأساعدك في ذلك.",
        }
        if not source:  # No phrases found → return multilingual defaults.
            return defaults.get(language, defaults["en"])

        # Apply pacing constraint if configured ("slow" mode requires ≥ min_words).
        # Prevents overly terse filler in slow TTS modes where "Okay" alone feels choppy.
        min_words = max(2, int(getattr(config, "TTS_PREFACE_MIN_WORDS", 2)))
        pacing = str(getattr(config, "TTS_PREFACE_PACING", "normal")).lower()
        if pacing == "slow":
            paced = [candidate for candidate in source if len(candidate.strip().split()) >= min_words]
            if paced:
                source = paced  # Use longer phrases in slow mode for better rhythm.

        # Select phrase: Random if randomization enabled, else deterministic (first in list).
        if config.TTS_CONTEXT_PREFACE_RANDOM and len(candidates) > 1:
            selected = random.choice(source)  # Variety: different filler each turn.
        else:
            selected = source[0]  # Consistency: same filler phrase each time.

        return selected

    @staticmethod
    def _detect_task(normalized_text, intent):
        """
        Detect the user's task intent (story, joke, poem, travel, weather, etc.) from input.
        Task detection enables task-lock mode: follow-ups within the same task stay in context.
        
        Algorithm:
        1. OCR wording is routed to the upload-based OCR task
        2. Extract tokens from text using Unicode regex (English/Hindi/Telugu/Malayalam/Arabic)
        3. Iterate through TASK_KEYWORDS dict, checking if any task vocabulary matches tokens
        4. First task with keyword match is returned; order matters (story checked before joke, etc)
        5. If no match found, return None (no task context)
        
        Matching Rules:
        - Multi-word keywords (e.g., "read this") matched as substrings in original text
        - Single-word keywords matched as whole words in token set (avoid "cal" matching "calculate")
        
        Returns: Task key string ("story", "joke", "travel", etc) or None if no task matched.
        """
        # Normalize and validate input.
        text = (normalized_text or "").strip().lower()
        if not text:
            return None

        # Appreciation about a previous story should not restart story mode.
        if any(
            phrase in text
            for phrase in {
                "manchi katha",
                "adi manchi katha",
                "chala manchi katha",
                "chaala manchi katha",
                "adi chala manchi katha",
                "adi chaala manchi katha",
                "అది చాలా మంచి కథ",
                "nice story",
                "good story",
                "achi kahani",
                "achhi kahani",
                "bahut achhi kahani",
            }
        ):
            return None

        # Extract tokens (words) using Unicode regex to support all 5 languages in one pass.
        # Regex ranges: [a-zA-Z] = English, [\u0900-\u097f] = Devanagari (Hindi), 
        # [\u0c00-\u0c7f] = Telugu, [\u0d00-\u0d7f] = Malayalam, [\u0600-\u06ff] = Arabic.
        tokens = set(re.findall(r"[a-zA-Z]+|[\u0900-\u097f]+|[\u0c00-\u0c7f]+|[\u0d00-\u0d7f]+|[\u0600-\u06ff]+", text))

        # Iterate TASK_KEYWORDS dict, checking if any keyword for any task matches input.
        # First task with a match is returned (order in dict matters; story before joke, etc).
        for task, words in Tarz.TASK_KEYWORDS.items():
            for word in words:
                token = word.lower()
                # Multi-word keywords (e.g., "read this") → substring match in original text.
                if " " in token:
                    if token in text:
                        return task  # Match found: return task immediately.
                # Single-word keywords (e.g., "travel") → whole-word match in token set.
                elif token in tokens:
                    return task  # Match found: return task immediately.

        # No task vocabulary found in input → return None (no task context).
        return None

    @staticmethod
    def _is_terse_followup(text):
        """
        Detect if a message is a terse follow-up (continuation of prior task) vs a new request.
        Used to decide whether to apply task-lock mode, keeping context from previous turn.
        
        Algorithm:
        - Check for explicit continuation vocabulary (also, add, more, next, continue, etc)
        - Includes English (also, more, and), Hindi (भी, और), Telugu (కూడా, ఇంకా), Malayalam (കൂടി)
        - Word count ALONE is not reliable: Telugu/Hindi phrases are naturally short (4-word sentences are complete)
        - Only explicit vocabulary triggers follow-up mode
        
        Returns: True if continuation vocabulary detected, False otherwise (is new standalone request).
        """
        # Normalize input: lowercase for case-insensitive matching.
        lowered = (text or "").strip().lower()
        if not lowered:
            return False
        
        # Explicit continuation signals in all supported languages.
        # NOTE: Word count alone is NOT reliable: Indic languages have naturally short sentences.
        # "flights to bombay" (4 words in Telugu) is a complete request, not a follow-up.
        # Only these explicit vocabulary tokens trigger follow-up mode.
        followup_signals = {
            # English
            "also", "add", "include", "more", "next", "continue", "too", "and",
            # Telugu (కూడా=also, ఇంకా=more)
            "కూడా", "ఇంకా", "అలాగే",
            # Hindi (भी=also, और=and)
            "भी", "और",
            # Nepali
            "पनि", "थप", "अझै", "pani", "thap", "ajhai",
            # Malayalam (കൂടി=more, കൂടെ=also)
            "കൂടി", "കൂടെ",
        }
        # Return True if any follow-up signal substring found in lowered input.
        return any(signal in lowered for signal in followup_signals)

    @staticmethod
    def _is_social_turn(text):
        """
        Detect if the turn is social/conversational (greeting, small-talk) vs transactional.
        Social turns receive conversational replies instead of task routing.
        
        Algorithm:
        1. Extract tokens from text using Unicode regex (all 5 languages)
        2. Check if any token matches SOCIAL_TOKENS (direct word lookup)
        3. Fallback: Check if social phrases matched as substrings (multi-word)
        4. Returns True if either check passes (is social turn)
        
        Examples:
        - "Hi, how are you?" → True (greeting + wellbeing query)
        - "Tell me a story" → False (task request, not social)
        - "നമസ്കാരം" (Malayalam hello) → True (social token match)
        
        Returns: True if social/greeting turn, False if transactional/task turn.
        """
        # Normalize and tokenize input using same Unicode regex as other methods.
        lowered = (text or "").strip().lower()
        if not lowered:
            return False
        
        # Extract tokens (words) for multilingual support.
        tokens = set(re.findall(r"[a-zA-Z]+|[\u0900-\u097f]+|[\u0c00-\u0c7f]+|[\u0d00-\u0d7f]+|[\u0600-\u06ff]+", lowered))
        if not tokens:
            return False
        
        # Fast path: Check if any token directly matches SOCIAL_TOKENS (single-word lookups).
        # E.g., if "hi" in tokens or "नमस्ते" in tokens → social turn detected.
        if tokens.intersection(Tarz.SOCIAL_TOKENS):
            return True
        
        # Extended social phrases: Multi-word patterns (e.g., "how are you", "i am good")
        # and language-specific responses (Hindi "में भी ठीक", Telugu "చాలా బాగుంది", etc).
        social_phrases = {
            # English wellbeing exchanges
            "how are you", "what about you", "i am good", "i'm good", "im good",
            # Hindi wellbeing questions, including the normalized Devanagari form.
            "aap kaise ho", "aap kese ho", "aap kaise hain", "aap kese hain",
            "आप कैसे हो", "आप कैसे हैं", "कैसे हो", "कैसे हैं",
            # Telugu phonetic and native
            "ela unnavu", "ela unnaru", "meeru ela unnaru", "nuvvu ela unnavu",
            "nenu bagunnanu", "nenu bagunnanu andi", "nenu kuda bagunnanu",
            # Nepali phonetic and native
            "tapai sanchai hunuhunchha", "tapai sanchai chha", "ma sanchai chu", "ma thik chu",
            "तपाईं सन्चै हुनुहुन्छ", "म सन्चै छु", "म ठिक छु",
            # Hindi phonetic and native ("me bhi" = I'm also fine)
            "me bhi", "mein bhi", "main bhi", "mai bhi", "me too", "same here",
            "me bhi thik", "mein bhi thik", "main bhi thik", "main bhi theek", "i am fine too",
            "में भी", "में भी ठीक", "मैं भी", "मैं भी ठीक", "मैं भी ठीक हूँ",
            # Hindi appreciation
            "bahut badiya", "bahut badhiya", "bahut badia", "बहुत बढ़िया", "बहुत बढिया",
            # Telugu appreciation
            "chaala bagundi", "chala bagundi", "చాలా బాగుంది",
            "chala manchi katha", "chaala manchi katha", "అది చాలా మంచి కథ",
            # Malayalam appreciation
            "valare nannayi", "വളരെ നല്ലത്",
            # Arabic appreciation
            "mumtaz", "ممتاز", "رائع",
            # Casual English
            "heymate", "wassup", "wassupp", "whatsup",
            "alright",
        }
        
        # Return True if any social phrase matches (multi-word as substring, single-word as token).
        return any(
            (phrase in lowered) if " " in phrase else (phrase in tokens)
            for phrase in social_phrases
        )

    def _build_effective_prompt(self, normalized_text, previous_task=None, detected_task=None, language=None):
        """
        Build the final prompt sent to LLM by optionally wrapping user input with context.
        This enables task-lock mode: terse follow-ups stay in the context of the previous task.
        
        Context Wrapping Rules:
        1. Social turn → Add conversational instruction ("Reply naturally in X language")
        2. New task detected → Use bare input (don't wrap)
        3. Previous task + terse follow-up → Wrap with "Continue the previous [task]" prefix
        4. Otherwise → Use bare input
        
        Example wrappings:
        - Input: "Show me flights to Paris" → Bare (detected_task="travel")
        - Input: "flights" (after travel turn) → "Continue previous travel task. Follow-up: flights..."
        - Input: "Hi" → "This is a greeting turn. Reply naturally in the selected language."
        
        Returns: Wrapped or bare prompt string for LLM consumption.
        """
        # Normalize input.
        text = (normalized_text or "").strip()
        lowered = text.lower()
        if not lowered:
            return text

        # Case 1: Social turn (greeting, small-talk) → Add conversational instruction.
        # Tells LLM to be chatty and natural, avoiding literal translation in non-English languages.
        if self._is_social_turn(text):
            language_name = config.SUPPORTED_LANGUAGES.get(language or "", "the selected language")
            return (
                "This is a greeting/small-talk turn. "
                f"Reply naturally in {language_name} in 1-2 short conversational sentences. "
                "Avoid literal translation artifacts. "
                f"User message: {text}"
            )

        # Case 2: New explicit task detected → Return bare input (no context wrapping needed).
        # E.g., "tell me a joke" triggers detected_task="joke" → no wrapping, just send input.
        if detected_task is not None:
            return text

        # Check for follow-up signals using two methods:
        # (1) Explicit vocabulary in FOLLOWUP_TOKENS (भी, கூடி, etc)
        # (2) _is_terse_followup() logic (also, add, more, continue, etc)
        has_followup_signal = any(token in lowered for token in self.FOLLOWUP_TOKENS)

        # Case 3: Task-lock mode — Keep prior task context for terse follow-ups.
        # Previous task exists AND no new task detected AND input looks like follow-up
        # → Wrap with "Continue the previous [task]..." instruction.
        if previous_task and detected_task is None and (self._is_terse_followup(text) or has_followup_signal):
            return (
                f"Continue the previous {previous_task} task. "
                f"User follow-up: {text}. "
                "Keep the same context and intent. "
                "Do not switch to stories or poems unless explicitly requested now."
            )

        # Default: No wrapping needed — return bare input.
        return text

    @staticmethod
    def _utility_fast_response(text, language):
        """Deterministic non-social fast path for common utility turns."""
        lowered = (text or "").strip().lower()
        if not lowered:
            return None

        # Normalize minor punctuation/typos so inputs like "what;s today's date"
        # match utility markers reliably.
        lowered = re.sub(r"[;]+", " ", lowered)
        lowered = lowered.replace("'", "")
        tokens = set(re.findall(r"[a-zA-Z]+|[\u0900-\u097f]+|[\u0c00-\u0c7f]+|[\u0d00-\u0d7f]+|[\u0600-\u06ff]+", lowered))

        def matches_markers(markers):
            # Multi-word markers use substring matching; single-word markers
            # must match whole tokens so "ok" does not match "oka".
            for marker in markers:
                m = marker.lower().strip()
                if not m:
                    continue
                if " " in m:
                    if m in lowered:
                        return True
                elif m in tokens:
                    return True
            return False

        def fetch_json(url, timeout=2.0):
            req = request.Request(url, headers={"User-Agent": "tarz/1.0"})
            with request.urlopen(req, timeout=timeout) as response:
                payload = response.read().decode("utf-8", errors="ignore")
            return json.loads(payload)

        def fetch_weather_snapshot():
            # wttr.in provides IP-based location and current weather without API keys.
            data = fetch_json("https://wttr.in/?format=j1")
            area = ((data.get("nearest_area") or [{}])[0])
            current = ((data.get("current_condition") or [{}])[0])
            return {
                "city": (((area.get("areaName") or [{}])[0]).get("value") or ""),
                "region": (((area.get("region") or [{}])[0]).get("value") or ""),
                "country": (((area.get("country") or [{}])[0]).get("value") or ""),
                "temp_c": current.get("temp_C"),
                "feels_c": current.get("FeelsLikeC"),
                "desc": (((current.get("weatherDesc") or [{}])[0]).get("value") or ""),
                "humidity": current.get("humidity"),
                "wind_kmph": current.get("windspeedKmph"),
            }

        time_markers = {
            "en": {"what time", "time now", "current time", "time"},
            "hi": {"कितने बजे", "समय", "टाइम"},
            "ne": {"कति बजे", "समय", "टाइम"},
            "te": {"ఎంత సమయం", "సమయం", "టైమ్"},
            "ml": {"എത്ര മണി", "സമയം", "ടൈം"},
            "ar": {"كم الساعة", "الوقت", "الساعه"},
        }
        date_markers = {
            "en": {"date", "todays date", "today date", "what is the date", "current date"},
            "hi": {"आज की तारीख", "तारीख", "दिनांक", "आज कौन सी तारीख"},
            "ne": {"आजको मिति", "मिति", "आज कुन मिति"},
            "te": {"ఈరోజు తేదీ", "తేదీ", "ఈ రోజు తేదీ"},
            "ml": {"ഇന്നത്തെ തീയതി", "തീയതി", "ഇന്ന് തീയതി"},
            "ar": {"تاريخ اليوم", "التاريخ", "ما هو التاريخ"},
        }
        weather_markers = {
            "en": {"weather", "temperature", "forecast", "climate"},
            "hi": {"मौसम", "तापमान", "पूर्वानुमान"},
            "ne": {"मौसम", "तापक्रम", "पूर्वानुमान"},
            "te": {"వాతావరణం", "ఉష్ణోగ్రత", "ఫోర్కాస్ట్"},
            "ml": {"കാലാവസ്ഥ", "താപനില", "ഫോർകാസ്റ്റ്"},
            "ar": {"الطقس", "درجة الحرارة", "توقعات"},
        }
        location_markers = {
            "en": {"location", "where am i", "my location", "current location"},
            "hi": {"लोकेशन", "स्थान", "मैं कहाँ हूँ"},
            "ne": {"स्थान", "लोकेसन", "म कहाँ छु"},
            "te": {"లోకేషన్", "నేను ఎక్కడ ఉన్నాను", "నా స్థానం"},
            "ml": {"ലൊക്കേഷൻ", "ഞാൻ എവിടെയാണ്", "എന്റെ സ്ഥലം"},
            "ar": {"موقعي", "الموقع", "أين أنا"},
        }
        thanks_markers = {
            "en": {"thanks", "thank you"},
            "hi": {"धन्यवाद", "शुक्रिया"},
            "ne": {"धन्यवाद", "धन्यबाद"},
            "te": {"ధన్యవాదాలు", "థాంక్స్"},
            "ml": {"നന്ദി", "താങ്ക്സ്"},
            "ar": {"شكرا", "شكرًا"},
        }
        confirmation_markers = {
            "en": {"ok", "okay", "done", "alright", "fine"},
            "hi": {"ठीक", "ठीक है", "हो गया"},
            "ne": {"ठिक", "ठिक छ", "भयो"},
            "te": {"సరే", "అయింది", "ఓకే"},
            "ml": {"ശരി", "കഴിഞ്ഞു", "ഓകെ"},
            "ar": {"تمام", "حسنًا", "حسنا", "تم"},
        }

        merged_date_markers = date_markers.get(language, set()) | date_markers["en"]
        if matches_markers(merged_date_markers):
            today = datetime.now().strftime("%A, %d %B %Y")
            replies = {
                "en": f"Today is {today}.",
                "hi": f"आज की तारीख {today} है।",
                "ne": f"आजको मिति {today} हो।",
                "te": f"ఈరోజు తేదీ {today}.",
                "ml": f"ഇന്നത്തെ തീയതി {today} ആണ്.",
                "ar": f"تاريخ اليوم هو {today}.",
            }
            return replies.get(language, replies["en"])

        merged_time_markers = time_markers.get(language, set()) | time_markers["en"]
        if matches_markers(merged_time_markers):
            now = datetime.now().strftime("%I:%M %p")
            replies = {
                "en": f"The current time is {now}.",
                "hi": f"अभी समय {now} है।",
                "ne": f"अहिले समय {now} हो।",
                "te": f"ఇప్పుడు సమయం {now}.",
                "ml": f"ഇപ്പോൾ സമയം {now} ആണ്.",
                "ar": f"الوقت الآن هو {now}.",
            }
            return replies.get(language, replies["en"])

        merged_location_markers = location_markers.get(language, set()) | location_markers["en"]
        if matches_markers(merged_location_markers):
            try:
                snap = fetch_weather_snapshot()
                city = snap.get("city") or ""
                region = snap.get("region") or ""
                country = snap.get("country") or ""
                location_text = ", ".join(part for part in (city, region, country) if part)
                if not location_text:
                    raise ValueError("location unavailable")
                replies = {
                    "en": f"Your current location appears to be {location_text}.",
                    "hi": f"आपकी वर्तमान लोकेशन {location_text} लग रही है।",
                    "ne": f"तपाईंको हालको स्थान {location_text} देखिन्छ।",
                    "te": f"మీ ప్రస్తుత స్థానం {location_text}గా కనిపిస్తోంది.",
                    "ml": f"നിങ്ങളുടെ നിലവിലെ സ്ഥലം {location_text} എന്നാണ് കാണുന്നത്.",
                    "ar": f"يبدو أن موقعك الحالي هو {location_text}.",
                }
                return replies.get(language, replies["en"])
            except Exception:
                fallback = {
                    "en": "I could not fetch your current location right now.",
                    "hi": "मैं अभी आपकी लोकेशन नहीं ला पाया।",
                    "ne": "अहिले तपाईंको स्थान ल्याउन सकिन।",
                    "te": "ప్రస్తుతం మీ లోకేషన్ పొందలేకపోయాను.",
                    "ml": "ഇപ്പോൾ നിങ്ങളുടെ ലൊക്കേഷൻ ലഭ്യമാക്കാനായില്ല.",
                    "ar": "تعذر جلب موقعك الحالي الآن.",
                }
                return fallback.get(language, fallback["en"])

        merged_weather_markers = weather_markers.get(language, set()) | weather_markers["en"]
        if matches_markers(merged_weather_markers):
            try:
                snap = fetch_weather_snapshot()
                city = snap.get("city") or "your area"
                desc = snap.get("desc") or ""
                temp_c = snap.get("temp_c")
                feels_c = snap.get("feels_c")
                humidity = snap.get("humidity")
                wind_kmph = snap.get("wind_kmph")
                replies = {
                    "en": f"Current weather in {city}: {desc}, temperature {temp_c} degree Celsius, feels like {feels_c}, humidity {humidity} percent, wind {wind_kmph} kilometers per hour.",
                    "hi": f"{city} में अभी मौसम {desc} है, तापमान {temp_c} डिग्री सेल्सियस है, महसूस {feels_c}, नमी {humidity} प्रतिशत, हवा {wind_kmph} किलोमीटर प्रति घंटा।",
                    "ne": f"{city} मा अहिले मौसम {desc} छ, तापक्रम {temp_c} डिग्री सेल्सियस, महसुस {feels_c}, आर्द्रता {humidity} प्रतिशत, हावा {wind_kmph} किलोमिटर प्रति घण्टा।",
                    "te": f"{city}లో ప్రస్తుతం వాతావరణం {desc}, ఉష్ణోగ్రత {temp_c} డిగ్రీ సెల్సియస్, ఫీల్స్ లైక్ {feels_c}, ఆర్ద్రత {humidity} శాతం, గాలి వేగం గంటకు {wind_kmph} కిలోమీటర్లు.",
                    "ml": f"{city}യിലെ ഇപ്പോഴത്തെ കാലാവസ്ഥ {desc} ആണ്, താപനില {temp_c} ഡിഗ്രി സെൽഷ്യസ്, അനുഭവപ്പെടുന്നത് {feels_c}, ഈർപ്പം {humidity} ശതമാനം, കാറ്റ് {wind_kmph} കിലോമീറ്റർ പ്രതി മണിക്കൂർ.",
                    "ar": f"الطقس الحالي في {city}: {desc}، الحرارة {temp_c} درجة مئوية، المحسوسة {feels_c}، الرطوبة {humidity} بالمئة، والرياح {wind_kmph} كيلومتر في الساعة.",
                }
                return replies.get(language, replies["en"])
            except Exception:
                fallback = {
                    "en": "I could not fetch weather details right now.",
                    "hi": "मैं अभी मौसम की जानकारी नहीं ला पाया।",
                    "ne": "अहिले मौसम जानकारी ल्याउन सकिन।",
                    "te": "ప్రస్తుతం వాతావరణ వివరాలు పొందలేకపోయాను.",
                    "ml": "ഇപ്പോൾ കാലാവസ്ഥ വിവരങ്ങൾ ലഭ്യമാക്കാനായില്ല.",
                    "ar": "تعذر جلب تفاصيل الطقس الآن.",
                }
                return fallback.get(language, fallback["en"])

        merged_thanks_markers = thanks_markers.get(language, set()) | thanks_markers["en"]
        if matches_markers(merged_thanks_markers):
            replies = {
                "en": "You are welcome.",
                "hi": "आपका स्वागत है।",
                "ne": "तपाईंलाई स्वागत छ।",
                "te": "మీకు స్వాగతం.",
                "ml": "സ്വാഗതം.",
                "ar": "على الرحب والسعة.",
            }
            return replies.get(language, replies["en"])

        merged_confirmation_markers = confirmation_markers.get(language, set()) | confirmation_markers["en"]
        if len(tokens) <= 3 and matches_markers(merged_confirmation_markers):
            replies = {
                "en": "Okay, done.",
                "hi": "ठीक है, हो गया।",
                "ne": "ठिक छ, भयो।",
                "te": "సరే, అయింది.",
                "ml": "ശരി, കഴിഞ്ഞു.",
                "ar": "حسنًا، تم.",
            }
            return replies.get(language, replies["en"])

        return None

    def process(
        self,
        text,
        stt_language_hint=None,
        stt_language_confidence=None,
        turn=None,
        input_mode=None,
        request_start=None,
        pipeline_metrics=None,
        media_source=None,
    ):
        """
        High-level process wrapper: Validates input and optionally creates a tracing turn.
        If owns_turn=True (caller didn't provide a turn), creates root turn and auto-flushes tracing.
        Otherwise delegates to _process() which does all the real work.
        
        This separation allows two call patterns:
        - Internal: process(text) → creates turn automatically
        - External: process(text, turn=existing_turn) → reuses existing trace context
        
        Parameters:
        - text: User input (voice transcription or text message)
        - stt_language_hint: Language hint from STT (e.g., "hi", "te"), None if text input
        - stt_language_confidence: Confidence [0-1] from STT engine
        - turn: Optional existing Langfuse turn object; if None, creates new one
        - input_mode: "voice" (from STT) or "text" (from console); auto-detected if not provided
        - request_start: Timestamp of request start; auto-set if not provided
        - pipeline_metrics: Dict of partial metrics (VAD, STT latencies) to merge into trace
        """

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
                        media_source,
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
                media_source,
            )

        if owns_turn:
            self.tracing.flush()

    def _process(
        self,
        text,
        stt_language_hint,
        stt_language_confidence,
        turn,
        input_mode,
        request_start,
        pipeline_metrics,
        media_source=None,
    ):
        """
        Core orchestration: Complete Tarz pipeline from user input to TTS playback.
        Implements all 7 processing stages with tracing at each step.
        
        Pipeline stages:
        1. Language detection & normalization: Identify language, script, and normalize text
        2. Intent routing: Classify intent (GENERAL, SEARCH, VISION, OCR)
        3. Task detection: Identify if user is asking for story/joke/travel/weather/etc
        4. Conversation memory: Retrieve historical context
        5. Optional vision: Capture image if VISION/OCR intent detected
        6. LLM → TTS pipeline: Stream LLM tokens → buffer to sentences → synthesize audio → playback
        7. Tracing & metrics: Log latencies, model engines, chunk counts to Langfuse
        
        Key behaviors:
        - Token-by-token printing: LLM tokens are printed to console as they stream in real time
        - Filler phrase plays while LLM generates (async): chat-GPT style responsiveness
        - Barge-in on ENTER: User can interrupt TTS playback and ask new question immediately
        - Task-lock mode: Terse follow-ups stay in prior task context ("flights" in travel mode)
        """

        print(f"\nYou: {text}")
        if config.DEBUG:
            print(f"[DEBUG APP] Input text: '{text}' | STT hint: {stt_language_hint} | Confidence: {stt_language_confidence}")

        # ── 1. Language detection & script classification ─────────────
        language_start = self.tracing.now()
        with self.tracing.start_step(
            "Language",
            input={
                "message": text,
                "stt_language_hint": stt_language_hint,
                "stt_language_confidence": stt_language_confidence,
            },
        ) as classification:
            script = detect_script(text)
            if config.DEBUG:
                print(f"[DEBUG APP] Script detected: {script}")
            language = detect_dominant_language(
                text,
                stt_hint=stt_language_hint,
                stt_confidence=stt_language_confidence,
                previous_language=self.llm.memory.get_language(),
            )
            if config.DEBUG:
                print(f"[DEBUG APP] Language detected: {language}")
            normalized_text = normalize_text(text, language)
            if config.DEBUG:
                print(f"[DEBUG APP] After normalization: '{normalized_text}' (changed: {normalized_text != text})")
            self.llm.memory.set_input_script(script)
            self.tracing.update_step(
                classification,
                output={
                    "language": language,
                    "script": script,
                    "normalized_message": normalized_text,
                },
                metadata={"latency_ms": self.tracing.elapsed_ms(language_start)},
            )

        language_latency_ms = self.tracing.elapsed_ms(language_start)

        if normalized_text != text:
            print(f"Normalized: {normalized_text}")

        language_name = config.SUPPORTED_LANGUAGES.get(language, language)
        print(f"Language: {language_name} | Script: {script}")

        # ── 2. Intent routing ────────────────────────────────────────
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
        is_social_turn = self._is_social_turn(normalized_text)
        detected_task = self._detect_task(normalized_text, intent)
        if detected_task is not None:
            self.current_task = detected_task
        elif is_social_turn:
            self.current_task = None
        elif self._is_terse_followup(normalized_text):
            # Keep prior task lock for terse follow-ups like "flights".
            pass
        else:
            self.current_task = None

        self.tracing.update_turn(turn, normalized_text, language, intent)

        if config.DEBUG:
            print(f"\n{'─'*55}")
            print(f"[DEBUG] Latest query   : {normalized_text}")
            print(f"[DEBUG] Intent         : {intent}")
            print(f"[DEBUG] Detected task  : {detected_task}")
            print(f"[DEBUG] Previous task  : {previous_task}")
            print(f"[DEBUG] Current task   : {self.current_task}")
            print(f"[DEBUG] Language       : {language}")
            print(f"{'─'*55}")

        # ── 3. Conversation memory ───────────────────────────────────
        memory_start = self.tracing.now()
        with self.tracing.start_step(
            "Memory",
            metadata={"type": "conversation-history", "rag_enabled": False},
        ) as memory:
            history = self.llm.memory.messages()
            self.tracing.update_step(
                memory,
                output={"message_count": len(history), "retrieved_chunks": 0},
                metadata={"latency_ms": self.tracing.elapsed_ms(memory_start)},
            )

        memory_latency_ms = self.tracing.elapsed_ms(memory_start)

        # ── 4. Optional uploaded media for vision intents ───────────
        image = None
        vision_requested = bool(media_source)
        sources = media_source if isinstance(media_source, (list, tuple)) else [media_source]
        ocr_requested = intent in ("OCR", "OCR_VISION") or any(
            isinstance(source, (str, Path)) and Path(source).suffix.lower() == ".pdf"
            for source in sources
        )
        extracted_ocr = ""
        if media_source and self.llm.supports_vision():
            media_start = self.tracing.now()
            with self.tracing.start_step(
                "Media",
                metadata={"source": str(media_source), "image_included_in_trace": False},
                as_type="tool",
            ) as media_step:
                print("Analyzing uploaded media...", flush=True)
                multimodal_inputs = self.vision.prepare_many(
                    list(sources),
                    use_ocr=ocr_requested,
                    ocr_lang=language,
                )
                image = [item.image for item in multimodal_inputs]
                extracted_ocr = "\n\n".join(
                    item.ocr_text for item in multimodal_inputs if item.ocr_text
                )
                self.tracing.update_step(
                    media_step,
                    output={
                        "source": [item.source for item in multimodal_inputs],
                        "pages": sum(item.pages for item in multimodal_inputs),
                        "media_count": len(multimodal_inputs),
                        "ocr_chars": len(extracted_ocr),
                    },
                    metadata={"latency_ms": self.tracing.elapsed_ms(media_start)},
                )
        elif media_source:
            print("Uploaded media analysis is unavailable with the configured text-only model.")

        # ── 5. Build filler phrase and effective LLM prompt ──────────
        context_preface = None
        # Creative requests should begin with the requested content immediately;
        # a canned bridge sounds like a duplicate answer before the LLM starts.
        preface_excluded_tasks = {"story", "joke", "poem"}
        if config.TTS_CONTEXT_PREFACE_ENABLED and self.current_task not in preface_excluded_tasks:
            context_preface = self._build_context_preface(normalized_text, language, self.current_task)
        effective_prompt = self._build_effective_prompt(
            normalized_text,
            previous_task=previous_task,
            detected_task=detected_task,
            language=language,
        )
        if extracted_ocr:
            effective_prompt = (
                f"{effective_prompt}\n\nExtracted text from the uploaded document:\n"
                f"{extracted_ocr}"
            )
        # After a spoken preface, prefer natural punctuation boundaries instead
        # of ultra-early single-word chunks so continuation sounds smoother.
        if config.DEBUG:
            print(f"[DEBUG] Bridge         : {context_preface!r}")
            print(f"[DEBUG] Effective prompt: {effective_prompt[:120]}..." if len(effective_prompt) > 120 else f"[DEBUG] Effective prompt: {effective_prompt}")
            hist = self.llm.memory.messages()
            print(f"[DEBUG] History turns  : {len(hist) // 2} turns")
            print(f"{'─'*55}\n")

        # Fast path: social-only turns should feel instant and should not repeat
        # an additional LLM greeting after a spoken bridge.
        if is_social_turn and detected_task is None and not vision_requested:
            social_reply = context_preface
            if not social_reply:
                defaults = {
                    "en": "Hi there. How can I help you?",
                    "hi": "नमस्ते। मैं आपकी कैसे मदद कर सकता हूँ?",
                    "ne": "नमस्ते। म कसरी सहयोग गर्न सक्छु?",
                    "te": "నమస్కారం. నేను ఎలా సహాయం చేయగలను?",
                    "ml": "നമസ്കാരം. ഞാൻ എങ്ങനെ സഹായിക്കാം?",
                    "ar": "مرحبًا، كيف يمكنني مساعدتك؟",
                }
                social_reply = defaults.get(language, defaults["en"])

            print(f"Tarz: {social_reply}")
            self.tts.speak(social_reply, language)
            self.llm.memory.set_language(language)
            self.llm.memory.add_user(normalized_text)
            self.llm.memory.add_assistant(social_reply)
            self.llm.last_metrics = {
                "first_token_ms": 0.0,
                "total_latency_ms": 0.0,
                "input_tokens": 0,
                "output_tokens": 0,
                "tokens_per_second": None,
                "first_request": False,
                "cold_start_ttft_ms": None,
            }

            if turn is not None:
                turn.update(output={"response": social_reply})
            return

        # Fast path: deterministic utility replies (time/thanks/confirmation)
        # skip LLM for lower latency and consistent behavior.
        utility_reply = None
        # Utility fast path should not override content-task flows like story/joke/poem.
        utility_allowed_tasks = {None, "weather"}
        if not vision_requested and detected_task in utility_allowed_tasks:
            utility_reply = self._utility_fast_response(normalized_text, language)
        if utility_reply:
            print(f"Tarz: {utility_reply}")
            self.tts.speak(utility_reply, language)
            self.llm.memory.set_language(language)
            self.llm.memory.add_user(normalized_text)
            self.llm.memory.add_assistant(utility_reply)
            self.llm.last_metrics = {
                "first_token_ms": 0.0,
                "total_latency_ms": 0.0,
                "input_tokens": 0,
                "output_tokens": 0,
                "tokens_per_second": None,
                "model": "utility-fastpath",
                "first_request": False,
                "cold_start_ttft_ms": None,
            }
            if turn is not None:
                turn.update(output={"response": utility_reply})
            return

        # Natural continuity mode: avoid ultra-early partial chunks that can
        # sound like broken words (especially in Indic scripts).
        # Exception: when a preface is spoken, queue the first LLM chunk early
        # so there is no audible gap after the filler.
        fast_start_enabled = False
        use_lead_words = False
        lead_words_count = config.TTS_LEAD_WORDS_COUNT
        first_chunk_min_chars = config.TTS_FIRST_CHUNK_MIN_CHARS
        first_chunk_min_words = config.TTS_FIRST_CHUNK_MIN_WORDS
        if context_preface:
            # Require enough words so the first chunk covers synthesis time of the next chunk.
            # Old values (chars=14, words=2) caused a 2-word "Delhi is" fragment then a long gap.
            first_chunk_min_chars = max(30, min(first_chunk_min_chars, 60))
            first_chunk_min_words = max(6, min(first_chunk_min_words, 10))
        # Always start TTS on the first sentence; first_chunk_min thresholds guard against tiny fragments.
        first_sentence_immediately = True
        first_word_immediately = False

        # Token queue for producer-consumer pattern between LLM and TTS threads.
        # Producer (LLM) puts tokens here; consumer (TTS sentence buffer) reads them.
        # Using sentinel object (stream_done) to signal end of stream (avoids None ambiguity).
        token_queue = queue.Queue()
        stream_done = object()  # Sentinel: not a real token; signals "stream finished".

        # Producer thread: Streams LLM output tokens into queue on background thread.
        # Allows LLM to run concurrently with TTS sentence buffering and audio playback.
        # Advantages: Tokens start flowing to TTS as soon as first word is generated (not after full response).
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
            except Exception as error:
                if config.DEBUG:
                    print(f"\n[LLM ERROR] {error}")
                error_replies = {
                    "en": "I could not finish analyzing that upload. Please try a shorter question.",
                    "hi": "मैं उस अपलोड का विश्लेषण पूरा नहीं कर पाया। कृपया छोटा सवाल पूछें।",
                    "ne": "मैले त्यो अपलोडको विश्लेषण पूरा गर्न सकिनँ। कृपया छोटो प्रश्न सोध्नुहोस्।",
                    "te": "ఆ అప్‌లోడ్‌ను పూర్తిగా విశ్లేషించలేకపోయాను. దయచేసి చిన్న ప్రశ్న అడగండి.",
                    "ml": "ആ അപ്‌ലോഡ് പൂർണ്ണമായി വിശകലനം ചെയ്യാൻ കഴിഞ്ഞില്ല. ദയവായി ചെറിയ ചോദ്യം ചോദിക്കുക.",
                    "ar": "لم أتمكن من تحليل الملف المرفوع بالكامل. يرجى طرح سؤال أقصر.",
                }
                token_queue.put(error_replies.get(language, error_replies["en"]))
            finally:
                token_queue.put(stream_done)

        producer_thread = threading.Thread(target=produce_tokens, daemon=True)
        producer_thread.start()

        # ── 6. LLM → sentence buffer → TTS pipeline ──────────────────
        # Capture how long the pre-TTS pipeline took so time_to_first_audio can be computed.
        pipeline_before_tts_ms = self.tracing.elapsed_ms(request_start)
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

        # Accumulator for raw LLM tokens (for memory/tracing, separate from TTS tokens).
        raw_tokens = []

        # Consumer function: Drains token_queue and yields tokens to sentence_stream().
        # Prints each token immediately as it streams, then prints a newline when done.
        # This is the "queued tokens" generator that bridges LLM output and TTS sentence buffering.
        def queued_tokens():
            first_token = True
            while True:
                token = token_queue.get()
                if token is stream_done:
                    print()  # Newline after streamed response ends.
                    return
                raw_tokens.append(token)
                if first_token:
                    print("Tarz: ", end="", flush=True)
                    first_token = False
                print(token, end="", flush=True)
                yield token

        # Sentence buffering: Groups tokens into sentences/chunks for TTS synthesis.
        # Parameters control chunking strategy:
        # - min/max_chars/words: Sentence size constraints (avoid too-short or too-long chunks)
        # - first_sentence_immediately: Send first sentence to TTS as soon as ready (doesn't wait for second)
        # - first_chunk_min_chars/words: Override min size for first sentence only (start TTS sooner)
        # - first_word_immediately: (Faster) Send single words initially, then group into sentences
        # - chunk_on_minor_punctuation: Split on commas/semicolons in addition to periods
        # - lead_words_immediate: Send first N words immediately without waiting for sentence boundary
        # - should_stop: Callback to check if barge-in/interrupt requested (stops buffering early)
        # Output: Generator yielding sentence strings to TTS
        sentences = sentence_stream(
            queued_tokens(),
            min_chars=config.TTS_MIN_CHARS,
            min_words=config.TTS_MIN_WORDS,
            max_chars=config.TTS_MAX_CHARS,
            max_words=config.TTS_MAX_WORDS,
            first_sentence_immediately=first_sentence_immediately,
            first_chunk_min_chars=first_chunk_min_chars,
            first_chunk_min_words=first_chunk_min_words,
            first_word_immediately=first_word_immediately,
            first_sentence_wordwise=config.TTS_FIRST_SENTENCE_WORDWISE,
            first_sentence_word_chunk_size=config.TTS_FIRST_SENTENCE_WORD_CHUNK_SIZE,
            chunk_on_minor_punctuation=config.TTS_CHUNK_ON_MINOR_PUNCTUATION,
            lead_words_immediate=use_lead_words,
            lead_words_count=lead_words_count,
            should_stop=self.tts.is_interrupted,
            min_force_chars=config.TTS_MIN_FORCE_CHARS,
            min_force_words=config.TTS_MIN_FORCE_WORDS,
        )

        # Accumulator for full TTS-synthesized response (sentences that actually got spoken).
        # Different from raw_tokens: these are buffered chunks sent to TTS, not raw LLM output.
        # Used for final console print, memory storage, and tracing.
        full_response = []

        # Injects context preface (filler phrase) at the start of TTS stream.
        # Enables ChatGPT-style behavior: "Okay, let me find that..." plays immediately while LLM generates.
        # Preface yields first, then all LLM sentences follow.
        # Checks tts.is_interrupted() to avoid speaking filler if user already barged in.
        def with_preface(stream):
            if context_preface and not self.tts.is_interrupted():
                print(f"Tarz: {context_preface}")  # Print filler to console immediately.
                yield context_preface  # Yield filler first so TTS speaks it before response.
            for item in stream:  # Then yield all LLM response sentences.
                yield item

        # Relay filter: Passes TTS sentences through while accumulating them in full_response.
        # Allows tracing and storage of what was actually synthesized (vs raw LLM tokens).
        # E.g., full_response = ["Sure, ", "let me help."] after relay finishes.
        def relay(stream):
            for sentence in stream:
                full_response.append(sentence)  # Track for memory/tracing.
                yield sentence  # Pass to TTS for synthesis.

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
            # TTS pipeline: Injects filler → sends LLM sentences through relay → synthesizes audio → streams to speaker.
            # This pipeline is streaming: first chunk starts synthesizing immediately (not after full response).
            # All happens concurrently: LLM generates → buffered to sentences → synthesized → played, all in parallel.
            self.tts.speak_stream(relay(with_preface(sentences)))
            # Wait for all audio to finish playback (blocks until speaker queue empty and audio ends).
            self.tts.wait_until_idle()
        finally:
            # Cleanup: Signal barge-in listener to stop polling keyboard.
            barge_in_stop.set()
            # Join listener thread with timeout (in case it's blocked on keyboard I/O).
            if barge_in_listener is not None:
                barge_in_listener.join(timeout=0.1)
            # Join LLM producer thread with timeout (should be done by now since we consumed all tokens).
            if producer_thread.is_alive():
                producer_thread.join(timeout=0.2)
        
        # ── 7. Post-turn tracing & metrics ───────────────────────────
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
            first_audio_latency_ms = tts_metrics.get("first_audio_latency_ms") or 0
            time_to_first_audio_ms = round(pipeline_before_tts_ms + first_audio_latency_ms, 2)
            llm_metrics = self.llm.last_metrics or {}
            timing = {
                # ── End-to-end ─────────────────────────────────────────────
                "total_task_ms": total_latency_ms,           # wall-clock from request received to playback complete
                "time_to_first_audio_ms": time_to_first_audio_ms,  # from request to first spoken word
                "pipeline_before_tts_ms": pipeline_before_tts_ms,  # language+router+memory+llm start overhead
                # ── Per-stage latencies ────────────────────────────────────
                "language_latency_ms": language_latency_ms,
                "router_latency_ms": self.tracing.elapsed_ms(router_start),
                "memory_latency_ms": memory_latency_ms,
                "llm": {
                    "first_token_ms": llm_metrics.get("first_token_ms"),         # time to first LLM token
                    "total_latency_ms": llm_metrics.get("total_latency_ms"),     # full generation time
                    "tokens_per_second": llm_metrics.get("tokens_per_second"),
                    "input_tokens": llm_metrics.get("input_tokens"),
                    "output_tokens": llm_metrics.get("output_tokens"),
                    "model": llm_metrics.get("model"),
                },
                "tts": {
                    "first_audio_latency_ms": first_audio_latency_ms,  # sentence buffer + synthesis
                    "synthesis_duration_ms": tts_metrics["synthesis_duration_ms"],
                    "playback_duration_ms": tts_metrics["playback_duration_ms"],
                    "queue_delay_ms": tts_metrics.get("queue_delay_ms"),
                    "audio_duration_ms": round(tts_metrics.get("audio_duration_ms") or 0, 2),
                    "chunk_count": tts_metrics.get("chunk_count"),
                },
            }
            timing.update(pipeline_metrics or {})
            # ── Percentage breakdown (what % of total each stage consumed) ─
            for name, duration in {
                "language_percent": language_latency_ms,
                "router_percent": timing["router_latency_ms"],
                "memory_percent": memory_latency_ms,
                "llm_percent": llm_metrics.get("total_latency_ms") or 0,
                "tts_percent": tts_metrics["synthesis_duration_ms"] or 0,
                "playback_percent": tts_metrics["playback_duration_ms"] or 0,
                "stt_percent": (pipeline_metrics or {}).get("stt_latency_ms") or 0,
            }.items():
                timing[name] = round(duration / max(total_latency_ms, 0.001) * 100, 2)
            self.tracing.update_turn_metrics(turn, timing)

    def run_voice(self):
        """
        Voice mode main loop: Continuously listen for speech, transcribe, process, and reply.
        Each iteration follows: VAD/listen → STT → Router → LLM → TTS → Play.
        User can press ENTER during playback to interrupt (barge-in) and ask next question.
        Exits when user says "back to menu" to return to mode selection screen.
        """
        print("Say 'back to menu' to choose a different mode.")
        pending_media_source = None

        while True:
            # Main voice interaction loop with exception handling for keyboard interrupt.
            try:
                # Timestamp the start of this interaction for latency tracking.
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
                                "model": config.WHISPER_SIZE,
                                "beam_size": config.WHISPER_BEAM_SIZE,
                                "compute_type": config.WHISPER_COMPUTE,
                                "audio_duration_seconds": round(audio_seconds, 3),
                                "audio_included_in_trace": False,
                                "partial_transcript_supported": False,
                            },
                        ) as transcription:
                            # Language hint strategy: Apply hints SELECTIVELY by language.
                            # 
                            # Hint eligibility:
                            # ✓ English (en) - Gets hints: Rare to code-switch FROM English
                            # ✓ Arabic (ar) - Gets hints: Standalone Arabic speakers, not Indic
                            # ✗ Hindi (hi), Telugu (te), Malayalam (ml) - NO hints: Code-switching common
                            #   (e.g., "Hi, namaste, कैसे हो?" is one sentence mixing languages)
                            # 
                            # Tradeoff: Indic without hints adds ~20ms latency but ensures 100% 
                            # language-switch robustness. English/Arabic keep ~20ms savings.
                            decode_hint = None
                            if config.STT_PREFER_PREVIOUS_LANGUAGE_HINT:
                                previous_language = self.llm.memory.get_language()
                                # Only apply hints for non-Indic languages (en, ar).
                                # Indic languages skip hints to enable language-switch detection.
                                if previous_language in ("en", "ar") and previous_language in config.STT_ALLOWED_LANGUAGES:
                                    decode_hint = previous_language

                            result = self.stt.transcribe(audio, language=decode_hint)
                            _engine = (
                                "indic-conformer"
                                if decode_hint in config.STT_INDIC_LANGUAGES and result.get("text")
                                else "faster-whisper"
                            )
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
                                metadata={"engine": _engine},
                            )

                        upload_request = result["text"].strip().lower() in {
                            "upload", "upload file", "upload image", "upload document",
                            "attach file", "attach an image", "add image",
                        }
                        if upload_request:
                            pending_media_source = self.media_chooser.choose()
                            if pending_media_source:
                                print("Uploaded media is ready. Ask your question.")
                            self.tracing.record_event(
                                turn,
                                "Media Selection Completed",
                                {"media_selected": bool(pending_media_source)},
                            )
                            continue

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
                            media_source=pending_media_source,
                        )
                        pending_media_source = None

                        self.tracing.record_event(
                            turn,
                            "Conversation Ended",
                            {"response_completed": True},
                        )

                # Let speaker echo decay before mic opens for the next turn.
                time.sleep(0.20)
                self.tracing.flush()

            except KeyboardInterrupt:
                self.tts.stop()
                print("\nVoice mode interrupted. Returning to mode selection...")
                self.tracing.flush()
                return

    def run_text(self):
        """
        Text mode main loop: Continuously prompt for user input from console, process, and reply.
        Each iteration: input() → Router → LLM → TTS → Play.
        User can type "/menu" or "back to menu" to exit and return to mode selection.
        
        Differences from run_voice():
        - No VAD/STT (skip audio recording and transcription)
        - No language hint (language auto-detected from text)
        - No barge-in (user controls timing via manual input)
        - Simpler error handling (no audio device/streaming issues)
        """
        while True:
            # Continuously prompt for user input.
            text = input("\nYou (/menu, 1=upload): ")

            media_source = None
            if text.strip().lower() in {"1", "/upload"}:
                media_source = self.media_chooser.choose()
                if not media_source:
                    continue
                text = input("Question about the uploaded media: ").strip()
                if not text:
                    continue

            if text.lower().startswith(("/image ", "/pdf ", "/media ")):
                command, _, remainder = text.partition(" ")
                media_source, separator, prompt = remainder.partition("|")
                media_source = media_source.strip()
                text = prompt.strip() if separator else "Describe or read this file."

            # Check if user requested menu exit.
            if self._return_to_menu_requested(text):
                print("Returning to mode selection...")
                return

            # Process text input (no language hint, no STT metrics).
            self.process(text, media_source=media_source or None)


if __name__ == "__main__":

    tarz = Tarz()
    tarz.speak_intro()

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


