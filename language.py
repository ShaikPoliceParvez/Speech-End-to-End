"""
Language detection + Hinglish/Telglish handling.

This implements the "Majority Language Policy" and "TTS-only preprocessing"
behaviour described in README.md:

- Language and script are treated separately.
- Dominant conversation language is decided by majority of meaningful tokens
  (punctuation, numbers, and technical borrowed words are excluded).
- Roman Hindi (Hinglish) is normalized to Devanagari before the transcript
    reaches the LLM; Roman Telugu (Telglish) is identified as Telugu and passed
    through for the LLM to interpret.
- Previous conversation language is used only for short, ambiguous input.

config.py already defines the Roman Hindi/Telugu word maps and
TECHNICAL_BORROWED_WORDS for exactly this purpose;
this module is what actually uses them.
"""

import re

from config import (
    SUPPORTED_LANGUAGES,
    DEFAULT_LANGUAGE,
    USER_PREFERRED_LANGUAGE,
    HINDI_ROMAN_CORE_WORDS,
    TELUGU_ROMAN_CORE_WORDS,
    ENGLISH_CORE_WORDS,
    AMBIGUOUS_LANGUAGE_TOKENS,
    LANGUAGE_SWITCH_TARGETS,
    LANGUAGE_SWITCH_ACTION_TOKENS,
    TECHNICAL_BORROWED_WORDS,
    HINGLISH_TOKEN_MAP,
    HINGLISH_PHRASE_MAP,
    WHISPER_LANGUAGE_CONFIDENCE_HIGH,
)

_WORD_RE = re.compile(r"[a-zA-Z]+|[\u0900-\u097f]+|[\u0c00-\u0c7f]+")
_LATIN_WORD_RE = re.compile(r"[a-zA-Z]+")


def detect_script(text: str) -> str:
    """Classify all supported Unicode scripts before considering language."""
    has_dev = any("\u0900" <= c <= "\u097F" for c in text)
    has_telugu = any("\u0C00" <= c <= "\u0C7F" for c in text)
    has_arabic = any("\u0600" <= c <= "\u06FF" for c in text)
    has_lat = any(c.isascii() and c.isalpha() for c in text)

    script_count = sum((has_dev, has_telugu, has_arabic, has_lat))
    if script_count > 1:
        return "mixed"
    if has_telugu:
        return "telugu"
    if has_dev:
        return "devanagari"
    if has_arabic:
        return "arabic"
    if has_lat:
        return "latin"
    return "unknown"


def _tokens(text: str):
    return _WORD_RE.findall(text.lower())


def _explicit_language_switch(text: str):
    """Return a requested language when the user explicitly asks to switch."""
    tokens = set(_tokens(text))
    if not tokens.intersection(LANGUAGE_SWITCH_ACTION_TOKENS):
        return None

    for language, aliases in LANGUAGE_SWITCH_TARGETS.items():
        if tokens.intersection(aliases):
            return language

    return None


def _roman_language_scores(tokens):
    """Score Latin tokens against extensible Roman Hindi, Telugu, and English banks."""
    meaningful_tokens = [token for token in tokens if token not in TECHNICAL_BORROWED_WORDS]
    return {
        "hi": sum(token in HINDI_ROMAN_CORE_WORDS for token in meaningful_tokens),
        "te": sum(token in TELUGU_ROMAN_CORE_WORDS for token in meaningful_tokens),
        "en": sum(token in ENGLISH_CORE_WORDS for token in meaningful_tokens),
    }, len(meaningful_tokens)


def _strong_roman_language(scores, token_count):
    """Return a language only when its Roman vocabulary is decisive."""
    if not token_count:
        return None

    winner = max(scores, key=scores.get)
    winner_score = scores[winner]
    tied = list(scores.values()).count(winner_score) > 1
    if winner_score and not tied and (winner_score >= 2 or winner_score / token_count >= 0.5):
        return winner
    return None


def resolve_language(
    text: str,
    stt_hint: str = None,
    stt_confidence: float = None,
    previous_language: str = None,
) -> dict:
    """
    Resolve the language using script, Roman vocabulary, Whisper, then history.

    Unicode script and decisive Roman-language evidence are reliable transcript
    signals, so they take priority over a conflicting Whisper ID. A high-
    confidence, supported Whisper result is used only when those signals do
    not disagree. Conversation history is the final fallback for genuinely
    ambiguous acknowledgements.
    """
    requested_language = _explicit_language_switch(text)
    if requested_language is not None:
        return {"language": requested_language, "script": detect_script(text), "reason": "explicit_switch"}

    script = detect_script(text)
    # Native scripts are unambiguous; mixed text retains the dominant native
    # script rather than letting a few English borrowed words change language.
    if script == "telugu" or (script == "mixed" and any("\u0C00" <= char <= "\u0C7F" for char in text)):
        return {"language": "te", "script": script, "reason": "telugu_script"}
    if script == "devanagari" or (script == "mixed" and any("\u0900" <= char <= "\u097F" for char in text)):
        return {"language": "hi", "script": script, "reason": "devanagari_script"}

    latin_tokens = [t for t in _tokens(text) if t.isascii()]
    scores, token_count = _roman_language_scores(latin_tokens)
    roman_language = _strong_roman_language(scores, token_count)
    if roman_language:
        return {"language": roman_language, "script": script, "reason": "roman_vocabulary"}

    supported_hint = stt_hint.lower() if isinstance(stt_hint, str) else None
    if (
        supported_hint in SUPPORTED_LANGUAGES
        and stt_confidence is not None
        and stt_confidence >= WHISPER_LANGUAGE_CONFIDENCE_HIGH
        and script != "arabic"
    ):
        return {"language": supported_hint, "script": script, "reason": "high_confidence_whisper"}

    # Clear non-ambiguous Latin input is English even if the previous turn was
    # Hindi or Telugu; this prevents language state from becoming sticky.
    if latin_tokens and not all(token in AMBIGUOUS_LANGUAGE_TOKENS for token in latin_tokens):
        return {"language": "en", "script": script, "reason": "latin_fallback"}

    # Arabic/Persian is unsupported. STT retries it; this final fallback keeps
    # the rest of the pipeline on a supported language if no recovery occurs.
    for candidate in (supported_hint, previous_language, USER_PREFERRED_LANGUAGE):
        if candidate and candidate.lower() in SUPPORTED_LANGUAGES:
            return {"language": candidate.lower(), "script": script, "reason": "fallback"}

    return {"language": DEFAULT_LANGUAGE, "script": script, "reason": "default"}


def detect_dominant_language(
    text: str,
    stt_hint: str = None,
    previous_language: str = None,
    stt_confidence: float = None,
) -> str:
    """Backward-compatible language-only facade for :func:`resolve_language`."""
    return resolve_language(text, stt_hint, stt_confidence, previous_language)["language"]


def normalize_text(text: str, language: str) -> str:
    """
    Normalize Roman-Hindi (Hinglish) transcripts to Devanagari before they
    reach the LLM. Text that is already Devanagari, or isn't Hindi, passes
    through unchanged.
    """

    if language != "hi" or detect_script(text) != "latin":
        return text

    lowered = re.sub(r"\s+", " ", text.lower().strip())

    # Full-phrase normalization first, for common complete sentences.
    if lowered in HINGLISH_PHRASE_MAP:
        return HINGLISH_PHRASE_MAP[lowered]

    # Otherwise normalize token by token, preserving punctuation/spacing.
    def replace_token(match):
        word = match.group(0)
        return HINGLISH_TOKEN_MAP.get(word.lower(), word)

    return _LATIN_WORD_RE.sub(replace_token, text)
