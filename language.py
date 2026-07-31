"""
Language detection + Hinglish normalization.

This implements the "Majority Language Policy" and "TTS-only preprocessing"
behaviour described in README.md:

- Language and script are treated separately.
- Dominant conversation language is decided by majority of meaningful tokens
  (punctuation, numbers, and technical borrowed words are excluded).
- Roman Hindi (Hinglish) is normalized to Devanagari before the transcript
  reaches the LLM.
- Ties are broken in order: previous conversation language -> STT language
  hint -> user preference -> default language.

config.py already defines HINGLISH_TOKEN_MAP / HINGLISH_PHRASE_MAP /
HINDI_ROMAN_CORE_WORDS / TECHNICAL_BORROWED_WORDS for exactly this purpose;
this module is what actually uses them.
"""

import re

from config import (
    SUPPORTED_LANGUAGES,
    DEFAULT_LANGUAGE,
    USER_PREFERRED_LANGUAGE,
    HINDI_ROMAN_CORE_WORDS,
    TECHNICAL_BORROWED_WORDS,
    HINGLISH_TOKEN_MAP,
    HINGLISH_PHRASE_MAP,
)

_WORD_RE = re.compile(r"[a-zA-Z]+|[\u0900-\u097f]+")
_LATIN_WORD_RE = re.compile(r"[a-zA-Z]+")


def detect_script(text: str) -> str:
    has_dev = any("\u0900" <= c <= "\u097F" for c in text)
    has_lat = any(c.isascii() and c.isalpha() for c in text)

    if has_dev and has_lat:
        return "mixed"
    if has_dev:
        return "devanagari"
    if has_lat:
        return "latin"
    return "unknown"


def _tokens(text: str):
    return _WORD_RE.findall(text.lower())


def detect_dominant_language(text: str, stt_hint: str = None, previous_language: str = None) -> str:
    """
    Decide the dominant conversation language for this turn.

    Devanagari script is always Hindi. Otherwise, Roman-script input is
    scored against the Roman-Hindi core word list (technical borrowed words
    like "wifi" or "laptop" are excluded so they don't tip the balance).
    If the turn itself is ambiguous (e.g. very short, or no strong signal),
    fall back to previous language -> STT hint -> user preference -> default.
    """

    script = detect_script(text)

    if script in ("devanagari", "mixed"):
        return "hi"

    tokens = [t for t in _tokens(text) if t not in TECHNICAL_BORROWED_WORDS]

    if tokens:
        hindi_hits = sum(1 for t in tokens if t in HINDI_ROMAN_CORE_WORDS)
        ratio = hindi_hits / len(tokens)

        # Majority of meaningful tokens are Roman-Hindi -> treat turn as Hindi.
        if hindi_hits >= 2 or ratio >= 0.34:
            return "hi"

    for candidate in (previous_language, stt_hint, USER_PREFERRED_LANGUAGE):
        if candidate and candidate.lower() in SUPPORTED_LANGUAGES:
            return candidate.lower()

    return DEFAULT_LANGUAGE


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
