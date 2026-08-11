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
    NEPALI_ROMAN_CORE_WORDS,
    TELUGU_ROMAN_CORE_WORDS,
    MALAYALAM_ROMAN_CORE_WORDS,
    ARABIC_ROMAN_CORE_WORDS,
    ENGLISH_CORE_WORDS,
    AMBIGUOUS_LANGUAGE_TOKENS,
    LANGUAGE_SWITCH_TARGETS,
    LANGUAGE_SWITCH_ACTION_TOKENS,
    TECHNICAL_BORROWED_WORDS,
    HINGLISH_TOKEN_MAP,
    HINGLISH_PHRASE_MAP,
    WHISPER_LANGUAGE_CONFIDENCE_HIGH,
)

_WORD_RE = re.compile(r"[a-zA-Z]+|[\u0900-\u097f]+|[\u0c00-\u0c7f]+|[\u0d00-\u0d7f]+|[\u0600-\u06ff]+")
_LATIN_WORD_RE = re.compile(r"[a-zA-Z]+")

# Common Roman vocabulary lives with the detector rather than in config. A
# token may appear in more than one set; each language receives that token's
# score and the majority of meaningful words decides the sentence language.
_ROMAN_LANGUAGE_VOCABULARY = {
    "en": ENGLISH_CORE_WORDS | {
        "cinema", "movie", "film", "what", "where", "when", "why", "how", "is", "are",
        "am", "was", "were", "do", "does", "did", "can", "could", "would", "will", "should",
        "want", "know", "about", "with", "for", "from", "this", "that", "these", "those",
        "i", "my", "we", "our", "he", "she", "they", "it", "help", "make", "give", "show",
        "explain", "read", "write", "watch", "listen", "find", "today", "tomorrow", "yesterday",
    },
    "hi": HINDI_ROMAN_CORE_WORDS | {
    # Pronouns
    "mai", "main", "me", "mein", "maii",
    "mujhe", "muje", "mujhko", "mjh", "mjhe",
    "ham", "hum", "hme", "hume", "humko",
    "aap", "ap", "aapko", "apko",
    "tum", "tm", "tumhe", "tumko", "tujhe",
    "tu", "tera", "teri", "tere",
    "mera", "meri", "mere",
    "hamara", "hamari", "hamare",
    "apna", "apni", "apne",

    # Question words
    "kya", "kya?", "ky", "kaise", "kese", "kaisa",
    "kab", "kahan", "kidhar", "kaun", "kon",
    "kis", "kisko", "jisko", "isko", "usko",
    "kitna", "kitni", "kitne", "kyun", "kyu",

    # Common verbs
    "hai", "h", "he", "hain",
    "ho", "hu", "hun", "hoo",
    "tha", "thi", "the",
    "kar", "karo", "karna", "karni",
    "karta", "karti", "karte",
    "kiya", "kiye", "ki",
    "bol", "bolo", "bolna",
    "bata", "batao", "batana",
    "sun", "suno", "sunao",
    "dekh", "dekho", "dekhna",
    "ja", "jao", "jana", "jaana",
    "aa", "aao", "aana",
    "aya", "aaya", "ayi", "aayi",
    "gaya", "gyi", "gayi", "gye",
    "chal", "chalo",
    "mil", "milte", "milna",
    "rakho", "rakhna",
    "le", "lelo", "lena",
    "de", "dedo", "dena",

    # Common words
    "ka", "ki", "ke",
    "ko", "se", "par", "pe",
    "liye", "keliye",
    "aur", "or",
    "lekin", "magar",
    "agar", "to", "phir", "fir",
    "kyunki", "isliye",

    # Time
    "ab", "abhi",
    "aaj", "kal", "parso",
    "baad", "pehle",
    "subah", "shaam", "raat",

    # Location
    "yaha", "yahaan", "yahan",
    "waha", "wahaan", "wahan",
    "ghar", "bahar",

    # State
    "acha", "achha", "accha",
    "acche", "ache",
    "badiya", "badhiya",
    "sahi", "galat",
    "thik", "theek", "thek",
    "nahi", "nahi", "nahin", "nai",
    "haan", "han", "ha", "haa",

    # Requests
    "please", "pls", "plz",
    "jara", "zara",
    "ek", "do",
    "madad", "help",
    "chahiye", "zarurat",

    # Greetings
    "namaste", "namaskar", 
    "shukriya", "thanks", "thankyou",
    "dhanyavad", "dhanyavaad",

    # Slang
    "bhai", "bro", "bros",
    "bhaiya", "bhayya",
    "yaar", "yar",
    "bhaii",
    "abe", "are", "arre",
    "oye", "oye",
    "boss",
    "mast", "masttt",
    "sahi", "jhakas",
    "lol",

    # Mixed English-Hindi
    "wifi", "internet",
    "phone", "mobile",
    "camera", "photo",
    "video", "browser",
    "login", "logout",
    "chatgpt", "python",
    "story", "meeting",
    "laptop", "computer",
    "file", "email",
    "message", "call",
    },
    "ne": NEPALI_ROMAN_CORE_WORDS | {
        "ma", "mero", "meri", "malai", "timro", "timi", "tapai", "tapain", "tapailai", "hami",
        "ke", "kina", "kin", "kasari", "kasto", "kati", "kahile", "kaha", "kata", "ko", "kasko",
        "cha", "chha", "chu", "chhu", "chan", "chhan", "thiyo", "hunchha", "huncha", "ho", "haina", "chaina",
        "garnu", "garna", "garda", "garchu", "garchau", "bhan", "bhannu", "sodh", "sunau", "hera",
        "aaja", "hijo", "bholi", "ramro", "dherai", "dhanyabad", "kripaya", "sanchai", "thikcha", "thik chha",
        "katha", "joke", "kavita", "mausam", "samachar", "anuwad", "hisab", "camera",
    },
    "te": TELUGU_ROMAN_CORE_WORDS | 
        {
        # Pronouns
        "nenu", "nen", "nuvvu", "nuvu", "nvu",
        "meeru", "meru", "manam", "memu",
        "atanu", "vadu", "vaadu", "vad", "vd",
        "ame", "aame",
        "ayana", "aayana",
        "vallu", "vaallu", "vall", "vl",

        "na", "naa", "ni", "nee", "niku", "niku", "nik",
        "naaku", "naku", "nak",
        "meeku", "miku", "mik",
        "meedi", "naadi", "vaadi",

        # Yes / No
        "avnu", "avunu", "avun", "avn",
        "kadu", "kadhu", "kaadu", "kad",
        "ledu", "ledhu", "led",

        # Questions
        "enti", "ent", "em", "emi", "emi ra",
        "ela", "elaa", "elaa", "elaa",
        "enduku", "endhuku", "endukuu", "endk",
        "ekkada", "ekkad", "ekkadiki",
        "eppudu", "epudu", "epd",
        "evaru", "evar", "evadu",
        "edi", "entha", "enni",

        # Common verbs
        "cheppu", "chepu", "chepp", "cheppandi",
        "chepta", "cheptanu", "cheptava",
        "cheppu ra", "cheppu anna",

        "chey", "cheyyi", "cheyi", "cheyy",
        "chesa", "chesanu", "chesadu",
        "chesava", "chesara", "chesindi",
        "chanipoyadu", "chanipoyindi",

        "matladu", "matladu", "matladuu",
        "matladandi", "matladava",
        "matladutunnava",

        "vellu", "velu", "vellu ra",
        "vellali", "vellandi",
        "vachanu", "vacha", "vachindi",
        "vastanu", "vasta", "vastunna",

        # State
        "undi", "undi ra", "unnanu",
        "unnava", "unnaru", "untundi",
        "untadu", "untanu",

        # Conversation
        "telusu", "telusa", "teliyadu",
        "teliyali", "vinu", "vinandi",
        "chudu", "choodu", "choosava",
        "kavali", "kaavali",
        "ivvu", "ivvandi",

        # Time
        "ippudu", "ipudu", "ippud",
        "appudu", "apudu",
        "ivala", "ivaala",
        "eroju", "repu", "ninna",

        # Connectors
        "inka", "kani", "aithe",
        "kabatti", "tarvata",
        "mundu", "kuda", "kooda",

        # Greetings
        "namaskaram", "namaste",
        "dhanyavadalu", "thanks",

        # Casual slang
        "anna", "ayya", "bro", "bros",
        "bava", "mama", "mowa",
        "rey", "ra", "raa", "orey",
        "oyi", "oi",
        "abba", "ammo", "ayyo",
        "super", "mast", "mastu",
        "chala", "chaala",
        "adbhutam", "adbutam", "adbhutham", "adbutham",
        "bagundi", "bagundhi",
        "bagunnava", "bagunnara",
        "bagoledu", "bagaledu",

        # Daily usage
        "tinava", "tinnava",
        "tinnara", "paduko",
        "nidra", "intiki",
        "bayata", "school",
        "college", "office",

        # Mixed English-Telugu
        "wifi", "internet",
        "phone", "mobile",
        "camera", "photo",
        "video", "browser",
        "login", "logout",
        "chatgpt", "python",
        "story", "meeting",
        "laptop", "file",
        "message", "call"
    },
    "ml": MALAYALAM_ROMAN_CORE_WORDS | {
        # Pronouns
        "njan", "njaan", "nee", "ni", "avan", "aval", "nammal", "avr", "avru",

        "njangal", "ningal", "ningalku", "avarkku",
        # Possessives — most common Manglish words
        "ente", "ende", "entey", "ninte", "ninde", "avante", "avalude", "nammalude",
        # Questions
        "enthu", "enth", "entha", "enthaa", "engane", "evidey", "evide", "eppol", "eppo",
        "enthukond", "enthukondu", "aar", "ethu", "etha",
        # Verbs / copula
        "undu", "und", "undo", "illa", "ille", "aanu", "anu", "aano", "aan", "aayirunnu",
        "agunnu", "agunn", "cheyyunnu", "cheyth", "cheythu", "poyyi", "vannu", "vann",
        # Commands / requests
        "para", "paraa", "parayoo", "parayou", "paranji", "paranjittu",
        "keel", "keelu", "keelunga", "nokkoo", "nokku", "varoo", "varuu",
        # Common words
        "peru", "peruu", "pera", "veedu", "veetu",
        "nalla", "nallath", "valare", "valara", "valareh", "illaa",
        "aavo", "aavoo", "shari", "shaari", "manasilayi", "manassilayi", "kupamilla",
        "vellam", "kazhikku", "veliye", "veliy", "velai", "neram", "divasam",
        "raatri", "raathri", "kalathu", "raavilae", "raavile",
        "innale", "innaleh", "naaley", "naale", "ippol", "ippo",
        "vegam", "vegath", "paadam", "valiya", "valiy", "cheriya", "cheriy",
        "ingey", "inge", "angey", "ange", "mukalil", "mukalile", "tazhey", "thazhey", "ullil",
        "enikku", "enikk", "ninakku", "ninak", "avannu", "avalku",
        "enthenkilum", "enthengilum", "ellarkum", "ellarum",
        "malayalam", "malayalee", "malayali", "kerala", "keralam",
        "oru", "katha", "parayu", "parayu", "parayoo", "ennikku", "cheyyu",
        "kochi", "trivandrum", "thiruvananthapuram", "kozhikode", "calicut",
        "vanakkam", "namaskaram", "namaskar", "ayyo", "aiyyo",
    },
    "ar": ARABIC_ROMAN_CORE_WORDS | {
        # Extra Arabizi variants not in core
        "maa3", "ind", "thumma",
        "na3am", "atakallam", "asma3", "akhbirni",
        "hadha", "hatha", "hadi", "hadhi", "kullu", "baad", "qabl",
        "gulf", "dubai", "riyadh", "arabi", "arabic", "masr", "misr",
    },
}

_NEPALI_DEVANAGARI_HINTS = frozenset({
    "तपाईं", "तिमी", "मलाई", "मेरो", "हामी", "किन", "कसरी", "कस्तो", "कहाँ", "कहिले",
    "छु", "छन्", "हुन्छ", "हुन्छ", "गर्नु", "गरेर", "भन्नु", "बोल्छु", "बोल्नु", "धन्यवाद", "कृपया", "नेपाली", "नेपालीमा",
})

_HINDI_DEVANAGARI_HINTS = frozenset({
    "आप", "तुम", "मुझे", "मेरा", "मेरी", "हम", "क्यों", "कैसे", "कहाँ", "कब",
    "हूँ", "हैं", "है", "करो", "करना", "बताओ", "धन्यवाद", "कृपया", "हिंदी",
})


def detect_script(text: str) -> str:
    """Classify all supported Unicode scripts before considering language."""
    has_dev = any("\u0900" <= c <= "\u097F" for c in text)
    has_telugu = any("\u0C00" <= c <= "\u0C7F" for c in text)
    has_malayalam = any("\u0D00" <= c <= "\u0D7F" for c in text)
    has_arabic = any("\u0600" <= c <= "\u06FF" for c in text)
    has_lat = any(c.isascii() and c.isalpha() for c in text)

    script_count = sum((has_dev, has_telugu, has_malayalam, has_arabic, has_lat))
    if script_count > 1:
        return "mixed"
    if has_telugu:
        return "telugu"
    if has_malayalam:
        return "malayalam"
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
    """Score each language and let the majority of meaningful words win."""
    meaningful_tokens = [token for token in tokens if token not in TECHNICAL_BORROWED_WORDS]
    return {
        language: sum(token in vocabulary for token in meaningful_tokens)
        for language, vocabulary in _ROMAN_LANGUAGE_VOCABULARY.items()
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


def _devanagari_language(text: str, stt_hint: str = None, previous_language: str = None):
    """Disambiguate Devanagari text between Hindi and Nepali using token hints.

    Safety policy: when ambiguous, prefer Hindi to avoid accidental Nepali
    switching because Hindi and Nepali share script and many function words.
    """
    tokens = set(re.findall(r"[\u0900-\u097f]+", (text or "").lower()))
    nepali_hits = len(tokens.intersection(_NEPALI_DEVANAGARI_HINTS))
    hindi_hits = len(tokens.intersection(_HINDI_DEVANAGARI_HINTS))

    # Require stronger Nepali evidence to classify as Nepali.
    if nepali_hits >= 2 and nepali_hits > hindi_hits:
        return "ne"

    # Any explicit Hindi evidence keeps Hindi.
    if hindi_hits > 0 and hindi_hits >= nepali_hits:
        return "hi"

    # If the hint says Nepali, still require at least one Nepali-specific token.
    hint = stt_hint.lower() if isinstance(stt_hint, str) else None
    if hint == "ne" and nepali_hits > 0 and hindi_hits == 0:
        return "ne"

    # Previous language can keep Nepali only when some Nepali evidence exists.
    previous = previous_language.lower() if isinstance(previous_language, str) else None
    if previous == "ne" and nepali_hits > 0 and hindi_hits == 0:
        return "ne"

    preferred = USER_PREFERRED_LANGUAGE.lower() if isinstance(USER_PREFERRED_LANGUAGE, str) else None
    if preferred == "ne" and nepali_hits > 0 and hindi_hits == 0:
        return "ne"

    return "hi"


def resolve_language(
    text: str,
    stt_hint: str = None,
    stt_confidence: float = None,
    previous_language: str = None,
) -> dict:
    """
    Resolve the language using script, Roman vocabulary, Whisper, then history.

    PRIORITY: High-confidence STT hint FIRST (≥0.80)
    Why? IndicConformer returns "adi chala bagundi" (Roman Telugu), but if it gets 
    normalized to Devanagari later, script detection sees "अदि चाला" and wrongly thinks Hindi.
    By trusting high-confidence hints upfront, we avoid this confusion.
    """
    supported_hint = stt_hint.lower() if isinstance(stt_hint, str) else None

    # ✓ PRIORITY 1: Explicit switch request — always honored, even over high-confidence STT hint.
    # Without this, an `ar` hint fed back to Whisper yields high-confidence Arabic even when the
    # user is clearly asking to switch to another language.
    requested_language = _explicit_language_switch(text)
    if requested_language is not None:
        return {"language": requested_language, "script": detect_script(text), "reason": "explicit_switch"}

    script = detect_script(text)
    latin_tokens = [t for t in _tokens(text) if t.isascii()]
    scores, token_count = _roman_language_scores(latin_tokens)
    roman_language = _strong_roman_language(scores, token_count)

    # ✓ PRIORITY 2: High-confidence STT hint unless strong script/roman evidence disagrees.
    # This preserves language switching while still allowing decisive Roman language
    # text (for example, Roman Telugu/Nepali) to beat a stale hint.
    if (
        supported_hint in SUPPORTED_LANGUAGES
        and stt_confidence is not None
        and stt_confidence >= WHISPER_LANGUAGE_CONFIDENCE_HIGH
    ):
        # Never let a confidence-only hint force Nepali on Devanagari input.
        # Re-run Hindi/Nepali disambiguation instead.
        if script == "devanagari":
            language = _devanagari_language(text, supported_hint, previous_language)
            reason = "devanagari_disambiguation_overrides_stt_hint"
            return {"language": language, "script": script, "reason": reason}
        if roman_language is not None and roman_language != supported_hint:
            return {"language": roman_language, "script": script, "reason": "roman_overrides_stt_hint"}
        return {"language": supported_hint, "script": script, "reason": "high_confidence_stt_hint"}

    # Native scripts are unambiguous; mixed text retains the dominant native
    # script rather than letting a few English borrowed words change language.
    if script == "telugu" or (script == "mixed" and any("\u0C00" <= char <= "\u0C7F" for char in text)):
        return {"language": "te", "script": script, "reason": "telugu_script"}
    if script == "devanagari" or (script == "mixed" and any("\u0900" <= char <= "\u097F" for char in text)):
        language = _devanagari_language(text, supported_hint, previous_language)
        reason = "devanagari_script_nepali" if language == "ne" else "devanagari_script"
        return {"language": language, "script": script, "reason": reason}
    if script == "malayalam" or (script == "mixed" and any("\u0D00" <= char <= "\u0D7F" for char in text)):
        return {"language": "ml", "script": script, "reason": "malayalam_script"}
    if script == "arabic" or (script == "mixed" and any("\u0600" <= char <= "\u06FF" for char in text)):
        return {"language": "ar", "script": script, "reason": "arabic_script"}

    if roman_language:
        return {"language": roman_language, "script": script, "reason": "roman_vocabulary"}

    # If any language vocabulary matched (even tied), prefer conversation history
    # over the English fallback — single shared words like greetings should stay
    # in the active language rather than resetting to English.
    any_lang_match = any(s > 0 for s in scores.values())
    if latin_tokens and not all(token in AMBIGUOUS_LANGUAGE_TOKENS for token in latin_tokens):
        if any_lang_match:
            for candidate in (supported_hint, previous_language, USER_PREFERRED_LANGUAGE):
                if candidate and candidate.lower() in SUPPORTED_LANGUAGES:
                    return {"language": candidate.lower(), "script": script, "reason": "shared_word_fallback"}
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
