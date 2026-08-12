from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class LanguageSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    DEFAULT_LANGUAGE: str = "en"              # fallback language when detection fails
    USER_PREFERRED_LANGUAGE: Optional[str] = None  # force a language e.g. "hi"; None = auto-detect


language_settings = LanguageSettings()

# ── Supported languages ──────────────────────────────────────────────────────
# Add a new entry here once STT + TTS support the language.

SUPPORTED_LANGUAGES: dict = {
    "en": "English",
    "hi": "Hindi",
    "ne": "Nepali",
    "te": "Telugu",
    "ml": "Malayalam",
    "ar": "Arabic",
}

# ── Script-agnostic vocabulary banks ────────────────────────────────────────
# Words shared across languages — excluded so they don't skew language detection.

TECHNICAL_BORROWED_WORDS: frozenset = frozenset({  # tech words that appear in every language
    "wifi", "wi-fi", "laptop", "internet", "browser", "file", "email",
    "meeting", "story", "login", "password", "weather", "chatgpt", "python",
    "router", "app", "camera", "photo", "image", "video", "joke",
    "slow", "fast", "net",
})

ENGLISH_CORE_WORDS: frozenset = frozenset({  # common English words that confirm English detection
    "hello", "hi", "thanks", "thank", "you", "wow", "nice", "good", "awesome",
    "fantastic", "tell", "me", "another", "story", "morning", "please",
})

AMBIGUOUS_LANGUAGE_TOKENS: frozenset = frozenset({  # words/tokens that could belong to any language — ignored during detection
    "ok", "okay", "hmm", "hm", "mm", "yes", "yeah", "yep", "no", "nope",
    # Place names that appear in multiple language vocabularies — not language signals.
    "dubai", "riyadh", "india", "mumbai", "delhi", "hyderabad", "bangalore",
    # Single letters are never reliable language indicators.
    "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m",
    "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z",
})

HINDI_ROMAN_CORE_WORDS: frozenset = frozenset({
    "aap", "ap", "mai", "main", "mein", "mujhe", "muje", "mera", "meri", "mere", "liye", "mereliye", "keliye",
    "mene", "maine", "liya", "suna", "sunliya", "sunli",
    "tum", "kaise", "kese", "kya", "hai", "hain", "ho", "kal", "naam",
    "kitne", "baje", "chal", "raha", "rahe", "batao", "sunao", "dekh",
    "jara", "zara", "kholo", "ek", "mein",
    "ab", "hindi", "baat", "karo", "milte", "bahut", "bohot", "ache", "acche",
    "acha", "achha", "accha", "badiya", "shukriya", "dhanyavad", "dhanyavaad",
    "bas", "bass", "hogaya", "hogayi", "hua", "hui", "nahi", "nahin", "haan",
    "han", "kyun", "kyu", "phir", "fir", "aur", "bhi",
    "ayyo", "aiyyo",
})

NEPALI_ROMAN_CORE_WORDS: frozenset = frozenset({
    "ma", "malai", "mero", "meri", "hami", "hamiro", "timilai", "timi", "tapai", "tapailai",
    "tapain", "tapaii", "tapaiko", "timro", "timilai", "tapaile", "uhale", "uhalai",
    "k", "ke", "kin", "kina", "kasari", "kasto", "kahile", "kaha", "kata", "ko", "kasko",
    "chha", "cha", "chu", "chhu", "chan", "chhan", "chaina", "chaina", "thiyo", "thie", "huncha", "hunchha", "hune",
    "garnu", "garna", "garera", "garne", "garda", "sodnu", "sodh", "bhanana", "bhan", "sunau",
    "hera", "hera na", "deu", "dinus", "lyau", "aau", "aaunu", "jaau", "janus", "aaja", "bholi", "hijo",
    "ramro", "dherai", "dhanyabad", "namaste", "sanchai", "thik", "thikai", "ho", "haina",
    "thikcha", "thik chha", "k cha", "k chha", "k xa", "sasto", "mahango", "chito", "dhilo",
    "katha", "hasaune", "joke", "mausam", "samachar", "anuwad", "ganit", "hisab", "program",
    "nepali", "नेपाल", "k garne", "k garchau", "ke bhayo",
})

MALAYALAM_ROMAN_CORE_WORDS: frozenset = frozenset({
    "njan", "njaan", "ente", "ende", "entey", "ninte", "ninde", "avante", "avalude",
    "enthu", "enth", "entha", "enthaa", "engane", "evidey", "evide", "eppol", "eppo",
    "enthukond", "enthukondu",
    "undu", "und", "aanu", "anu", "aano", "aan", "illa", "ille",
    "nalla", "valare", "shari", "aavo", "manasilayi", "kupamilla",
    "veedu", "veetu", "peru", "neram", "ippol", "ippo", "innale", "naaley",
    "enikku", "ninakku", "nammal", "ningal", "avarkku",
    "malayalam", "malayalee", "malayali", "kerala", "keralam",
    "oru", "katha", "parayu", "parayu", "parayoo", "ennikku", "cheyyu",
    "vanakkam", "namaskaram", "namaskar", "ayyo", "aiyyo", "ente amma",
})

TELUGU_ROMAN_CORE_WORDS: frozenset = frozenset({
    "nenu", "nuvvu", "meeru", "mee", "ni", "nee", "naaku", "naku", "mana", "maaku",
    "nuv", "sahayam", "chestava", "chesthava",
    "ela", "enti", "em", "undi", "unnaru", "unnavu", "unnava", "unnanu",
    "alaga", "oho", "cheppagalava", "cheppagalara", "oka", "chinna",
    "cheppu", "cheppandi", "cheppavu", "cheppava", "chey", "cheyyi", "choodu", "choosi", "kanipistundi", "kavali", "avunu", "kaadu",
    "balega", "balegaa", "balegara", "balegandi",
    "ledu", "namaskaram", "dhanyavadalu", "peru", "pairu", "evaru", "eppudu", "enduku", "katha", "vinali", "inko",
    "ekkada", "ikkada", "akkada", "telugu", "lo", "matladu", "matladandi", "idi", "chaduvu", "rasundi",
    "namaskaram", "namaste", "ayyo", "aiyyo", "sare", "sarey",
    "kachitanga", "kacchitanga", "cheddam", "chedham", "plan cheddam", "plan chedham",
})

ARABIC_ROMAN_CORE_WORDS: frozenset = frozenset({
    "ana", "inta", "inti", "inty", "huwwa", "huwa", "hiyya", "hiya", "nahnu", "intum", "hum",
    "ma", "man", "hal", "kaif", "kayf", "kaifa", "kaifak",
    "mata", "ayna", "wein", "wayn", "wen",
    "kam", "leish", "lesh", "meen", "qaddesh",
    "fi", "min", "ila", "maa", "ala", "li", "bi", "ind",
    "wa", "aw", "fa", "thumma", "lakin",
    "la", "mush", "mesh", "mafi",
    "na3am", "aywa", "aywah", "aiwa",
    "kan", "yakun", "akun",
    "ureed", "urid", "beddi", "bedi",
    "atakallam", "takallam",
    "afham", "tafham",
    "asma", "asma3",
    "akhbirni", "qul", "rooh", "jeeb", "ta", "haki",
    "ismi", "ism", "ismak", "ismik",
    "hadha", "hatha", "hadi", "hadhi",
    "hina", "hunak", "huna",
    "kullu", "baad", "qabl",
    "shu", "shlonak", "shlonk", "shlonich", "shlon", "kif", "kifak", "kifik",
    "marhaba", "marhaban", "ahlan", "salam", "shukran", "afwan", "habibi", "habibti",
    "inshallah", "mashallah", "wallah", "bismillah",
    "tayeb", "tamam", "zain", "mzyan", "yalla", "khalas", "bass",
    "akhi", "ukhti",
    "sabah", "sabahalkheir", "masaa", "masaalkheir",
    "arabi", "arabic", "masr", "misr",
})

# ── Language-switching vocabulary ────────────────────────────────────────────

LANGUAGE_SWITCH_TARGETS: dict = {
    "en": {"english"},
    "hi": {"hindi", "hindhi"},
    "ne": {"nepali", "nepalee", "nepalese"},
    "te": {"telugu", "telagum", "telugum"},
    "ml": {"malayalam", "malayalee", "malayali"},
    "ar": {"arabic", "arabi"},
}

LANGUAGE_SWITCH_ACTION_TOKENS: frozenset = frozenset({
    "speak", "talk", "switch", "language", "baat", "bolo", "bol", "mein", "me", "matladu", "bolnu", "bhan", "bhanne", "ma",
})

# ── Hinglish normalisation ────────────────────────────────────────────────────

HINGLISH_TOKEN_MAP: dict = {
    "aap": "आप", "ap": "आप", "app": "आप",
    "iss": "इस", "is": "इस", "me": "में",
    "likha": "लिखा", "he": "है", "kr": "कर",
    "mujhe": "मुझे", "muje": "मुझे", "tum": "तुम",
    "ek": "एक", "batao": "बताओ", "sunao": "सुनाओ",
    "dekh": "देख", "jara": "जरा", "zara": "ज़रा",
    "kholo": "खोलो", "mein": "मैं", "main": "मैं",
    "mera": "मेरा", "meri": "मेरी", "mere": "मेरे",
    "liye": "लिए", "mereliye": "मेरे लिए", "keliye": "के लिए",
    "naam": "नाम", "kaise": "कैसे", "kese": "कैसे",
    "kya": "क्या", "ho": "हो", "hai": "है", "hain": "हैं",
    "kar": "कर", "karo": "करो", "rahe": "रहे", "raha": "रहा",
    "chal": "चल", "kal": "कल", "milte": "मिलते",
    "bahut": "बहुत", "bohot": "बहुत",
    "ache": "अच्छे", "acche": "अच्छे",
    "acha": "अच्छा", "achha": "अच्छा", "accha": "अच्छा",
    "badiya": "बढ़िया", "shukriya": "शुक्रिया",
    "dhanyavad": "धन्यवाद", "dhanyavaad": "धन्यवाद",
    "bas": "बस", "bass": "बस",
    "hogaya": "हो गया", "hogayi": "हो गई",
    "hua": "हुआ", "hui": "हुई",
    "nahi": "नहीं", "nahin": "नहीं",
    "haan": "हाँ", "han": "हाँ",
    "kyun": "क्यों", "kyu": "क्यों",
    "phir": "फिर", "fir": "फिर",
    "aur": "और", "bhi": "भी",
    "kitne": "कितने", "baje": "बजे",
    "internet": "इंटरनेट", "net": "नेट",
    "slow": "स्लो", "laptop": "लैपटॉप",
    "bill": "बिल", "story": "स्टोरी",
    "meeting": "मीटिंग", "weather": "वेदर",
    "rahul": "राहुल",
}

HINGLISH_PHRASE_MAP: dict = {
    "aap kese ho": "आप कैसे हो",
    "aap kaise ho": "आप कैसे हो",
    "aap kaise hain": "आप कैसे हैं",
    "mera naam rahul hai": "मेरा नाम राहुल है",
    "kya kar rahe ho": "क्या कर रहे हो",
    "tum kya dekh rahe ho jara batao": "तुम क्या देख रहे हो जरा बताओ",
    "tum kya dekh rahe ho zara batao": "तुम क्या देख रहे हो ज़रा बताओ",
    "mujhe ek story sunao": "मुझे एक स्टोरी सुनाओ",
    "muje ek story sunao": "मुझे एक स्टोरी सुनाओ",
    "kal milte hain": "कल मिलते हैं",
    "muje ek story batao": "मुझे एक स्टोरी बताओ",
    "mujhe ek story batao": "मुझे एक स्टोरी बताओ",
    "mera laptop slow chal raha hai": "मेरा लैपटॉप स्लो चल रहा है",
    "please mujhe weather batao": "प्लीज मुझे वेदर बताओ",
    "mera internet slow hai": "मेरा इंटरनेट स्लो है",
    "kal meeting kitne baje hai": "कल मीटिंग कितने बजे है",
}
