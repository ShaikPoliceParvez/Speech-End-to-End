# ========= MODELS =========

STT_MODEL = "whisper"      # whisper | parakeet
LLM_MODEL = "gemma3:4b" # gemma3:4b, qwen2.5:1.5b, gemma2:2b-instruct-q2_K is also available.
VOICE = "M1" 				 # M1 | M2 | F1 | F2 | F3 | F4 | F5 | F6

# ========= LLM RESPONSE LENGTH =========

# Upper bound on generated tokens so long, detailed answers (stories,
# explanations, step-by-step help) are not truncated by Ollama's default cap.
# The model still keeps simple answers short via the adaptive-length prompt.
LLM_MAX_TOKENS = 1024

# ========= LANGUAGE =========

# Keep this map extensible. Add a new language code/name here to enable it
# once STT + TTS can handle the language.
SUPPORTED_LANGUAGES = {
	"en": "English",
	"hi": "Hindi",
	"te": "Telugu",
	# "ta": "Tamil",
	# "kn": "Kannada",
	# "ml": "Malayalam",
	# "bn": "Bengali",
}

DEFAULT_LANGUAGE = "en"
USER_PREFERRED_LANGUAGE = None

# Whisper's language ID is accepted directly only at this confidence when it
# agrees with the transcript's script and Roman-language evidence.
WHISPER_LANGUAGE_CONFIDENCE_HIGH = 0.80

# Words commonly mixed in speech that should not decide dominant language.
TECHNICAL_BORROWED_WORDS = {
	"wifi", "wi-fi", "laptop", "internet", "browser", "file", "email",
	"meeting", "story", "login", "password", "weather", "chatgpt", "python",
	"router", "app", "camera", "photo", "image", "video", "joke",
	"slow", "fast", "net",
}

# Roman-Hindi core words used for majority-language detection.
HINDI_ROMAN_CORE_WORDS = {
	"aap", "ap", "mai", "main", "mein", "mujhe", "muje", "mera", "meri",
	"tum", "kaise", "kese", "kya", "hai", "hain", "ho", "kal", "naam",
	"kitne", "baje", "chal", "raha", "rahe", "batao", "sunao", "dekh",
	"jara", "zara", "kholo", "ek", "mein",
	"ab", "hindi", "baat", "karo", "milte", "bahut", "bohot", "ache", "acche",
	"acha", "achha", "accha", "badiya", "shukriya", "dhanyavad", "dhanyavaad",
	"bas", "bass", "hogaya", "hogayi", "hua", "hui", "nahi", "nahin", "haan",
	"han", "kyun", "kyu", "phir", "fir", "aur", "bhi",
}

# Roman-Telugu core words used to identify Telugu written in Latin script.
TELUGU_ROMAN_CORE_WORDS = {
	"nenu", "nuvvu", "meeru", "mee", "ni", "nee", "naaku", "naku", "mana", "maaku",
	"ela", "enti", "em", "undi", "unnaru", "unnavu", "unnava", "unnanu",
	"cheppu", "cheppandi", "chey", "cheyyi", "choodu", "choosi", "kanipistundi", "kavali", "avunu", "kaadu",
	"ledu", "namaskaram", "dhanyavadalu", "peru", "pairu", "evaru", "eppudu", "enduku", "katha", "vinali", "inko",
	"ekkada", "ikkada", "akkada", "telugu", "lo", "matladu", "matladandi", "idi", "chaduvu", "rasundi",
}

# Clear English vocabulary prevents brief English turns such as "wow" or
# "thanks" from inheriting the prior conversation language.
ENGLISH_CORE_WORDS = {
	"hello", "hi", "thanks", "thank", "you", "wow", "nice", "good", "awesome",
	"fantastic", "tell", "me", "another", "story", "morning", "please",
}

# Only these short acknowledgements may inherit the previous conversation
# language. Every other clear Latin-script message is treated as English unless
# it matches the Roman Hindi or Telugu word banks above.
AMBIGUOUS_LANGUAGE_TOKENS = {
	"ok", "okay", "hmm", "hm", "mm", "yes", "yeah", "yep", "no", "nope",
}

# Explicit language requests override the language used to ask them. Add a
# target alias here when supporting a new language or common STT misspelling.
LANGUAGE_SWITCH_TARGETS = {
	"en": {"english"},
	"hi": {"hindi", "hindhi"},
	"te": {"telugu", "telagum", "telugum"},
}
LANGUAGE_SWITCH_ACTION_TOKENS = {
	"speak", "talk", "switch", "language", "baat", "bolo", "bol", "mein", "me", "matladu",
}

# Common Roman-Hindi words to normalize into Devanagari before LLM.
HINGLISH_TOKEN_MAP = {
	"aap": "आप",
	"ap": "आप",
	"app": "आप",
	"iss": "इस",
	"is": "इस",
	"me": "में",
	"likha": "लिखा",
	"he": "है",
	"kr": "कर",
	"mujhe": "मुझे",
	"muje": "मुझे",
	"tum": "तुम",
	"ek": "एक",
	"batao": "बताओ",
	"sunao": "सुनाओ",
	"dekh": "देख",
	"jara": "जरा",
	"zara": "ज़रा",
	"kholo": "खोलो",
	"mein": "मैं",
	"main": "मैं",
	"mera": "मेरा",
	"meri": "मेरी",
	"naam": "नाम",
	"kaise": "कैसे",
	"kese": "कैसे",
	"kya": "क्या",
	"ho": "हो",
	"hai": "है",
	"hain": "हैं",
	"kar": "कर",
	"rahe": "रहे",
	"raha": "रहा",
	"chal": "चल",
	"kal": "कल",
	"milte": "मिलते",
	"bahut": "बहुत",
	"bohot": "बहुत",
	"ache": "अच्छे",
	"acche": "अच्छे",
	"acha": "अच्छा",
	"achha": "अच्छा",
	"accha": "अच्छा",
	"badiya": "बढ़िया",
	"shukriya": "शुक्रिया",
	"dhanyavad": "धन्यवाद",
	"dhanyavaad": "धन्यवाद",
	"bas": "बस",
	"bass": "बस",
	"hogaya": "हो गया",
	"hogayi": "हो गई",
	"hua": "हुआ",
	"hui": "हुई",
	"nahi": "नहीं",
	"nahin": "नहीं",
	"haan": "हाँ",
	"han": "हाँ",
	"kyun": "क्यों",
	"kyu": "क्यों",
	"phir": "फिर",
	"fir": "फिर",
	"aur": "और",
	"bhi": "भी",
	"kitne": "कितने",
	"baje": "बजे",
	"internet": "इंटरनेट",
	"net": "नेट",
	"slow": "स्लो",
	"laptop": "लैपटॉप",
	"bill": "बिल",
	"story": "स्टोरी",
	"meeting": "मीटिंग",
	"weather": "वेदर",
	"rahul": "राहुल",
}

# Phrase-level normalization for common Hinglish patterns.
HINGLISH_PHRASE_MAP = {
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

# ========= WHISPER =========

# "small" is the realistic minimum for usable Hindi accuracy. The multilingual
# "base" model is notably weak on Devanagari; "small" runs fine on cpu/int8.
WHISPER_SIZE = "small" #tiny | base | small | medium | large
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE = "int8"

# Beam search improves accuracy for non-English speech. Greedy (beam_size=1) is
# used for fast partials; the final transcript uses a wider beam.
WHISPER_BEAM_SIZE = 5

# Temperature fallback: if a decode is low-confidence / repetitive, Whisper
# retries at higher temperatures instead of emitting garbage.
WHISPER_TEMPERATURES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

# Quality gates used with the temperature fallback above.
WHISPER_COMPRESSION_RATIO_THRESHOLD = 2.4
WHISPER_LOG_PROB_THRESHOLD = -1.0
WHISPER_NO_SPEECH_THRESHOLD = 0.6

# Devanagari initial prompt biases the decoder toward correct Hindi script
# instead of romanising or dropping words. Applied only when language == "hi".
# AFTER
WHISPER_HINDI_PROMPT = "हिंदी, देवनागरी"
WHISPER_TELUGU_PROMPT = "తెలుగు, తెలుగు లిపి"


# Tarz accepts speech only in these languages. Whisper initially auto-detects
# the language, then retries unsupported detections using the default language.
STT_ALLOWED_LANGUAGES = ("en", "hi", "te")

# Pseudo-streaming configuration
# First partial after 2.0s — ensures Whisper has real speech, not leading silence
STT_MIN_PARTIAL_SECONDS = 2.0

# Update partial transcript every 750ms
STT_PARTIAL_INTERVAL = 0.75

# Keep a longer rolling context
STT_ROLLING_SECONDS = 3.5

# Rolling overlap to stabilize hypotheses
STT_OVERLAP_SECONDS = 0.8

# ========= AUDIO =========

SAMPLE_RATE = 16000
CHANNELS = 1
TTS_SPEED = 0.92
TTS_MIN_CHARS = 72
TTS_MIN_WORDS = 12
TTS_MAX_CHARS = 220
TTS_MAX_WORDS = 36
# Send the first completed LLM sentence to TTS immediately. Later sentences
# still use the normal buffer to avoid choppy playback.
TTS_FIRST_SENTENCE_IMMEDIATELY = True
TTS_PREFETCH_TEXT = 2
TTS_PREFETCH_AUDIO = 1

# TTS routing is configured in one place. To add a language, add an entry
# here; Piper languages also specify their downloaded model path.
TTS_LANGUAGE_BACKENDS = {
	"en": {"backend": "supertonic"},
	"hi": {"backend": "supertonic"},
	"te": {
		"backend": "piper",
		"model": "models/piper/te_IN-venkatesh-medium.onnx",
	},
}

# ========= VOICE ACTIVITY DETECTION (VAD) =========

# Energy-based VAD for endpoint detection in the microphone.
# Silero VAD is used separately inside Whisper (vad_filter=True in stt.py)
# for audio cleaning before transcription — not for endpoint timing.
VAD_SILENCE_THRESHOLD = 0.01       # Normalised RMS energy below which a chunk is silent
VAD_SILENCE_DURATION  = 0.8        # Seconds of continuous silence to declare end-of-speech
VAD_MIN_SPEECH_DURATION = 0.3      # Minimum speech before endpoint is considered
VAD_GRACE_PERIOD = 0.2             # Extra silence buffer to protect short mid-sentence pauses
VAD_MAX_RECORD_SECONDS = 30.0      # Hard cap: force-stop if VAD never fires

# Add custom entries to improve name pronunciation in TTS.
TTS_PRONUNCIATION_MAP = {
    "tarz": "taarz",
    "Parvez": "par vez",
}

# ========= PERFORMANCE =========

MAX_PARTIAL_UPDATES_PER_SECOND = 2

ENABLE_LIVE_TRANSCRIPT = True

ENABLE_PARTIAL_TRANSCRIPTS = True

DEBUG = False

# ========= CAMERA =========

CAMERA_INDEX = 0
CAPTURE_SAVE_IMAGES = True
CAPTURE_MAX_FILES = 20

# ========= MEMORY =========

MAX_HISTORY = 10

# ========= ROUTER =========

ROUTER_CONFIDENCE_THRESHOLD = 0.60
ROUTER_CLARIFICATION_PROMPT = "Do you want me to read the document or describe the scene?"