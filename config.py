# ========= MODELS =========

STT_MODEL = "whisper"      # whisper | parakeet
LLM_MODEL = "qwen2.5:1.5b" # gemma3:4b, qwen2.5:1.5b, gemma2:2b-instruct-q2_K is also available.
VOICE = "M1" 				 # M1 | M2 | F1 | F2 | F3 | F4 | F5 | F6
#any OCR
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
	# "te": "Telugu",
	# "ta": "Tamil",
	# "kn": "Kannada",
	# "ml": "Malayalam",
	# "bn": "Bengali",
}

DEFAULT_LANGUAGE = "en"
USER_PREFERRED_LANGUAGE = None

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
	"ab", "hindi", "baat", "karo", "milte",
}

# Common Roman-Hindi words to normalize into Devanagari before LLM.
HINGLISH_TOKEN_MAP = {
	"aap": "आप",
	"ap": "आप",
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
WHISPER_SIZE = "base" #tiny | base | small | medium | large
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE = "int8"

# Beam search improves accuracy for non-English speech. Greedy (beam_size=1) is
# used for fast partials; the final transcript uses a wider beam.
WHISPER_BEAM_SIZE = 1

# Temperature fallback: if a decode is low-confidence / repetitive, Whisper
# retries at higher temperatures instead of emitting garbage.
WHISPER_TEMPERATURES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

# Quality gates used with the temperature fallback above.
WHISPER_COMPRESSION_RATIO_THRESHOLD = 2.4
WHISPER_LOG_PROB_THRESHOLD = -1.0
WHISPER_NO_SPEECH_THRESHOLD = 0.6

# Devanagari initial prompt biases the decoder toward correct Hindi script
# instead of romanising or dropping words. Applied only when language == "hi".
WHISPER_HINDI_PROMPT = "यह एक हिंदी बातचीत है। कृपया इसे देवनागरी लिपि में लिखें।"

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
TTS_PREFETCH_TEXT = 2
TTS_PREFETCH_AUDIO = 1

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