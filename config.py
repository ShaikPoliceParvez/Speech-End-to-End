# ========= MODELS =========

STT_MODEL = "whisper"      # whisper | parakeet
LLM_MODEL = "gemma3:4b" # qwen2.5:3b, gemma3:4b, qwen2.5:1.5b, gemma2:2b-instruct-q2_K is also available.
VOICE = "F2" 				 # M1 | M2 | F1 | F2 | F3 | F4 | F5 | F6

# ========= LLM RESPONSE LENGTH =========

# Upper bound on generated tokens. A lower cap improves responsiveness and
# keeps spoken answers concise for voice-first interactions.
LLM_MAX_TOKENS = 512
# Short social/acknowledgement turns do not need long generations.
LLM_SOCIAL_MAX_TOKENS = 96

# Warm the model once at startup so first live query has a lower TTFT.
LLM_WARMUP_ON_STARTUP = True

# History policy for response generation:
# - "strict": ignore prior chat unless explicitly requested or a follow-up lock prompt is present.
# - "full": always include conversation history.
LLM_HISTORY_MODE = "strict"
# Number of prior user+assistant turns to include when history is allowed.
LLM_HISTORY_TURNS = 2

# ========= LANGUAGE =========

# Keep this map extensible. Add a new language code/name here to enable it
# once STT + TTS can handle the language.
SUPPORTED_LANGUAGES = {
	"en": "English",
	"hi": "Hindi",
	"te": "Telugu",
	"ml": "Malayalam",
	"ar": "Arabic",
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
	"aap", "ap", "mai", "main", "mein", "mujhe", "muje", "mera", "meri", "mere", "liye", "mereliye", "keliye",
	"tum", "kaise", "kese", "kya", "hai", "hain", "ho", "kal", "naam",
	"kitne", "baje", "chal", "raha", "rahe", "batao", "sunao", "dekh",
	"jara", "zara", "kholo", "ek", "mein",
	"ab", "hindi", "baat", "karo", "milte", "bahut", "bohot", "ache", "acche",
	"acha", "achha", "accha", "badiya", "shukriya", "dhanyavad", "dhanyavaad",
	"bas", "bass", "hogaya", "hogayi", "hua", "hui", "nahi", "nahin", "haan",
	"han", "kyun", "kyu", "phir", "fir", "aur", "bhi",
	"ayyo", "aiyyo",
}

# Roman-Malayalam core words used to identify Malayalam written in Latin script.
MALAYALAM_ROMAN_CORE_WORDS = {
	"njan", "njaan", "ente", "ende", "entey", "ninte", "ninde", "avante", "avalude",
	"enthu", "enth", "entha", "enthaa", "engane", "evidey", "evide", "eppol", "eppo",
	"enthukond", "enthukondu",
	"undu", "und", "aanu", "anu", "aano", "aan", "illa", "ille",
	"nalla", "valare", "shari", "aavo", "manasilayi", "kupamilla",
	"veedu", "veetu", "peru", "neram", "ippol", "ippo", "innale", "naaley",
	"enikku", "ninakku", "nammal", "ningal", "avarkku",
	"malayalam", "malayalee", "malayali", "kerala", "keralam",
	"vanakkam", "namaskaram", "namaskar", "ayyo", "aiyyo", "ente amma",
}

# Roman-Telugu core words used to identify Telugu written in Latin script.
TELUGU_ROMAN_CORE_WORDS = {
	"nenu", "nuvvu", "meeru", "mee", "ni", "nee", "naaku", "naku", "mana", "maaku",
	"ela", "enti", "em", "undi", "unnaru", "unnavu", "unnava", "unnanu",
	"cheppu", "cheppandi", "chey", "cheyyi", "choodu", "choosi", "kanipistundi", "kavali", "avunu", "kaadu",
	"ledu", "namaskaram", "dhanyavadalu", "peru", "pairu", "evaru", "eppudu", "enduku", "katha", "vinali", "inko",
	"ekkada", "ikkada", "akkada", "telugu", "lo", "matladu", "matladandi", "idi", "chaduvu", "rasundi",
	"namaskaram", "namaste", "ayyo", "aiyyo", "sare", "sarey",
}

# Roman-Arabic (Arabizi) core words used to identify Arabic written in Latin script.
ARABIC_ROMAN_CORE_WORDS = {
	# Pronouns
	"ana", "inta", "inti", "inty", "huwwa", "huwa", "hiyya", "hiya", "nahnu", "intum", "hum",
	# Question words
	"ma", "man", "hal", "kaif", "kayf", "kaifa", "kaifak",
	"mata", "ayna", "wein", "wayn", "wen",
	"kam", "leish", "lesh", "meen", "qaddesh",
	# Prepositions / conjunctions
	"fi", "min", "ila", "maa", "ala", "li", "bi", "ind",
	"wa", "aw", "fa", "thumma", "lakin",
	# Negation / affirmation
	"la", "mush", "mesh", "mafi",
	"na3am", "aywa", "aywah", "aiwa",
	# Common verbs
	"kan", "yakun", "akun",
	"ureed", "urid", "beddi", "bedi",
	"atakallam", "takallam",
	"afham", "tafham",
	"asma", "asma3",
	"akhbirni", "qul", "rooh", "jeeb", "ta", "haki",
	# Names / identity
	"ismi", "ism", "ismak", "ismik",
	# Demonstratives / location
	"hadha", "hatha", "hadi", "hadhi",
	"hina", "hunak", "huna",
	"kullu", "baad", "qabl",
	# Greetings / social
	"shu", "shlonak", "shlonk", "shlonich", "shlon", "kif", "kifak", "kifik",
	"marhaba", "marhaban", "ahlan", "salam", "shukran", "afwan", "habibi", "habibti",
	"inshallah", "mashallah", "wallah", "bismillah",
	"tayeb", "tamam", "zain", "mzyan", "yalla", "khalas", "bass",
	"akhi", "ukhti",
	# Time
	"sabah", "sabahalkheir", "masaa", "masaalkheir",
	# Places
	"arabi", "arabic", "masr", "misr",
}

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
	"ml": {"malayalam", "malayalee", "malayali"},
	"ar": {"arabic", "arabi"},
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
	"mere": "मेरे",
	"liye": "लिए",
	"mereliye": "मेरे लिए",
	"keliye": "के लिए",
	"naam": "नाम",
	"kaise": "कैसे",
	"kese": "कैसे",
	"kya": "क्या",
	"ho": "हो",
	"hai": "है",
	"hain": "हैं",
	"kar": "कर",
	"karo": "करो",
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
# AFTER
WHISPER_HINDI_PROMPT = "हिंदी, देवनागरी"
WHISPER_TELUGU_PROMPT = "తెలుగు, తెలుగు లిపి"
WHISPER_MALAYALAM_PROMPT = "മലയാളം, മലയാളം ലിപി"
WHISPER_ARABIC_PROMPT = "العربية، اللغة العربية"

# Prefix forces the decoder to commit to the correct script before transcribing.
# Unlike hotwords (soft nudge), prefix is a hard constraint — the model MUST
# start with these characters and is then very likely to continue in that script.
WHISPER_HINDI_PREFIX = "मैं"
WHISPER_TELUGU_PREFIX = "నేను"
WHISPER_MALAYALAM_PREFIX = "ഞാൻ"
WHISPER_ARABIC_PREFIX = "أنا"

# Hotwords fed to faster-whisper's beam search during forced-language re-decode.
# They boost token log-probabilities for native-script words, nudging the
# decoder away from wrong-script outputs without the echo risk of initial_prompt.
WHISPER_HINDI_HOTWORDS = (
    "नमस्ते,आप,मैं,हम,वो,यह,क्या,कैसे,कहाँ,कब,क्यों,कौन,"
    "है,हैं,था,थे,होगा,होगी,करो,करें,बताओ,सुनो,देखो,जाओ,"
    "अच्छा,बहुत,नहीं,हाँ,ठीक,समझ,पानी,खाना,घर,काम,वक्त,"
    "दिन,रात,सुबह,शाम,आज,कल,अभी,जल्दी,धीरे,बड़ा,छोटा,"
    "मेरा,मेरी,तुम्हारा,उसका,हमारा,यहाँ,वहाँ,ऊपर,नीचे,"
    "मुझे,तुम्हें,उसे,हमें,किसे,क्या,सब,कुछ,बात,समय,लोग,"
    "सरकार,पैसा,बाज़ार,मोबाइल,गाना,फ़िल्म,दोस्त,परिवार"
)

WHISPER_TELUGU_HOTWORDS = (
    "నమస్కారం,నేను,నువ్వు,మీరు,అతను,ఆమె,మనం,వాళ్ళు,"
    "ఏమి,ఎలా,ఎక్కడ,ఎప్పుడు,ఎందుకు,ఎవరు,ఏది,ఏం,"
    "ఉంది,ఉన్నారు,అవుతుంది,చేస్తున్నాను,చేస్తున్నారు,వెళ్ళాను,"
    "చెప్పు,చెప్పండి,వినండి,చూడు,రండి,వెళ్ళండి,తెండి,"
    "బాగుంది,బాగా,చాలా,లేదు,అవును,సరే,అర్థమైంది,"
    "నీళ్ళు,తిండి,ఇల్లు,పని,సమయం,రోజు,రాత్రి,ఉదయం,"
    "నిన్న,రేపు,ఇప్పుడు,త్వరగా,నెమ్మదిగా,పెద్ద,చిన్న,"
    "నా,నీ,మీ,వాళ్ళ,ఇక్కడ,అక్కడ,పైన,కింద,లోపల,బయట,"
    "నాకు,నీకు,మీకు,అతనికి,ఆమెకు,అందరికి,ఏదైనా,అన్నీ,"
    "ప్రభుత్వం,డబ్బు,బజారు,మొబైల్,పాట,సినిమా,స్నేహితుడు,"
    "ఆరోగ్యం,చదువు,ఉద్యోగం,వ్యాపారం,ముఖ్యం,విషయం,సమస్య,"
    "తెలుగు,ఆంధ్ర,తెలంగాణ,హైదరాబాద్,విజయవాడ,విశాఖపట్నం"
)

WHISPER_MALAYALAM_HOTWORDS = (
    "നമസ്കാരം,ഞാൻ,നീ,അവൻ,അവൾ,നമ്മൾ,അവർ,"
    "എന്ത്,എങ്ങനെ,എവിടെ,എപ്പോൾ,എന്തുകൊണ്ട്,ആര്,ഏത്,"
    "ഉണ്ട്,ഉണ്ടായിരുന്നു,ഇല്ല,ആണ്,ആകുന്നു,ചെയ്യുന്നു,"
    "പറ,പറയൂ,കേൾ,കേൾക്കൂ,നോക്ക്,നോക്കൂ,വരൂ,"
    "നല്ലത്,വളരെ,ഇല്ല,ആണ്,ശരി,മനസ്സിലായി,കുഴപ്പമില്ല,"
    "വെള്ളം,കഴിക്ക്,വീട്,പുറത്ത്,ജോലി,സമയം,ദിവസം,രാത്രി,കാലത്ത്,"
    "ഇന്നലെ,നാളെ,ഇപ്പോൾ,വേഗം,പതുക്കെ,വലിയ,ചെറിയ,"
    "എന്റെ,നിന്റെ,അവന്റെ,അവളുടെ,ഇവിടെ,അവിടെ,മുകളിൽ,താഴെ,ഉള്ളിൽ,പുറത്ത്,"
    "എനിക്ക്,നിനക്ക്,അവനു,അവൾക്ക്,എല്ലാർക്കും,"
    "സർക്കാർ,പണം,കട,മൊബൈൽ,പാട്ട്,സിനിമ,സുഹൃത്ത്,കുടുംബം,"
    "മലയാളം,കേരളം,തിരുവനന്തപുരം,കൊച്ചി,കോഴിക്കോട്"
)

WHISPER_ARABIC_HOTWORDS = (
    "مرحبا,أنا,أنت,هو,هي,نحن,أنتم,هم,"
    "ماذا,كيف,أين,متى,لماذا,من,كم,"
    "نعم,لا,في,على,إلى,مع,من,ب,"
    "يكون,يكونون,أريد,أحتاج,قل,قول,اسمع,شوف,"
    "كويس,كثير,صغير,كبير,جميل,سريع,بطيء,"
    "ماء,طعام,بيت,عمل,وقت,يوم,ليل,صباح,"
    "اليوم,غدا,أمس,الآن,بسرعة,ببطء,"
    "معي,معك,معه,هنا,هناك,فوق,تحت,داخل,خارج,"
    "شكراً,عفواً,من فضلك,أهلاً,سهلاً"
)


# Tarz accepts speech only in these languages. Whisper initially auto-detects
# the language, then retries unsupported detections using the default language.
STT_ALLOWED_LANGUAGES = ("en", "hi", "te", "ml", "ar")

# Use previous conversation language as STT decode hint for faster follow-up
# turns in multilingual conversations.
STT_PREFER_PREVIOUS_LANGUAGE_HINT = False

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
# Emit the first spoken chunk earlier so voice starts quickly.
TTS_FIRST_CHUNK_MIN_CHARS = 8
TTS_FIRST_CHUNK_MIN_WORDS = 1
# If enabled, start speaking after the first stable word boundary.
TTS_FIRST_WORD_IMMEDIATELY = True
# If enabled, stream only the first sentence word-by-word, then continue with
# normal sentence buffering for smoother natural playback.
TTS_FIRST_SENTENCE_WORDWISE = False
# Group the first sentence into small word chunks (2-3) to reduce choppiness
# while preserving low startup latency.
TTS_FIRST_SENTENCE_WORD_CHUNK_SIZE = 2
# If enabled, split TTS chunks on minor punctuation too (comma/semicolon/colon)
# for more natural pacing and less choppy starts.
TTS_CHUNK_ON_MINOR_PUNCTUATION = True
# Start speaking with a tiny initial phrase (1-2 words) before waiting for
# punctuation boundaries. This lowers perceived latency without full word-wise
# choppy playback.
TTS_LEAD_WORDS_IMMEDIATE = True
TTS_LEAD_WORDS_COUNT = 2
# Optional short spoken preface before model stream (multilingual + context aware)
# e.g. "Okay, here is a story for you." then continue generated content.
TTS_CONTEXT_PREFACE_ENABLED = True
TTS_CONTEXT_PREFACE_RANDOM = True
# Filler pacing controls.
# - normal: default preface selection
# - slow: prefer longer, naturally paced preface variants
TTS_PREFACE_PACING = "slow"
TTS_PREFACE_MIN_WORDS = 6

LANGUAGE_PREFACES = {
	"en": {
		"greeting": ["Hello. It is really nice to hear from you.", "Hi there. Glad we are speaking right now."],
		"wellbeing_query": ["Thanks for asking.", "I am doing well, thank you."],
		"smalltalk": ["Glad to hear that.", "Nice!", "That sounds good."],
		"appreciation": ["Great!", "Awesome.", "Happy to hear that."],
		"generic": ["Sure, I can help with that.", "Alright, let me help you.", "Okay, let us go through it."],
		"answer": ["Sure, here is what I can tell you.", "Certainly, here is the answer.", "Alright, let me explain clearly."],
		"story": ["Sure, let us begin a story.", "Alright, here comes a story.", "Great, let us start the story."],
		"joke": ["Here is a joke for you.", "This one should make you smile."],
		"poem": ["Here's a poem.", "I hope you enjoy it."],
		"weather": ["Let me check.", "Here's the weather update."],
		"news": ["Here's what's happening.", "Let's take a look."],
		"camera": ["Opening the camera.", "Let me have a look."],
		"translation": ["Here's the translation."],
		"math": ["Let's calculate that."],
		"coding": ["Let's solve it.", "Here's the code."],
		"search": ["Looking into that."],
		"thanks": ["You're welcome!", "Happy to help!"],
		"goodbye": ["See you soon!", "Take care!"],
		"apology": ["No worries.", "That's alright."],
		"confirmation": ["Done.", "Consider it done."],
		"clarification": ["Could you clarify that?"],
		"fallback": ["Let me think about that."],
	},
	"hi": {
		"greeting": ["नमस्ते। आपसे फिर बात करके अच्छा लगा।", "नमस्कार। आपकी आवाज़ सुनकर अच्छा लगा।"],
		"wellbeing_query": ["पूछने के लिए धन्यवाद।", "मैं ठीक हूँ, धन्यवाद।"],
		"smalltalk": ["अच्छा लगा सुनकर।", "बहुत बढ़िया।", "यह अच्छा है।"],
		"appreciation": ["बहुत अच्छा!", "शानदार।", "यह सुनकर खुशी हुई।"],
		"generic": ["ज़रूर, मैं मदद करता हूँ।", "ठीक है, मैं मदद करता हूँ।"],
		"answer": ["ज़रूर, यह रहा जवाब।", "ठीक है, मैं साफ़-साफ़ बताता हूँ।"],
		"story": ["नमस्ते, कहानी शुरू करते हैं।"],
		"joke": ["यह रहा एक मज़ेदार चुटकुला।", "यह चुटकुला आपको पसंद आएगा।"],
		"poem": ["यह रही एक कविता।"],
		"weather": ["मौसम की जानकारी देखता हूँ।"],
		"news": ["यह रही ताज़ा जानकारी।"],
		"camera": ["कैमरा खोल रहा हूँ।"],
		"translation": ["यह रहा अनुवाद।"],
		"math": ["आइए गणना करते हैं।"],
		"coding": ["आइए इसे हल करते हैं।"],
		"search": ["देखता हूँ।"],
		"thanks": ["कोई बात नहीं!", "खुशी हुई मदद करके!"],
		"goodbye": ["फिर मिलेंगे!", "अपना ध्यान रखिए।"],
		"apology": ["कोई बात नहीं।"],
		"confirmation": ["हो गया।"],
		"clarification": ["क्या आप थोड़ा और स्पष्ट करेंगे?"],
		"fallback": ["एक क्षण सोचने दीजिए।"],
	},
	"te": {
		"greeting": ["నమస్కారం. మళ్లీ మాట్లాడటం చాలా ఆనందంగా ఉంది.", "హాయ్. మీతో మాట్లాడటం చాలా బాగుంది."],
		"wellbeing_query": ["అడిగినందుకు ధన్యవాదాలు.", "నేను బాగున్నాను, ధన్యవాదాలు."],
		"smalltalk": ["వినడానికి బాగుంది.", "చాలా బాగుంది.", "అది మంచి విషయం."],
		"appreciation": ["అద్భుతం!", "సూపర్.", "అది విని సంతోషంగా ఉంది."],
		"generic": ["సరే, నేను సహాయం చేస్తాను.", "అలాగే, దీనిని కలిసి చూసేద్దాం."],
		"answer": ["సరే, ఇది మీకు సమాధానం.", "అలాగే, స్పష్టంగా వివరిస్తాను."],
		"story": ["సరే, కథను ప్రారంభిద్దాం.", "బాగుంది, ఇప్పుడు ఒక కథ చెబుతాను."],
		"joke": ["మీ కోసం ఒక జోక్ ఉంది.", "ఇది వింటే నవ్వొస్తుంది."],
		"poem": ["ఇది ఒక కవిత.", "మీకు నచ్చుతుందని ఆశిస్తున్నాను."],
		"weather": ["సరే, వాతావరణ వివరాలు చెక్ చేస్తాను.", "ఇప్పుడు వాతావరణ సమాచారం చెబుతాను."],
		"news": ["సరే, తాజా వార్తలు చెబుతాను.", "ఇప్పుడు ఏముంది చూద్దాం."],
		"camera": ["సరే, కెమెరా తెరిచి చూస్తాను.", "అలాగే, ఒకసారి చూసి చెబుతాను."],
		"translation": ["సరే, ఇది అనువాదం.", "అలాగే, మీకు అనువదించి చెబుతాను."],
		"math": ["సరే, లెక్క చేద్దాం.", "అలాగే, దాన్ని దశలవారీగా గణిస్తాను."],
		"coding": ["సరే, దీన్ని పరిష్కరిద్దాం.", "అలాగే, కోడ్‌తో స్పష్టంగా చూపిస్తాను."],
		"search": ["సరే, దీనిని చూసి చెబుతాను.", "అలాగే, వివరాలు వెతికి చెబుతాను."],
		"thanks": ["స్వాగతం!", "సహాయం చేసినందుకు ఆనందంగా ఉంది!"],
		"goodbye": ["మళ్లీ కలుద్దాం!", "జాగ్రత్త!"],
		"apology": ["పర్లేదు."],
		"confirmation": ["అయింది.", "పూర్తయ్యింది."],
		"clarification": ["దయచేసి మరింత స్పష్టంగా చెబుతారా?"],
		"fallback": ["ఒక్కసారి ఆలోచిస్తాను."],
	},
	"ml": {
		"greeting": ["നമസ്കാരം. വീണ്ടും നിങ്ങളോടു സംസാരിക്കുന്നത് സന്തോഷമാണ്.", "ഹലോ. നിങ്ങളുമായി സംസാരിക്കാൻ കഴിഞ്ഞത് സന്തോഷം."],
		"wellbeing_query": ["ചോദിച്ചതിന് നന്ദി.", "എനിക്ക് സുഖമാണ്, നന്ദി."],
		"smalltalk": ["അത് കേട്ട് സന്തോഷം.", "നല്ലതാണ്.", "അത് നല്ല കാര്യമാണ്."],
		"appreciation": ["വളരെ നല്ലത്!", "അദ്ഭുതം.", "അത് കേട്ട് സന്തോഷം."],
		"generic": ["ശരി, ഞാൻ സഹായിക്കാം.", "അങ്ങനെ ചെയ്യാം, നമുക്ക് നോക്കാം."],
		"answer": ["ശരി, ഇതാണ് ഉത്തരം.", "അങ്ങനെ ചെയ്യാം, ഞാൻ വ്യക്തമായി വിശദീകരിക്കാം."],
		"story": ["ശരി, കഥ തുടങ്ങാം.", "ഇപ്പോൾ ഒരു കഥ പറയാം."],
		"joke": ["നിങ്ങൾക്കായി ഒരു തമാശ പറയാം.", "ഇത് കേട്ടാൽ നിങ്ങൾ ചിരിക്കും."],
		"poem": ["ഇതാണ് ഒരു കവിത.", "നിങ്ങൾക്ക് ഇഷ്ടപ്പെടുമെന്ന് കരുതുന്നു."],
		"weather": ["ശരി, കാലാവസ്ഥ വിവരങ്ങൾ നോക്കാം.", "ഇപ്പോൾ കാലാവസ്ഥ അപ്ഡേറ്റ് പറയുന്നു."],
		"news": ["ശരി, പുതിയ വാർത്തകൾ പറയുന്നു.", "ഇപ്പോൾ എന്തുണ്ട് എന്ന് നോക്കാം."],
		"camera": ["ശരി, ക്യാമറ തുറന്ന് നോക്കാം.", "അങ്ങനെ ചെയ്യാം, ഒന്ന് കണ്ടു പറയുന്നു."],
		"translation": ["ശരി, ഇതാണ് വിവർത്തനം.", "അങ്ങനെ ചെയ്യാം, ഞാൻ ഇത് വിവർത്തനം ചെയ്ത് പറയുന്നു."],
		"math": ["ശരി, നമുക്ക് കണക്കാക്കാം.", "അങ്ങനെ ചെയ്യാം, ഘട്ടം ഘട്ടമായി കണക്കാക്കാം."],
		"coding": ["ശരി, ഇത് പരിഹരിക്കാം.", "അങ്ങനെ ചെയ്യാം, കോഡോടെ വിശദീകരിക്കാം."],
		"search": ["ശരി, ഞാൻ അത് തിരഞ്ഞ് പറയുന്നു.", "അങ്ങനെ ചെയ്യാം, വിവരങ്ങൾ കണ്ടെത്തി പറയുന്നു."],
		"thanks": ["സ്വാഗതം!", "സഹായിക്കാൻ സന്തോഷം!"],
		"goodbye": ["വീണ്ടും കാണാം!", "ശ്രദ്ധിക്കുക!"],
		"apology": ["പ്രശ്നമില്ല."],
		"confirmation": ["കഴിഞ്ഞു.", "ചെയ്തു."],
		"clarification": ["കുറച്ച് കൂടുതൽ വ്യക്തമാക്കാമോ?"],
		"fallback": ["ഒരു നിമിഷം ആലോചിക്കാം."],
	},
	"ar": {
		"greeting": ["مرحبًا. يسعدني جدًا التحدث معك من جديد.", "أهلًا. من الجميل سماعك والتواصل معك الآن."],
		"wellbeing_query": ["شكرًا لسؤالك.", "أنا بخير، شكرًا لك."],
		"smalltalk": ["سعيد بسماع ذلك.", "هذا جميل.", "رائع."],
		"appreciation": ["ممتاز!", "رائع جدًا.", "سعيد بذلك."],
		"generic": ["حسنًا، سأساعدك في ذلك.", "تمام، دعنا نراجع هذا معًا."],
		"answer": ["حسنًا، إليك الإجابة.", "تمام، سأوضح لك الأمر بشكل واضح."],
		"story": ["حسنًا، لنبدأ القصة.", "رائع، سأحكي لك قصة الآن."],
		"joke": ["إليك نكتة لطيفة.", "هذه نكتة قد تعجبك."],
		"poem": ["إليك قصيدة.", "أتمنى أن تنال إعجابك."],
		"weather": ["حسنًا، سأتفقد حالة الطقس الآن.", "تمام، إليك تحديث الطقس."],
		"news": ["حسنًا، إليك آخر الأخبار.", "تمام، دعنا نرى ما الجديد."],
		"camera": ["حسنًا، سأفتح الكاميرا الآن.", "تمام، دعني أنظر وأخبرك."],
		"translation": ["حسنًا، إليك الترجمة.", "تمام، سأترجم ذلك لك بوضوح."],
		"math": ["حسنًا، دعنا نحسب ذلك.", "تمام، سأحسبها خطوة بخطوة."],
		"coding": ["حسنًا، دعنا نحل ذلك.", "تمام، سأشرح لك الحل مع الكود."],
		"search": ["حسنًا، سأبحث عن ذلك الآن.", "تمام، سأجمع لك التفاصيل."],
		"thanks": ["على الرحب والسعة!", "سعيد بمساعدتك!"],
		"goodbye": ["إلى اللقاء!", "اعتنِ بنفسك!"],
		"apology": ["لا بأس."],
		"confirmation": ["تم.", "تم التنفيذ."],
		"clarification": ["هل يمكنك التوضيح أكثر؟"],
		"fallback": ["دعني أفكر في ذلك."],
	},
}
TTS_PREFETCH_TEXT = 2
TTS_PREFETCH_AUDIO = 1

# TTS routing is configured in one place. To add a language, add an entry
# here; Piper languages also specify their downloaded model path.
TTS_LANGUAGE_BACKENDS = {
	"en": {"backend": "supertonic"},
	"hi": {"backend": "supertonic"},
	"te": {
		"backend": "piper",
		"model": "models/piper/te_IN-maya-medium.onnx",
	},
	"ml": {
		"backend": "piper",
		"model": "models/piper/ml_IN-arjun-medium.onnx",
	},
	"ar": {
		"backend": "piper",
		"model": "models/piper/ar_JO-kareem-medium.onnx",
	},
}

# ========= VOICE ACTIVITY DETECTION (VAD) =========

# Energy-based VAD for endpoint detection in the microphone.
# Silero VAD is used separately inside Whisper (vad_filter=True in stt.py)
# for audio cleaning before transcription — not for endpoint timing.
VAD_SILENCE_THRESHOLD = 0.01       # Normalised RMS energy below which a chunk is silent
VAD_SILENCE_DURATION  = 0.5        # Seconds of continuous silence to declare end-of-speech
VAD_MIN_SPEECH_DURATION = 0.3      # Minimum speech before endpoint is considered
VAD_GRACE_PERIOD = 0.12            # Extra silence buffer to protect short mid-sentence pauses
VAD_MAX_RECORD_SECONDS = 30.0      # Hard cap: force-stop if VAD never fires

# When False, skip expensive second-pass decode on confidence alone and only
# retry on stronger failure signals (unsupported language/script mismatch/etc.).
STT_RETRY_ON_LOW_CONFIDENCE = False

# Add custom entries to improve name pronunciation in TTS.
TTS_PRONUNCIATION_MAP = {
    "tarz": "taarz",
    "Parvez": "par vez",
	"బాగ ఉన్నాను": "బాగున్నాను",
	"నేను బాగ ఉన్నాను": "నేను బాగున్నాను",
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