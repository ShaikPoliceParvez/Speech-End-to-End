from typing import Tuple
from pydantic_settings import BaseSettings, SettingsConfigDict


class STTSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    STT_MODEL: str = "whisper"  # whisper | parakeet
    WHISPER_SIZE: str = "small"  # tiny | base | small | medium | large
    WHISPER_DEVICE: str = "cpu"
    WHISPER_COMPUTE: str = "int8"
    WHISPER_BEAM_SIZE: int = 1
    WHISPER_TEMPERATURES: Tuple[float, ...] = (0.0, 0.2)
    WHISPER_COMPRESSION_RATIO_THRESHOLD: float = 2.4
    WHISPER_LOG_PROB_THRESHOLD: float = -1.0
    WHISPER_NO_SPEECH_THRESHOLD: float = 0.6
    WHISPER_LANGUAGE_CONFIDENCE_HIGH: float = 0.80

    # Script-biasing prompts per language
    WHISPER_HINDI_PROMPT: str = "हिंदी, देवनागरी"
    WHISPER_NEPALI_PROMPT: str = "नेपाली, देवनागरी"
    WHISPER_TELUGU_PROMPT: str = "తెలుగు, తెలుగు లిపి"
    WHISPER_MALAYALAM_PROMPT: str = "മലയാളം, മലയാളം ലിപി"
    WHISPER_ARABIC_PROMPT: str = "العربية، اللغة العربية"

    # Hard-constraint prefix tokens to commit the decoder to the correct script
    WHISPER_HINDI_PREFIX: str = "मैं"
    WHISPER_NEPALI_PREFIX: str = "म"
    WHISPER_TELUGU_PREFIX: str = "నేను"
    WHISPER_MALAYALAM_PREFIX: str = "ഞാൻ"
    WHISPER_ARABIC_PREFIX: str = "أنا"

    STT_ALLOWED_LANGUAGES: Tuple[str, ...] = ("en", "hi", "ne", "te", "ml", "ar")
    STT_PREFER_PREVIOUS_LANGUAGE_HINT: bool = True
    STT_RETRY_ON_LOW_CONFIDENCE: bool = False
    STT_INDIC_ASR_ENABLED: bool = True

    # Pseudo-streaming windows
    STT_MIN_PARTIAL_SECONDS: float = 2.0
    STT_PARTIAL_INTERVAL: float = 0.75
    STT_ROLLING_SECONDS: float = 3.5
    STT_OVERLAP_SECONDS: float = 0.8


stt_settings = STTSettings()

# ── Hotwords (beam-search token boosts) ─────────────────────────────────────
# Too large and script-rich for env-var overrides; kept as module constants.

WHISPER_HINDI_HOTWORDS = (
    "नमस्ते,आप,मैं,हम,वो,यह,क्या,कैसे,कहाँ,कब,क्यों,कौन,"
    "है,हैं,था,थे,होगा,होगी,करो,करें,बताओ,सुनो,देखो,जाओ,"
    "अच्छा,बहुत,नहीं,हाँ,ठीक,समझ,पानी,खाना,घर,काम,वक्त,"
    "दिन,रात,सुबह,शाम,आज,कल,अभी,जल्दी,धीरे,बड़ा,छोटा,"
    "मेरा,मेरी,तुम्हारा,उसका,हमारा,यहाँ,वहाँ,ऊपर,नीचे,"
    "मुझे,तुम्हें,उसे,हमें,किसे,क्या,सब,कुछ,बात,समय,लोग,"
    "सरकार,पैसा,बाज़ार,मोबाइल,गाना,फ़िल्म,दोस्त,परिवार"
)

WHISPER_NEPALI_HOTWORDS = (
    "नमस्ते,म,मलाई,तिमी,तपाईं,हामी,उहाँ,यो,त्यो,के,किन,कसरी,कहाँ,कहिले,को,कसको,"
    "छ,छु,छन्,हुन्छ,भयो,भएको,भन्नु,बोल्नु,सोध्नु,सुन्नु,हेर्नु,गर्नु,जानु,आउनु,"
    "राम्रो,धेरै,ठिक,हो,होइन,धन्यवाद,कृपया,मदत,कथा,ठट्टा,कविता,"
    "मौसम,समाचार,अनुवाद,हिसाब,कोड,कार्यक्रम,क्यामेरा,फोटो,कागज,बिल"
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

STT_INDIC_LANGUAGES = frozenset({"hi", "te", "ml"})
