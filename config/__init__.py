# Re-exports every name from the layer configs so that
#   from config import X
# keeps working across the whole codebase unchanged.

from config.llm_config import llm_settings
from config.stt_config import (
    stt_settings,
    WHISPER_HINDI_HOTWORDS,
    WHISPER_NEPALI_HOTWORDS,
    WHISPER_TELUGU_HOTWORDS,
    WHISPER_MALAYALAM_HOTWORDS,
    WHISPER_ARABIC_HOTWORDS,
    STT_INDIC_LANGUAGES,
)
from config.tts_config import (
    tts_settings,
    TTS_LANGUAGE_BACKENDS,
    TTS_PRONUNCIATION_MAP,
    LANGUAGE_PREFACES,
)
from config.language_config import (
    language_settings,
    SUPPORTED_LANGUAGES,
    TECHNICAL_BORROWED_WORDS,
    ENGLISH_CORE_WORDS,
    AMBIGUOUS_LANGUAGE_TOKENS,
    HINDI_ROMAN_CORE_WORDS,
    NEPALI_ROMAN_CORE_WORDS,
    MALAYALAM_ROMAN_CORE_WORDS,
    TELUGU_ROMAN_CORE_WORDS,
    ARABIC_ROMAN_CORE_WORDS,
    LANGUAGE_SWITCH_TARGETS,
    LANGUAGE_SWITCH_ACTION_TOKENS,
    HINGLISH_TOKEN_MAP,
    HINGLISH_PHRASE_MAP,
)
from config.audio_config import audio_settings
from config.app_config import app_settings

# ── Flat aliases (preserve backward-compatible bare names) ───────────────────

# LLM
LLM_MODEL                 = llm_settings.LLM_MODEL
LLM_ENGLISH_MODEL         = llm_settings.LLM_ENGLISH_MODEL
LLM_INDIC_MODEL           = llm_settings.LLM_INDIC_MODEL
LLM_MULTILINGUAL_MODEL    = llm_settings.LLM_MULTILINGUAL_MODEL
LLM_CAMERA_MODEL          = llm_settings.LLM_CAMERA_MODEL
LLM_AVAILABLE_MODELS      = llm_settings.LLM_AVAILABLE_MODELS
LLM_MAX_TOKENS            = llm_settings.LLM_MAX_TOKENS
LLM_SOCIAL_MAX_TOKENS     = llm_settings.LLM_SOCIAL_MAX_TOKENS
LLM_ROUTINE_MAX_TOKENS    = llm_settings.LLM_ROUTINE_MAX_TOKENS
LLM_WARMUP_ON_STARTUP     = llm_settings.LLM_WARMUP_ON_STARTUP
LLM_KEEP_ALIVE            = llm_settings.LLM_KEEP_ALIVE
LLM_TEMPERATURE           = llm_settings.LLM_TEMPERATURE
LLM_TOP_K                 = llm_settings.LLM_TOP_K
LLM_TOP_P                 = llm_settings.LLM_TOP_P
LLM_REPEAT_PENALTY        = llm_settings.LLM_REPEAT_PENALTY
LLM_NUM_CTX               = llm_settings.LLM_NUM_CTX
LLM_INDIC_NUM_CTX         = llm_settings.LLM_INDIC_NUM_CTX
LLM_HISTORY_MODE          = llm_settings.LLM_HISTORY_MODE
LLM_HISTORY_TURNS         = llm_settings.LLM_HISTORY_TURNS
LLM_ROUTINE_HISTORY_TURNS = llm_settings.LLM_ROUTINE_HISTORY_TURNS

# STT / Whisper
STT_MODEL                           = stt_settings.STT_MODEL
WHISPER_SIZE                        = stt_settings.WHISPER_SIZE
WHISPER_DEVICE                      = stt_settings.WHISPER_DEVICE
WHISPER_COMPUTE                     = stt_settings.WHISPER_COMPUTE
WHISPER_BEAM_SIZE                   = stt_settings.WHISPER_BEAM_SIZE
WHISPER_TEMPERATURES                = stt_settings.WHISPER_TEMPERATURES
WHISPER_COMPRESSION_RATIO_THRESHOLD = stt_settings.WHISPER_COMPRESSION_RATIO_THRESHOLD
WHISPER_LOG_PROB_THRESHOLD          = stt_settings.WHISPER_LOG_PROB_THRESHOLD
WHISPER_NO_SPEECH_THRESHOLD         = stt_settings.WHISPER_NO_SPEECH_THRESHOLD
WHISPER_LANGUAGE_CONFIDENCE_HIGH    = stt_settings.WHISPER_LANGUAGE_CONFIDENCE_HIGH
WHISPER_HINDI_PROMPT                = stt_settings.WHISPER_HINDI_PROMPT
WHISPER_NEPALI_PROMPT               = stt_settings.WHISPER_NEPALI_PROMPT
WHISPER_TELUGU_PROMPT               = stt_settings.WHISPER_TELUGU_PROMPT
WHISPER_MALAYALAM_PROMPT            = stt_settings.WHISPER_MALAYALAM_PROMPT
WHISPER_ARABIC_PROMPT               = stt_settings.WHISPER_ARABIC_PROMPT
WHISPER_HINDI_PREFIX                = stt_settings.WHISPER_HINDI_PREFIX
WHISPER_NEPALI_PREFIX               = stt_settings.WHISPER_NEPALI_PREFIX
WHISPER_TELUGU_PREFIX               = stt_settings.WHISPER_TELUGU_PREFIX
WHISPER_MALAYALAM_PREFIX            = stt_settings.WHISPER_MALAYALAM_PREFIX
WHISPER_ARABIC_PREFIX               = stt_settings.WHISPER_ARABIC_PREFIX
STT_ALLOWED_LANGUAGES               = stt_settings.STT_ALLOWED_LANGUAGES
STT_PREFER_PREVIOUS_LANGUAGE_HINT   = stt_settings.STT_PREFER_PREVIOUS_LANGUAGE_HINT
STT_RETRY_ON_LOW_CONFIDENCE         = stt_settings.STT_RETRY_ON_LOW_CONFIDENCE
STT_INDIC_ASR_ENABLED               = stt_settings.STT_INDIC_ASR_ENABLED
STT_MIN_PARTIAL_SECONDS             = stt_settings.STT_MIN_PARTIAL_SECONDS
STT_PARTIAL_INTERVAL                = stt_settings.STT_PARTIAL_INTERVAL
STT_ROLLING_SECONDS                 = stt_settings.STT_ROLLING_SECONDS
STT_OVERLAP_SECONDS                 = stt_settings.STT_OVERLAP_SECONDS

# TTS
VOICE                           = tts_settings.VOICE
TTS_SPEED                       = tts_settings.TTS_SPEED
TTS_MIN_CHARS                   = tts_settings.TTS_MIN_CHARS
TTS_MIN_WORDS                   = tts_settings.TTS_MIN_WORDS
TTS_MAX_CHARS                   = tts_settings.TTS_MAX_CHARS
TTS_MAX_WORDS                   = tts_settings.TTS_MAX_WORDS
TTS_FIRST_SENTENCE_IMMEDIATELY  = tts_settings.TTS_FIRST_SENTENCE_IMMEDIATELY
TTS_FIRST_CHUNK_MIN_CHARS       = tts_settings.TTS_FIRST_CHUNK_MIN_CHARS
TTS_FIRST_CHUNK_MIN_WORDS       = tts_settings.TTS_FIRST_CHUNK_MIN_WORDS
TTS_FIRST_WORD_IMMEDIATELY      = tts_settings.TTS_FIRST_WORD_IMMEDIATELY
TTS_FIRST_SENTENCE_WORDWISE     = tts_settings.TTS_FIRST_SENTENCE_WORDWISE
TTS_FIRST_SENTENCE_WORD_CHUNK_SIZE = tts_settings.TTS_FIRST_SENTENCE_WORD_CHUNK_SIZE
TTS_CHUNK_ON_MINOR_PUNCTUATION  = tts_settings.TTS_CHUNK_ON_MINOR_PUNCTUATION
TTS_SENTENCE_PAUSE_MS           = tts_settings.TTS_SENTENCE_PAUSE_MS
TTS_LEAD_WORDS_IMMEDIATE        = tts_settings.TTS_LEAD_WORDS_IMMEDIATE
TTS_LEAD_WORDS_COUNT            = tts_settings.TTS_LEAD_WORDS_COUNT
TTS_MIN_FORCE_CHARS             = tts_settings.TTS_MIN_FORCE_CHARS
TTS_MIN_FORCE_WORDS             = tts_settings.TTS_MIN_FORCE_WORDS
TTS_CONTEXT_PREFACE_ENABLED     = tts_settings.TTS_CONTEXT_PREFACE_ENABLED
TTS_CONTEXT_PREFACE_RANDOM      = tts_settings.TTS_CONTEXT_PREFACE_RANDOM
TTS_PREFACE_PACING              = tts_settings.TTS_PREFACE_PACING
TTS_PREFACE_MIN_WORDS           = tts_settings.TTS_PREFACE_MIN_WORDS
TTS_PREFETCH_TEXT               = tts_settings.TTS_PREFETCH_TEXT
TTS_PREFETCH_AUDIO              = tts_settings.TTS_PREFETCH_AUDIO

# Language
DEFAULT_LANGUAGE        = language_settings.DEFAULT_LANGUAGE
USER_PREFERRED_LANGUAGE = language_settings.USER_PREFERRED_LANGUAGE

# Audio / VAD
SAMPLE_RATE                  = audio_settings.SAMPLE_RATE
CHANNELS                     = audio_settings.CHANNELS
VAD_SILENCE_THRESHOLD        = audio_settings.VAD_SILENCE_THRESHOLD
VAD_SILENCE_DURATION         = audio_settings.VAD_SILENCE_DURATION
VAD_MIN_SPEECH_DURATION      = audio_settings.VAD_MIN_SPEECH_DURATION
VAD_GRACE_PERIOD             = audio_settings.VAD_GRACE_PERIOD
VAD_MAX_RECORD_SECONDS       = audio_settings.VAD_MAX_RECORD_SECONDS
MAX_PARTIAL_UPDATES_PER_SECOND = audio_settings.MAX_PARTIAL_UPDATES_PER_SECOND
ENABLE_LIVE_TRANSCRIPT       = audio_settings.ENABLE_LIVE_TRANSCRIPT
ENABLE_PARTIAL_TRANSCRIPTS   = audio_settings.ENABLE_PARTIAL_TRANSCRIPTS

# App
DEBUG                        = app_settings.DEBUG
MAX_HISTORY                  = app_settings.MAX_HISTORY
ROUTER_CONFIDENCE_THRESHOLD  = app_settings.ROUTER_CONFIDENCE_THRESHOLD
ROUTER_CLARIFICATION_PROMPT  = app_settings.ROUTER_CLARIFICATION_PROMPT
INTRO_ENABLED                = app_settings.INTRO_ENABLED
INTRO_LANGUAGE               = app_settings.INTRO_LANGUAGE

