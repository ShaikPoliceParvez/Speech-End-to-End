from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Model selection — each route picks the best model for the job
    LLM_MODEL: str = "gemma3:4b"               # default model; used when no specific route matches
    LLM_ENGLISH_MODEL: str = "qwen2.5:1.5b"    # lighter/faster model for English-only turns
    LLM_INDIC_MODEL: str = "gemma3:4b"          # used for Hindi, Telugu, Malayalam, Nepali turns
    LLM_MULTILINGUAL_MODEL: str = "gemma3:4b"   # used when language switches mid-conversation
    LLM_CAMERA_MODEL: str = "gemma3:4b"         # vision-capable model used for uploaded images
    LLM_AVAILABLE_MODELS: str = "qwen2.5:1.5b,mashriram/sarvam-1,gemma3:4b,qwen2.5:3b,gemma2:2b,gemma2:2b-instruct-q2_K"  # reference list; not loaded automatically

    # Response length — lower = faster reply, shorter answer
    LLM_MAX_TOKENS: int = 512          # max tokens for a normal response
    LLM_SOCIAL_MAX_TOKENS: int = 96    # short cap for greetings and small talk
    LLM_ROUTINE_MAX_TOKENS: int = 128  # short cap for simple queries like time or thanks

    # Model lifecycle
    LLM_WARMUP_ON_STARTUP: bool = True  # send a dummy prompt at startup so first real reply is fast
    LLM_KEEP_ALIVE: str = "10m"         # how long Ollama keeps the model loaded after last use

    # Sampling — controls how creative or focused the reply is
    LLM_TEMPERATURE: float = 0.2        # 0 = deterministic, 1 = more random/creative
    LLM_TOP_K: int = 30                 # only consider the top-K most likely next tokens
    LLM_TOP_P: float = 0.85             # keep tokens until their combined probability hits 85%
    LLM_REPEAT_PENALTY: float = 1.15    # penalise repeating the same words; raise to reduce looping

    # Context window — bigger = more history but uses more memory
    LLM_NUM_CTX: int = 4096             # context window in tokens for main models
    LLM_INDIC_NUM_CTX: int = 2048       # smaller window for Indic model to reduce first-token latency

    # History sent to the model each turn
    LLM_HISTORY_MODE: str = "full"      # "full" = all turns; "strict" = system + latest only
    LLM_HISTORY_TURNS: int = 6          # number of past conversation turns included
    LLM_ROUTINE_HISTORY_TURNS: int = 1  # fewer turns for quick utility queries to save tokens


llm_settings = LLMSettings()
