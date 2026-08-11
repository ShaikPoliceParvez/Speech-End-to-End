from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    LLM_MODEL: str = "gemma3:4b"
    LLM_ENGLISH_MODEL: str = "qwen2.5:1.5b"
    LLM_INDIC_MODEL: str = "gemma3:4b"
    LLM_MULTILINGUAL_MODEL: str = "gemma3:4b"
    LLM_CAMERA_MODEL: str = "gemma3:4b"
    # Comma-separated reference list shown in config so users can replace quickly.
    LLM_AVAILABLE_MODELS: str = "qwen2.5:1.5b,mashriram/sarvam-1,gemma3:4b,qwen2.5:3b,gemma2:2b,gemma2:2b-instruct-q2_K"
    LLM_MAX_TOKENS: int = 512
    LLM_SOCIAL_MAX_TOKENS: int = 96
    LLM_ROUTINE_MAX_TOKENS: int = 128
    LLM_WARMUP_ON_STARTUP: bool = True
    LLM_KEEP_ALIVE: str = "10m"
    LLM_TEMPERATURE: float = 0.2
    LLM_TOP_K: int = 30
    LLM_TOP_P: float = 0.85
    LLM_REPEAT_PENALTY: float = 1.15
    LLM_NUM_CTX: int = 4096
    LLM_INDIC_NUM_CTX: int = 2048  # smaller KV-cache for sarvam-1 to reduce TTFT
    # "strict" | "full"
    LLM_HISTORY_MODE: str = "full"
    LLM_HISTORY_TURNS: int = 6
    LLM_ROUTINE_HISTORY_TURNS: int = 1


llm_settings = LLMSettings()
