from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    LLM_MODEL: str = "gemma3:4b"
    LLM_MAX_TOKENS: int = 512
    LLM_SOCIAL_MAX_TOKENS: int = 96
    LLM_WARMUP_ON_STARTUP: bool = True
    # "strict" | "full"
    LLM_HISTORY_MODE: str = "full"
    LLM_HISTORY_TURNS: int = 6


llm_settings = LLMSettings()
