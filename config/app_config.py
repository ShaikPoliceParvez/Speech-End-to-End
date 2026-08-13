from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    DEBUG: bool = False  # set True to print all internal events to the console

    # Conversation memory
    MAX_HISTORY: int = 10  # how many past turns the assistant remembers per session

    # Intent router
    ROUTER_CONFIDENCE_THRESHOLD: float = 0.60  # below this score the assistant asks the user to clarify
    ROUTER_CLARIFICATION_PROMPT: str = (
        "Do you want me to read the document or describe the scene?"
    )  # shown when the assistant can't tell if you want vision or OCR

    # Startup greeting spoken after all models are loaded
    INTRO_ENABLED: bool = True   # set False to skip the spoken greeting at startup
    INTRO_LANGUAGE: str = "en"   # language for the greeting: "en" | "hi" | "ne" | "te" | "ml" | "ar"


app_settings = AppSettings()
