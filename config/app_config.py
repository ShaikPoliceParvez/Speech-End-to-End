from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    DEBUG: bool = False

    # Camera
    CAMERA_INDEX: int = 0
    CAPTURE_SAVE_IMAGES: bool = True
    CAPTURE_MAX_FILES: int = 20

    # Conversation memory
    MAX_HISTORY: int = 10

    # Intent router
    ROUTER_CONFIDENCE_THRESHOLD: float = 0.60
    ROUTER_CLARIFICATION_PROMPT: str = (
        "Do you want me to read the document or describe the scene?"
    )


app_settings = AppSettings()
