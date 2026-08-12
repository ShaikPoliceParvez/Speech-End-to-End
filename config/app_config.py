from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    DEBUG: bool = False  # set True to print all internal events to the console

    # Camera
    CAMERA_INDEX: int = 0              # 0 = first/default camera; change if you have multiple
    CAPTURE_SAVE_IMAGES: bool = True   # save every captured photo to the captures/ folder
    CAPTURE_MAX_FILES: int = 20        # oldest file is deleted once this limit is reached

    # Conversation memory
    MAX_HISTORY: int = 10  # how many past turns the assistant remembers per session

    # Intent router
    ROUTER_CONFIDENCE_THRESHOLD: float = 0.60  # below this score the assistant asks the user to clarify
    ROUTER_CLARIFICATION_PROMPT: str = (
        "Do you want me to read the document or describe the scene?"
    )  # shown when the assistant can't tell if you want vision or OCR


app_settings = AppSettings()
