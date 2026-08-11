from pydantic_settings import BaseSettings, SettingsConfigDict


class AudioSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    SAMPLE_RATE: int = 16000
    CHANNELS: int = 1

    # Voice-activity detection (energy-based endpoint detector)
    VAD_SILENCE_THRESHOLD: float = 0.015
    VAD_SILENCE_DURATION: float = 0.5
    VAD_MIN_SPEECH_DURATION: float = 0.3
    VAD_GRACE_PERIOD: float = 0.12
    VAD_MAX_RECORD_SECONDS: float = 30.0

    MAX_PARTIAL_UPDATES_PER_SECOND: int = 2
    ENABLE_LIVE_TRANSCRIPT: bool = True
    ENABLE_PARTIAL_TRANSCRIPTS: bool = True


audio_settings = AudioSettings()
