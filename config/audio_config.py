from pydantic_settings import BaseSettings, SettingsConfigDict


class AudioSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    SAMPLE_RATE: int = 16000  # mic recording rate; 16000 Hz is standard for all speech models
    CHANNELS: int = 1         # mono audio; speech models don't need stereo

    # Voice-activity detection — controls when recording starts and stops
    VAD_SILENCE_THRESHOLD: float = 0.025   # mic energy below this = silence; raised from 0.015 to sit above typical speaker echo (~0.02)
    VAD_SILENCE_DURATION: float = 0.5      # seconds of silence after speech before recording stops
    VAD_MIN_SPEECH_DURATION: float = 0.3   # clips shorter than this are dropped as accidental noise
    VAD_GRACE_PERIOD: float = 0.12         # extra buffer after silence so word endings aren't clipped
    VAD_MAX_RECORD_SECONDS: float = 30.0   # hard cap per recording; stops runaway capture

    MAX_PARTIAL_UPDATES_PER_SECOND: int = 2  # how often the live transcript refreshes on screen
    ENABLE_LIVE_TRANSCRIPT: bool = True      # show what the mic hears while you speak
    ENABLE_PARTIAL_TRANSCRIPTS: bool = True  # show rolling text before the final decode finishes


audio_settings = AudioSettings()
