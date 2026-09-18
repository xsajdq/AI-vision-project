from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="QUALITYSCOPE_")

    app_name: str = "QualityScope AI"
    max_upload_mb: int = 15
    history_limit: int = 25


settings = Settings()
