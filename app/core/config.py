from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Strongly-typed configuration with environment variable overrides.

    All values can be overridden via environment variables or a ``.env`` file.
    """

    REDIS_URL: str = "redis://localhost:6379"
    OLLAMA_URL: str = "http://localhost:11434"
    CACHE_THRESHOLD: float = 0.12
    VECTOR_DIMENSION: int = 384

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
