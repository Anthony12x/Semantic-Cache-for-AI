from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application configuration loaded from environment variables and .env file."""

    REDIS_URL: str = "redis://localhost:6379"
    OLLAMA_URL: str = "http://localhost:11434"
    CACHE_THRESHOLD: float = 0.22
    CACHE_TTL_SECONDS: int = 86400
    VECTOR_DIMENSION: int = 768
    EMBEDDING_MODEL_NAME: str = "bge_base_en_v1_5"
    EMBEDDING_MODEL_ID: str = "BAAI/bge-base-en-v1.5"
    EMBEDDING_PROVIDER: str = "onnx"
    COLBERT_MODEL_ID: str = "colbert-ir/colbertv2.0"
    MAXSIM_THRESHOLD: float = 12.0
    OPENAI_API_KEY: str | None = None
    ANTHROPIC_API_KEY: str | None = None

    LLM_CIRCUIT_BREAKER_ERRORS: int = 5
    LLM_CIRCUIT_BREAKER_WINDOW: int = 60

    CORS_ALLOWED_ORIGINS: str = ""

    LOG_FORMAT: str = "text"
    LOG_LEVEL: str = "INFO"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
