from functools import lru_cache
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # Application
    PROJECT_NAME: str = "High-Performance RAG Backend"
    API_V1_PREFIX: str = "/api/v1"
    DEBUG: bool = False
    API_KEY: Optional[str] = None  # Optional API Key for service-to-service auth

    # Google Gemini Configuration
    GEMINI_API_KEY: Optional[str] = None
    GEMINI_CHAT_MODEL: str = "gemini-3-flash-preview"
    GEMINI_EMBEDDING_MODEL: str = "gemini-embedding-001"
    EMBEDDING_DIMENSION: int = 768

    # Database (PostgreSQL + pgvector)
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/rag_db"
    DB_POOL_SIZE: int = 20
    DB_MAX_OVERFLOW: int = 10

    # Redis Cache
    REDIS_URL: str = "redis://localhost:6379/0"
    CACHE_TTL_SECONDS: int = 3600
    CACHE_ENABLED: bool = True

    # Hybrid Search RRF Weights
    RRF_K: int = 60
    DENSE_TOP_K: int = 10
    SPARSE_TOP_K: int = 10
    FINAL_TOP_K: int = 5


@lru_cache
def get_settings() -> Settings:
    return Settings()
