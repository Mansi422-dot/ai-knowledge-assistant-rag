from urllib.parse import urlsplit

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_model: str = Field(
        default="qwen2.5:3b",
        min_length=1,
        max_length=150,
    )

    llm_base_url: str = Field(
        default="http://localhost:11434",
        min_length=1,
        max_length=500,
    )

    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        min_length=1,
        max_length=200,
    )

    data_dir: str = Field(default="./data", min_length=1, max_length=500)

    max_upload_mb: int = Field(default=15, ge=1, le=100)
    chunk_size: int = Field(default=900, ge=100, le=10000)
    chunk_overlap: int = Field(default=150, ge=0, le=5000)
    top_k: int = Field(default=4, ge=1, le=10)
    similarity_threshold: float = Field(default=0.25, ge=0.0, le=1.0)

    @field_validator("llm_base_url")
    @classmethod
    def validate_llm_base_url(cls, value: str) -> str:
        value = value.strip()
        parsed = urlsplit(value)

        if parsed.scheme not in {"http", "https"}:
            raise ValueError("LLM URL must use HTTP or HTTPS")

        if not parsed.hostname:
            raise ValueError("LLM URL must contain a hostname")

        if parsed.username or parsed.password:
            raise ValueError("Credentials must not be embedded in the LLM URL")

        if parsed.query or parsed.fragment:
            raise ValueError("LLM URL must not contain a query or fragment")

        try:
            if parsed.port is not None and not 1 <= parsed.port <= 65535:
                raise ValueError("Invalid LLM URL port")
        except ValueError as exc:
            raise ValueError("Invalid LLM URL port") from exc

        return value.rstrip("/")

    @model_validator(mode="after")
    def validate_chunk_settings(self):
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        return self


settings = Settings()
