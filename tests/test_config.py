import pytest
from pydantic import ValidationError

from app.config import Settings


def test_default_settings():
    settings = Settings(_env_file=None)

    assert settings.llm_model == "qwen2.5:3b"
    assert settings.max_upload_mb == 15
    assert settings.chunk_size == 900
    assert settings.chunk_overlap == 150


def test_invalid_upload_limit():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, max_upload_mb=0)


def test_invalid_chunk_overlap():
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            chunk_size=100,
            chunk_overlap=100,
        )


def test_invalid_similarity_threshold():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, similarity_threshold=1.5)


def test_reject_non_http_llm_url():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, llm_base_url="ftp://localhost:11434")


def test_reject_credentials_in_llm_url():
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            llm_base_url="http://user:password@localhost:11434",
        )
