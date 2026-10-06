import pytest
from app.documents import chunk_text

def test_empty_text():
    assert chunk_text("   ") == []

def test_short_text():
    assert chunk_text("hello world", 20, 2) == ["hello world"]

def test_chunks_cover_text():
    text = "word " * 500
    chunks = chunk_text(text, 100, 20)
    assert len(chunks) > 1
    assert all(chunks)

def test_invalid_overlap():
    with pytest.raises(ValueError):
        chunk_text("hello", 10, 10)
