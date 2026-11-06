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

def test_invalid_chunk_size():
    with pytest.raises(ValueError):
        chunk_text("hello world", 0, 0)


def test_negative_overlap():
    with pytest.raises(ValueError):
        chunk_text("hello world", 10, -1)


def test_zero_overlap():
    text = "word " * 20
    chunks = chunk_text(text, 20, 0)

    assert len(chunks) > 1
    assert all(chunks)


def test_markdown_extraction(tmp_path):
    from app.documents import extract_text

    path = tmp_path / "sample.md"
    path.write_text("# Security\nAccess control matters.", encoding="utf-8")

    result = extract_text(path)

    assert "Security" in result
    assert "Access control matters." in result


def test_unsupported_file(tmp_path):
    from app.documents import extract_text

    path = tmp_path / "sample.exe"
    path.write_text("not supported")

    with pytest.raises(ValueError):
        extract_text(path)
