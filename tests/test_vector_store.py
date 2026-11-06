import json

import numpy as np
import pytest

from app.vector_store import VectorStore


class FakeModel:
    def get_sentence_embedding_dimension(self):
        return 3

    def encode(self, texts, normalize_embeddings=True):
        vectors = []

        for text in texts:
            if "security" in text.lower():
                vector = [1.0, 0.0, 0.0]
            else:
                vector = [0.0, 1.0, 0.0]

            vectors.append(vector)

        return np.asarray(vectors, dtype=np.float32)


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "app.vector_store.SentenceTransformer",
        lambda model_name: FakeModel(),
    )

    return VectorStore(str(tmp_path), "test-model")


def test_add_document(store):
    count = store.add_document(
        "security.txt",
        ["security controls", "security monitoring"],
    )

    assert count == 2
    assert store.documents() == [
        {"filename": "security.txt", "chunks": 2}
    ]


def test_search_returns_relevant_results(store):
    store.add_document(
        "security.txt",
        ["security monitoring"],
    )

    results = store.search("security question")

    assert len(results) == 1
    assert results[0]["document"] == "security.txt"


def test_similarity_threshold_is_respected(store):
    store.add_document(
        "network.txt",
        ["network monitoring"],
    )

    results = store.search(
        "security question",
        min_score=0.9,
    )

    assert results == []


def test_delete_document(store):
    store.add_document("security.txt", ["security controls"])

    assert store.delete_document("security.txt") is True
    assert store.documents() == []


def test_missing_document_deletion(store):
    assert store.delete_document("missing.txt") is False


def test_failed_write_preserves_memory(store, monkeypatch):
    store.add_document("security.txt", ["security controls"])

    original_records = list(store.records)

    def fail_write(records):
        raise OSError("Simulated disk failure")

    monkeypatch.setattr(store, "_write_records", fail_write)

    with pytest.raises(OSError):
        store.add_document("network.txt", ["network monitoring"])

    assert store.records == original_records


def test_corrupted_index_fails_safely(tmp_path, monkeypatch):
    index = tmp_path / "index.json"
    index.write_text("{invalid json", encoding="utf-8")

    monkeypatch.setattr(
        "app.vector_store.SentenceTransformer",
        lambda model_name: FakeModel(),
    )

    with pytest.raises(RuntimeError):
        VectorStore(str(tmp_path), "test-model")

    assert index.read_text(encoding="utf-8") == "{invalid json"


def test_invalid_embedding_is_rejected(store, tmp_path):
    bad_records = [
        {
            "document": "bad.txt",
            "chunk": 0,
            "text": "bad embedding",
            "embedding": [float("nan"), 0.0, 0.0],
        }
    ]

    with pytest.raises(ValueError):
        store._validate_records(bad_records)


def test_invalid_filename_is_rejected(store):
    with pytest.raises(ValueError):
        store.add_document("../outside.txt", ["test"])


def test_atomic_write_leaves_valid_json(store, tmp_path):
    store.add_document("security.txt", ["security controls"])

    data = json.loads(
        (tmp_path / "index.json").read_text(encoding="utf-8")
    )

    assert len(data) == 1
    assert data[0]["document"] == "security.txt"


def test_embedding_count_mismatch_does_not_modify_index(store, monkeypatch):
    store.add_document("existing.txt", ["security controls"])
    original = list(store.records)

    class ShortModel(FakeModel):
        def encode(self, texts, normalize_embeddings=True):
            return np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)

    monkeypatch.setattr(store, "model", ShortModel())

    with pytest.raises(ValueError, match="unexpected number"):
        store.add_document("new.txt", ["one", "two"])

    assert store.records == original


def test_windows_path_filename_is_rejected(store):
    with pytest.raises(ValueError):
        store.add_document(r"..\\outside.txt", ["test"])


def test_invalid_search_parameters_are_rejected(store):
    with pytest.raises(ValueError):
        store.search("security", top_k=True)
    with pytest.raises(ValueError):
        store.search("security", min_score=float("nan"))
