import pytest
from fastapi.testclient import TestClient

from app import main
from app.vector_store import VectorStore


class FakeModel:
    def get_sentence_embedding_dimension(self):
        return 3

    def encode(self, texts, normalize_embeddings=True):
        import numpy as np

        return np.asarray(
            [[1.0, 0.0, 0.0] for _ in texts],
            dtype=np.float32,
        )


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "app.vector_store.SentenceTransformer",
        lambda model_name: FakeModel(),
    )

    monkeypatch.setattr(main.settings, "data_dir", str(tmp_path))

    with TestClient(main.app) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_upload_document(client):
    response = client.post(
        "/documents",
        files={
            "file": (
                "security.txt",
                b"Security controls and access management.",
                "text/plain",
            )
        },
    )

    assert response.status_code == 201
    assert response.json()["filename"] == "security.txt"
    assert response.json()["chunks_indexed"] > 0


def test_duplicate_upload_is_rejected(client):
    first = client.post(
        "/documents",
        files={"file": ("security.txt", b"First content.", "text/plain")},
    )

    second = client.post(
        "/documents",
        files={"file": ("security.txt", b"Second content.", "text/plain")},
    )

    assert first.status_code == 201
    assert second.status_code == 409

    content = (
        main.store.root / "security.txt"
    ).read_text(encoding="utf-8")

    assert content == "First content."


def test_empty_upload_is_rejected(client):
    response = client.post(
        "/documents",
        files={"file": ("empty.txt", b"", "text/plain")},
    )

    assert response.status_code == 400
    assert not (main.store.root / "empty.txt").exists()


def test_unsupported_extension_is_rejected(client):
    response = client.post(
        "/documents",
        files={"file": ("payload.exe", b"not allowed", "application/octet-stream")},
    )

    assert response.status_code == 400


def test_path_traversal_filename_is_normalized(client):
    response = client.post(
        "/documents",
        files={"file": ("../../security.txt", b"Safe content.", "text/plain")},
    )

    assert response.status_code == 201
    assert response.json()["filename"] == "security.txt"
    assert (main.store.root / "security.txt").exists()


def test_delete_document(client):
    upload = client.post(
        "/documents",
        files={"file": ("security.txt", b"Security content.", "text/plain")},
    )

    assert upload.status_code == 201

    response = client.delete("/documents/security.txt")

    assert response.status_code == 200
    assert not (main.store.root / "security.txt").exists()
    assert main.store.documents() == []


def test_delete_missing_document(client):
    response = client.delete("/documents/missing.txt")

    assert response.status_code == 404


def test_list_documents(client):
    client.post(
        "/documents",
        files={"file": ("security.txt", b"Security content.", "text/plain")},
    )

    response = client.get("/documents")

    assert response.status_code == 200
    assert len(response.json()) == 1
    assert response.json()[0]["filename"] == "security.txt"


def test_batch_upload_two_files(client):
    response = client.post("/documents/batch", files=[
        ("files", ("alpha.txt", b"Alpha security policy.", "text/plain")),
        ("files", ("beta.md", b"# Beta\nIncident response.", "text/markdown")),
    ])
    assert response.status_code == 200
    body = response.json()
    assert (body["total"], body["successful"], body["failed"]) == (2, 2, 0)
    assert [x["status"] for x in body["results"]] == ["indexed", "indexed"]
    assert len(client.get("/documents").json()) == 2


def test_batch_partial_failure_and_duplicate(client):
    response = client.post("/documents/batch", files=[
        ("files", ("same.txt", b"First version.", "text/plain")),
        ("files", ("same.txt", b"Second version.", "text/plain")),
        ("files", ("bad.exe", b"Nope.", "application/octet-stream")),
    ])
    assert response.status_code == 200
    body = response.json()
    assert (body["successful"], body["failed"]) == (1, 2)
    assert [x["status"] for x in body["results"]] == ["indexed", "duplicate", "failed"]
    assert (main.store.root / "same.txt").read_bytes() == b"First version."
    assert not (main.store.root / "bad.exe").exists()


def test_batch_rejects_more_than_five_files(client):
    response = client.post("/documents/batch", files=[
        ("files", (f"doc{i}.txt", b"content", "text/plain")) for i in range(6)
    ])
    assert response.status_code == 400
    assert client.get("/documents").json() == []


def test_batch_empty_file_does_not_block_valid_file(client):
    response = client.post("/documents/batch", files=[
        ("files", ("empty.txt", b"", "text/plain")),
        ("files", ("valid.txt", b"Valid content.", "text/plain")),
    ])
    assert response.status_code == 200
    assert response.json()["successful"] == 1
    assert response.json()["failed"] == 1
    assert not (main.store.root / "empty.txt").exists()
    assert (main.store.root / "valid.txt").exists()


def test_batch_oversize_file_does_not_block_valid_file(client, monkeypatch):
    monkeypatch.setattr(main.settings, "max_upload_mb", 1)
    response = client.post("/documents/batch", files=[
        ("files", ("huge.txt", b"X" * (1024 * 1024 + 1), "text/plain")),
        ("files", ("small.txt", b"Small valid document.", "text/plain")),
    ])
    assert response.status_code == 200
    assert response.json()["successful"] == 1
    assert response.json()["failed"] == 1
    assert not (main.store.root / "huge.txt").exists()
    assert (main.store.root / "small.txt").exists()
