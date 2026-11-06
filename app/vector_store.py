import json
import math
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer


def _is_safe_filename(value: str) -> bool:
    return (
        isinstance(value, str)
        and bool(value.strip())
        and value not in {".", ".."}
        and "/" not in value
        and "\\" not in value
        and Path(value).name == value
    )


class VectorStore:
    """Persistent, single-process cosine-similarity document index."""

    def __init__(self, data_dir: str, model_name: str):
        self.root = Path(data_dir).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.file = self.root / "index.json"
        self.model = SentenceTransformer(model_name)
        self.lock = threading.RLock()
        self.records: list[dict[str, Any]] = []

        if self.file.exists():
            self.records = self._load_records()

    def _model_dimension(self) -> int | None:
        getter = getattr(self.model, "get_embedding_dimension", None)
        if callable(getter):
            return getter()
        getter = getattr(self.model, "get_sentence_embedding_dimension", None)
        return getter() if callable(getter) else None

    def _validate_records(self, records: Any) -> list[dict[str, Any]]:
        if not isinstance(records, list):
            raise ValueError("Index must contain a JSON list.")

        validated: list[dict[str, Any]] = []
        dimensions: set[int] = set()

        for index, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(f"Invalid record at position {index}.")

            required = {"document", "chunk", "text", "embedding"}
            if not required.issubset(record):
                raise ValueError(f"Missing fields in record {index}.")

            document = record["document"]
            chunk = record["chunk"]
            content = record["text"]
            embedding = record["embedding"]

            if not _is_safe_filename(document):
                raise ValueError(f"Invalid document name in record {index}.")
            if not isinstance(chunk, int) or isinstance(chunk, bool) or chunk < 0:
                raise ValueError(f"Invalid chunk number in record {index}.")
            if not isinstance(content, str) or not content.strip():
                raise ValueError(f"Invalid text in record {index}.")
            if not isinstance(embedding, list) or not embedding:
                raise ValueError(f"Invalid embedding in record {index}.")
            if not all(
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(value)
                for value in embedding
            ):
                raise ValueError(f"Invalid embedding values in record {index}.")

            dimensions.add(len(embedding))
            validated.append({
                "document": document,
                "chunk": chunk,
                "text": content,
                "embedding": [float(value) for value in embedding],
            })

        if len(dimensions) > 1:
            raise ValueError("Index contains inconsistent embedding dimensions.")

        expected = self._model_dimension()
        if dimensions and expected is not None and next(iter(dimensions)) != expected:
            raise ValueError("Stored embeddings do not match the configured model.")

        return validated

    def _load_records(self) -> list[dict[str, Any]]:
        try:
            data = json.loads(self.file.read_text(encoding="utf-8"))
            return self._validate_records(data)
        except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
            raise RuntimeError(
                f"Could not safely load vector index '{self.file}'. "
                "The original index has not been modified. Restore a valid backup "
                "or correct the index before restarting."
            ) from exc

    def _write_records(self, records: list[dict[str, Any]]) -> None:
        validated = self._validate_records(records)
        temporary_path: Path | None = None

        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.root,
                prefix=".index-",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                json.dump(
                    validated,
                    temporary,
                    ensure_ascii=False,
                    indent=2,
                    allow_nan=False,
                )
                temporary.flush()
                os.fsync(temporary.fileno())

            os.replace(temporary_path, self.file)
            temporary_path = None
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

    def add_document(self, filename: str, chunks: list[str]) -> int:
        if not _is_safe_filename(filename):
            raise ValueError("Invalid document filename.")
        if not isinstance(chunks, list) or not chunks or not all(
            isinstance(chunk, str) and chunk.strip() for chunk in chunks
        ):
            raise ValueError("Document must contain non-empty text chunks.")

        vectors = np.asarray(
            self.model.encode(chunks, normalize_embeddings=True),
            dtype=np.float32,
        )

        if vectors.ndim == 1 and len(chunks) == 1:
            vectors = vectors.reshape(1, -1)
        if vectors.ndim != 2 or vectors.shape[0] != len(chunks):
            raise ValueError(
                "Embedding model returned an unexpected number of vectors."
            )
        if vectors.shape[1] == 0 or not np.isfinite(vectors).all():
            raise ValueError("Embedding model returned invalid vector values.")

        expected = self._model_dimension()
        if expected is not None and vectors.shape[1] != expected:
            raise ValueError("Embedding model returned an unexpected vector size.")

        new_records = [
            {
                "document": filename,
                "chunk": index,
                "text": chunk,
                "embedding": vector.astype(float).tolist(),
            }
            for index, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True))
        ]

        with self.lock:
            updated = [
                record for record in self.records
                if record["document"] != filename
            ]
            updated.extend(new_records)
            self._write_records(updated)
            self.records = updated

        return len(new_records)

    def delete_document(self, filename: str, save: bool = True) -> bool:
        if not _is_safe_filename(filename):
            raise ValueError("Invalid document filename.")

        with self.lock:
            updated = [
                record for record in self.records
                if record["document"] != filename
            ]
            if len(updated) == len(self.records):
                return False
            if save:
                self._write_records(updated)
            self.records = updated
            return True

    def documents(self) -> list[dict[str, Any]]:
        with self.lock:
            grouped: dict[str, int] = {}
            for record in self.records:
                name = record["document"]
                grouped[name] = grouped.get(name, 0) + 1
            return [
                {"filename": name, "chunks": count}
                for name, count in sorted(grouped.items())
            ]

    def search(
        self,
        query: str,
        top_k: int = 4,
        min_score: float = 0.25,
    ) -> list[dict[str, Any]]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("Search query cannot be empty.")
        if not isinstance(top_k, int) or isinstance(top_k, bool) or not 1 <= top_k <= 10:
            raise ValueError("top_k must be an integer between 1 and 10.")
        if (
            not isinstance(min_score, (int, float))
            or isinstance(min_score, bool)
            or not math.isfinite(min_score)
            or not 0.0 <= min_score <= 1.0
        ):
            raise ValueError("min_score must be between 0 and 1.")

        with self.lock:
            records = list(self.records)
        if not records:
            return []

        query_vectors = np.asarray(
            self.model.encode([query.strip()], normalize_embeddings=True),
            dtype=np.float32,
        )
        if query_vectors.shape != (1, len(records[0]["embedding"])):
            raise ValueError("Query embedding has an unexpected dimension.")
        if not np.isfinite(query_vectors).all():
            raise ValueError("Query embedding contains invalid values.")

        matrix = np.asarray(
            [record["embedding"] for record in records],
            dtype=np.float32,
        )
        scores = matrix @ query_vectors[0]
        indices = np.argsort(scores)[::-1]

        results = []
        for index in indices:
            score = float(scores[index])
            if not math.isfinite(score):
                continue
            if score < min_score:
                continue
            results.append({**records[int(index)], "score": score})
            if len(results) >= top_k:
                break
        return results
