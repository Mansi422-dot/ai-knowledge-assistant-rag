import json
from pathlib import Path
from typing import Any
import numpy as np
from sentence_transformers import SentenceTransformer

class VectorStore:
    """Small persistent cosine-similarity store suitable for a local demo."""
    def __init__(self, data_dir: str, model_name: str):
        self.root = Path(data_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.file = self.root / "index.json"
        self.model = SentenceTransformer(model_name)
        self.records: list[dict[str, Any]] = []
        if self.file.exists():
            try:
                self.records = json.loads(self.file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                self.records = []

    def _save(self) -> None:
        self.file.write_text(json.dumps(self.records, ensure_ascii=False, indent=2), encoding="utf-8")

    def add_document(self, filename: str, chunks: list[str]) -> int:
        self.delete_document(filename, save=False)
        if not chunks:
            self._save()
            return 0
        vectors = self.model.encode(chunks, normalize_embeddings=True).tolist()
        for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
            self.records.append({"document": filename, "chunk": i, "text": chunk, "embedding": vector})
        self._save()
        return len(chunks)

    def delete_document(self, filename: str, save: bool = True) -> bool:
        old = len(self.records)
        self.records = [r for r in self.records if r["document"] != filename]
        if save:
            self._save()
        return len(self.records) != old

    def documents(self) -> list[dict[str, Any]]:
        grouped: dict[str, int] = {}
        for r in self.records:
            grouped[r["document"]] = grouped.get(r["document"], 0) + 1
        return [{"filename": name, "chunks": count} for name, count in sorted(grouped.items())]

    def search(self, query: str, top_k: int = 4, min_score: float = 0.25) -> list[dict[str, Any]]:
        if not self.records:
            return []

        q = self.model.encode(
            [query], normalize_embeddings=True
        )[0]

        matrix = np.asarray(
            [r["embedding"] for r in self.records],
            dtype=np.float32
        )

        scores = matrix @ q
        indices = np.argsort(scores)[::-1][:top_k]

        results = [
            {**self.records[int(i)], "score": float(scores[int(i)])}
            for i in indices
            if scores[int(i)] >= 0.25
        ]

        return results

