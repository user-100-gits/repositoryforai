import json
import os
import uuid
from typing import Any, Dict, List, Optional
import numpy as np
from .document import DocumentChunk


class VectorStore:
    def __init__(self, path: str):
        self.path = path
        self._vectors: List[Dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as handle:
                self._vectors = json.load(handle)
        else:
            self._vectors = []

    def _save(self) -> None:
        with open(self.path, "w", encoding="utf-8") as handle:
            json.dump(self._vectors, handle, ensure_ascii=False, indent=2)

    def add(self, chunk: DocumentChunk, embedding: List[float]) -> None:
        self._vectors.append({
            "id": chunk.id,
            "source": chunk.source,
            "text": chunk.text,
            "embedding": embedding,
        })
        self._save()

    def similarity_search(self, query_embedding: List[float], top_k: int = 4) -> List[Dict[str, Any]]:
        if not self._vectors:
            return []

        query_vec = np.array(query_embedding, dtype=np.float32)
        scores = []
        for entry in self._vectors:
            stored = np.array(entry["embedding"], dtype=np.float32)
            similarity = self._cosine_similarity(query_vec, stored)
            scores.append((similarity, entry))

        results = sorted(scores, key=lambda item: item[0], reverse=True)[:top_k]
        return [entry for _, entry in results]

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            return 0.0
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    def clear(self) -> None:
        self._vectors = []
        self._save()
