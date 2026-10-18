"""A tiny per-document vector store: NumPy matrices on disk, exact cosine search in memory.

For a few hundred chunks per document, exact search is instant and avoids extra
dependencies. The interface is small enough to swap for FAISS or Chroma later.
"""
import json
import shutil
import threading
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


class VectorStore:
    def __init__(self, index_dir: Path):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._cache: Dict[str, dict] = {}  # doc_id -> {"vectors", "chunks", "meta"}
        self._load_all()

    # ---------- persistence ----------
    def _load_all(self) -> None:
        for folder in sorted(self.index_dir.iterdir()):
            if folder.is_dir():
                self._load_one(folder.name)

    def _load_one(self, doc_id: str) -> None:
        folder = self.index_dir / doc_id
        try:
            vectors = np.load(folder / "vectors.npy")
            chunks = json.loads((folder / "chunks.json").read_text(encoding="utf-8"))
            meta = json.loads((folder / "meta.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return  # skip half-written or corrupt indexes instead of crashing at startup
        self._cache[doc_id] = {"vectors": vectors, "chunks": chunks, "meta": meta}

    # ---------- public API ----------
    def add_document(self, meta: dict, chunks: List[dict], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        doc_id = meta["doc_id"]
        folder = self.index_dir / doc_id
        vectors = vectors.astype("float32")
        with self._lock:
            folder.mkdir(parents=True, exist_ok=True)
            np.save(folder / "vectors.npy", vectors)
            (folder / "chunks.json").write_text(json.dumps(chunks), encoding="utf-8")
            (folder / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
            self._cache[doc_id] = {"vectors": vectors, "chunks": chunks, "meta": meta}

    def list_documents(self) -> List[dict]:
        with self._lock:
            docs = [entry["meta"] for entry in self._cache.values()]
        return sorted(docs, key=lambda m: m.get("uploaded_at", ""))

    def get_document(self, doc_id: str) -> Optional[dict]:
        with self._lock:
            entry = self._cache.get(doc_id)
            return entry["meta"] if entry else None

    def delete_document(self, doc_id: str) -> bool:
        with self._lock:
            existed = self._cache.pop(doc_id, None) is not None
            shutil.rmtree(self.index_dir / doc_id, ignore_errors=True)
        return existed

    def search(self, query_vector: np.ndarray, doc_ids: Optional[List[str]], top_k: int) -> List[dict]:
        """Return the top_k chunks across the chosen documents, best first, with a 'score'."""
        with self._lock:
            ids = doc_ids if doc_ids else list(self._cache.keys())
            entries = [self._cache[d] for d in ids if d in self._cache]

        hits: List[dict] = []
        for entry in entries:
            if len(entry["vectors"]) == 0:
                continue
            scores = entry["vectors"] @ query_vector  # cosine similarity (unit vectors)
            k = min(top_k, len(scores))
            best = np.argpartition(-scores, k - 1)[:k]
            for idx in best:
                hits.append({**entry["chunks"][int(idx)], "score": float(scores[idx])})

        hits.sort(key=lambda h: h["score"], reverse=True)
        return hits[:top_k]
