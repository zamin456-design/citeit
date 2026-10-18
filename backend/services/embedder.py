"""Turn text into normalized vectors. Default: a small local model (no API, no rate limit)."""
from functools import lru_cache
from typing import List, Protocol

import numpy as np


class Embedder(Protocol):
    def embed(self, texts: List[str]) -> np.ndarray:
        """Return an array of shape (len(texts), dim), L2-normalized, float32."""


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None  # loaded lazily so the API starts fast

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 1), dtype="float32")
        vectors = self._load().encode(
            texts,
            batch_size=32,
            normalize_embeddings=True,  # unit length -> dot product == cosine similarity
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype="float32")


@lru_cache(maxsize=1)
def get_embedder(model_name: str) -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder(model_name)
