"""Offline stand-ins so tests need no model download and no API key."""
import re
import zlib
from typing import List

import numpy as np


class HashEmbedder:
    """Bag-of-words hashing embedder: deterministic, tiny, good enough to test retrieval logic."""

    STOP = {"the", "a", "an", "of", "is", "to", "and", "in", "for", "what", "how", "are", "on", "at", "be", "by"}

    def __init__(self, dim: int = 512):
        self.dim = dim

    def embed(self, texts: List[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, text in enumerate(texts):
            for tok in re.findall(r"[a-z0-9]+", text.lower()):
                if tok not in self.STOP:
                    out[i, zlib.crc32(tok.encode()) % self.dim] += 1.0
            norm = np.linalg.norm(out[i])
            if norm:
                out[i] /= norm
        return out


class ScriptedLLM:
    """Returns canned replies and records every prompt it received."""

    def __init__(self, reply: str = "Answer [Doc: x, Page: 1]"):
        self.reply = reply
        self.prompts: List[str] = []

    def generate(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return self.reply
