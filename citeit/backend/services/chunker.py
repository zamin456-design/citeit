"""Split page text into overlapping word-based chunks. Chunks never cross page boundaries."""
from typing import Dict, List, Tuple


def chunk_text(text: str, chunk_words: int = 500, overlap: int = 50) -> List[str]:
    if chunk_words <= 0:
        raise ValueError("chunk_words must be positive")
    if not 0 <= overlap < chunk_words:
        raise ValueError("overlap must be >= 0 and smaller than chunk_words")

    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_words:  # short pages stay as one chunk, never discarded
        return [" ".join(words)]

    step = chunk_words - overlap
    chunks: List[str] = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start : start + chunk_words]))
        if start + chunk_words >= len(words):
            break
    return chunks


def chunk_pages(
    pages: List[Tuple[int, str]],
    doc_id: str,
    doc_name: str,
    chunk_words: int = 500,
    overlap: int = 50,
) -> List[Dict]:
    """Return chunk records with the metadata the data model requires."""
    records: List[Dict] = []
    for page_number, text in pages:
        for i, piece in enumerate(chunk_text(text, chunk_words, overlap)):
            records.append(
                {
                    "chunk_id": f"{doc_id}_p{page_number}_c{i}",
                    "doc_id": doc_id,
                    "doc_name": doc_name,
                    "page": page_number,
                    "text": piece,
                }
            )
    return records
