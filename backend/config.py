"""Central configuration. Every value can be overridden with an environment variable."""
import os
from dataclasses import dataclass
from pathlib import Path

try:  # optional: load a local .env file during development
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover
    pass

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def _float(name: str, default: float) -> float:
    return float(os.getenv(name, default))


@dataclass(frozen=True)
class Settings:
    # --- LLM (Gemini free tier) ---
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")
    # Free-tier model names change over time. Check Google AI Studio and set LLM_MODEL.
    llm_model: str = os.getenv("LLM_MODEL", "gemini-3.8-flash")
    llm_temperature: float = _float("LLM_TEMPERATURE", 0.1)
    llm_max_retries: int = _int("LLM_MAX_RETRIES", 2)

    # --- Embeddings (local, free, CPU-friendly) ---
    embedding_model: str = os.getenv(
        "EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )

    # --- Chunking and retrieval ---
    chunk_words: int = _int("CHUNK_WORDS", 500)
    chunk_overlap: int = _int("CHUNK_OVERLAP", 50)
    top_k: int = _int("TOP_K", 4)
    # Below this best-chunk similarity the app answers "not found" without calling the LLM.
    min_similarity: float = _float("MIN_SIMILARITY", 0.30)
    rewrite_followups: bool = os.getenv("REWRITE_FOLLOWUPS", "true").lower() == "true"

    # --- Upload limits ---
    max_upload_mb: int = _int("MAX_UPLOAD_MB", 20)
    max_pages: int = _int("MAX_PAGES", 300)

    # --- Storage ---
    data_dir: Path = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
    sample_docs_dir: Path = Path(os.getenv("SAMPLE_DOCS_DIR", PROJECT_ROOT / "sample_docs"))

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    @property
    def indexes_dir(self) -> Path:
        return self.data_dir / "indexes"


settings = Settings()

NOT_FOUND_MESSAGE = (
    "I couldn't find this in your documents. Try rephrasing the question "
    "or check that the right document is selected."
)
