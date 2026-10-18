"""Read a PDF and return its text page by page, keeping real page numbers."""
import re
from pathlib import Path
from typing import List, Tuple

from pypdf import PdfReader

Page = Tuple[int, str]  # (page_number starting at 1, cleaned text)


class PDFError(ValueError):
    """Raised for files we cannot or will not process. The message is user-friendly."""


def clean_text(text: str) -> str:
    """Fix hyphenated line breaks and collapse whitespace."""
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # "inter-\nnational" -> "international"
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_pages(path: Path, max_pages: int = 300) -> List[Page]:
    try:
        reader = PdfReader(str(path))
        encrypted = reader.is_encrypted
        n_pages = None if encrypted else len(reader.pages)
    except Exception as exc:  # corrupted or not a PDF at all
        raise PDFError("This file could not be read as a PDF.") from exc

    if encrypted:
        raise PDFError("This PDF is password protected.")
    if n_pages == 0:
        raise PDFError("This PDF has no pages.")
    if n_pages > max_pages:
        raise PDFError(f"This PDF has {n_pages} pages. The limit is {max_pages}.")

    pages: List[Page] = []
    for number, page in enumerate(reader.pages, start=1):
        try:
            raw = page.extract_text() or ""
        except Exception:  # one bad page should not kill the whole upload
            raw = ""
        text = clean_text(raw)
        if text:
            pages.append((number, text))

    total_words = sum(len(t.split()) for _, t in pages)
    if total_words < 30:
        raise PDFError("This PDF looks scanned. Text-based PDFs only.")
    return pages


def page_count(path: Path) -> int:
    return len(PdfReader(str(path)).pages)
