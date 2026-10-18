"""Thin wrapper around the CiteIt backend so the UI code stays clean."""
import os
from typing import List, Optional

import requests

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")


class BackendError(Exception):
    """A readable error to show in the UI."""


def _request(method: str, path: str, timeout: int = 120, **kwargs):
    try:
        response = requests.request(method, f"{BACKEND_URL}{path}", timeout=timeout, **kwargs)
    except requests.ConnectionError:
        raise BackendError("Cannot reach the backend. Is it running?")
    except requests.Timeout:
        raise BackendError("The backend took too long to respond. Please try again.")
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        if isinstance(detail, list):  # FastAPI validation errors
            detail = "Invalid request."
        raise BackendError(str(detail))
    return response.json()


def health() -> Optional[dict]:
    try:
        return _request("GET", "/health", timeout=5)
    except BackendError:
        return None


def list_documents() -> List[dict]:
    return _request("GET", "/documents")


def upload_document(name: str, data: bytes) -> dict:
    return _request("POST", "/upload", timeout=300, files={"file": (name, data, "application/pdf")})


def delete_document(doc_id: str) -> None:
    _request("DELETE", f"/documents/{doc_id}")


def ask(question: str, doc_ids: List[str], history: List[dict]) -> dict:
    return _request("POST", "/ask", json={"question": question, "doc_ids": doc_ids, "history": history})


def evaluate(doc_id: str, cases: List[dict]) -> dict:
    return _request("POST", "/evaluate", json={"doc_id": doc_id, "cases": cases})
