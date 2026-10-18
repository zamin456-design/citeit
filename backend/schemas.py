"""Request and response models for the API."""
from typing import List, Optional

from pydantic import BaseModel, Field


class DocumentInfo(BaseModel):
    doc_id: str
    doc_name: str
    pages: int
    chunks: int
    uploaded_at: Optional[str] = None


class ChatMessage(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str


class AskRequest(BaseModel):
    question: str
    doc_ids: Optional[List[str]] = None
    top_k: Optional[int] = Field(default=None, ge=1, le=10)
    history: List[ChatMessage] = []


class Source(BaseModel):
    doc_name: str
    page: int
    text: str
    score: float


class AskResponse(BaseModel):
    answer: str
    found: bool
    sources: List[Source]


class EvalCase(BaseModel):
    question: str
    expected_page: int = Field(ge=1)


class EvalRequest(BaseModel):
    doc_id: str
    cases: List[EvalCase] = Field(min_length=1, max_length=50)
    top_k: Optional[int] = Field(default=None, ge=1, le=10)


class EvalResult(BaseModel):
    question: str
    expected_page: int
    retrieved_pages: List[int]
    hit: bool


class EvalResponse(BaseModel):
    total: int
    hits: int
    hit_rate: float
    results: List[EvalResult]
