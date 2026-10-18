"""CiteIt FastAPI backend: /upload /documents /ask /evaluate /health."""
import logging
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile

from backend.config import settings
from backend.schemas import (
    AskRequest,
    AskResponse,
    DocumentInfo,
    EvalRequest,
    EvalResponse,
)
from backend.services.embedder import get_embedder
from backend.services.llm_client import GeminiClient, LLMError, RateLimitError
from backend.services.pdf_loader import PDFError
from backend.services.rag_pipeline import RagPipeline
from backend.services.vector_store import VectorStore

log = logging.getLogger("citeit")
logging.basicConfig(level=logging.INFO)

_pipeline: Optional[RagPipeline] = None
_pipeline_lock = threading.Lock()


def get_pipeline() -> RagPipeline:
    """Build the pipeline once. Tests can replace it with set_pipeline()."""
    global _pipeline
    with _pipeline_lock:
        if _pipeline is None:
            settings.uploads_dir.mkdir(parents=True, exist_ok=True)
            _pipeline = RagPipeline(
                embedder=get_embedder(settings.embedding_model),
                store=VectorStore(settings.indexes_dir),
                llm=GeminiClient(
                    settings.gemini_api_key,
                    settings.llm_model,
                    settings.llm_temperature,
                    settings.llm_max_retries,
                ),
                settings=settings,
            )
        return _pipeline


def set_pipeline(pipeline: Optional[RagPipeline]) -> None:
    global _pipeline
    _pipeline = pipeline


def _preload_samples() -> None:
    """If nothing is indexed yet, index the PDFs in sample_docs/ so the demo is never empty."""
    try:
        pipeline = get_pipeline()
        if pipeline.store.list_documents():
            return
        for pdf in sorted(settings.sample_docs_dir.glob("*.pdf")):
            doc_id = uuid.uuid4().hex[:12]
            target = settings.uploads_dir / f"{doc_id}.pdf"
            target.write_bytes(pdf.read_bytes())
            pipeline.ingest(target, pdf.name, doc_id)
            log.info("Preloaded sample document %s", pdf.name)
    except Exception:  # never block startup because of the demo document
        log.exception("Could not preload sample documents")


@asynccontextmanager
async def lifespan(_: FastAPI):
    threading.Thread(target=_preload_samples, daemon=True).start()
    yield


app = FastAPI(title="CiteIt API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "llm_configured": bool(settings.gemini_api_key)}


@app.get("/documents", response_model=List[DocumentInfo])
def list_documents():
    return get_pipeline().store.list_documents()


@app.post("/upload", response_model=DocumentInfo)
def upload(file: UploadFile = File(...)):
    pipeline = get_pipeline()
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    name = Path(file.filename or "").name
    if not name.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")

    max_bytes = settings.max_upload_mb * 1024 * 1024
    data = file.file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(413, f"File is larger than {settings.max_upload_mb} MB.")
    if not data.startswith(b"%PDF"):
        raise HTTPException(400, "This file is not a valid PDF.")

    # Re-uploading a file with the same name replaces the old copy.
    for existing in pipeline.store.list_documents():
        if existing["doc_name"] == name:
            _remove_document(existing["doc_id"])

    doc_id = uuid.uuid4().hex[:12]
    path = settings.uploads_dir / f"{doc_id}.pdf"
    path.write_bytes(data)
    try:
        return pipeline.ingest(path, name, doc_id)
    except PDFError as exc:
        _remove_document(doc_id)
        raise HTTPException(400, str(exc))
    except Exception:
        log.exception("Indexing failed for %s", name)
        _remove_document(doc_id)
        raise HTTPException(500, "Indexing failed. Please try another file.")


def _remove_document(doc_id: str) -> bool:
    existed = get_pipeline().store.delete_document(doc_id)
    (settings.uploads_dir / f"{doc_id}.pdf").unlink(missing_ok=True)
    return existed


@app.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    if not _remove_document(doc_id):
        raise HTTPException(404, "Document not found.")
    return {"deleted": doc_id}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    pipeline = get_pipeline()
    question = request.question.strip()
    if not question:
        raise HTTPException(400, "Please enter a question.")

    docs = pipeline.store.list_documents()
    if not docs:
        raise HTTPException(400, "Upload a document first.")
    if request.doc_ids:
        known = {d["doc_id"] for d in docs}
        if not known.intersection(request.doc_ids):
            raise HTTPException(400, "None of the selected documents exist.")

    try:
        return pipeline.ask(
            question,
            request.doc_ids,
            request.top_k,
            [m.model_dump() for m in request.history],
        )
    except RateLimitError as exc:
        raise HTTPException(429, str(exc))
    except LLMError as exc:
        raise HTTPException(502, str(exc))


@app.post("/evaluate", response_model=EvalResponse)
def evaluate(request: EvalRequest):
    pipeline = get_pipeline()
    if pipeline.store.get_document(request.doc_id) is None:
        raise HTTPException(404, "Document not found.")
    return pipeline.evaluate(
        request.doc_id,
        [c.model_dump() for c in request.cases],
        request.top_k,
    )
