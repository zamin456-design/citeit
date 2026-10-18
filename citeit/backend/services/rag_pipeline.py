"""The heart of CiteIt: ingest documents, retrieve evidence, answer only from it."""
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from backend.config import NOT_FOUND_MESSAGE, Settings
from backend.services.chunker import chunk_pages
from backend.services.embedder import Embedder
from backend.services.llm_client import LLMClient, LLMError
from backend.services.pdf_loader import extract_pages, page_count
from backend.services.vector_store import VectorStore

ANSWER_PROMPT = """You are a careful assistant that answers questions using ONLY the context below.

Rules:
1. Use only the information in the context. Do not use outside knowledge.
2. If the context does not contain the answer, reply exactly: NOT_FOUND
3. Keep the answer short and clear.
4. After each fact, mention the source in square brackets like [Doc: name, Page: n].

Context:
{context}

Question: {question}

Answer:"""

REWRITE_PROMPT = """Rewrite the user's last question so it can be understood without the earlier conversation.
Replace pronouns and vague references using the conversation. Do not answer it.
If it is already clear, return it unchanged. Return ONLY the rewritten question.

Conversation:
{conversation}

Last question: {question}

Rewritten question:"""


def build_context(hits: List[dict]) -> str:
    blocks = [f"[{i}] (Doc: {h['doc_name']}, Page: {h['page']})\n{h['text']}" for i, h in enumerate(hits, 1)]
    return "\n\n".join(blocks)


def is_not_found(reply: str) -> bool:
    """True when the model followed rule 2 and replied NOT_FOUND (tolerates quotes, dots, bold)."""
    return reply.strip().strip("`*_. \n\"'").upper().startswith("NOT_FOUND")


def to_source(hit: dict) -> dict:
    return {
        "doc_name": hit["doc_name"],
        "page": hit["page"],
        "text": hit["text"],
        "score": round(hit["score"], 4),
    }


class RagPipeline:
    def __init__(self, embedder: Embedder, store: VectorStore, llm: LLMClient, settings: Settings):
        self.embedder = embedder
        self.store = store
        self.llm = llm
        self.settings = settings

    # ---------------- ingestion ----------------
    def ingest(self, path: Path, doc_name: str, doc_id: str) -> Dict:
        """PDF -> pages -> chunks -> embeddings -> stored index. Raises PDFError for bad files."""
        s = self.settings
        pages = extract_pages(path, max_pages=s.max_pages)
        chunks = chunk_pages(pages, doc_id, doc_name, s.chunk_words, s.chunk_overlap)
        vectors = self.embedder.embed([c["text"] for c in chunks])
        meta = {
            "doc_id": doc_id,
            "doc_name": doc_name,
            "pages": page_count(path),
            "chunks": len(chunks),
            "uploaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        self.store.add_document(meta, chunks, vectors)
        return meta

    # ---------------- retrieval ----------------
    def retrieve(self, question: str, doc_ids: Optional[List[str]], top_k: int) -> List[dict]:
        query_vector = self.embedder.embed([question])[0]
        return self.store.search(query_vector, doc_ids, top_k)

    # ---------------- question answering ----------------
    def _rewrite_followup(self, question: str, history: List[dict]) -> str:
        if not history or not self.settings.rewrite_followups:
            return question
        recent = history[-6:]
        conversation = "\n".join(f"{m['role'].capitalize()}: {m['content'][:500]}" for m in recent)
        try:
            rewritten = self.llm.generate(REWRITE_PROMPT.format(conversation=conversation, question=question))
        except LLMError:
            return question  # never fail the whole request because of the optional rewrite
        rewritten = rewritten.strip().strip('"')
        return rewritten if 0 < len(rewritten) < 500 else question

    def ask(
        self,
        question: str,
        doc_ids: Optional[List[str]] = None,
        top_k: Optional[int] = None,
        history: Optional[List[dict]] = None,
    ) -> Dict:
        s = self.settings
        top_k = top_k or s.top_k
        standalone = self._rewrite_followup(question, history or [])

        hits = self.retrieve(standalone, doc_ids, top_k)
        sources = [to_source(h) for h in hits]

        # Gate 1: weak retrieval -> do not even call the LLM.
        if not hits or hits[0]["score"] < s.min_similarity:
            return {"answer": NOT_FOUND_MESSAGE, "found": False, "sources": sources}

        reply = self.llm.generate(ANSWER_PROMPT.format(context=build_context(hits), question=standalone))

        # Gate 2: the LLM itself says the context has no answer.
        if is_not_found(reply):
            return {"answer": NOT_FOUND_MESSAGE, "found": False, "sources": sources}
        return {"answer": reply, "found": True, "sources": sources}

    # ---------------- evaluation ----------------
    def evaluate(self, doc_id: str, cases: List[dict], top_k: Optional[int] = None) -> Dict:
        """Retrieval-only accuracy: is the expected page among the top-k retrieved pages?"""
        top_k = top_k or self.settings.top_k
        vectors = self.embedder.embed([c["question"] for c in cases])
        results = []
        for case, vector in zip(cases, vectors):
            hits = self.store.search(vector, [doc_id], top_k)
            pages: List[int] = []
            for h in hits:  # keep order, drop duplicate pages
                if h["page"] not in pages:
                    pages.append(h["page"])
            results.append(
                {
                    "question": case["question"],
                    "expected_page": case["expected_page"],
                    "retrieved_pages": pages,
                    "hit": case["expected_page"] in pages,
                }
            )
        hits_count = sum(r["hit"] for r in results)
        total = len(results)
        return {
            "total": total,
            "hits": hits_count,
            "hit_rate": round(hits_count / total, 4) if total else 0.0,
            "results": results,
        }
