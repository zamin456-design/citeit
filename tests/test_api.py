"""API tests. Need fastapi + httpx installed (see requirements.txt). The LLM is faked, so no key is used."""
import tempfile
from dataclasses import replace
from pathlib import Path

from tests.fakes import HashEmbedder, ScriptedLLM
from tests.test_core import SAMPLE


class RateLimitedLLM:
    def generate(self, prompt: str) -> str:
        from backend.services.llm_client import RateLimitError

        raise RateLimitError("The free Gemini quota is temporarily used up.")


def build_client(tmp: str, llm):
    from fastapi.testclient import TestClient

    from backend import main
    from backend.config import settings as base
    from backend.services.rag_pipeline import RagPipeline
    from backend.services.vector_store import VectorStore

    s = replace(base, data_dir=Path(tmp), min_similarity=0.2)
    main.settings = s  # point the app at the temp folder
    main.set_pipeline(RagPipeline(HashEmbedder(), VectorStore(s.indexes_dir), llm, s))
    return TestClient(main.app)  # no `with`: skips the sample-preload startup hook


def upload_sample(client, name="Handbook.pdf"):
    return client.post("/upload", files={"file": (name, SAMPLE.read_bytes(), "application/pdf")})


def test_full_api_flow():
    with tempfile.TemporaryDirectory() as tmp:
        c = build_client(tmp, ScriptedLLM("Sixty percent [Doc: Handbook.pdf, Page: 2]"))
        assert c.get("/health").json()["status"] == "ok"

        # nothing uploaded yet
        assert c.post("/ask", json={"question": "anything"}).status_code == 400

        # bad uploads
        assert c.post("/upload", files={"file": ("notes.txt", b"hello", "text/plain")}).status_code == 400
        assert c.post("/upload", files={"file": ("fake.pdf", b"not a pdf", "application/pdf")}).status_code == 400

        # good upload, and re-uploading the same name replaces it
        first = upload_sample(c)
        assert first.status_code == 200 and first.json()["pages"] == 8
        second = upload_sample(c)
        docs = c.get("/documents").json()
        assert len(docs) == 1 and docs[0]["doc_id"] == second.json()["doc_id"]
        doc_id = docs[0]["doc_id"]

        # ask
        assert c.post("/ask", json={"question": "   "}).status_code == 400
        ok = c.post("/ask", json={"question": "What is the minimum percentage for admission?", "doc_ids": [doc_id]})
        body = ok.json()
        assert ok.status_code == 200 and body["found"] is True and body["sources"][0]["page"] == 2

        gibberish = c.post("/ask", json={"question": "zebra quantum volleyball xylophone", "doc_ids": [doc_id]}).json()
        assert gibberish["found"] is False

        # evaluate
        ev = c.post("/evaluate", json={"doc_id": doc_id, "cases": [{"question": "What is the last date to submit an application?", "expected_page": 3}]})
        assert ev.status_code == 200 and ev.json()["hit_rate"] == 1.0
        assert c.post("/evaluate", json={"doc_id": "nope", "cases": [{"question": "x", "expected_page": 1}]}).status_code == 404

        # delete
        assert c.delete(f"/documents/{doc_id}").status_code == 200
        assert c.delete(f"/documents/{doc_id}").status_code == 404
        assert c.get("/documents").json() == []


def test_rate_limit_becomes_429():
    with tempfile.TemporaryDirectory() as tmp:
        c = build_client(tmp, RateLimitedLLM())
        doc_id = upload_sample(c).json()["doc_id"]
        r = c.post("/ask", json={"question": "What is the minimum percentage for admission?", "doc_ids": [doc_id]})
        assert r.status_code == 429 and "quota" in r.json()["detail"]
