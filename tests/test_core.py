"""Core tests. Plain asserts, so they run with pytest or `python -m tests.test_core`."""
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np

from backend.config import NOT_FOUND_MESSAGE, settings as base_settings
from backend.services.chunker import chunk_pages, chunk_text
from backend.services.pdf_loader import PDFError, extract_pages
from backend.services.rag_pipeline import RagPipeline, is_not_found
from backend.services.vector_store import VectorStore
from tests.fakes import HashEmbedder, ScriptedLLM

SAMPLE = Path(__file__).resolve().parent.parent / "sample_docs" / "Northridge_Student_Handbook_Demo.pdf"


def make_pipeline(tmp: str, reply: str = "Sixty percent [Doc: s.pdf, Page: 2]", **overrides):
    s = replace(base_settings, data_dir=Path(tmp), min_similarity=0.2, **overrides)
    llm = ScriptedLLM(reply)
    pipe = RagPipeline(HashEmbedder(), VectorStore(s.indexes_dir), llm, s)
    return pipe, llm


# ---------- chunker ----------
def test_chunk_sizes_and_overlap():
    text = " ".join(f"w{i}" for i in range(1200))
    chunks = chunk_text(text, chunk_words=500, overlap=50)
    assert all(len(c.split()) <= 500 for c in chunks)
    assert chunks[0].split()[-50:] == chunks[1].split()[:50]  # overlap is exact
    assert chunks[-1].split()[-1] == "w1199"  # nothing is lost at the end


def test_short_text_is_single_chunk_and_empty_is_none():
    assert chunk_text("just a few words") == ["just a few words"]
    assert chunk_text("   ") == []


def test_chunks_never_cross_pages():
    pages = [(1, " ".join(["alpha"] * 700)), (2, " ".join(["beta"] * 700))]
    for rec in chunk_pages(pages, "d", "d.pdf", 500, 50):
        words = set(rec["text"].split())
        assert words == ({"alpha"} if rec["page"] == 1 else {"beta"})


# ---------- pdf loader ----------
def test_pdf_pages_match_source():
    pages = extract_pages(SAMPLE)
    assert [n for n, _ in pages] == list(range(1, 9))
    assert "sixty percent" in dict(pages)[2].lower()


def test_bad_pdf_rejected():
    with tempfile.TemporaryDirectory() as tmp:
        bad = Path(tmp) / "bad.pdf"
        bad.write_bytes(b"%PDF-1.4 this is not really a pdf")
        try:
            extract_pages(bad)
            raise AssertionError("should have failed")
        except PDFError:
            pass


# ---------- vector store ----------
def test_store_add_search_delete_and_reload():
    with tempfile.TemporaryDirectory() as tmp:
        store = VectorStore(Path(tmp))
        emb = HashEmbedder()
        chunks = [
            {"chunk_id": "a", "doc_id": "d1", "doc_name": "d1.pdf", "page": 1, "text": "cats purr softly"},
            {"chunk_id": "b", "doc_id": "d1", "doc_name": "d1.pdf", "page": 2, "text": "rockets launch into orbit"},
        ]
        store.add_document({"doc_id": "d1", "doc_name": "d1.pdf", "pages": 2, "chunks": 2}, chunks, emb.embed([c["text"] for c in chunks]))
        top = store.search(emb.embed(["rocket orbit launch"])[0], ["d1"], 1)[0]
        assert top["page"] == 2 and top["score"] > 0.5
        assert len(VectorStore(Path(tmp)).list_documents()) == 1  # survives restart
        assert store.delete_document("d1") and store.list_documents() == []
        assert store.search(emb.embed(["rocket"])[0], None, 3) == []


# ---------- pipeline ----------
def test_ingest_and_cited_answer():
    with tempfile.TemporaryDirectory() as tmp:
        pipe, llm = make_pipeline(tmp)
        meta = pipe.ingest(SAMPLE, SAMPLE.name, "doc1")
        assert meta["pages"] == 8 and meta["chunks"] == 8
        out = pipe.ask("What is the minimum percentage marks needed for admission?", ["doc1"])
        assert out["found"] is True
        assert out["sources"][0]["page"] == 2
        assert "Page: 2" in llm.prompts[0]  # evidence reached the prompt with its page number
        assert "ONLY the context" in llm.prompts[0]


def test_weak_retrieval_skips_llm():
    with tempfile.TemporaryDirectory() as tmp:
        pipe, llm = make_pipeline(tmp)
        pipe.ingest(SAMPLE, SAMPLE.name, "doc1")
        out = pipe.ask("zebra quantum volleyball xylophone", ["doc1"])
        assert out["found"] is False and out["answer"] == NOT_FOUND_MESSAGE
        assert llm.prompts == []  # gate 1: no API call was spent


def test_llm_not_found_is_converted():
    with tempfile.TemporaryDirectory() as tmp:
        pipe, _ = make_pipeline(tmp, reply="NOT_FOUND")
        pipe.ingest(SAMPLE, SAMPLE.name, "doc1")
        out = pipe.ask("What is the tuition fee per year?", ["doc1"])
        assert out["found"] is False and out["answer"] == NOT_FOUND_MESSAGE
        assert out["sources"]  # closest chunks still shown for transparency
    assert is_not_found("`NOT_FOUND`.") and is_not_found("not_found") and not is_not_found("The fee is high")


def test_followup_is_rewritten_before_retrieval():
    with tempfile.TemporaryDirectory() as tmp:
        pipe, llm = make_pipeline(tmp, reply="tuition fee for the Artificial Intelligence program")
        pipe.ingest(SAMPLE, SAMPLE.name, "doc1")
        history = [{"role": "user", "content": "Tell me about the AI program"}, {"role": "assistant", "content": "It is offered."}]
        out = pipe.ask("and the fee?", ["doc1"], history=history)
        assert out["sources"][0]["page"] == 4  # the rewritten question found the fee page
        assert len(llm.prompts) == 2  # one rewrite call + one answer call


def test_evaluation_hit_rate():
    with tempfile.TemporaryDirectory() as tmp:
        pipe, llm = make_pipeline(tmp)
        pipe.ingest(SAMPLE, SAMPLE.name, "doc1")
        cases = [
            {"question": "What is the last date to submit an application?", "expected_page": 3},
            {"question": "How many books can students borrow from the library?", "expected_page": 7},
            {"question": "What is the tuition fee?", "expected_page": 1},  # deliberately wrong
        ]
        res = pipe.evaluate("doc1", cases, top_k=2)
        assert [r["hit"] for r in res["results"]] == [True, True, False]
        assert res["total"] == 3 and res["hits"] == 2 and abs(res["hit_rate"] - 0.6667) < 1e-3
        assert llm.prompts == []  # evaluation never spends API quota


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn(); print("PASS", name)
        except Exception as exc:
            failed += 1; print("FAIL", name, "->", repr(exc))
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(failed)
