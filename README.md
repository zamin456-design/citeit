---
title: CiteIt
emoji: 📚
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
---

# 📚 CiteIt — document Q&A that never makes things up

Upload PDFs, ask questions, and get answers with **page-level citations**. If the answer is not in your documents, CiteIt says so instead of guessing. A built-in **evaluation tab** measures how well retrieval finds the right page.

*(The block at the very top is only needed for Hugging Face Spaces. It is harmless elsewhere.)*

## Why it is different

1. **Every answer cites its source** (document, page, and the exact retrieved text).
2. **Two "not found" gates.** If retrieval is weak, the language model is never called. If the model itself finds no answer, it must reply `NOT_FOUND`.
3. **Self-measurement.** Add test questions with the page where the answer lives, and the app reports a hit rate. The test uses no API quota.

## Architecture

```
Streamlit UI  --HTTP/JSON-->  FastAPI backend
 upload / chat /               /upload /documents /ask /evaluate /health
 sources / evaluation               |
                      PDF loader -> chunker -> embedder -> vector store
                                                   |
                      question -> embed -> top-k chunks -> gate -> Gemini -> answer + citations
```

| Part | Choice |
|------|--------|
| Frontend | Streamlit |
| Backend | FastAPI |
| LLM | Gemini free tier (Google AI Studio key) |
| Embeddings | `all-MiniLM-L6-v2`, local, CPU, no limits (PyTorch under the hood) |
| Vector store | NumPy matrices on disk, exact cosine search (swap for FAISS/Chroma if documents grow large) |
| PDF parsing | `pypdf` |

## Quick start (local)

```bash
git clone <your-repo-url> citeit && cd citeit
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env        # then paste your GEMINI_API_KEY into .env
```

Get a free key at **Google AI Studio** (search "Google AI Studio API key"). Free-tier model names and limits change, so check which model is currently free and set `LLM_MODEL` in `.env`.

Run the two parts in two terminals, both from the project root:

```bash
# terminal 1: backend
uvicorn backend.main:app --reload --port 8000

# terminal 2: frontend
streamlit run frontend/app.py
```

Open the Streamlit link. On first start the backend downloads the small embedding model (about 90 MB) and indexes the demo handbook in `sample_docs/`, so you can ask questions immediately. Interactive API docs are at `http://localhost:8000/docs`.

## Configuration

All settings live in `.env` (see `.env.example`): `GEMINI_API_KEY`, `LLM_MODEL`, `EMBEDDING_MODEL`, `CHUNK_WORDS`, `CHUNK_OVERLAP`, `TOP_K`, `MIN_SIMILARITY`, `REWRITE_FOLLOWUPS`, `MAX_UPLOAD_MB`, `MAX_PAGES`, `BACKEND_URL`.

### Tuning `MIN_SIMILARITY` (important)

This threshold decides when the app answers "not found" without calling the model. The right value depends on the embedding model and your documents.

1. Ask 5 questions whose answers are in the document and 5 whose answers are not.
2. Open "Show sources" on each answer and note the top similarity score.
3. Set `MIN_SIMILARITY` between the highest score of the out-of-document questions and the lowest score of the in-document ones.

## Using the Evaluation tab

For the demo handbook, try these cases (question → page):

| Question | Page |
|----------|------|
| What is the minimum percentage for the AI program? | 2 |
| What is the last date to submit an application? | 3 |
| How much is the tuition fee per year? | 4 |
| Which scholarship covers fifty percent of tuition? | 5 |
| What attendance is required to sit the final exam? | 6 |
| How many books can a student borrow? | 7 |
| When is the admissions office open? | 8 |

Record the hit rate you get here, after tuning, in this README.

**Evaluation results:** _fill in after your first run_

## Tests

```bash
pytest -q
# core tests also run without pytest:
python -m tests.test_core
```

Tests use fake embedders and a fake language model, so they need no model download and no API key.

## Deploy for free on a Hugging Face Space

1. Create a new Space, choose **Docker**, and push this repository to it (the front matter above is already in place).
2. In *Settings → Variables and secrets*, add a **Secret** named `GEMINI_API_KEY`. Optionally add `LLM_MODEL`.
3. Wait for the build. The Dockerfile downloads the embedding model during the build, and `start.sh` runs FastAPI and Streamlit together.

Free Spaces do not guarantee persistent disk, so uploaded files can disappear after a restart. The demo handbook in `sample_docs/` is re-indexed automatically whenever the index is empty.

**Alternative:** Streamlit Community Cloud for `frontend/app.py` plus Render (free) for the backend. Set `BACKEND_URL` in the Streamlit secrets. Render's free tier sleeps when idle, so the first request after a pause is slow.

## Project layout

```
backend/   main.py (routes) · config.py · schemas.py · services/ (pdf_loader, chunker, embedder, vector_store, llm_client, rag_pipeline)
frontend/  app.py (Streamlit) · api_client.py
tests/     fakes.py · test_core.py · test_api.py
sample_docs/  fictional demo handbook (PDF)
```

## Known limitations

- Scanned PDFs are rejected (no OCR). Tables and multi-column layouts may extract imperfectly.
- Free-tier Gemini limits can slow or block heavy use. The app shows a friendly message and retries twice.
- Re-uploading a file with the same name replaces the old copy.
- The default embedding model works best in English.
- Follow-up question rewriting costs one extra model call per follow-up (turn it off with `REWRITE_FOLLOWUPS=false`).
