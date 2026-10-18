"""CiteIt Streamlit interface: Documents, Ask, Evaluation."""
import streamlit as st

import api_client as api

st.set_page_config(page_title="CiteIt", page_icon="📚", layout="wide")

# ---------- session state ----------
st.session_state.setdefault("messages", [])  # {"role", "content", "found", "sources"}
st.session_state.setdefault("eval_cases", [])  # {"question", "expected_page"}


# ---------- helpers ----------
def render_sources(sources: list) -> None:
    if not sources:
        return
    with st.expander(f"Show sources ({len(sources)})"):
        for s in sources:
            st.markdown(f"**{s['doc_name']}** · page **{s['page']}** · similarity `{s['score']:.2f}`")
            st.text(s["text"])
            st.divider()


def render_message(msg: dict) -> None:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and msg.get("found") is False:
            st.warning(msg["content"])
        else:
            st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_sources(msg.get("sources", []))


def fetch_documents() -> list:
    try:
        return api.list_documents()
    except api.BackendError as exc:
        st.error(str(exc))
        return []


# ---------- header + sidebar ----------
st.title("📚 CiteIt")
st.caption("Ask questions about your PDFs. Every answer cites the page it came from, and if it isn't in your documents, CiteIt says so.")

status = api.health()
with st.sidebar:
    st.header("Status")
    if status is None:
        st.error("Backend not reachable")
    else:
        st.success("Backend online")
        if not status.get("llm_configured"):
            st.warning("No GEMINI_API_KEY set on the server. Retrieval works, but answers will fail.")
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()
    st.markdown("---")
    st.caption("Tip: use the Evaluation tab to measure how well retrieval finds the right page.")

tab_docs, tab_ask, tab_eval = st.tabs(["📄 Documents", "💬 Ask", "🧪 Evaluation"])

# ---------- tab 1: documents ----------
with tab_docs:
    st.subheader("Upload PDFs")
    files = st.file_uploader("Text-based PDFs only (max 20 MB each)", type=["pdf"], accept_multiple_files=True)
    if files and st.button("Index selected files", type="primary"):
        for f in files:
            with st.spinner(f"Indexing {f.name}..."):
                try:
                    info = api.upload_document(f.name, f.getvalue())
                    st.success(f"{info['doc_name']}: {info['pages']} pages, {info['chunks']} chunks")
                except api.BackendError as exc:
                    st.error(f"{f.name}: {exc}")

    st.subheader("Your documents")
    docs = fetch_documents()
    if not docs:
        st.info("No documents yet. Upload a PDF above.")
    for d in docs:
        c1, c2, c3, c4 = st.columns([5, 1, 1, 1])
        c1.markdown(f"**{d['doc_name']}**")
        c2.write(f"{d['pages']} pages")
        c3.write(f"{d['chunks']} chunks")
        if c4.button("Delete", key=f"del_{d['doc_id']}"):
            try:
                api.delete_document(d["doc_id"])
                st.rerun()
            except api.BackendError as exc:
                st.error(str(exc))

# ---------- tab 2: ask ----------
with tab_ask:
    docs = fetch_documents()
    if not docs:
        st.info("Upload a document in the Documents tab to start asking questions.")
    else:
        names = {d["doc_name"]: d["doc_id"] for d in docs}
        chosen = st.multiselect("Search in", list(names), default=list(names))

        for m in st.session_state.messages:
            render_message(m)

        question = st.chat_input("Ask something about your documents...")
        if question:
            if not chosen:
                st.warning("Select at least one document to search.")
            else:
                history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages if m.get("found", True)]
                st.session_state.messages.append({"role": "user", "content": question})
                render_message(st.session_state.messages[-1])
                with st.spinner("Searching your documents..."):
                    try:
                        result = api.ask(question, [names[n] for n in chosen], history[-6:])
                        reply = {"role": "assistant", "content": result["answer"], "found": result["found"], "sources": result["sources"]}
                    except api.BackendError as exc:
                        reply = {"role": "assistant", "content": f"⚠️ {exc}", "found": False, "sources": []}
                st.session_state.messages.append(reply)
                render_message(reply)

# ---------- tab 3: evaluation ----------
with tab_eval:
    st.subheader("How good is retrieval on your document?")
    st.write(
        "Add test questions together with the page where the answer really is. "
        "CiteIt checks whether that page appears among the retrieved pages. "
        "This test does not call the language model, so it uses no API quota."
    )
    docs = fetch_documents()
    if not docs:
        st.info("Upload a document first.")
    else:
        names = {d["doc_name"]: d["doc_id"] for d in docs}
        target = st.selectbox("Document to test", list(names))
        max_page = next(d["pages"] for d in docs if d["doc_name"] == target)

        with st.form("add_case", clear_on_submit=True):
            q = st.text_input("Test question")
            p = st.number_input("Page where the answer is", min_value=1, max_value=max_page, value=1, step=1)
            if st.form_submit_button("Add test case"):
                if not q.strip():
                    st.warning("Enter a question.")
                elif len(st.session_state.eval_cases) >= 20:
                    st.warning("Maximum 20 test cases.")
                else:
                    st.session_state.eval_cases.append({"question": q.strip(), "expected_page": int(p)})
                    st.rerun()

        cases = st.session_state.eval_cases
        if cases:
            st.table([{"Question": c["question"], "Expected page": c["expected_page"]} for c in cases])
            col_a, col_b = st.columns(2)
            if col_b.button("Clear test cases"):
                st.session_state.eval_cases = []
                st.rerun()
            if col_a.button("Run evaluation", type="primary"):
                if len(cases) < 5:
                    st.info("Tip: 5 or more test cases give a more meaningful score.")
                try:
                    res = api.evaluate(names[target], cases)
                    st.metric("Retrieval hit rate", f"{res['hit_rate'] * 100:.0f}%", f"{res['hits']} of {res['total']} found")
                    st.dataframe(
                        [
                            {
                                "Result": "✅ hit" if r["hit"] else "❌ miss",
                                "Question": r["question"],
                                "Expected page": r["expected_page"],
                                "Retrieved pages": ", ".join(map(str, r["retrieved_pages"])),
                            }
                            for r in res["results"]
                        ],
                        hide_index=True,
                    )
                except api.BackendError as exc:
                    st.error(str(exc))
