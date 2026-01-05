import os
import streamlit as st
from document_loader import load_document
from rag_engine import build_vector_store, similarity_search, rag_pipeline

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Mini RAG Chatbot",
    layout="wide"
)

st.title("Mini RAG Chatbot (Local Ollama)")

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Upload Document")
    uploaded = st.file_uploader(
        "Upload a TXT or PDF file",
        type=["txt", "pdf"]
    )

    if st.button("Reset / Clear"):
        st.session_state.clear()
        st.rerun()

# ---------------- Session State ----------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
    st.session_state.doc_name = None

# ---------------- Document Processing ----------------
if uploaded is not None:
    file_path = os.path.join("data", uploaded.name)

    with open(file_path, "wb") as f:
        f.write(uploaded.getbuffer())

    st.success(f"Saved `{uploaded.name}`")

    try:
        text = load_document(file_path)
        st.session_state.vector_store = build_vector_store(text)
        st.session_state.doc_name = uploaded.name
        st.success("Document processed. You can now ask questions.")
    except Exception as e:
        st.error(f"Failed to process document: {e}")

# ---------------- Question Answering ----------------
st.markdown("---")
st.subheader("Ask a question (answers only from the uploaded document)")

query = st.text_input("Your question")

if st.button("Ask"):
    if not query.strip():
        st.warning("Please enter a question.")
    elif st.session_state.vector_store is None:
        st.error("Please upload a document first.")
    else:
        hits = similarity_search(query, st.session_state.vector_store, k=4)

        st.markdown("### Retrieved Context")
        for i, h in enumerate(hits, start=1):
            st.markdown(f"**Chunk {i} (score: {h['score']:.3f})**")
            st.text(h["text"][:1000])
            st.markdown("---")

        answer = rag_pipeline(query, st.session_state.vector_store)
        st.markdown("### AI Answer")
        st.write(answer)

# ---------------- Footer ----------------
st.markdown("---")
st.caption(
    "Built with Sentence-Transformers, FAISS, and Ollama (local LLM)."
)
