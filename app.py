# app.py
import streamlit as st
from document_loader import load_document
from rag_engine import build_vector_store, similarity_search, rag_pipeline

st.set_page_config(page_title="Mini RAG Chatbot", layout="wide")

st.title("mini RAG chatbot (Local Ollama)")

# Sidebar - upload and controls
with st.sidebar:
    st.header("Upload Document")
    uploaded = st.file_uploader("Upload a TXT or PDF", type=["txt", "pdf"])
    if st.button("Reset / Clear"):
        st.session_state.clear()
        st.experimental_rerun()

# Main
if "vector_store" not in st.session_state:
    st.session_state["vector_store"] = None
    st.session_state["doc_name"] = None

if uploaded:
    # save uploaded file to disk (temporary)
    file_path = f"data/{uploaded.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded.getbuffer())
    st.success(f"Saved `{uploaded.name}`")
    text = load_document(file_path)
    st.session_state["vector_store"] = build_vector_store(text)
    st.session_state["doc_name"] = uploaded.name
    st.success("Document processed and vector store created. Ready to ask questions.")

st.markdown("---")
st.subheader("Ask a question (answers only from uploaded document)")
query = st.text_input("Your question", key="query_input")

if st.button("Ask") and query.strip():
    if st.session_state.get("vector_store") is None:
        st.error("Upload a document first.")
    else:
        # run retrieval to show context and then RAG to get the LLM answer
        hits = similarity_search(query, st.session_state["vector_store"], k=4)
        st.markdown("**Retrieved context (top chunks):**")
        for i, h in enumerate(hits):
            st.markdown(f"**Chunk {i+1} (score {h['score']:.3f}):**")
            st.text(h["text"][:1000])  # show first 1000 chars to keep UI tidy
            st.markdown("---")
        # get final answer
        answer = rag_pipeline(query, st.session_state["vector_store"])
        st.markdown("**AI Answer:**")
        st.write(answer)

st.markdown("---")
st.caption("Built with Sentence-Transformers + FAISS + Ollama (local LLM).")
