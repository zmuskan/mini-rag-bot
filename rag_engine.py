# rag_engine.py
# Simple, dependency-minimal RAG backend using sentence-transformers + faiss + Ollama

from langchain_ollama import ChatOllama
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from typing import List, Dict

# --------- Embedding model (sentence-transformers) ----------
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

# --------- Local LLM via Ollama ----------
# Make sure this name matches the model you ran with `ollama run <name>`
LLM_MODEL_NAME = "llama3"   # change to "llama2" or whatever you used if needed
llm = ChatOllama(model=LLM_MODEL_NAME, temperature=0.2)

# --------- Simple chunker (naive) ----------
def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> List[str]:
    tokens = text.split()
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = tokens[i : i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks

# --------- Build FAISS index + store metadata ----------
def build_vector_store(text: str):
    """
    Returns a dict with:
      - 'index' : faiss index (IndexFlatIP normalized)
      - 'metas' : list of dicts { 'text': chunk_text, 'id': idx }
    """
    chunks = chunk_text(text, chunk_size=200, overlap=50)  # use smaller chunk tokens for short docs
    # create embeddings (numpy array)
    embeddings = embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
    # normalize (for inner-product similarity which behaves like cosine when normalized)
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product
    index.add(embeddings.astype('float32'))
    metas = [{"id": i, "text": chunks[i]} for i in range(len(chunks))]
    return {"index": index, "metas": metas, "embeddings": embeddings}

# --------- Similarity search helper ----------
def similarity_search(query: str, store: Dict, k: int = 4):
    """
    Returns top-k chunk texts for the query.
    """
    q_emb = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = store["index"].search(q_emb.astype('float32'), k)
    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0:
            continue
        results.append({"score": float(score), "text": store["metas"][idx]["text"]})
    return results

# --------- RAG pipeline ----------
def rag_pipeline(query: str, vector_store):
    docs = similarity_search(query, vector_store, k=4)
    context = "\n\n".join([d["text"] for d in docs])

    prompt = f"""You are an assistant answering ONLY from the provided context.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{query}
"""
    # call the local LLM (Ollama)
    response = llm.invoke(prompt)
    try:
        return response.content
    except Exception:
        return str(response)
