from document_loader import load_document
from rag_engine import build_vector_store, rag_pipeline

text = load_document("sample.txt")
vector_store = build_vector_store(text)
query = "What is this document about?"
answer = rag_pipeline(query, vector_store)
print("\nAnswer:", answer)
