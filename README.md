Mini RAG Bot (Local Document-Based Chatbot)

This project is a local Retrieval-Augmented Generation (RAG) chatbot built to understand how modern document-based question answering systems work in practice.

The system allows a user to upload a document and ask questions. Answers are generated only from the content of the uploaded document, using semantic search and a local large language model. No cloud APIs are used.

This project was developed as a learning exercise by a final-year Chemical Engineering student exploring Data Science, Machine Learning, and applied AI systems.

Project Overview

The chatbot follows a standard RAG pipeline:

A document is uploaded by the user.

The document text is extracted and split into smaller chunks.

Each chunk is converted into embeddings using a sentence-transformer model.

The embeddings are stored in a FAISS vector index.

When a question is asked, the most relevant chunks are retrieved.

The retrieved context and the question are passed to a local LLM via Ollama.

The final answer is generated based only on the retrieved context.

Key Features

Runs completely locally (no external API calls)

Uses semantic search instead of keyword matching

Answers are grounded in the uploaded document

Simple Streamlit-based user interface

Modular code structure for learning and experimentation

Technology Stack

Python

Streamlit (user interface)

Sentence-Transformers (text embeddings)

FAISS (vector similarity search)

Ollama (local LLM runtime)

Llama 3 (language model)

LangChain (Ollama wrapper)

Repository Structure

mini-rag-bot/
│── app.py                 # Streamlit application
│── rag_engine.py          # Embedding, FAISS, and RAG logic
│── document_loader.py     # Document loading and text extraction
│── run_loader_tests.py    # Manual loader test script
│── test_rag.py            # End-to-end RAG test
│── sample.txt             # Sample document
│── README.md
│── .gitignore


How to Run the Project Locally
Prerequisites

Python 3.9 or later

Git

Ollama installed locally

Install Ollama from:
https://ollama.com

After installation, pull a model:
ollama run llama3

Clone the Repository
git clone https://github.com/zmuskan/mini-rag-bot.git
cd mini-rag-bot

Create and Activate Virtual Environment
python -m venv venv


Windows:

venv\Scripts\activate


Linux / macOS:

source venv/bin/activate

Install Dependencies
pip install -r requirements.txt


Note:
OCR-related libraries are optional and only required for scanned PDFs or image documents.

Run the Application
streamlit run app.py


The application will open in the browser at:

http://localhost:8501

Testing

To test document loading:

python run_loader_tests.py


To test the full RAG pipeline:

python test_rag.py

Current Limitations

Vector index is created in memory and not persisted

Chunking strategy is basic

Single-document support only

No citation highlighting in answers

These limitations are intentional to keep the project simple and focused on learning core concepts.

Learning Outcome

Through this project, I learned:

How Retrieval-Augmented Generation systems work internally

The role of embeddings and vector similarity search

How local LLMs can be integrated using Ollama

Structuring an AI application with clear separation of concerns

Author

Zaiba Muskan
Final-year B.Tech Chemical Engineering student
Exploring Data Science, Machine Learning, and applied AI systems