# 📚 RAG Document Assistant

Upload PDFs, ask questions, get answers with page-level citations —
fully local, fully free, no API keys required.

## Overview

This project is a Retrieval-Augmented Generation (RAG) pipeline: it extracts
text from PDFs, chunks and embeds it, stores it in a vector database, and
uses an LLM to answer questions grounded in that content — with every claim
traceable back to the exact source page.

Available both as a web app and a CLI.

## Features

- 📄 PDF upload and text extraction (page numbers tracked per chunk)
- ✂️ Overlapping chunking to preserve context across boundaries
- 🧠 Local embeddings via Ollama (`nomic-embed-text`)
- 🔍 Vector similarity search via ChromaDB
- 💬 Local LLM answer generation via Ollama (`llama3.2`) through LangChain
- 📌 Citations mapped by code, not trusted from the LLM — so they can't be hallucinated
- 🌐 Minimal web UI, plus a CLI for scripting

## Tech Stack

| Layer | Tool |
|---|---|
| Backend framework | FastAPI |
| PDF extraction | PyMuPDF |
| Embeddings | Ollama (`nomic-embed-text`) via LangChain |
| Vector database | ChromaDB (local, persisted to disk) |
| LLM | Ollama (`llama3.2`) via LangChain |
| Frontend | Vanilla HTML/JS (no build step) |
| Package management | uv |

## Architecture

```
                 ┌─────────────┐
   PDF Upload →  │ pdf_processor│ → text + page numbers
                 └──────┬──────┘
                        ↓ chunking (overlapping windows)
                 ┌─────────────┐
                 │ vector_store │ → embed (Ollama) → store (ChromaDB)
                 └──────┬──────┘
                        ↓
   Question →  embed query → similarity search → top-k chunks
                        ↓
                 ┌─────────────┐
                 │  rag_engine  │ → numbered context + question → LLM
                 └──────┬──────┘
                        ↓
              Answer with [N] citations mapped back to real page numbers
```

`rag_engine.py` holds all of this logic once; `app.py` (FastAPI) and
`rag_cli.py` are thin interfaces on top of it, sharing the same document
store.

## Project Structure

```
rag_assistant/
├── app.py              # FastAPI web backend (/documents, /ask, /status)
├── rag_cli.py          # CLI entry point
├── rag_engine.py       # shared ingestion + Q&A logic
├── pdf_processor.py    # PDF extraction + chunking
├── vector_store.py     # ChromaDB storage + Ollama embeddings
├── requirements.txt
├── README.md
└── static/
    └── index.html      # browser UI
```

## Setup

1. **Install [Ollama](https://ollama.com/download)** (one-time).

2. **Pull the models:**
   ```bash
   ollama pull llama3.2
   ollama pull nomic-embed-text
   ```

3. **Install dependencies:**
   ```bash
   uv add -r requirements.txt
   ```
   > If your project was created with `uv init` in library mode, add this
   > to `pyproject.toml` first:
   > ```toml
   > [tool.uv]
   > package = false
   > ```

## Usage

**Web app (recommended):**
```bash
uv run uvicorn app:app --reload
```
Open `http://localhost:8000` — upload a PDF, ask questions in the browser.

**CLI:**
```bash
uv run python rag_cli.py add /path/to/document.pdf
uv run python rag_cli.py ask "What does the document say about pricing?"
uv run python rag_cli.py count
```

Both share the same `chroma_db/` store — documents added via one are
searchable from the other.

## How Citations Stay Accurate

A common failure mode in RAG demos is the LLM inventing plausible-looking
page numbers. This project avoids that: retrieved chunks are numbered
`[1]`, `[2]`, etc. in the prompt, the LLM is only asked to cite by number,
and the mapping from `[N]` → actual page number is done in code — never
trusted from model output.

## Roadmap

- [x] Core pipeline: extraction → chunking → embeddings → retrieval → generation
- [x] Web app (FastAPI + minimal UI)
- [ ] Reranking / hybrid search for better retrieval quality
- [ ] Background ingestion queue for large documents + OCR for scanned PDFs
- [ ] Multi-user support (auth, per-user document scoping)

## What This Project Demonstrates

Python → REST APIs → embeddings → vector search → RAG → prompt engineering,
end to end, running entirely on local/free tooling.
