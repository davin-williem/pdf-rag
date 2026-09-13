# RAG Document Assistant

Ask questions about PDFs with page-level citations. Fully free/local -- no
API keys required. Available as both a CLI and a web app.

## Setup

1. **Install Ollama** (one-time): https://ollama.com/download

2. **Pull the models** (one-time):
   ollama pull llama3.2
   ollama pull nomic-embed-text

3. **Install dependencies** (using uv):
   uv add -r requirements.txt
   If your project was created with `uv init` in library mode, add this to
   `pyproject.toml` first so uv doesn't try to build the project itself as
   an installable package:
   [tool.uv]
   package = false

## Usage
### Web app (recommended)

uv run uvicorn app:app --reload
Then open http://localhost:8000 -- upload a PDF and ask questions in the browser.

### CLI

uv run python rag_cli.py add /path/to/document.pdf
uv run python rag_cli.py ask "What does the document say about pricing?"
uv run python rag_cli.py count

Both the CLI and the web app share the same document store (`./chroma_db/`)
and the same logic (`rag_engine.py`), so documents added via one are
searchable from the other.

## Architecture

pdf_processor.py   -- PDF text extraction + chunking (tracks page numbers)
vector_store.py     -- ChromaDB storage + Ollama embeddings (nomic-embed-text)
rag_engine.py         -- shared ingestion + question-answering logic
rag_cli.py             -- CLI entry point (thin wrapper over rag_engine)
app.py                  -- FastAPI entry point (thin wrapper over rag_engine)
static/index.html        -- minimal browser UI (vanilla JS, no build step)

**How citations stay accurate:** retrieved chunks are numbered `[1]`, `[2]`,
etc. in the prompt, and the LLM is instructed to cite by number. The mapping
from `[N]` back to the real page number is done by our own code, not the
LLM -- so citations can't be hallucinated.

## Scaling roadmap

This project can grow in a few directions, roughly in order of value:
2. **Better answer quality** -- add a reranking step (e.g. a cross-encoder)
   after initial retrieval, or hybrid search (combine vector search with
   keyword/BM25 search) for queries where exact terms matter.
3. **Handle more/larger documents** -- background job queue for ingestion
   (so large PDFs don't block the API), batching embedding calls, OCR for
   scanned PDFs.
4. **Multi-user support** -- auth, and scoping each user's documents to
   their own ChromaDB collection instead of one shared collection.