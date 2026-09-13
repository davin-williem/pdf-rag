"""
Shared RAG engine -- used by both the CLI (rag_cli.py) and the web API (app.py)
so ingestion and question-answering logic lives in exactly one place.
"""

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

from pdf_processor import process_pdf
from vector_store import VectorStore

LLM_MODEL = "llama3.2"
TOP_K = 4

SYSTEM_PROMPT = """You are a document assistant. You answer questions using ONLY the \
numbered context snippets provided below.

Rules:
- Every factual claim in your answer must end with a citation marker like [1] or [2], \
referring to the snippet number it came from.
- If the answer isn't in the provided context, say so plainly. Do not guess or use \
outside knowledge.
- Keep answers concise and directly responsive to the question.
"""

# One shared store instance -- avoids reloading the embedding client on every call.
_store: VectorStore | None = None


def get_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store


def ingest_document(pdf_path: str, source_name: str | None = None) -> dict:
    """Extract, chunk, embed, and store a PDF. Returns a small summary dict."""
    chunks = process_pdf(pdf_path, source_name=source_name)
    if not chunks:
        return {"success": False, "chunks_added": 0, "message": "No extractable text found."}

    store = get_store()
    store.add_chunks(chunks)
    return {"success": True, "chunks_added": len(chunks), "message": "Document ingested."}


def answer_question(question: str, top_k: int = TOP_K) -> dict:
    """Retrieve relevant chunks and generate a cited answer. Returns answer + sources."""
    store = get_store()
    if store.document_count() == 0:
        return {
            "answer": "No documents have been uploaded yet.",
            "sources": [],
        }

    hits = store.search(question, top_k=top_k)

    context_block = "\n\n".join(
        f"[{i+1}] (source: {h['source_file']}, page {h['page_number']})\n{h['text']}"
        for i, h in enumerate(hits)
    )

    llm = ChatOllama(model=LLM_MODEL)
    response = llm.invoke(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Context:\n{context_block}\n\nQuestion: {question}"),
        ]
    )

    sources = [
        {
            "label": i + 1,
            "source_file": h["source_file"],
            "page_number": h["page_number"],
            "relevance": round(1 - h["distance"], 3),
            "excerpt": h["text"][:200],
        }
        for i, h in enumerate(hits)
    ]

    return {"answer": response.content, "sources": sources}