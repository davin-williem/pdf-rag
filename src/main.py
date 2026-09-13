"""
RAG Document Assistant -- CLI

Usage:
    python main.py add <path_to_pdf>       # ingest a PDF
    python main.py ask "<question>"        # ask a question
    python main.py count                   # show how many chunks are stored

Requires Ollama running locally with a model pulled, e.g.:
    ollama pull llama3.2
"""

import sys

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from rich import print as rprint
from rich.panel import Panel

from pdf_processor import process_pdf
from vector_store import VectorStore

LLM_MODEL = "llama3.2"  # change to whatever you've pulled in Ollama
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


def add_document(pdf_path: str) -> None:
    store = VectorStore()
    chunks = process_pdf(pdf_path)
    if not chunks:
        rprint(f"[red]No extractable text found in {pdf_path}.[/red] "
               "(It may be a scanned/image-only PDF -- would need OCR.)")
        return
    store.add_chunks(chunks)


def ask_question(question: str) -> None:
    store = VectorStore()
    if store.document_count() == 0:
        rprint("[red]No documents in the store yet. Run `add <pdf_path>` first.[/red]")
        return

    hits = store.search(question, top_k=TOP_K)

    # Build a numbered context block. We control the mapping from [N] -> real
    # page number ourselves, rather than trusting the LLM to output correct
    # page numbers -- this is the safer citation pattern.
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
    answer = response.content

    rprint(Panel(answer, title="Answer", border_style="cyan"))

    rprint("\n[bold]Sources:[/bold]")
    for i, h in enumerate(hits):
        rprint(f"  [{i+1}] {h['source_file']} -- page {h['page_number']} "
               f"(relevance: {1 - h['distance']:.2f})")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]

    if command == "add" and len(sys.argv) == 3:
        add_document(sys.argv[2])
    elif command == "ask" and len(sys.argv) == 3:
        ask_question(sys.argv[2])
    elif command == "count":
        store = VectorStore()
        rprint(f"Chunks stored: {store.document_count()}")
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()