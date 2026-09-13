"""
RAG Document Assistant -- CLI

Usage:
    python rag_cli.py add <path_to_pdf>       # ingest a PDF
    python rag_cli.py ask "<question>"        # ask a question
    python rag_cli.py count                   # show how many chunks are stored

Requires Ollama running locally with these models pulled:
    ollama pull llama3.2
    ollama pull nomic-embed-text
"""

import sys

from rich import print as rprint
from rich.panel import Panel

import rag_engine


def add_document(pdf_path: str) -> None:
    result = rag_engine.ingest_document(pdf_path)
    if result["success"]:
        rprint(f"[green]Stored {result['chunks_added']} chunks from {pdf_path}[/green]")
    else:
        rprint(f"[red]{result['message']}[/red] "
               "(It may be a scanned/image-only PDF -- would need OCR.)")


def ask_question(question: str) -> None:
    result = rag_engine.answer_question(question)

    rprint(Panel(result["answer"], title="Answer", border_style="cyan"))

    if result["sources"]:
        rprint("\n[bold]Sources:[/bold]")
        for s in result["sources"]:
            rprint(f"  [{s['label']}] {s['source_file']} -- page {s['page_number']} "
                   f"(relevance: {s['relevance']})")


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
        rprint(f"Chunks stored: {rag_engine.get_store().document_count()}")
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()