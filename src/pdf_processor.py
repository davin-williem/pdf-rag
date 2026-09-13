"""
PDF extraction and chunking.

Key design decision: we track page numbers PER CHUNK from the very start,
because reconstructing "which page did this text come from" after the
fact is unreliable. Each chunk carries its source page as metadata.
"""

from dataclasses import dataclass
import fitz  # pymupdf


@dataclass
class Chunk:
    text: str
    page_number: int  # 1-indexed, matches what a human sees in a PDF viewer
    source_file: str
    chunk_id: str


def extract_pages(pdf_path: str) -> list[tuple[int, str]]:
    """Returns a list of (page_number, page_text) tuples, 1-indexed."""
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text().strip()
        if text:  # skip blank pages (e.g. scanned-image-only pages with no OCR)
            pages.append((i + 1, text))
    doc.close()
    return pages


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """
    Simple word-based sliding window chunker.
    chunk_size / overlap are in words, not tokens (good enough approximation
    for a first version -- ~0.75 words per token on average English text).
    """
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = end - overlap  # step forward, keeping overlap words of context
    return chunks


def process_pdf(pdf_path: str, source_name: str | None = None) -> list[Chunk]:
    """Full pipeline: extract pages -> chunk each page -> return Chunk objects."""
    source_name = source_name or pdf_path.split("/")[-1]
    pages = extract_pages(pdf_path)

    all_chunks: list[Chunk] = []
    for page_number, page_text in pages:
        for i, piece in enumerate(chunk_text(page_text)):
            chunk_id = f"{source_name}::p{page_number}::c{i}"
            all_chunks.append(
                Chunk(
                    text=piece,
                    page_number=page_number,
                    source_file=source_name,
                    chunk_id=chunk_id,
                )
            )
    return all_chunks


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python pdf_processor.py <path_to_pdf>")
        sys.exit(1)

    chunks = process_pdf(sys.argv[1])
    print(f"Extracted {len(chunks)} chunks")
    for c in chunks[:3]:
        print(f"\n--- {c.chunk_id} (page {c.page_number}) ---")
        print(c.text[:200] + "...")