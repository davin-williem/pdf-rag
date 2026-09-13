"""
Vector storage and retrieval using ChromaDB (local, embedded -- no server needed)
and sentence-transformers for embeddings (local, free, no API key).
"""

import chromadb
from sentence_transformers import SentenceTransformer

from pdf_processor import Chunk

EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # small, fast, good enough to start (~80MB download)
DB_PATH = "./chroma_db"
COLLECTION_NAME = "documents"


class VectorStore:
    def __init__(self):
        print(f"Loading embedding model '{EMBEDDING_MODEL}' (first run downloads it)...")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        self.client = chromadb.PersistentClient(path=DB_PATH)
        self.collection = self.client.get_or_create_collection(COLLECTION_NAME)

    def add_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        texts = [c.text for c in chunks]
        embeddings = self.embedder.encode(texts, show_progress_bar=True).tolist()

        self.collection.add(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=texts,
            metadatas=[
                {"page_number": c.page_number, "source_file": c.source_file}
                for c in chunks
            ],
        )
        print(f"Stored {len(chunks)} chunks in vector DB.")

    def search(self, query: str, top_k: int = 4) -> list[dict]:
        query_embedding = self.embedder.encode([query]).tolist()
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
        )

        hits = []
        for i in range(len(results["ids"][0])):
            hits.append(
                {
                    "chunk_id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "page_number": results["metadatas"][0][i]["page_number"],
                    "source_file": results["metadatas"][0][i]["source_file"],
                    "distance": results["distances"][0][i],
                }
            )
        return hits

    def document_count(self) -> int:
        return self.collection.count()