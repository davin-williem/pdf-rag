"""
Vector storage and retrieval using ChromaDB (local, embedded -- no server needed)
and Ollama's nomic-embed-text for embeddings (local, free, no API key, no torch
dependency -- reuses the same Ollama service you already run for the LLM).
"""

import chromadb
from langchain_ollama import OllamaEmbeddings

from pdf_processor import Chunk

EMBEDDING_MODEL = "nomic-embed-text"  # pull first: `ollama pull nomic-embed-text`
DB_PATH = "./chroma_db"
COLLECTION_NAME = "documents"


class VectorStore:
    def __init__(self):
        self.embedder = OllamaEmbeddings(model=EMBEDDING_MODEL)
        self.client = chromadb.PersistentClient(path=DB_PATH)
        self.collection = self.client.get_or_create_collection(COLLECTION_NAME)

    def add_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed_documents(texts)

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
        query_embedding = self.embedder.embed_query(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
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