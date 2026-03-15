import chromadb

from src.config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME
from src.chunker import Chunk


class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
        self.collection = self.client.get_or_create_collection(
            name=CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def has_data(self) -> bool:
        """Check if the collection already has indexed data."""
        return self.collection.count() > 0

    def index_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Store chunks with their embeddings in ChromaDB."""
        # ChromaDB has a batch limit, so we insert in groups
        batch_size = 500
        for i in range(0, len(chunks), batch_size):
            end = min(i + batch_size, len(chunks))
            batch_chunks = chunks[i:end]
            batch_embeddings = embeddings[i:end]

            ids = [f"chunk_{i + j}" for j in range(len(batch_chunks))]
            documents = [c.text for c in batch_chunks]
            metadatas = [
                {k: str(v) for k, v in c.metadata.items()}
                for c in batch_chunks
            ]

            self.collection.add(
                ids=ids,
                documents=documents,
                embeddings=batch_embeddings,
                metadatas=metadatas,
            )
            print(f"  Indexed batch {i // batch_size + 1}")

        print(f"Indexed {len(chunks)} chunks in ChromaDB")

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[dict]:
        """Search for the most similar chunks to the query embedding."""
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )

        hits = []
        for i in range(len(results["ids"][0])):
            hits.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "distance": results["distances"][0][i] if results["distances"] else None,
            })

        return hits
