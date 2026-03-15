from src.embedder import embed_texts
from src.vector_store import VectorStore
from src.config import TOP_K


class Retriever:
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = TOP_K) -> list[dict]:
        """Embed a query and retrieve the most relevant chunks."""
        query_embedding = embed_texts([query])[0]
        results = self.vector_store.search(query_embedding, top_k=top_k)
        return results
