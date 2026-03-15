from src.loader import load_harry_potter_dataset
from src.chunker import chunk_documents
from src.embedder import embed_texts
from src.vector_store import VectorStore
from src.retriever import Retriever
from src.generator import generate


class RAGPipeline:
    def __init__(self):
        self.vector_store = VectorStore()
        self.retriever = Retriever(self.vector_store)

    def ingest(self) -> None:
        """Load, chunk, embed, and index the Harry Potter dataset."""
        if self.vector_store.has_data():
            count = self.vector_store.collection.count()
            print(f"ChromaDB already has {count} chunks indexed. Skipping ingestion.")
            print("Delete the ./chroma_db directory to re-ingest.")
            return

        print("=== Loading dataset ===")
        documents = load_harry_potter_dataset()

        print("\n=== Chunking documents ===")
        chunks = chunk_documents(documents)

        print("\n=== Generating embeddings ===")
        texts = [c.text for c in chunks]
        embeddings = embed_texts(texts)

        print("\n=== Indexing in ChromaDB ===")
        self.vector_store.index_chunks(chunks, embeddings)

        print("\nIngestion complete!")

    def query(self, question: str) -> str:
        """Retrieve relevant context and generate an answer."""
        if not self.vector_store.has_data():
            return "No data indexed yet. Run 'ingest' first."

        print(f"\nRetrieving relevant passages...")
        results = self.retriever.retrieve(question)

        print(f"Found {len(results)} relevant passages. Generating answer...\n")
        answer = generate(question, results)

        return answer

    def query_with_sources(self, question: str) -> tuple[str, list[dict]]:
        """Like query(), but also returns the source chunks."""
        if not self.vector_store.has_data():
            return "No data indexed yet. Run 'ingest' first.", []

        results = self.retriever.retrieve(question)
        answer = generate(question, results)
        return answer, results
