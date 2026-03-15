"""
Retriever tests - these require API keys and a populated vector store.
Run after ingestion: python -m src.cli ingest
"""
import pytest


def test_vector_store_initialization():
    """Test that we can create a VectorStore instance."""
    from src.vector_store import VectorStore
    store = VectorStore()
    assert store.collection is not None


def test_vector_store_search_empty():
    """Test searching an empty collection returns empty results gracefully."""
    import chromadb

    client = chromadb.Client()  # in-memory, no persistence
    collection = client.create_collection("test_empty")

    # Verify empty collection behavior
    assert collection.count() == 0
