from src.chunker import chunk_text, chunk_documents, Chunk
from src.loader import Document


def test_chunk_text_splits_long_text():
    # Create a text that's definitely longer than the default chunk size
    long_text = "Hello world. " * 1000
    chunks = chunk_text(long_text, chunk_size=100, overlap=10)
    assert len(chunks) > 1
    # Each chunk should be non-empty
    for chunk in chunks:
        assert len(chunk.strip()) > 0


def test_chunk_text_short_text_single_chunk():
    short_text = "This is a short sentence."
    chunks = chunk_text(short_text, chunk_size=100, overlap=10)
    assert len(chunks) == 1
    assert chunks[0].strip() == short_text


def test_chunk_text_overlap():
    # With overlap, consecutive chunks should share some content
    text = "word " * 500
    chunks = chunk_text(text, chunk_size=50, overlap=10)
    assert len(chunks) > 1


def test_chunk_documents():
    docs = [
        Document(text="Short text", metadata={"source": "test"}),
        Document(text="Another short text", metadata={"source": "test2"}),
    ]
    chunks = chunk_documents(docs)
    assert len(chunks) >= 2
    assert all(isinstance(c, Chunk) for c in chunks)
    assert chunks[0].metadata["source"] == "test"
    assert chunks[0].metadata["chunk_index"] == 0
