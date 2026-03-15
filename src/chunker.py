from dataclasses import dataclass, field
import tiktoken

from src.config import CHUNK_SIZE, CHUNK_OVERLAP


@dataclass
class Chunk:
    text: str
    metadata: dict = field(default_factory=dict)


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks based on token count."""
    encoder = tiktoken.get_encoding("cl100k_base")
    tokens = encoder.encode(text)

    chunks = []
    start = 0
    while start < len(tokens):
        end = start + chunk_size
        chunk_tokens = tokens[start:end]
        chunk_text = encoder.decode(chunk_tokens)
        chunks.append(chunk_text)

        start += chunk_size - overlap

    return chunks


def chunk_documents(documents: list) -> list[Chunk]:
    """Chunk a list of Document objects into smaller Chunk objects."""
    chunks = []
    for doc in documents:
        text_chunks = chunk_text(doc.text)
        for i, text in enumerate(text_chunks):
            metadata = {**doc.metadata, "chunk_index": i}
            chunks.append(Chunk(text=text, metadata=metadata))

    print(f"Created {len(chunks)} chunks from {len(documents)} documents")
    return chunks
