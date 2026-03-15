from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL

model = SentenceTransformer(EMBEDDING_MODEL)

BATCH_SIZE = 256


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts using a local sentence-transformers model."""
    all_embeddings = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        print(f"  Embedding batch {i // BATCH_SIZE + 1}/{(len(texts) - 1) // BATCH_SIZE + 1}...")

        batch_embeddings = model.encode(batch, show_progress_bar=False)
        all_embeddings.extend(batch_embeddings.tolist())

    print(f"Generated {len(all_embeddings)} embeddings")
    return all_embeddings
