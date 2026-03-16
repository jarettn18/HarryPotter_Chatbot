import ollama

from src.config import OLLAMA_MODEL

SYSTEM_PROMPT = """You are a knowledgeable assistant that answers questions about Harry Potter.
Use ONLY the provided context passages to answer the question.
If the context doesn't contain enough information to answer, say so honestly.
Cite which passage(s) you're drawing from when possible."""


def format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a context string for the prompt."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(f"[Passage {i}]\n{chunk['text']}\n")
    return "\n".join(parts)


def generate(query: str, context_chunks: list[dict]) -> str:
    """Generate an answer using Ollama with retrieved context."""
    context = format_context(context_chunks)

    user_message = f"""Context passages:
{context}

Question: {query}

Answer based on the context above:"""

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
    )

    return response.message.content
