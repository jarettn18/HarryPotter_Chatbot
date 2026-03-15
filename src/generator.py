from anthropic import Anthropic

from src.config import ANTHROPIC_API_KEY, CLAUDE_MODEL

client = Anthropic(api_key=ANTHROPIC_API_KEY)

SYSTEM_PROMPT = """You are a knowledgeable assistant that answers questions about Harry Potter.
Use ONLY the provided context passages to answer the question.
If the context doesn't contain enough information to answer, say so honestly.
Cite which passage(s) you're drawing from when possible."""


def format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a context string for the prompt."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        source_info = chunk.get("metadata", {})
        parts.append(f"[Passage {i}]\n{chunk['text']}\n")
    return "\n".join(parts)


def generate(query: str, context_chunks: list[dict]) -> str:
    """Generate an answer using Claude with retrieved context."""
    context = format_context(context_chunks)

    user_message = f"""Context passages:
{context}

Question: {query}

Answer based on the context above:"""

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    return response.content[0].text
