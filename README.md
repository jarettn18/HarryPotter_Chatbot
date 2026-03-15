# Harry Potter RAG Pipeline

A Retrieval-Augmented Generation (RAG) pipeline that lets you ask questions about Harry Potter. It loads the full text from a HuggingFace dataset, chunks and embeds it into a local vector database, then retrieves relevant passages to ground Claude's answers in actual source text.

## How It Works

RAG solves a core LLM limitation: models can only answer from their training data. By retrieving relevant documents at query time and injecting them into the prompt, the model can answer grounded in specific source material.

This pipeline has two phases:

### Ingestion (one-time)

```
HuggingFace Dataset → Chunking → Embedding → ChromaDB
```

1. **Load** — Downloads the Harry Potter text dataset from HuggingFace (`AlekseyKorshuk/harry-potter`)
2. **Chunk** — Splits the text into overlapping segments of ~512 tokens using a token-aware splitter (via `tiktoken`). Overlap ensures context isn't lost at chunk boundaries.
3. **Embed** — Each chunk is converted into a vector (list of floats) using `all-MiniLM-L6-v2`, an open-source sentence-transformers model that runs locally. These vectors capture semantic meaning — similar text produces similar vectors.
4. **Store** — Chunks and their embeddings are stored in ChromaDB, a local vector database that supports fast cosine similarity search.

### Query (per question)

```
Question → Embed → Search ChromaDB → Build Prompt → Claude → Answer
```

1. **Embed the question** — The user's question is embedded using the same model
2. **Retrieve** — ChromaDB finds the top 5 most semantically similar chunks via cosine similarity
3. **Generate** — The retrieved passages are formatted into a prompt and sent to Claude (`claude-sonnet-4-6`) with instructions to answer using only the provided context
4. **Return** — Claude's response is returned, grounded in the actual Harry Potter text

## Project Structure

```
src/
├── config.py         # API keys, model names, chunk size settings
├── loader.py         # Loads the HuggingFace dataset into Document objects
├── chunker.py        # Token-aware text splitting with overlap
├── embedder.py       # Local embedding via sentence-transformers
├── vector_store.py   # ChromaDB wrapper for indexing and similarity search
├── retriever.py      # Combines embedding + vector search into one step
├── generator.py      # Builds the RAG prompt and calls Claude
├── pipeline.py       # Orchestrates the full ingest and query flows
└── cli.py            # Command-line interface
```

## Setup

### Prerequisites

- Python 3.12+
- An [Anthropic API key](https://console.anthropic.com/) (for generation)

### Install

```bash
cd hp-rag
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Configure API Keys

```bash
cp .env.example .env
```

Edit `.env` and add your key:

```
ANTHROPIC_API_KEY=sk-ant-...
```

## Usage

### 1. Ingest the dataset

This downloads the Harry Potter text, chunks it, generates embeddings, and stores everything in a local ChromaDB database. You only need to do this once — the data persists in the `./chroma_db/` directory.

```bash
python -m src.cli ingest
```

### 2. Ask questions

**Interactive chat:**

```bash
python -m src.cli chat
```

    Harry Potter RAG Chat (type 'quit' to exit)

    You: What are the Deathly Hallows?
    Assistant: Based on the text, the Deathly Hallows are three legendary objects...

**Single question:**

```bash
python -m src.cli query "Who is Dobby?"
```

**Show retrieved source passages:**

```bash
python -m src.cli chat --show-sources
python -m src.cli query "What is a horcrux?" --show-sources
```

## Configuration

Settings are in `src/config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local sentence-transformers model for embeddings |
| `CLAUDE_MODEL` | `claude-sonnet-4-6-20250514` | Claude model used for generation |
| `CHUNK_SIZE` | `512` | Max tokens per chunk |
| `CHUNK_OVERLAP` | `50` | Token overlap between consecutive chunks |
| `TOP_K` | `5` | Number of passages retrieved per query |

## Re-ingesting

To re-ingest the dataset (e.g., after changing chunk settings), delete the ChromaDB directory and run ingest again:

```bash
rm -rf ./chroma_db
python -m src.cli ingest
```

## Running Tests

```bash
python -m pytest tests/ -v
```