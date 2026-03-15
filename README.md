# Harry Potter RAG Pipeline

A Retrieval-Augmented Generation (RAG) pipeline that lets you ask questions about Harry Potter. It loads the full text from a HuggingFace dataset, chunks and embeds it into a local vector database, then retrieves relevant passages to ground Claude's answers in actual source text.

## How It Works

RAG solves a core LLM limitation: models can only answer from their training data. By retrieving relevant documents at query time and injecting them into the prompt, the model can answer grounded in specific source material.

This pipeline has two phases:

### Ingestion (one-time)

```
HuggingFace Dataset → Chunking → Embedding → ChromaDB
```

1. **Load** — Downloads the Harry Potter text dataset from HuggingFace (`elricwan/HarryPotter`)
2. **Chunk** — Splits the text into overlapping segments of ~512 tokens using a token-aware splitter (via `tiktoken`). Overlap ensures context isn't lost at chunk boundaries.
3. **Embed** — Each chunk is converted into a vector (list of floats) using `all-MiniLM-L6-v2`, an open-source sentence-transformers model that runs locally. These vectors capture semantic meaning — similar text produces similar vectors.
4. **Store** — Chunks and their embeddings are stored in ChromaDB, a local vector database that supports fast cosine similarity search.

### Query (per question)

```
Question → Embed → Search ChromaDB → Build Prompt → Claude → Answer
```

1. **Embed the question** — The user's question is embedded using the same model
2. **Retrieve** — ChromaDB finds the top 5 most semantically similar chunks via cosine similarity
3. **Generate** — The retrieved passages are formatted into a prompt and sent to Claude (`claude-haiku-4-5`) with instructions to answer using only the provided context
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
| `CLAUDE_MODEL` | `claude-haiku-4-5-20251001` | Claude model used for generation |
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

## Chatbot in action

```
$ python -m src.cli chat

Harry Potter RAG Chat (type 'quit' to exit)
```
### Query: What are the Deathly Hallows?

### Response:
```
# The Deathly Hallows

Based on the provided passages, the Deathly Hallows are three magical objects from "The Tale of the Three Brothers," a children's story in *The Tales of Beedle the Bard*.

According to Xenophilius Lovegood [Passage 2], the three Hallows are:

1. **The Elder Wand** - represented by a straight vertical line
2. **The Resurrection Stone** - represented by a circle
3. **The Cloak of Invisibility** - represented by a triangle enclosing the other two symbols

[Passage 2] explains that "the ancient story refers to three objects, or Hallows, which, if united, will make the possessor master of Death."

However, it's important to note that [Passage 2] clarifies the original story contains "no mention of the words 'Deathly Hallows'" — this interpretation comes from those who believe in their significance. The story itself is described [Passage 5] as "a morality tale" told to teach lessons to children, though some characters, like Xenophilius, believe the Hallows are real objects that actually exist.
```

### Query: How is polyjuice potion made?

### Response:
```
The passages tell us that:

- **Polyjuice Potion is a transformation potion** that changes you into somebody else, and the effects wear off after a while (Passage 2)
- **The recipe is found in a book called "Moste Potente Potions"** which is located in the Restricted Section of the library (Passage 2)
- **The potion has a mudlike appearance** when being brewed (Passage 1)
- **It requires human hair as an ingredient** - specifically hair from the person you wish to transform into (Passages 3 and 4)
- **The potion must be drunk** to take effect, and the transformation process is physically uncomfortable and involves writhing, burning sensations, and a melting feeling (Passage 4)

However, the context passages do not contain the actual step-by-step instructions or complete ingredient list for making Polyjuice Potion itself. To find that information, you would need to consult "Moste Potente Potions" as mentioned in Passage 2.
```

### Query: Describe the relationship between Snape and Lily Potter.

### Response:
```
# Snape and Lily Potter's Relationship

Based on the provided passages, Snape and Lily had a complicated relationship that evolved over time:

## Early Friendship
In their younger years at Hogwarts, Snape and Lily were **close friends**. Snape refers to her as his "best friend" (Passage 2), and they spent time together, including a memorable encounter on the Hogwarts Express during their first year (Passage 3).

## Growing Tension
However, their friendship became strained due to several conflicts:
- Lily disapproved of Snape's association with students involved in Dark Magic, particularly friends like Avery and Mulciber (Passage 2)
- Snape harbored resentment toward James Potter and his friends, while Lily defended them
- Snape was obsessed with exposing what he saw as the wrongdoings of Potter and his group (Passage 2)

## Unrequited Feelings
The passages suggest Snape had deeper romantic feelings for Lily. When she and James "put their faith in the wrong person," Dumbledore asks Snape: "Weren't you hoping that Lord Voldemort would spare her?" (Passage 4). After Lily's death, Snape expresses profound grief and guilt, saying "I wish . . . I wish I were dead" (Passage 4).

## Legacy
Dumbledore ultimately asks Snape to protect Harry Potter "if you loved Lily Evans, if you truly loved her" (Passage 4), connecting Snape's devotion to Lily with his subsequent protection of her son.
```

