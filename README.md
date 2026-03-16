# Harry Potter RAG Pipeline

A Retrieval-Augmented Generation (RAG) pipeline that lets you ask questions about Harry Potter. It loads the full text from a HuggingFace dataset, chunks and embeds it into a local vector database, then retrieves relevant passages to ground Ollama's answers in actual source text.

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
Question → Embed → Search ChromaDB → Build Prompt → Ollama (llama3) → Answer
```

1. **Embed the question** — The user's question is embedded using the same model
2. **Retrieve** — ChromaDB finds the top 5 most semantically similar chunks via cosine similarity
3. **Generate** — The retrieved passages are formatted into a prompt and sent to Ollama (`llama3`) with instructions to answer using only the provided context
4. **Return** — The model's response is returned, grounded in the actual Harry Potter text

## Project Structure

```
src/
├── config.py         # API keys, model names, chunk size settings
├── loader.py         # Loads the HuggingFace dataset into Document objects
├── chunker.py        # Token-aware text splitting with overlap
├── embedder.py       # Local embedding via sentence-transformers
├── vector_store.py   # ChromaDB wrapper for indexing and similarity search
├── retriever.py      # Combines embedding + vector search into one step
├── generator.py      # Builds the RAG prompt and calls Ollama
├── pipeline.py       # Orchestrates the full ingest and query flows
└── cli.py            # Command-line interface
```

## Setup

### Prerequisites

- Python 3.12+
- [Ollama](https://ollama.com/) installed and running locally

### Install

```bash
cd hp-rag
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### Set up Ollama

1. Install Ollama from [ollama.com](https://ollama.com/) or via `brew install ollama`
2. Start the Ollama server (open the app, or run `ollama serve`)
3. Pull the llama3 model:

```bash
ollama pull llama3
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
| `OLLAMA_MODEL` | `llama3` | Ollama model used for generation |
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

Harry Potter RAG Chat (type ‘quit’ to exit)
```
### Query: What are the Deathly Hallows?

### Response:
```
  Embedding batch 1/1...
Generated 1 embeddings


According to the passages provided, the Deathly Hallows refer to three objects or Hallows that, if united, will make the possessor master of Death. The three objects are:

1. The Elder Wand
2. The Resurrection Stone
3. The Cloak of Invisibility

(Xenophilius Lovegood, Passage 2)

These objects are mentioned in “The Tale of the Three Brothers” story, which Xenophilius is familiar with and suggests that Hermione read aloud from “The Tales of Beedle the Bard”.
```

### Query: How is polyjuice potion made?

### Response:
```
  Embedding batch 1/1...
Generated 1 embeddings


According to Passage 2, Polyjuice Potion can be made by obtaining the recipe from a book called “Moste Potente Potions” which is located in the Restricted Section of the library. The book requires a signed note of permission from a teacher to access it. Additionally, according to Passage 4, Hermione mentioned that she had already made Polyjuice Potion before, implying that she may have found another way to obtain the recipe or learned how to make it through other means.
```

### Query: Describe the relationship between Snape and Lily Potter.

### Response:
```
  Embedding batch 1/1...
Generated 1 embeddings


Based on Passage 1, Passage 2, and Passage 4, the relationship between Snape and Lily Potter can be described as follows:

Snape and Lily had a complicated history. They were once friends, but their friendship was tainted by Snape’s resentment towards James Potter and his friends. Snape felt that they were getting special treatment and that he was being left out. This led to Snape becoming bitter and jealous.

Lily tried to reason with Snape, pointing out the unfairness of his behavior and telling him not to be ungrateful for her help in saving him from whatever was down the Whomping Willow tunnel. However, Snape’s feelings towards James and his friends remained strong, and he continued to harbor resentment towards them.

Despite their differences, Lily seemed to have a certain affection for Snape, as evidenced by her willingness to talk to him on the Hogwarts Express (Passage 3) and her attempt to defend him from Dumbledore’s criticism (Passage 4).

Snape’s feelings towards Lily were complex. While he was jealous of James and his friends, he also seemed to have a deep affection for Lily herself. He was devastated by her death and felt guilty about not being able to protect her or keep her safe.

In the end, Dumbledore used Snape’s love for Lily as leverage to persuade him to help protect Harry Potter, who was in danger due to his connection to Voldemort (Passage 4).
```

### Query: Give me a couple quotes from Dudley

### Response:
```
  Embedding batch 1/1...
Generated 1 embeddings


Based on the provided context passages, here are a couple of quotes from Dudley:

1. “He was asking for it.” - This quote appears when Harry asks Dudley who he’s been beating up tonight.
2. “Point that thing somewhere else!” - This quote appears when Harry points his wand at Dudley and Dudley tries to avoid getting hit.

Please note that these are just a couple of quotes from the provided context passages, and there might be more quotes from Dudley in the actual text or story.
```

## Query with --show-sources
```
python -m src.cli query “How is polyjuice potion made?” --show-sources
```
### Response
```
  Embedding batch 1/1...
Generated 1 embeddings


According to Passage 2, Polyjuice Potion can be made by transforming into someone else. Hermione explains that it “transforms you into somebody else” and suggests that they could change into three of the Slytherins. However, it’s implied that the recipe for making Polyjuice Potion is difficult to obtain, as Snape mentioned it being in a book called Moste Potente Potions, which is bound to be in the Restricted Section of the library.

Passage 4 mentions that Harry and his friends drink separate portions of Polyjuice Potion, but it doesn’t provide details on how the potion is made.

--- Retrieved Sources ---

[1] ‘t made ‘em yet. Anyone tell me what this one is?”

He indicated the cauldron nearest the Slytherin table. Harry raised himself slightly in his seat and saw what looked like plain water boiling away i...

[2] ,” said Hermione coldly. “What we’d need to do is to get inside the Slytherin common room and ask Malfoy a few questions without him realizing it’s us.”

“But that’s impossible,” Harry said as Ron lau...

[3]  enjoy the unexpected warmth, however, before Hermione’s silent Stunning Spell hit her in the chest and she toppled over.

“Nicely done, Hermione,” said Ron, emerging from behind a bin beside the thea...

[4]  Hermione reached for their glasses. “We’d better not all drink them in here. . . . Once we turn into Crabbe and Goyle we won’t fit. And Millicent Bulstrode’s no pixie.”

“Good thinking,” said Ron, un...

[5]  a heavy and sometimes irreversible sleep, so you will need to pay close attention to what you are doing.” On Harry’s left, Hermione sat up a little straighter, her expression one of the utmost attent...
--- End Sources ---
```
