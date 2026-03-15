import os
from dotenv import load_dotenv

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CLAUDE_MODEL = "claude-haiku-4-5-20251001"

CHUNK_SIZE = 512       # tokens
CHUNK_OVERLAP = 50     # tokens
TOP_K = 5              # number of chunks to retrieve

CHROMA_PERSIST_DIR = "./chroma_db"
CHROMA_COLLECTION_NAME = "harry_potter"
