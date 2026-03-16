import os
from dotenv import load_dotenv

load_dotenv()

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

CHUNK_SIZE = 512       # tokens
CHUNK_OVERLAP = 50     # tokens
TOP_K = 5              # number of chunks to retrieve

CHROMA_PERSIST_DIR = "./chroma_db"
CHROMA_COLLECTION_NAME = "harry_potter"
