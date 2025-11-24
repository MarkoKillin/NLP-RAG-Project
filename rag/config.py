from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

RAW_DATA_DIR = BASE_DIR / "data" / "raw"
INDEX_DIR = BASE_DIR / "index" / "lucene_index"

CHUNK_SIZE = 400
CHUNK_OVERLAP = 50 

TOP_K = 5  # number of chunks to retrieve

EMBEDDING_MODEL_NAME = "nomic-embed-text"
EMBEDDING_DIM = 768

import os

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "mistral")

