import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent

RAW_DATA_DIR = Path(os.getenv("RAW_DATA_DIR", str(BASE_DIR / "data" / "raw")))
INDEX_DIR = Path(os.getenv("INDEX_DIR", str(BASE_DIR / "index" / "lucene_index")))

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "400"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

TOP_K = int(os.getenv("TOP_K", "5"))

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "hf.co/Snowflake/snowflake-arctic-embed-m-v1.5:BF16")
EMBEDDING_DIM = int(os.getenv("EMBEDDING_DIM", "768"))

# Base Ollama URL without the OpenAI-compatible /v1 suffix.
# The chat model appends /v1 itself; the embedding endpoint uses the bare host.
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434").rstrip("/")
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "hf.co/google/gemma-2b-it")