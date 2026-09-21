import os
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent


def _env_flag(name: str, default: bool) -> bool:
    return os.getenv(name, "1" if default else "0").strip().lower() in {"1", "true", "yes", "on"}


RAW_DATA_DIR = Path(os.getenv("RAW_DATA_DIR", str(BASE_DIR / "data" / "raw")))
INDEX_DIR = Path(os.getenv("INDEX_DIR", str(BASE_DIR / "index")))

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "400"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

TOP_K = int(os.getenv("TOP_K", "5"))

# Recorded in the index so queries tokenize the way the corpus did. Defaults are
# what `scripts/evaluate.py --ablation` measured best on the sample corpus; the
# numbers are in the README. Re-run it on your own corpus.
BM25_STEM = _env_flag("BM25_STEM", True)
BM25_REMOVE_STOPWORDS = _env_flag("BM25_REMOVE_STOPWORDS", False)

# RRF_K damps low-ranked hits; 60 comes from Cormack et al. Fusion needs deeper
# lists than the final top_k, so each retriever returns
# top_k * HYBRID_CANDIDATE_MULTIPLIER.
RRF_K = int(os.getenv("RRF_K", "60"))
HYBRID_CANDIDATE_MULTIPLIER = int(os.getenv("HYBRID_CANDIDATE_MULTIPLIER", "4"))

EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "hf.co/Snowflake/snowflake-arctic-embed-m-v1.5:BF16")

# arctic-embed is asymmetric: the query takes this prefix, documents are embedded
# raw. Prefixing both sides hurts retrieval (applied in VectorRetriever.search).
# Set empty for a symmetric model.
EMBEDDING_QUERY_PREFIX = os.getenv(
    "EMBEDDING_QUERY_PREFIX",
    "Represent this sentence for searching relevant passages: ",
)

# Base Ollama URL without the OpenAI-compatible /v1 suffix.
# The chat model appends /v1 itself; the embedding endpoint uses the bare host.
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434").rstrip("/")
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "hf.co/google/gemma-2b-it")