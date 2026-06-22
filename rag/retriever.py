import pickle
from pathlib import Path

import numpy as np

from rag.bm25 import BM25Index, tokenize
from rag.config import EMBEDDING_DIM, EMBEDDING_QUERY_PREFIX
from rag.embedding_model import EmbeddingModel
from rag.ingestion import INDEX_FILE, VECTORS_FILE
from rag.models import RetrievedChunk


def _load_index(index_dir: Path) -> tuple[BM25Index, list[dict], np.ndarray]:
    index_path = index_dir / INDEX_FILE
    vectors_path = index_dir / VECTORS_FILE
    if not index_path.exists() or not vectors_path.exists():
        raise FileNotFoundError(
            f"Index files not found in {index_dir}. "
            "Please build the index first using: python -m scripts.build_index"
        )
    with open(index_path, "rb") as f:
        data = pickle.load(f)
    vectors = np.load(vectors_path)
    return data["bm25"], data["chunks"], vectors


def _chunk_to_result(meta: dict, score: float) -> RetrievedChunk:
    return {
        "id": meta["id"],
        "source": meta["source"],
        "chunk_index": meta["chunk_index"],
        "content": meta["content"],
        "score": float(score),
    }


class BM25Retriever:
    def __init__(self, index_dir: Path):
        self.index_dir = Path(index_dir)
        self.bm25, self.chunks, _ = _load_index(self.index_dir)

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        ranked = self.bm25.search(tokenize(query), top_k=top_k)
        return [_chunk_to_result(self.chunks[doc_id], score) for doc_id, score in ranked]

    def close(self) -> None:
        pass


class VectorRetriever:
    def __init__(self, index_dir: Path, embedding_model: EmbeddingModel):
        self.index_dir = Path(index_dir)
        _, self.chunks, self.vectors = _load_index(self.index_dir)
        self.embedding_model = embedding_model
        self._dim_checked = False

    def _check_dimension(self, query_vec: np.ndarray) -> None:
        # Verify lazily, on the first real query, so constructing the retriever
        # doesn't require the embedding backend (Ollama) to be reachable yet.
        if self._dim_checked:
            return
        if query_vec.shape[0] != EMBEDDING_DIM:
            raise ValueError(
                f"Embedding dimension mismatch: model produced {query_vec.shape[0]}, "
                f"but EMBEDDING_DIM is configured as {EMBEDDING_DIM}. "
                "The index was built against EMBEDDING_DIM; rebuild it or update the env var."
            )
        self._dim_checked = True

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        # arctic-embed is asymmetric: prefix the query so it lands in the same
        # space as the raw-embedded documents. The prefix is empty for symmetric
        # models (configured via EMBEDDING_QUERY_PREFIX).
        query_vec = self.embedding_model.encode([EMBEDDING_QUERY_PREFIX + query])[0].astype(np.float32)
        self._check_dimension(query_vec)
        norm = float(np.linalg.norm(query_vec))
        if norm > 0:
            query_vec = query_vec / norm

        scores = self.vectors @ query_vec
        k = min(top_k, len(scores))
        if k == 0:
            return []
        top_indices = np.argpartition(-scores, k - 1)[:k]
        top_indices = top_indices[np.argsort(-scores[top_indices])]

        return [_chunk_to_result(self.chunks[int(idx)], scores[idx]) for idx in top_indices]

    def close(self) -> None:
        pass
