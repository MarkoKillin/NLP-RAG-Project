import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rag.bm25 import BM25Index
from rag.config import (
    EMBEDDING_MODEL_NAME,
    EMBEDDING_QUERY_PREFIX,
    HYBRID_CANDIDATE_MULTIPLIER,
    INDEX_DIR,
    RRF_K,
)
from rag.embedding_model import EmbeddingModel
from rag.ingestion import INDEX_FILE, INDEX_FORMAT_VERSION, VECTORS_FILE
from rag.models import RetrievedChunkModel, Retrievers

_REBUILD_HINT = "Rebuild it with: python -m scripts.build_index (or REBUILD_INDEX=1 in Docker)."


@dataclass
class LoadedIndex:
    """One in-memory copy of the index, shared by all retrievers."""

    bm25: BM25Index
    chunks: list[dict]
    vectors: np.ndarray


def load_index(index_dir: Path) -> LoadedIndex:
    index_dir = Path(index_dir)
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

    version = data.get("format_version")
    if version != INDEX_FORMAT_VERSION:
        raise ValueError(
            f"Index format v{version} was built by an older version of this code "
            f"(current is v{INDEX_FORMAT_VERSION}). {_REBUILD_HINT}"
        )

    # Compare model names, not shapes: two models at the same dimension give a
    # dot product that computes fine and ranks garbage.
    built_with = data.get("embedding_model")
    if built_with != EMBEDDING_MODEL_NAME:
        raise ValueError(
            f"Index was built with embedding model {built_with!r} but "
            f"EMBEDDING_MODEL_NAME is now {EMBEDDING_MODEL_NAME!r}. "
            f"Query and document vectors would come from different models. {_REBUILD_HINT}"
        )

    if vectors.shape[0] != len(data["chunks"]):
        raise ValueError(
            f"Corrupt index: {vectors.shape[0]} vectors for {len(data['chunks'])} chunks. "
            f"{_REBUILD_HINT}"
        )

    return LoadedIndex(bm25=data["bm25"], chunks=data["chunks"], vectors=vectors)


def _to_chunk(meta: dict, score: float) -> RetrievedChunkModel:
    return RetrievedChunkModel(
        id=meta["id"],
        source=meta["source"],
        chunk_index=meta["chunk_index"],
        content=meta["content"],
        score=float(score),
    )


class BM25Retriever:
    def __init__(self, index: LoadedIndex):
        self.bm25 = index.bm25
        self.chunks = index.chunks

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunkModel]:
        ranked = self.bm25.search(query, top_k=top_k)
        return [_to_chunk(self.chunks[doc_id], score) for doc_id, score in ranked]


class VectorRetriever:
    def __init__(self, index: LoadedIndex, embedding_model: EmbeddingModel):
        self.chunks = index.chunks
        self.vectors = index.vectors
        self.embedding_model = embedding_model

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunkModel]:
        # arctic-embed is asymmetric: prefix the query so it matches how the
        # documents were embedded. The prefix is empty for symmetric models
        # (configured via EMBEDDING_QUERY_PREFIX).
        query_vec = self.embedding_model.encode([EMBEDDING_QUERY_PREFIX + query])[0].astype(np.float32)
        if query_vec.shape[0] != self.vectors.shape[1]:
            raise ValueError(
                f"Embedding dimension mismatch: the model returned {query_vec.shape[0]} dims "
                f"but the index holds {self.vectors.shape[1]}-dim vectors. {_REBUILD_HINT}"
            )

        norm = float(np.linalg.norm(query_vec))
        if norm > 0:
            query_vec = query_vec / norm

        scores = self.vectors @ query_vec
        k = min(top_k, len(scores))
        if k == 0:
            return []
        top_indices = np.argpartition(-scores, k - 1)[:k]
        top_indices = top_indices[np.argsort(-scores[top_indices])]

        return [_to_chunk(self.chunks[int(idx)], scores[idx]) for idx in top_indices]


class HybridRetriever:
    """Reciprocal Rank Fusion over the BM25 and vector rankings.

    Fuses by position, not by score, so BM25's unbounded scores need no
    calibration against cosine. A chunk at rank r contributes 1 / (rrf_k + r).
    Returned scores are RRF scores, roughly 0.01 to 0.03, and do not compare to
    the other two modes.
    """

    def __init__(
        self,
        bm25: BM25Retriever,
        vector: VectorRetriever,
        rrf_k: int = RRF_K,
        candidate_multiplier: int = HYBRID_CANDIDATE_MULTIPLIER,
    ):
        self.bm25 = bm25
        self.vector = vector
        self.rrf_k = rrf_k
        self.candidate_multiplier = max(1, candidate_multiplier)

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunkModel]:
        # Fuse deeper than we return, or a chunk ranked 8th by one retriever and
        # 2nd by the other can never win.
        n = top_k * self.candidate_multiplier
        rankings = [self.bm25.search(query, top_k=n), self.vector.search(query, top_k=n)]

        fused: dict[int, float] = defaultdict(float)
        by_id: dict[int, RetrievedChunkModel] = {}
        for ranking in rankings:
            for rank, chunk in enumerate(ranking):
                fused[chunk.id] += 1.0 / (self.rrf_k + rank + 1)
                by_id[chunk.id] = chunk

        ordered = sorted(fused.items(), key=lambda kv: -kv[1])[:top_k]
        return [by_id[chunk_id].model_copy(update={"score": score}) for chunk_id, score in ordered]


def build_retrievers(
    index_dir: Path | None = None, index: LoadedIndex | None = None
) -> Retrievers:
    """Build all three retrievers over one in-memory copy of the index.

    Call once per process and reuse. Pass an already-loaded ``index`` to reuse it
    (scripts/evaluate.py does this, avoiding a second unpickle); otherwise the
    index is loaded from ``index_dir`` and the same copy is handed to every
    retriever instead of letting each load its own.
    """
    if index is None:
        index = load_index(index_dir or INDEX_DIR)
    bm25 = BM25Retriever(index)
    vector = VectorRetriever(index, EmbeddingModel(EMBEDDING_MODEL_NAME))
    return Retrievers(bm25=bm25, vector=vector, hybrid=HybridRetriever(bm25, vector))