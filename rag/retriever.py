import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rag.bm25 import BM25Index
from rag.config import settings
from rag.embedding_model import EmbeddingModel
from rag.ingestion import INDEX_FILE, VECTORS_FILE
from rag.models import RetrievedChunkModel, Retrievers


@dataclass
class LoadedIndex:
    bm25: BM25Index
    chunks: list[dict]
    vectors: np.ndarray


def load_index(index_dir: Path) -> LoadedIndex:
    index_dir = Path(index_dir)
    index_path = index_dir / INDEX_FILE
    vectors_path = index_dir / VECTORS_FILE
    if not index_path.exists() or not vectors_path.exists():
        raise FileNotFoundError(
            f"No index files in {index_dir}. "
            "Build the index first with: python -m scripts.build_index"
        )
    with open(index_path, "rb") as f:
        data = pickle.load(f)
    return LoadedIndex(bm25=data["bm25"], chunks=data["chunks"], vectors=np.load(vectors_path))


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
        query_vec = self.embedding_model.encode([query])[0].astype(np.float32)
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
    """Reciprocal Rank Fusion of the BM25 and vector rankings."""

    def __init__(
        self,
        bm25: BM25Retriever,
        vector: VectorRetriever,
        rrf_k: int = settings.rrf_k,
        candidates: int = settings.hybrid_candidates,
    ):
        self.bm25 = bm25
        self.vector = vector
        self.rrf_k = rrf_k
        self.candidates = candidates

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunkModel]:
        n = max(top_k, self.candidates)
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
    if index is None:
        index = load_index(index_dir or settings.index_dir)
    bm25 = BM25Retriever(index)
    vector = VectorRetriever(index, EmbeddingModel(settings.embedding_model_name))
    return Retrievers(bm25=bm25, vector=vector, hybrid=HybridRetriever(bm25, vector))