from typing import Any, Literal, TypedDict
from pydantic import BaseModel


class RetrievedChunkModel(BaseModel):
    id: int
    source: str
    chunk_index: int
    content: str
    score: float


class RAGResult(BaseModel):
    answer: str
    retrieval_mode: Literal["bm25", "vector"]
    chunks: list[RetrievedChunkModel]


class RAGDeps(BaseModel):
    model_config = {"arbitrary_types_allowed": True}
    bm25: Any
    vector: Any
    mode: Literal["bm25", "vector"] = "bm25"
    top_k: int = 5
    # Populated by the retrieve_chunks tool so the actual retrieved chunks can
    # be attached to the result instead of round-tripping through the LLM.
    retrieved: list[RetrievedChunkModel] = []


class RetrievedChunk(TypedDict):
    id: int
    source: str
    chunk_index: int
    content: str
    score: float