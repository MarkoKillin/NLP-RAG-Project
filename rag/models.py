from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import BaseModel

RetrievalMode = Literal["bm25", "vector", "hybrid"]


class RetrievedChunkModel(BaseModel):
    id: int
    source: str
    chunk_index: int
    content: str
    score: float


class RAGResult(BaseModel):
    answer: str
    retrieval_mode: RetrievalMode
    chunks: list[RetrievedChunkModel]


class Retriever(Protocol):
    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunkModel]: ...


@dataclass
class Retrievers:
    bm25: Retriever
    vector: Retriever
    hybrid: Retriever

    def get(self, mode: RetrievalMode) -> Retriever:
        return {"bm25": self.bm25, "vector": self.vector, "hybrid": self.hybrid}[mode]