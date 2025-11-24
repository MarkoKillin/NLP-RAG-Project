from typing import Any, Literal, TypedDict
from pydantic import BaseModel, ConfigDict

# from rag.retriever import LuceneBM25Retriever, LuceneVectorRetriever

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
    # model_config = ConfigDict(arbitrary_types_allowed=True)
    bm25: Any
    vector: Any

class RetrievedChunk(TypedDict):
    id: int
    source: str
    chunk_index: int
    content: str
    score: float
    