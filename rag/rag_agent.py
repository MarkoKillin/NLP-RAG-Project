# rag/pydantic_rag_agent.py

from typing import Literal
from pathlib import Path

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider

from rag.models import RAGDeps, RAGResult, RetrievedChunkModel
from rag.retriever import LuceneBM25Retriever, LuceneVectorRetriever
from rag.config import INDEX_DIR, OLLAMA_BASE_URL, OLLAMA_MODEL_NAME, EMBEDDING_MODEL_NAME


ollama_model = OpenAIChatModel(
    model_name=OLLAMA_MODEL_NAME,
    provider=OllamaProvider(base_url=OLLAMA_BASE_URL),  
)


SYSTEM_PROMPT = """
You are a retrieval-augmented assistant.

- You MUST first call the `retrieve_chunks` tool to get relevant document chunks.
- The user message will explicitly tell you which retrieval mode to use: "bm25" or "vector".
- Use only the information from the retrieved chunks when answering.
- If the chunks do not contain an answer, say that you don't know.
- Return a concise, clear answer in the `answer` field and include all chunks you used.
"""


rag_agent = Agent(
    model=ollama_model,
    deps_type=RAGDeps,
    output_type=RAGResult,
    system_prompt=SYSTEM_PROMPT,
)


@rag_agent.tool
def retrieve_chunks(
    ctx: RunContext[RAGDeps],
    query: str,
    mode: Literal["bm25", "vector"] = "bm25",
    top_k: int = 5,
) -> list[RetrievedChunkModel]:
    """
    Retrieve relevant chunks from the Lucene index.

    mode="bm25"   -> LuceneBM25Retriever
    mode="vector" -> LuceneVectorRetriever
    """
    if mode == "bm25":
        results = ctx.deps.bm25.search(query, top_k=top_k)
    else:
        results = ctx.deps.vector.search(query, top_k=top_k)

    return [RetrievedChunkModel(**r) for r in results]

def run_rag(
    question: str,
    mode: Literal["bm25", "vector"],
    top_k: int = 5,
    index_dir: Path | None = None,
) -> RAGResult:
    index_path = index_dir or Path(INDEX_DIR)

    bm25_retriever = LuceneBM25Retriever(index_path)
    from rag.embedding_model import EmbeddingModel
    embedding_model = EmbeddingModel(EMBEDDING_MODEL_NAME)
    vector_retriever = LuceneVectorRetriever(index_path, embedding_model)

    deps = RAGDeps(bm25=bm25_retriever, vector=vector_retriever)

    user_message = (
        f"Retrieval mode: {mode}. Top_k: {top_k}. "
        f"User question: {question}"
    )

    result = rag_agent.run_sync(
        user_message,
        deps=deps,
    )

    return result.data
