from typing import Literal
from pathlib import Path

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider

from rag.models import RAGDeps, RAGResult, RetrievedChunkModel
from rag.retriever import LuceneBM25Retriever, LuceneVectorRetriever
from rag.embedding_model import EmbeddingModel
from rag.config import INDEX_DIR, OLLAMA_BASE_URL, OLLAMA_MODEL_NAME, EMBEDDING_MODEL_NAME


ollama_model = OpenAIChatModel(
    model_name=OLLAMA_MODEL_NAME,
    provider=OllamaProvider(base_url=f"{OLLAMA_BASE_URL}/v1"),
)


SYSTEM_PROMPT = """
You are a retrieval-augmented assistant.

- You MUST first call the `retrieve_chunks` tool with the user's question to get relevant document chunks.
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
) -> list[RetrievedChunkModel]:
    """Retrieve relevant chunks from the Lucene index for the given query."""
    if ctx.deps.mode == "bm25":
        results = ctx.deps.bm25.search(query, top_k=ctx.deps.top_k)
    else:
        results = ctx.deps.vector.search(query, top_k=ctx.deps.top_k)

    return [RetrievedChunkModel(**r) for r in results]


def build_rag_deps(
    index_dir: Path | None = None,
    mode: Literal["bm25", "vector"] = "bm25",
    top_k: int = 5,
) -> RAGDeps:
    """Construct the BM25 + vector retrievers once. Reuse across queries."""
    index_path = index_dir or Path(INDEX_DIR)
    embedding_model = EmbeddingModel(EMBEDDING_MODEL_NAME)
    bm25_retriever = LuceneBM25Retriever(index_path)
    vector_retriever = LuceneVectorRetriever(index_path, embedding_model)
    return RAGDeps(bm25=bm25_retriever, vector=vector_retriever, mode=mode, top_k=top_k)


def run_rag(
    question: str,
    mode: Literal["bm25", "vector"],
    deps: RAGDeps,
    top_k: int = 5,
) -> RAGResult:
    deps.mode = mode
    deps.top_k = top_k

    result = rag_agent.run_sync(question, deps=deps)
    output = result.output
    output.retrieval_mode = mode
    return output
