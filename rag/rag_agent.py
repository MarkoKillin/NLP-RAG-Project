from typing import Literal
from pathlib import Path
import asyncio
import threading

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider

from rag.models import RAGDeps, RAGResult, RetrievedChunkModel
from rag.retriever import BM25Retriever, VectorRetriever
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
- Return a concise, clear answer. The citations are attached automatically; you do not need to repeat the chunks.
"""


rag_agent = Agent(
    model=ollama_model,
    deps_type=RAGDeps,
    output_type=str,
    system_prompt=SYSTEM_PROMPT,
)


@rag_agent.tool
def retrieve_chunks(
    ctx: RunContext[RAGDeps],
    query: str,
) -> list[RetrievedChunkModel]:
    """Retrieve relevant chunks from the index for the given query."""
    if ctx.deps.mode == "bm25":
        results = ctx.deps.bm25.search(query, top_k=ctx.deps.top_k)
    else:
        results = ctx.deps.vector.search(query, top_k=ctx.deps.top_k)

    chunks = [RetrievedChunkModel(**r) for r in results]
    # Capture the actual retrieved chunks so the result carries verbatim
    # citations (content + scores) rather than whatever the LLM reproduces.
    # De-dupe by chunk id: the model may call this tool more than once per run
    # (re-queries / retries), which would otherwise show duplicate sources.
    seen = {c.id for c in ctx.deps.retrieved}
    ctx.deps.retrieved.extend(c for c in chunks if c.id not in seen)
    return chunks


def build_rag_deps(index_dir: Path | None = None) -> RAGDeps:
    """Construct the BM25 + vector retrievers once. Reuse across queries.

    The per-request mode/top_k are set later in run_rag, so they are left at
    their model defaults here.
    """
    index_path = index_dir or Path(INDEX_DIR)
    embedding_model = EmbeddingModel(EMBEDDING_MODEL_NAME)
    bm25_retriever = BM25Retriever(index_path)
    vector_retriever = VectorRetriever(index_path, embedding_model)
    return RAGDeps(bm25=bm25_retriever, vector=vector_retriever)


def _run_agent(question: str, run_deps: RAGDeps):
    """Run the agent, tolerating a thread that already has an event loop.

    ``run_sync`` raises if called from a thread with a running event loop. That
    does not normally happen under Streamlit, but to stay robust we detect a
    running loop and, if present, drive the async ``run`` in a dedicated thread
    with its own loop instead.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        # No loop running in this thread — the simple path.
        return rag_agent.run_sync(question, deps=run_deps)

    box: dict = {}

    def _worker() -> None:
        try:
            box["result"] = asyncio.run(rag_agent.run(question, deps=run_deps))
        except BaseException as exc:  # noqa: BLE001 - re-raised on caller thread
            box["error"] = exc

    thread = threading.Thread(target=_worker)
    thread.start()
    thread.join()
    if "error" in box:
        raise box["error"]
    return box["result"]


def run_rag(
    question: str,
    mode: Literal["bm25", "vector"],
    deps: RAGDeps,
    top_k: int = 5,
) -> RAGResult:
    # Build a fresh per-request deps that reuses the (expensive) shared
    # retrievers. The shared deps from build_rag_deps may be cached and used
    # concurrently, so we must not mutate its mode/top_k or its retrieved list.
    run_deps = RAGDeps(bm25=deps.bm25, vector=deps.vector, mode=mode, top_k=top_k)

    result = _run_agent(question, run_deps)

    return RAGResult(
        answer=result.output,
        retrieval_mode=mode,
        chunks=run_deps.retrieved,
    )
