import asyncio
import threading

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider

from rag.config import OLLAMA_BASE_URL, OLLAMA_MODEL_NAME
from rag.models import RAGResult, RetrievalMode, RetrievedChunkModel, Retrievers

ollama_model = OpenAIChatModel(
    model_name=OLLAMA_MODEL_NAME,
    provider=OllamaProvider(base_url=f"{OLLAMA_BASE_URL}/v1"),
)


SYSTEM_PROMPT = """
You are a retrieval-augmented assistant. You will be given numbered context
passages followed by a question.

- Answer using only the information in the passages. Do not use prior knowledge.
- If the passages do not contain the answer, reply exactly: I don't know.
- Be concise. Do not quote the passages back; citations are attached separately.
""".strip()


PROMPT_TEMPLATE = """\
Context passages:

{context}

Question: {question}"""


NO_CONTEXT_ANSWER = (
    "I could not find anything relevant in the indexed documents, so I can't answer that."
)


# Retrieval is not a tool. A model that can call a tool can also decline to, and
# small local models often do, answering from their own weights while the UI
# still shows the reply as grounded. Retrieving first removes the choice.
rag_agent = Agent(model=ollama_model, output_type=str, system_prompt=SYSTEM_PROMPT)


def format_context(chunks: list[RetrievedChunkModel]) -> str:
    return "\n\n".join(
        f"[{i}] (source: {c.source}, chunk {c.chunk_index})\n{c.content}"
        for i, c in enumerate(chunks, start=1)
    )


def _run_agent(prompt: str):
    """Run the agent, tolerating a thread that already has an event loop.

    ``run_sync`` raises if called from a thread with a running loop. Streamlit's
    script runner does not have one, but the fallback keeps this usable from
    notebooks and from async callers.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return rag_agent.run_sync(prompt)

    box: dict = {}

    def _worker() -> None:
        try:
            box["result"] = asyncio.run(rag_agent.run(prompt))
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
    mode: RetrievalMode,
    retrievers: Retrievers,
    top_k: int = 5,
) -> RAGResult:
    chunks = retrievers.get(mode).search(question, top_k=top_k)

    # No passages means no grounds for an answer, so skip the LLM.
    if not chunks:
        return RAGResult(answer=NO_CONTEXT_ANSWER, retrieval_mode=mode, chunks=[])

    prompt = PROMPT_TEMPLATE.format(context=format_context(chunks), question=question)
    result = _run_agent(prompt)

    return RAGResult(answer=result.output, retrieval_mode=mode, chunks=chunks)