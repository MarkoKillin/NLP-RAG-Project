import asyncio
import threading
from typing import Callable, TypeVar

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider

from rag.config import settings
from rag.models import RAGResult, RetrievalMode, RetrievedChunkModel, Retrievers

T = TypeVar("T")

ollama_model = OpenAIChatModel(
    model_name=settings.ollama_model_name,
    provider=OllamaProvider(base_url=f"{settings.ollama_base_url}/v1"),
)


SYSTEM_PROMPT = """
You are a retrieval-augmented assistant. You will be given numbered context
passages followed by a question.

- Answer using only the information in the passages. Do not use prior knowledge.
- Cite the passages you use with their bracketed numbers, e.g. [1] or [2][3],
  placed right after the sentence they support.
- If the passages do not contain the answer, reply exactly: I don't know.
- Be concise. Do not quote passages verbatim; refer to them by their number.
""".strip()


PROMPT_TEMPLATE = """\
Context passages:

{context}

Question: {question}"""


NO_CONTEXT_ANSWER = (
    "I could not find anything relevant in the indexed documents, so I can't answer that."
)

# Shown when the model runs but returns nothing usable (empty or whitespace).
EMPTY_ANSWER = "I don't know."


class RAGError(RuntimeError):
    """The retrieval or generation step failed. Carries a user-safe message."""


# Retrieval is not a tool. A model that can call a tool can also decline to, and
# small local models often do, answering from their own weights while the UI
# still shows the reply as grounded. Retrieving first removes the choice.
rag_agent = Agent(model=ollama_model, output_type=str, system_prompt=SYSTEM_PROMPT)


def format_context(chunks: list[RetrievedChunkModel]) -> str:
    return "\n\n".join(
        f"[{i}] (source: {c.source}, chunk {c.chunk_index})\n{c.content}"
        for i, c in enumerate(chunks, start=1)
    )


def _run_off_thread(coro_factory: Callable[[], T]) -> T:
    """Run a coroutine factory on a fresh thread with its own event loop.

    Used when the caller already has a running loop, which ``run_sync`` refuses.
    """
    box: dict = {}

    def _worker() -> None:
        try:
            box["result"] = asyncio.run(coro_factory())
        except BaseException as exc:  # noqa: BLE001 - re-raised on caller thread
            box["error"] = exc

    thread = threading.Thread(target=_worker)
    thread.start()
    thread.join()
    if "error" in box:
        raise box["error"]
    return box["result"]


def _run_agent(prompt: str) -> str:
    """Run the agent and return its text output, tolerating an active loop.

    ``run_sync`` raises if called from a thread with a running loop. Streamlit's
    script runner does not have one, but the off-thread fallback keeps this
    usable from notebooks and async callers.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        result = rag_agent.run_sync(prompt)
    else:
        result = _run_off_thread(lambda: rag_agent.run(prompt))

    return (result.output or "").strip()


def run_rag(
    question: str,
    mode: RetrievalMode,
    retrievers: Retrievers,
    top_k: int = 5,
) -> RAGResult:
    question = question.strip()
    if not question:
        raise RAGError("Please enter a question.")

    try:
        chunks = retrievers.get(mode).search(question, top_k=top_k)
    except Exception as exc:  # noqa: BLE001 - wrapped for a user-safe message
        raise RAGError(f"Retrieval failed: {exc}") from exc

    # No passages means no grounds for an answer, so skip the LLM.
    if not chunks:
        return RAGResult(answer=NO_CONTEXT_ANSWER, retrieval_mode=mode, chunks=[])

    prompt = PROMPT_TEMPLATE.format(context=format_context(chunks), question=question)
    try:
        answer = _run_agent(prompt)
    except Exception as exc:  # noqa: BLE001 - wrapped for a user-safe message
        raise RAGError(f"Answer generation failed: {exc}") from exc

    return RAGResult(
        answer=answer or EMPTY_ANSWER,
        retrieval_mode=mode,
        chunks=chunks,
    )