import re

from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_ai.settings import ModelSettings

from rag.config import settings
from rag.models import RAGResult, RetrievalMode, RetrievedChunkModel, Retrievers

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

EMPTY_ANSWER = "I don't know."

_CITATION_RE = re.compile(r"\[([\d,\s]+)\]")


class RAGError(RuntimeError):
    """Error class for RAG."""


rag_agent = Agent(
    model=ollama_model,
    output_type=str,
    system_prompt=SYSTEM_PROMPT,
    model_settings=ModelSettings(temperature=0.0),
)


def format_context(chunks: list[RetrievedChunkModel]) -> str:
    return "\n\n".join(
        f"[{i}] (source: {c.source}, chunk {c.chunk_index})\n{c.content}"
        for i, c in enumerate(chunks, start=1)
    )


def _run_agent(prompt: str) -> str:
    return (rag_agent.run_sync(prompt).output or "").strip()


def _extract_citations(answer: str, chunk_count: int) -> list[int]:
    seen: set[int] = set()
    valid: list[int] = []
    for group in _CITATION_RE.findall(answer):
        for n in map(int, re.findall(r"\d+", group)):
            if 1 <= n <= chunk_count and n not in seen:
                seen.add(n)
                valid.append(n)
    return valid


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

    if not chunks:
        return RAGResult(answer=NO_CONTEXT_ANSWER, chunks=[])

    prompt = PROMPT_TEMPLATE.format(context=format_context(chunks), question=question)
    try:
        answer = _run_agent(prompt)
    except Exception as exc:
        raise RAGError(f"Answer generation failed: {exc}") from exc

    return RAGResult(
        answer=answer or EMPTY_ANSWER,
        chunks=chunks,
        citations=_extract_citations(answer, len(chunks)),
    )