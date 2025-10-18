from pydantic_ai import Agent
from .model import RAGDeps, ModelResponse


rag_agent = Agent[RAGDeps, ModelResponse](
    model = "openai:gpt-4o-mini",
    system_prompt="",
)

@rag_agent.tool
def retrieve():
    """Retrieves the files from database."""
    raise NotImplementedError