from typing import Literal, Any
from pathlib import Path

from rag.config import INDEX_DIR, TOP_K, EMBEDDING_MODEL_NAME, OLLAMA_MODEL_NAME, OLLAMA_BASE_URL
from rag.retriever import LuceneBM25Retriever, LuceneVectorRetriever, RetrievedChunk
from rag.embedding_model import EmbeddingModel
from rag.llm_client import OllamaLLMClient


def build_rag_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[Document {i} - Source: {chunk['source']}, Chunk {chunk['chunk_index']}]\n"
            f"{chunk['content']}\n"
        )

    context = "\n".join(context_parts)

    prompt = f"""You are a helpful assistant that answers questions based on the provided context documents.

        Context documents:
        {context}

        Question: {question}

        Instructions:
        - Answer the question using ONLY the information provided in the context documents above.
        - If the context does not contain enough information to answer the question, say "I don't have enough information in the provided context to answer this question."
        - Be concise and accurate.
        - Cite the source document when possible (e.g., "According to [Document X]...").

        Answer:"""

    return prompt


def answer_question(
    question: str,
    mode: Literal["bm25", "vector"] = "bm25",
    index_dir: Path = None,
    embedding_model: EmbeddingModel = None,
    llm_client: OllamaLLMClient = None,
    top_k: int = None,
) -> dict[str, Any]:
    if index_dir is None:
        index_dir = INDEX_DIR
    if top_k is None:
        top_k = TOP_K

    if mode == "bm25":
        retriever = LuceneBM25Retriever(index_dir)
    elif mode == "vector":
        if embedding_model is None:
            embedding_model = EmbeddingModel(EMBEDDING_MODEL_NAME)
        retriever = LuceneVectorRetriever(index_dir, embedding_model)
 
    try:
        chunks = retriever.search(question, top_k=top_k)

        if not chunks:
            return {
                "answer": "I couldn't find any relevant documents to answer your question.",
                "sources": [],
            }

        prompt = build_rag_prompt(question, chunks)

        if llm_client is None:
            llm_client = OllamaLLMClient(OLLAMA_MODEL_NAME, OLLAMA_BASE_URL)

        answer = llm_client.generate(prompt)

        sources = [
            {
                "source": chunk["source"],
                "chunk_index": chunk["chunk_index"],
                "score": chunk["score"],
            }
            for chunk in chunks
        ]

        return {
            "answer": answer,
            "sources": sources,
        }
    finally:
        retriever.close()

