import streamlit as st

from rag.config import settings
from rag.models import RETRIEVAL_MODES
from rag.rag_agent import run_rag
from rag.retriever import build_retrievers

MAX_TOP_K = 10

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("RAG Chatbot")


def render_sources(sources: list[dict], citations: list[int] | None = None) -> None:
    cited = set(citations or [])
    count = f"{len(sources)}, {len(cited)} cited" if cited else f"{len(sources)}"
    with st.expander(f"View sources ({count})"):
        for i, src in enumerate(sources, start=1):
            marker = "**[cited]** " if i in cited else ""
            st.markdown(
                f"- {marker}**[{i}] {src['source']}** "
                f"(chunk {src['chunk_index']}, score={src['score']:.4f})"
            )


@st.cache_resource
def get_retrievers():
    return build_retrievers()


if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Configuration")
    mode = st.selectbox("Retrieval mode", RETRIEVAL_MODES)
    top_k = st.slider(
        "Chunks retrieved (top_k)",
        min_value=1,
        max_value=MAX_TOP_K,
        value=min(settings.top_k, MAX_TOP_K),
    )
    st.info(
        "**BM25.** Keyword search over stemmed words.\n\n"
        "**Vector.** Embedding similarity.\n\n"
        "**Hybrid.** Reciprocal Rank Fusion of both rankings.\n\n"
        f"**Rerank.** Hybrid's top {settings.rerank_candidates}, re-scored by a cross-encoder."
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        sources = message.get("sources") or []
        if sources:
            render_sources(sources, message.get("citations"))
            if message["role"] == "assistant" and not message.get("citations"):
                st.caption("The model cited no passages, so the answer may not be grounded.")
        elif message["role"] == "assistant" and message.get("ungrounded"):
            st.warning("Retrieval found no passages for this question.")

if prompt := st.chat_input("Ask a question about the indexed documents"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner(f"Searching with {mode.upper()} and writing the answer..."):
            try:
                result = run_rag(
                    question=prompt,
                    mode=mode,
                    retrievers=get_retrievers(),
                    top_k=top_k,
                )

                answer = result.answer
                sources = [c.model_dump() for c in result.chunks]
                citations = result.citations

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "citations": citations,
                        "ungrounded": not sources,
                    }
                )

                st.markdown(answer)
                if sources:
                    render_sources(sources, citations)
                    if not citations:
                        st.caption("The model cited no passages, so the answer may not be grounded.")
                else:
                    st.warning("Retrieval found no passages for this question.")
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_msg,
                    }
                )
                st.error(error_msg)