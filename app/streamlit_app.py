import streamlit as st

from rag.config import settings
from rag.rag_agent import run_rag
from rag.retriever import build_retrievers

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("RAG Chatbot")


def render_sources(sources: list[dict]) -> None:
    with st.expander(f"View sources ({len(sources)})"):
        for src in sources:
            st.markdown(
                f"- **{src['source']}** "
                f"(chunk {src['chunk_index']}, score={src['score']:.4f})"
            )


@st.cache_resource
def get_retrievers():
    return build_retrievers()


if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Configuration")
    mode = st.selectbox("Retrieval mode", ["bm25", "vector", "hybrid"])
    slider_max = max(20, settings.top_k)
    top_k = st.slider(
        "Chunks retrieved (top_k)",
        min_value=1,
        max_value=slider_max,
        value=min(max(settings.top_k, 1), slider_max),
    )
    st.info(
        "**BM25**: lexical search over stemmed, stopword-filtered tokens\n\n"
        "**Vector**: semantic search over embeddings\n\n"
        "**Hybrid**: Reciprocal Rank Fusion of both. Its scores are RRF scores, "
        "so they are much smaller than the other two modes and not comparable to them."
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        sources = message.get("sources") or []
        if sources:
            render_sources(sources)
        elif message["role"] == "assistant" and message.get("ungrounded"):
            st.warning("No passages were retrieved for this question.")

if prompt := st.chat_input("Ask a question about the indexed documents:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner(f"Searching using {mode.upper()} and generating answer..."):
            try:
                result = run_rag(
                    question=prompt,
                    mode=mode,
                    retrievers=get_retrievers(),
                    top_k=top_k,
                )

                answer = result.answer
                sources = [c.model_dump() for c in result.chunks]

                st.markdown(answer)
                if sources:
                    render_sources(sources)
                else:
                    st.warning("No passages were retrieved for this question.")

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "ungrounded": not sources,
                    }
                )
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_msg,
                    }
                )