import streamlit as st

from rag.config import TOP_K
from rag.rag_agent import build_rag_deps, run_rag


st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("RAG Chatbot")


def render_sources(sources: list[dict]) -> None:
    with st.expander("View sources"):
        for src in sources:
            st.markdown(
                f"- **{src['source']}** "
                f"(chunk {src['chunk_index']}, score={src['score']:.3f})"
            )


@st.cache_resource
def get_rag_deps():
    return build_rag_deps()


if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Configuration")
    mode = st.selectbox("Retrieval mode", ["bm25", "vector"])
    st.info(
        "**BM25**: Lexical search using keyword matching\n\n"
        "**Vector**: Semantic search using embeddings"
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            sources = message["sources"] or []
            if sources:
                render_sources(sources)

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
                    deps=get_rag_deps(),
                    top_k=TOP_K,
                )

                answer = result.answer
                sources = [c.model_dump() for c in result.chunks]

                st.markdown(answer)

                if sources:
                    render_sources(sources)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
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
