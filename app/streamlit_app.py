import streamlit as st
from rag.rag_agent import answer_question

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")

st.title("RAG Chatbot")

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("Configuration")
    mode = st.selectbox("Retrieval mode", ["bm25", "vector"])
    st.info(
        f"**BM25**: Lexical search using keyword matching\n\n"
        f"**Vector**: Semantic search using embeddings"
    )

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("View sources"):
                for src in message["sources"]:
                    st.markdown(
                        f"- **{src['source']}** (chunk {src['chunk_index']}, "
                        f"score={src['score']:.3f})"
                    )

if prompt := st.chat_input("Ask a question about the indexed documents:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner(f"Searching using {mode.upper()} and generating answer..."):
            try:
                result = answer_question(prompt, mode=mode)
                answer = result["answer"]
                sources = result["sources"]

                st.markdown(answer)

                if sources:
                    with st.expander("View sources"):
                        for src in sources:
                            st.markdown(
                                f"- **{src['source']}** (chunk {src['chunk_index']}, "
                                f"score={src['score']:.3f})"
                            )

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
                    {"role": "assistant", "content": error_msg}
                )

