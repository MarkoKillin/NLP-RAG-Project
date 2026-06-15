# RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot powered by:

-   **Pure-Python BM25 lexical search**
-   **Numpy cosine k-NN vector search**
-   **Local Ollama LLM**
-   **PydanticAI agent**
-   **Streamlit UI**

Designed as a full local RAG pipeline --- no external APIs required.

## Features

-   **Two retrieval modes**: BM25 (keyword) and Vector (semantic)
-   **Single on-disk index** (`index.pkl` + `vectors.npy`) holding tokenized text and pre-normalized dense vectors
-   **Local LLM and embedding model via Ollama** (e.g., Mistral, Llama, etc.)
-   **Streamlit chat interface** with citations
-   **Modular RAG architecture** (retriever, embeddings, ingestion, agent)

## Project Structure

    root/
      app/
        streamlit_app.py
      data/raw/                # Input documents (.txt, .md)
      index/                   # Auto-built index (index.pkl, vectors.npy)
      rag/
        bm25.py
        config.py
        embedding_model.py
        ingestion.py
        models.py
        rag_agent.py
        retriever.py
      scripts/
        build_index.py
        docker-entrypoint.sh
      docker-compose.yml
      Dockerfile
      README.md

## Requirements

### Docker

Install: - Docker Desktop or other Docker VMs

## Usage

### Add documents

Place `.txt` or `.md` files in `data/raw/`.

### Run 

Docker:

``` bash
docker-compose up --build
```

### UI

Open:

    http://localhost:8501

### Retrieval mode

Choose **BM25** or **Vector** in sidebar.
