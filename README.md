# RAG Chatbot

A retrieval-augmented chatbot for an NLP and IR course. Ask it about the documents
in `data/raw/`. It retrieves passages with BM25, vector search or a Reciprocal Rank
Fusion of both, then a local Ollama model writes the answer and cites the passages
it used.

## Requirements

- Docker with Docker Compose
- 8 GB of memory available to Docker
- About 10 GB of free disk for the images and models

## Installation

    git clone https://github.com/MarkoKillin/NLP-RAG-Project.git
    cd NLP-RAG-Project
    docker compose up --build

The first start downloads about 4 GB of models, `llama3.2:3b-instruct-q8_0` and
`nomic-embed-text`. Then open http://localhost:8501 and pick a retrieval mode in
the sidebar.

Docker on macOS has no GPU access, so each answer takes about 20 seconds there.

## Adding documents

Put `.txt`, `.md`, `.csv` or `.xlsx` files in `data/raw/` and restart the app:

    docker compose restart rag-app

The app rebuilds the index on every start. Each table row becomes one passage.

## Configuration

Defaults are in `docker-compose.yml`. To change them, copy `.env.example` to `.env`,
edit it, and run `docker compose up -d`.

## Evaluation

`eval/questions.json` holds labelled test questions. With the app running:

    docker compose exec rag-app python -m scripts.evaluate --k 3
    docker compose exec rag-app python -m scripts.evaluate --ablation
    docker compose exec rag-app python -m scripts.evaluate --grounding

The first command prints Recall@k, MRR@k and Hit@k for each retrieval mode.
`--ablation` adds BM25 runs with stemming and stopword removal switched on and off.
`--grounding` runs the full pipeline and checks whether each answer cites a passage
that holds the answer.

## Project structure

    app/streamlit_app.py     Streamlit UI
    rag/                     Chunking, BM25, embeddings, retrieval, answering
    scripts/build_index.py   Builds the index from data/raw
    scripts/evaluate.py      Evaluation
    eval/questions.json      Labelled questions
    data/raw/                Documents