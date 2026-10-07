# RAG Chatbot

A retrieval-augmented chatbot for our NLP and IR course. Ask it about the
documents in `data/raw/` and it answers with citations to the passages it used.

It has four retrieval modes. BM25 is keyword search. Vector search compares
`nomic-embed-text` embeddings. Hybrid fuses the two with Reciprocal Rank Fusion,
and rerank re-scores hybrid's top 20 with a cross-encoder run through Hugging
Face `transformers`. A local `llama3.2:3b` model in Ollama writes the answers.

Slides on installing and using it are in `slides.pdf`.

## Requirements

- Docker with Docker Compose
- 8 GB of memory for Docker
- About 11 GB of free disk

No API keys needed. Everything runs locally.

## Running

    git clone https://github.com/MarkoKillin/NLP-RAG-Project.git
    cd NLP-RAG-Project
    docker compose up --build

The first start downloads about 4 GB of models, so give it a few minutes. Once
the log says `Starting Streamlit app`, open http://localhost:8501, pick a mode
in the sidebar and ask something. "View sources" under an answer shows the
passages it came from.

On a Mac, Docker can't use the GPU, so each answer takes about 20 seconds.

## Your own documents

Drop `.txt`, `.md`, `.csv` or `.xlsx` files into `data/raw/` and run
`docker compose restart rag-app`. The app rebuilds the index on every start, and
each table row becomes its own passage.

## Settings

The defaults are in `docker-compose.yml`. To change one, copy `.env.example` to
`.env`, edit it, and run `docker compose up -d`.

## Evaluation

With the app running:

    docker compose exec rag-app python -m scripts.evaluate --k 3
    docker compose exec rag-app python -m scripts.evaluate --grounding

The first scores each mode on the 32 questions in `eval/questions.json` with
Recall@k, MRR@k and Hit@k. The second also generates answers and checks that
they cite a passage containing the answer. Add `--ablation` to compare BM25 with
and without stemming and stopword removal.
