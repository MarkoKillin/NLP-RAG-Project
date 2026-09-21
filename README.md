# RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot with three retrieval modes, a local
Ollama LLM, and a retrieval evaluation harness.

- Pure-Python BM25 lexical search with Porter stemming
- Numpy cosine k-NN vector search
- Reciprocal Rank Fusion hybrid of the two
- Local Ollama LLM and embedding model, no external APIs
- PydanticAI for the LLM call, Streamlit for the UI

## How retrieval works

Retrieval runs before the LLM call, and the retrieved passages go into the prompt.
The LLM has no tools, so it cannot skip retrieval and answer from its own weights
while the UI still shows the reply as grounded. If retrieval returns nothing, the
LLM never runs and the UI says so.

The three modes:

| Mode | Method | Good at | Weak at |
| --- | --- | --- | --- |
| `bm25` | BM25 over stemmed tokens | exact terms, rare names, identifiers | paraphrase, synonyms |
| `vector` | cosine k-NN over embeddings | paraphrase, loose wording | exact identifiers, numbers |
| `hybrid` | RRF over both rankings | both, in general | costs one embedding call plus a BM25 pass |

`hybrid` fuses by rank, not by score, so nothing needs calibrating between BM25's
unbounded scores and cosine similarities. It fuses over `top_k * 4` candidates from
each retriever, so a passage that one retriever ranks well and the other ranks mid
can still win. Its reported scores are RRF scores, roughly 0.01 to 0.03, and do not
compare to the other two modes.

## Project structure

    root/
      app/streamlit_app.py
      data/raw/                # Input documents (.txt, .md)
      eval/questions.json      # Labelled questions for retrieval evaluation
      index/                   # Built index (index.pkl, vectors.npy)
      rag/
        bm25.py                # BM25 + tokenizer (stemming, stopwords)
        config.py
        embedding_model.py     # Ollama /api/embed client
        ingestion.py           # Chunking + index build
        models.py
        rag_agent.py           # Retrieve-then-read pipeline
        retriever.py           # BM25 / vector / hybrid retrievers
      scripts/
        build_index.py
        evaluate.py            # Retrieval metrics
        docker-entrypoint.sh

## Running

Put `.txt` or `.md` files in `data/raw/`, then:

``` bash
docker-compose up --build
```

Open http://localhost:8501 and pick a retrieval mode in the sidebar.

The first run pulls both models into the `ollama_data` volume, which takes a few
minutes and several GB of disk.

The default chat model needs one extra step. `hf.co/google/gemma-2b-it` is a gated
Hugging Face repository, so `ollama pull` fails with a 401 unless you have accepted
the Gemma licence and configured Hugging Face credentials for Ollama. Point
`OLLAMA_MODEL_NAME` at an ungated model such as `gemma2:2b` or `qwen2.5:3b-instruct`
to skip that.

## Evaluation

`scripts/evaluate.py` scores each retrieval mode against the labelled questions in
`eval/questions.json`:

``` bash
python -m scripts.evaluate                    # all modes
python -m scripts.evaluate --k 3              # tighter cutoff
python -m scripts.evaluate --modes bm25       # no Ollama needed
python -m scripts.evaluate --ablation         # compare BM25 tokenizer settings
```

It reports Recall@k, MRR@k and Hit@k overall and broken down by question type
(`lexical`, `semantic`, `mixed`), so the lexical/semantic trade-off shows up instead of
averaging away.

Relevance comes from content. A chunk is relevant if it comes from one of the listed
`sources` and contains the `must_contain` string. Judgements then survive re-chunking
and re-indexing, which chunk-id labels would not. A question whose `must_contain`
matches nothing gets reported and skipped, not scored as a retrieval failure.

Pick `k` relative to your corpus size. On the 9-chunk sample corpus, Recall@5 sits at
1.000 for every mode because retrieving 5 of 9 chunks is nearly free. Use `--k 1` or
`--k 3` to tell the modes apart.

### BM25 tokenizer ablation

Measured on the sample corpus with 20 labelled questions. `--ablation` rebuilds
BM25 in memory under each setting, so it needs no re-indexing and no embeddings.

| Stemming | Stopwords removed | MRR@1 | MRR@2 | MRR@3 |
| --- | --- | --- | --- | --- |
| no | no | 0.850 | 0.875 | 0.892 |
| no | yes | 0.800 | 0.825 | 0.842 |
| **yes** | **no** | **0.900** | **0.950** | **0.950** |
| yes | yes | 0.750 | 0.825 | 0.842 |

Stemming helps at every cutoff, which is what you would expect, since it lets
`embeddings` match `embedding`. Stopword removal hurts at every cutoff, which is less
obvious. The IDF term already discounts frequent words, so removing them buys nothing
there, and the eval questions are full sentences whose function words still match. The
defaults in `config.py` follow this result: stemming on, stopword removal off. Re-run
the ablation on your own corpus before trusting them.

## Configuration

All settings are environment variables, passed through in `docker-compose.yml`.

| Variable | Default | Notes |
| --- | --- | --- |
| `OLLAMA_BASE_URL` | `http://ollama:11434` | Without the `/v1` suffix |
| `OLLAMA_MODEL_NAME` | `hf.co/google/gemma-2b-it` | Gated, see note above |
| `EMBEDDING_MODEL_NAME` | `hf.co/Snowflake/...-m-v1.5:BF16` | Changing this requires a rebuild |
| `EMBEDDING_QUERY_PREFIX` | arctic-embed instruction | Set empty for symmetric models |
| `RAW_DATA_DIR` | `data/raw` | |
| `INDEX_DIR` | `index` | |
| `CHUNK_SIZE` | `400` | Words per chunk, not tokens |
| `CHUNK_OVERLAP` | `50` | Words |
| `TOP_K` | `5` | Sidebar slider overrides per query |
| `BM25_STEM` | `1` | |
| `BM25_REMOVE_STOPWORDS` | `0` | See ablation above |
| `RRF_K` | `60` | Rank damping for hybrid fusion |
| `HYBRID_CANDIDATE_MULTIPLIER` | `4` | Candidates per retriever = `top_k * this` |
| `REBUILD_INDEX` | `0` | Set to `1` to force a rebuild |

### When you must rebuild the index

Each index belongs to one embedding model, one chunking setting, and one BM25
tokenizer. Change any of `EMBEDDING_MODEL_NAME`, `CHUNK_SIZE`, `CHUNK_OVERLAP`,
`BM25_STEM` or `BM25_REMOVE_STOPWORDS` and you need `REBUILD_INDEX=1`, because
`docker-entrypoint.sh` skips the build whenever the index files exist and `./index`
is a mounted volume that survives `docker-compose up --build`.

The index records the embedding model that built it, and loading refuses to proceed
once that stops matching `EMBEDDING_MODEL_NAME`. Skip that check and query vectors
from the new model get dotted against document vectors from the old one. The
dimensions agree, nothing errors, and the rankings mean nothing.

## Limitations

- Retrieval returns something whenever any query term matches anything. There is no
  relevance threshold, so a weak query pulls in low-scoring passages and fills the
  prompt with them as if they were relevant. On the sample corpus a good query scores
  around 4.3 and a junk one around 1.1, so a threshold would work, but the right value
  depends on the corpus and BM25 scores have no upper bound. For now the only defence
  is the system prompt telling the model to say "I don't know", and the model has to
  comply.
- Only retrieval gets measured. Nothing checks whether the answer the model produces
  is supported by the passages it was handed.
- Chunking splits on whitespace at a fixed word count, so chunks can start and end
  mid-sentence. Paragraph-aware splitting would be better.
- `top_k=5` at 400-word chunks puts roughly 3.5k tokens into the prompt, which is a
  lot for a 2B model. Reduce `CHUNK_SIZE` or `TOP_K` if answers degrade.
- The index is a pickle, so it does not survive refactors of `rag/bm25.py`.
  `INDEX_FORMAT_VERSION` at least makes that fail loudly.