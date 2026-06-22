import pickle
from pathlib import Path

import numpy as np

from rag.bm25 import BM25Index, tokenize
from rag.config import EMBEDDING_DIM
from rag.embedding_model import EmbeddingModel


INDEX_FILE = "index.pkl"
VECTORS_FILE = "vectors.npy"


def chunk_text(text: str, chunk_size: int = 400, chunk_overlap: int = 50) -> list[str]:
    if chunk_size <= 0:
        raise ValueError(f"Chunk_size must be positive, got {chunk_size}")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError(
            f"Chunk_overlap must satisfy 0 <= chunk_overlap < chunk_size, "
            f"got chunk_overlap={chunk_overlap}, chunk_size={chunk_size}"
        )

    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - chunk_overlap

    return chunks


def load_documents(raw_data_dir: Path) -> list[tuple[str, str]]:
    documents = []
    raw_data_dir = Path(raw_data_dir)

    if not raw_data_dir.exists():
        print(f"Warning: Raw data directory {raw_data_dir} does not exist.")
        return documents

    for ext in ["*.txt", "*.md"]:
        for file_path in raw_data_dir.glob(ext):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except (OSError, UnicodeDecodeError) as e:
                print(f"Error loading {file_path}: {e}")
                continue
            if content.strip():
                documents.append((file_path.name, content))

    return documents


def build_index(
    raw_data_dir: Path,
    index_dir: Path,
    embedding_model: EmbeddingModel,
    chunk_size: int = 400,
    chunk_overlap: int = 50,
) -> None:
    print(f"Loading documents from {raw_data_dir}...")
    documents = load_documents(raw_data_dir)
    if not documents:
        raise ValueError(f"No documents found in {raw_data_dir}")
    print(f"Loaded {len(documents)} documents")

    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    print("Chunking documents...")
    all_chunks: list[str] = []
    chunk_metadata: list[dict] = []
    for filename, content in documents:
        chunks = chunk_text(content, chunk_size, chunk_overlap)
        for idx, chunk in enumerate(chunks):
            chunk_metadata.append({
                "id": len(all_chunks),
                "source": filename,
                "chunk_index": idx,
                "content": chunk,
            })
            all_chunks.append(chunk)
    print(f"Created {len(all_chunks)} chunks")

    print("Building BM25 index...")
    bm25 = BM25Index()
    for chunk in all_chunks:
        bm25.add(tokenize(chunk))
    bm25.finalize()

    print("Computing embeddings...")
    batch_size = 32
    batches = []
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i : i + batch_size]
        batches.append(embedding_model.encode(batch))
        if (i // batch_size + 1) % 10 == 0:
            print(f"Computed embeddings for {min(i + batch_size, len(all_chunks))} chunks...")
    vectors = np.vstack(batches).astype(np.float32)
    print(f"Computed {len(vectors)} embeddings of dimension {vectors.shape[1]}")

    if vectors.shape[1] != EMBEDDING_DIM:
        raise ValueError(
            f"Embedding dimension mismatch: model produced {vectors.shape[1]}, "
            f"but EMBEDDING_DIM is configured as {EMBEDDING_DIM}. "
            "Update EMBEDDING_DIM to match the embedding model."
        )

    # Pre-normalize so cosine similarity reduces to a dot product at query time.
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors = vectors / norms

    print(f"Writing index to {index_dir}...")
    with open(index_dir / INDEX_FILE, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunk_metadata}, f)
    np.save(index_dir / VECTORS_FILE, vectors)
    print(f"Index built with {len(all_chunks)} chunks in {index_dir}")
