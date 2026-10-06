import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from rag.bm25 import BM25Index
from rag.config import settings
from rag.embedding_model import EmbeddingModel


INDEX_FILE = "index.pkl"
VECTORS_FILE = "vectors.npy"


def chunk_text(text: str, chunk_size: int = 400, chunk_overlap: int = 50) -> list[str]:
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if not 0 <= chunk_overlap < chunk_size:
        raise ValueError(
            f"chunk_overlap must be 0 <= chunk_overlap < chunk_size, "
            f"got chunk_overlap={chunk_overlap}, chunk_size={chunk_size}"
        )

    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        # was bug, if 400-word doc at size=400, overlap=50
        # makes a second chunk that is pure overlap
        if end == len(words):
            break
        start += chunk_size - chunk_overlap

    return chunks


def _row_to_text(row: dict) -> str:
    """col: value | col: value"""
    return " | ".join(
        f"{col}: {val}" for col, val in row.items() if str(val).strip()
    )


def _chunk_text_file(path: Path, chunk_size: int, chunk_overlap: int) -> list[str]:
    content = path.read_text(encoding="utf-8")
    if not content.strip():
        return []
    return chunk_text(content, chunk_size, chunk_overlap)


def _chunk_table_file(path: Path, chunk_size: int, chunk_overlap: int) -> list[str]:
    # one chunk per row
    if path.suffix.lower() == ".csv":
        frames = [pd.read_csv(path, dtype=str, keep_default_na=False)]
    else:
        frames = list(pd.read_excel(path, sheet_name=None, dtype=str, keep_default_na=False).values())
    chunks = [_row_to_text(row) for frame in frames for row in frame.to_dict(orient="records")]
    return [c for c in chunks if c]


_LOADERS = {
    ".txt": _chunk_text_file,
    ".md": _chunk_text_file,
    ".csv": _chunk_table_file,
    ".xlsx": _chunk_table_file,
}


def load_documents(
    raw_data_dir: Path, chunk_size: int = 400, chunk_overlap: int = 50
) -> list[tuple[str, list[str]]]:
    documents = []
    raw_data_dir = Path(raw_data_dir)

    if not raw_data_dir.exists():
        print(f"Warning: Raw data directory {raw_data_dir} does not exist.")
        return documents

    for file_path in sorted(raw_data_dir.iterdir()):
        loader = _LOADERS.get(file_path.suffix.lower())
        if loader is None:
            continue
        try:
            chunks = loader(file_path, chunk_size, chunk_overlap)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            continue
        if chunks:
            documents.append((file_path.name, chunks))

    return documents


def build_index(
    raw_data_dir: Path,
    index_dir: Path,
    embedding_model: EmbeddingModel,
    chunk_size: int = 400,
    chunk_overlap: int = 50,
) -> None:
    print(f"Loading docs from {raw_data_dir}...")
    documents = load_documents(raw_data_dir, chunk_size, chunk_overlap)
    if not documents:
        raise ValueError(f"No docs found in {raw_data_dir}")
    print(f"Loaded {len(documents)} docs")

    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    all_chunks: list[str] = []
    chunk_metadata: list[dict] = []
    for filename, chunks in documents:
        for idx, chunk in enumerate(chunks):
            chunk_metadata.append({
                "id": len(all_chunks),
                "source": filename,
                "chunk_index": idx,
                "content": chunk,
            })
            all_chunks.append(chunk)
    print(f"Created {len(all_chunks)} chunks")

    print(f"Building BM25 index (stem={settings.bm25_stem}, remove_stopwords={settings.bm25_remove_stopwords})...")
    bm25 = BM25Index(stem=settings.bm25_stem, remove_stopwords=settings.bm25_remove_stopwords)
    for chunk in all_chunks:
        bm25.add(chunk)
    bm25.finalize()

    print("Calculating embeddings...")
    batch_size = 32
    batches = []
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i : i + batch_size]
        batches.append(embedding_model.encode(batch))
        if (i // batch_size + 1) % 10 == 0:
            print(f"Calculated embeddings for {min(i + batch_size, len(all_chunks))} chunks...")
    vectors = np.vstack(batches).astype(np.float32)
    print(f"Calculated {len(vectors)} embeddings of dimension {vectors.shape[1]}")

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors = vectors / norms

    print(f"Writing index to {index_dir}...")
    with open(index_dir / INDEX_FILE, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunk_metadata}, f)
    np.save(index_dir / VECTORS_FILE, vectors)
    print(f"Index built with {len(all_chunks)} chunks in {index_dir}")