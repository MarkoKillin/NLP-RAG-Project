import sys
from pathlib import Path

# Add parent directory to path to import rag module
sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.config import RAW_DATA_DIR, INDEX_DIR, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL_NAME
from rag.ingestion import build_lucene_index
from rag.embedding_model import EmbeddingModel


def main():
    """Build the Lucene index from raw documents."""
    print("=" * 60)
    print("Building Lucene Index")
    print("=" * 60)
    print(f"Raw data directory: {RAW_DATA_DIR}")
    print(f"Index directory: {INDEX_DIR}")
    print(f"Chunk size: {CHUNK_SIZE}")
    print(f"Chunk overlap: {CHUNK_OVERLAP}")
    print(f"Embedding model: {EMBEDDING_MODEL_NAME}")
    print("=" * 60)

    if not RAW_DATA_DIR.exists():
        print(f"Error: Raw data directory {RAW_DATA_DIR} does not exist.")
        print(f"Please create it and add .txt or .md files.")
        sys.exit(1)

    print("\nInitializing embedding model...")
    embedding_model = EmbeddingModel(EMBEDDING_MODEL_NAME)

    try:
        build_lucene_index(
            raw_data_dir=RAW_DATA_DIR,
            index_dir=INDEX_DIR,
            embedding_model=embedding_model,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        print("\n" + "=" * 60)
        print("Index built successfully!")
        print("=" * 60)
    except Exception as e:
        print(f"\nError building index: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

