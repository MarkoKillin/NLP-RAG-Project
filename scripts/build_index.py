import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.config import settings
from rag.ingestion import build_index
from rag.embedding_model import EmbeddingModel


def main():
    """Build the search index from raw documents."""
    print("=" * 60)
    print("Building Index")
    print("=" * 60)
    print(f"Raw data directory: {settings.raw_data_dir}")
    print(f"Index directory: {settings.index_dir}")
    print(f"Chunk size: {settings.chunk_size}")
    print(f"Chunk overlap: {settings.chunk_overlap}")
    print(f"Embedding model: {settings.embedding_model_name}")
    print("=" * 60)

    if not settings.raw_data_dir.exists():
        print(f"Error: Raw data directory {settings.raw_data_dir} does not exist.")
        print("Please create it and add .txt or .md files.")
        sys.exit(1)

    print("\nInitializing embedding model...")
    embedding_model = EmbeddingModel(settings.embedding_model_name)

    try:
        build_index(
            raw_data_dir=settings.raw_data_dir,
            index_dir=settings.index_dir,
            embedding_model=embedding_model,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
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

