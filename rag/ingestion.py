from pathlib import Path

try:
    import lucene  # type: ignore
    from lucene import JArray  # type: ignore
    from java.nio.file import Paths # type: ignore
    from org.apache.lucene.analysis.standard import StandardAnalyzer # type: ignore
    from org.apache.lucene.document import Document, Field, FieldType, StoredField # type: ignore
    from org.apache.lucene.index import IndexWriter, IndexWriterConfig, IndexOptions # type: ignore
    from org.apache.lucene.store import FSDirectory # type: ignore
    from org.apache.lucene.search.similarities import BM25Similarity # type: ignore
    from org.apache.lucene.document import KnnVectorField # type: ignore
    LUCENE_AVAILABLE = True
except ImportError:
    LUCENE_AVAILABLE = False
    print("Warning: PyLucene not available. Please install PyLucene.")

import numpy as np

from rag.config import RAW_DATA_DIR, INDEX_DIR, CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_DIM
from rag.embedding_model import EmbeddingModel


def chunk_text(text: str, chunk_size: int = 400, chunk_overlap: int = 50) -> list[str]:
    words = text.split()
    chunks = []

    if len(words) <= chunk_size:
        return [text]

    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
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
                    if content.strip():
                        documents.append((file_path.name, content))
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

    return documents


def build_lucene_index(
    raw_data_dir: Path,
    index_dir: Path,
    embedding_model: EmbeddingModel,
    chunk_size: int = 400,
    chunk_overlap: int = 50,
) -> None:
    if not LUCENE_AVAILABLE:
        raise ImportError(
            "PyLucene is not available. Please install PyLucene to build the index."
        )

    # Initialize Lucene VM
    if not lucene.getVMEnv():
        lucene.initVM(vmargs=["-Xmx2g"])

    print(f"Loading documents from {raw_data_dir}...")
    documents = load_documents(raw_data_dir)
    if not documents:
        raise ValueError(f"No documents found in {raw_data_dir}")

    print(f"Loaded {len(documents)} documents")

    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)

    print("Chunking documents and computing embeddings...")
    all_chunks = []
    all_embeddings = []
    chunk_metadata = []  # (source, chunk_index)

    for filename, content in documents:
        chunks = chunk_text(content, chunk_size, chunk_overlap)
        for idx, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            chunk_metadata.append((filename, idx))

    print(f"Created {len(all_chunks)} chunks")

    # Compute embeddings in batches
    batch_size = 32
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i : i + batch_size]
        embeddings = embedding_model.encode(batch)
        all_embeddings.append(embeddings)
        if (i // batch_size + 1) % 10 == 0:
            print(f"Computed embeddings for {min(i + batch_size, len(all_chunks))} chunks...")

    all_embeddings = np.vstack(all_embeddings)
    print(f"Computed {len(all_embeddings)} embeddings of dimension {all_embeddings.shape[1]}")

    print(f"Building Lucene index in {index_dir}...")

    directory = FSDirectory.open(Paths.get(str(index_dir)))
    analyzer = StandardAnalyzer()

    config = IndexWriterConfig(analyzer)
    config.setSimilarity(BM25Similarity())

    writer = IndexWriter(directory, config)

    # Define field types
    # Text field for BM25 search (stored and indexed)
    text_field_type = FieldType()
    text_field_type.setIndexOptions(IndexOptions.DOCS_AND_FREQS_AND_POSITIONS)
    text_field_type.setStored(True)
    text_field_type.setTokenized(True)
    text_field_type.freeze()

    # String field for source (stored, not tokenized)
    string_field_type = FieldType()
    string_field_type.setIndexOptions(IndexOptions.DOCS)
    string_field_type.setStored(True)
    string_field_type.setTokenized(False)
    string_field_type.freeze()

    # Add documents to index
    doc_id = 0
    for (chunk, embedding), (source, chunk_idx) in zip(
        zip(all_chunks, all_embeddings), chunk_metadata
    ):
        doc = Document()

        # Text field for BM25
        doc.add(Field("content", chunk, text_field_type))

        # Source and chunk index
        doc.add(Field("source", source, string_field_type))
        doc.add(StoredField("chunk_index", chunk_idx))

        # Vector field for k-NN search
        # Convert numpy array to Java float array
        vector = embedding.astype(np.float32).tolist() 
        java_vector = JArray('float')(vector)
        doc.add(KnnVectorField("embedding", java_vector))

        # Store document ID
        doc.add(StoredField("doc_id", doc_id))

        writer.addDocument(doc)
        doc_id += 1

        if (doc_id + 1) % 100 == 0:
            print(f"Indexed {doc_id + 1} chunks...")

    writer.commit()
    writer.close()

    print(f"Index built successfully with {doc_id} documents in {index_dir}")

