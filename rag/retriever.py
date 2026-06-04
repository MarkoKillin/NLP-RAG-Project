from typing import Any
from pathlib import Path
import numpy as np
from rag.models import RetrievedChunk
from rag.config import EMBEDDING_DIM

try:
    import lucene  # type: ignore
    from lucene import JArray  # type: ignore

    from java.nio.file import Paths  # type: ignore
    from org.apache.lucene.analysis.standard import StandardAnalyzer  # type: ignore
    from org.apache.lucene.index import DirectoryReader  # type: ignore
    from org.apache.lucene.queryparser.classic import QueryParser  # type: ignore
    from org.apache.lucene.search import IndexSearcher, TopDocs  # type: ignore
    from org.apache.lucene.store import FSDirectory  # type: ignore
    from org.apache.lucene.search.similarities import BM25Similarity  # type: ignore
    from org.apache.lucene.search import KnnFloatVectorQuery  # type: ignore

    LUCENE_AVAILABLE = True
except ImportError:
    LUCENE_AVAILABLE = False
    print("Warning: PyLucene not available. Please install PyLucene.")


def ensure_lucene_env() -> Any:
    env = lucene.getVMEnv()
    if env is None:
        lucene.initVM(vmargs=["-Xmx2g"])
        env = lucene.getVMEnv()

    env.attachCurrentThread()
    return env


class LuceneBM25Retriever:
    def __init__(self, index_dir: Path):
        if not LUCENE_AVAILABLE:
            raise ImportError("PyLucene is required for BM25 retrieval.")

        ensure_lucene_env()

        self.index_dir = Path(index_dir)
        if not self.index_dir.exists():
            self.index_dir.mkdir(parents=True, exist_ok=True)

        if not any(self.index_dir.iterdir()):
            raise FileNotFoundError(
                f"Index directory {self.index_dir} exists but is empty. "
                "Please build the index first using: python -m scripts.build_index"
            )

        self.directory = FSDirectory.open(Paths.get(str(self.index_dir)))
        self.reader = DirectoryReader.open(self.directory)
        self.searcher = IndexSearcher(self.reader)
        self.searcher.setSimilarity(BM25Similarity())
        self.analyzer = StandardAnalyzer()
        self.query_parser = QueryParser("content", self.analyzer)

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        ensure_lucene_env()

        parsed_query = self.query_parser.parse(query)
        top_docs: TopDocs = self.searcher.search(parsed_query, top_k)
        stored_fields = self.searcher.storedFields()

        results: list[RetrievedChunk] = []
        for score_doc in top_docs.scoreDocs:
            doc = stored_fields.document(score_doc.doc)
            chunk: RetrievedChunk = {
                "id": int(doc.get("doc_id")),
                "source": doc.get("source"),
                "chunk_index": int(doc.get("chunk_index")),
                "content": doc.get("content"),
                "score": float(score_doc.score),
            }
            results.append(chunk)

        return results

    def close(self):
        if hasattr(self, "reader"):
            self.reader.close()


class LuceneVectorRetriever:
    def __init__(self, index_dir: Path, embedding_model):
        if not LUCENE_AVAILABLE:
            raise ImportError("PyLucene is required for vector retrieval.")

        ensure_lucene_env()

        self.index_dir = Path(index_dir)
        if not self.index_dir.exists():
            self.index_dir.mkdir(parents=True, exist_ok=True)

        if not any(self.index_dir.iterdir()):
            raise FileNotFoundError(
                f"Index directory {self.index_dir} exists but is empty. "
                "Please build the index first using: python -m scripts.build_index"
            )

        self.embedding_model = embedding_model
        probe = self.embedding_model.encode(["dimension probe"])
        if probe.shape[1] != EMBEDDING_DIM:
            raise ValueError(
                f"Embedding dimension mismatch: model produced {probe.shape[1]}, "
                f"but EMBEDDING_DIM is configured as {EMBEDDING_DIM}. "
                "The index was built against EMBEDDING_DIM; rebuild it or update the env var."
            )

        self.directory = FSDirectory.open(Paths.get(str(self.index_dir)))
        self.reader = DirectoryReader.open(self.directory)
        self.searcher = IndexSearcher(self.reader)

    def _numpy_to_java_float_array(self, vector: np.ndarray) -> Any:
        vec = vector.astype(np.float32)
        return JArray('float')(vec.tolist())

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        ensure_lucene_env()

        query_vector = self.embedding_model.encode([query])[0]  # shape: (dim,)
        java_vector = self._numpy_to_java_float_array(query_vector)

        knn_query = KnnFloatVectorQuery("embedding", java_vector, top_k)
        top_docs: TopDocs = self.searcher.search(knn_query, top_k)
        stored_fields = self.searcher.storedFields()

        results: list[RetrievedChunk] = []
        for score_doc in top_docs.scoreDocs:
            doc = stored_fields.document(score_doc.doc)
            chunk: RetrievedChunk = {
                "id": int(doc.get("doc_id")),
                "source": doc.get("source"),
                "chunk_index": int(doc.get("chunk_index")),
                "content": doc.get("content"),
                "score": float(score_doc.score),
            }
            results.append(chunk)

        return results

    def close(self):
        if hasattr(self, "reader"):
            self.reader.close()
