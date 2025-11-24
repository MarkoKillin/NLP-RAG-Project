from typing import TypedDict, Any
from pathlib import Path
import numpy as np

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
    from org.apache.lucene.search import KnnVectorQuery  # type: ignore

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


class RetrievedChunk(TypedDict):
    id: int
    source: str
    chunk_index: int
    content: str
    score: float


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

        try:
            parsed_query = self.query_parser.parse(query)
            top_docs: TopDocs = self.searcher.search(parsed_query, top_k)

            results: list[RetrievedChunk] = []
            for score_doc in top_docs.scoreDocs:
                doc = self.searcher.doc(score_doc.doc)
                chunk: RetrievedChunk = {
                    "id": score_doc.doc,
                    "source": doc.get("source"),
                    "chunk_index": int(doc.get("chunk_index")),
                    "content": doc.get("content"),
                    "score": float(score_doc.score),
                }
                results.append(chunk)

            return results
        except Exception as e:
            print(f"Error during BM25 search: {e}")
            return []

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
        self.directory = FSDirectory.open(Paths.get(str(self.index_dir)))
        self.reader = DirectoryReader.open(self.directory)
        self.searcher = IndexSearcher(self.reader)

    def _numpy_to_java_float_array(self, vector: np.ndarray) -> Any:
        """
        Pretvara np.ndarray u Java float[] preko JArray; bez JNI glupiranja sa getEnv/NewFloatArray.
        """
        vec = vector.astype(np.float32)
        return JArray('float')(vec.tolist())

    def search(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        ensure_lucene_env()

        try:
            query_vector = self.embedding_model.encode([query])[0]  # shape: (dim,)
            java_vector = self._numpy_to_java_float_array(query_vector)

            knn_query = KnnVectorQuery("embedding", java_vector, top_k)
            top_docs: TopDocs = self.searcher.search(knn_query, top_k)

            results: list[RetrievedChunk] = []
            for score_doc in top_docs.scoreDocs:
                doc = self.searcher.doc(score_doc.doc)
                chunk: RetrievedChunk = {
                    "id": score_doc.doc,
                    "source": doc.get("source"),
                    "chunk_index": int(doc.get("chunk_index")),
                    "content": doc.get("content"),
                    "score": float(score_doc.score),
                }
                results.append(chunk)

            return results
        except Exception as e:
            print(f"Error during vector search: {e}")
            import traceback
            traceback.print_exc()
            return []

    def close(self):
        if hasattr(self, "reader"):
            self.reader.close()
