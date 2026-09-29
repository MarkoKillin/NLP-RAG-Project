import re
import threading
from collections import Counter
from functools import lru_cache
from math import log
from pathlib import Path

import snowballstemmer

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

_STOPWORDS_FILE = Path(__file__).parent / "stopwords.txt"


@lru_cache(maxsize=1)
def _load_stopwords() -> frozenset[str]:
    text = _STOPWORDS_FILE.read_text(encoding="utf-8")
    return frozenset(line.strip() for line in text.splitlines() if line.strip())


_thread_local = threading.local()


def _stem_all(tokens: list[str]) -> list[str]:
    stemmer = getattr(_thread_local, "stemmer", None)
    if stemmer is None:
        stemmer = snowballstemmer.stemmer("english")
        _thread_local.stemmer = stemmer
    return stemmer.stemWords(tokens)


class BM25Index:
    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        stem: bool = True,
        remove_stopwords: bool = False,
    ):
        self.k1 = k1
        self.b = b
        self.stem = stem
        self.remove_stopwords = remove_stopwords
        self.postings: dict[str, list[tuple[int, int]]] = {}
        self.doc_lengths: list[int] = []
        self.avgdl: float = 0.0
        self.N: int = 0

    def tokenize(self, text: str) -> list[str]:
        tokens = _TOKEN_RE.findall(text.lower())
        if self.remove_stopwords:
            stop = _load_stopwords()
            stripped = [t for t in tokens if t not in stop]
            # An all-stopword query ("what is it about?") would return nothing.
            tokens = stripped or tokens
        if self.stem:
            tokens = _stem_all(tokens)
        return tokens

    def add(self, text: str) -> None:
        doc_tokens = self.tokenize(text)
        doc_id = len(self.doc_lengths)
        self.doc_lengths.append(len(doc_tokens))
        for term, tf in Counter(doc_tokens).items():
            self.postings.setdefault(term, []).append((doc_id, tf))

    def finalize(self) -> None:
        self.N = len(self.doc_lengths)
        self.avgdl = sum(self.doc_lengths) / self.N if self.N else 0.0
        if self.avgdl == 0.0:
            self.avgdl = 1.0

    def search(self, query: str, top_k: int = 5) -> list[tuple[int, float]]:
        if self.N == 0:
            return []
        scores: dict[int, float] = {}
        for term in self.tokenize(query):
            postings = self.postings.get(term)
            if not postings:
                continue
            idf = log((self.N - len(postings) + 0.5) / (len(postings) + 0.5) + 1)
            for doc_id, tf in postings:
                dl = self.doc_lengths[doc_id]
                denom = tf + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * tf * (self.k1 + 1) / denom
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return ranked[:top_k]