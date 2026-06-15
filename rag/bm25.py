import re
from collections import Counter
from math import log

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.postings: dict[str, list[tuple[int, int]]] = {}
        self.doc_lengths: list[int] = []
        self.avgdl: float = 0.0
        self.N: int = 0

    def add(self, doc_tokens: list[str]) -> None:
        doc_id = len(self.doc_lengths)
        self.doc_lengths.append(len(doc_tokens))
        for term, tf in Counter(doc_tokens).items():
            self.postings.setdefault(term, []).append((doc_id, tf))

    def finalize(self) -> None:
        self.N = len(self.doc_lengths)
        self.avgdl = sum(self.doc_lengths) / self.N if self.N else 0.0

    def search(self, query_tokens: list[str], top_k: int = 5) -> list[tuple[int, float]]:
        if self.N == 0:
            return []
        scores: dict[int, float] = {}
        for term in query_tokens:
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
