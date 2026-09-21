import re
from collections import Counter
from math import log

import snowballstemmer

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)

# Standard English stopword list. Kept in-repo rather than pulled from NLTK so
# the tokenizer has no data-download step.
STOPWORDS = frozenset("""
a about above after again against all am an and any are aren't as at be because
been before being below between both but by can't cannot could couldn't did
didn't do does doesn't doing don't down during each few for from further had
hadn't has hasn't have haven't having he he'd he'll he's her here here's hers
herself him himself his how how's i i'd i'll i'm i've if in into is isn't it
it's its itself let's me more most mustn't my myself no nor not of off on once
only or other ought our ours ourselves out over own same shan't she she'd
she'll she's should shouldn't so some such than that that's the their theirs
them themselves then there there's these they they'd they'll they're they've
this those through to too under until up very was wasn't we we'd we'll we're
we've were weren't what what's when when's where where's which while who who's
whom why why's with won't would wouldn't you you'd you'll you're you've your
yours yourself yourselves
""".split())

# Built on first use and kept at module level so BM25Index stays picklable. A
# stemmer stored as an instance attribute would get serialized with the index.
_stemmer = None


def _stem_all(tokens: list[str]) -> list[str]:
    global _stemmer
    if _stemmer is None:
        _stemmer = snowballstemmer.stemmer("english")
    return _stemmer.stemWords(tokens)


class BM25Index:
    """BM25 over a tokenized corpus.

    Tokenizer settings live on the index, not in module-level config. The query
    path calls ``index.tokenize``, so documents and queries always go through
    the same pipeline, and the settings travel with the pickled index.
    """

    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        stem: bool = True,
        remove_stopwords: bool = True,
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
            stripped = [t for t in tokens if t not in STOPWORDS]
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
        # Guard against an all-empty corpus: avgdl of 0 would divide-by-zero in
        # the length-normalization term during search.
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