"""Retrieval evaluation.

Scores each retrieval mode against the labelled questions in eval/questions.json
and prints Recall@k, MRR@k and Hit@k, broken down by question type.

Relevance comes from content, not from chunk ids. A chunk is relevant to a
question if it comes from one of the listed sources and contains the
`must_contain` string. Judgements then survive re-chunking and re-indexing, which
chunk-id labels would not.

Only the vector and hybrid modes need Ollama. `--modes bm25` runs offline.

Usage:
    python -m scripts.evaluate
    python -m scripts.evaluate --k 3 --modes bm25 vector
    python -m scripts.evaluate --ablation      # BM25 tokenizer ablation
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.bm25 import BM25Index
from rag.config import BM25_REMOVE_STOPWORDS, BM25_STEM, INDEX_DIR, TOP_K
from rag.models import Retriever
from rag.retriever import LoadedIndex, build_retrievers, load_index

DEFAULT_QUESTIONS = Path(__file__).parent.parent / "eval" / "questions.json"


def _normalize(text: str) -> str:
    """Chunking joins words on single spaces, so a `must_contain` string that
    spans a line break in the source file would never match the chunk text."""
    return re.sub(r"\s+", " ", text).strip().lower()


def relevant_ids(chunks: list[dict], question: dict) -> set[int]:
    needle = _normalize(question["must_contain"])
    sources = question.get("sources")
    return {
        c["id"]
        for c in chunks
        if (sources is None or c["source"] in sources) and needle in _normalize(c["content"])
    }


def score_mode(
    retriever: Retriever, questions: list[dict], judgements: list[set[int]], k: int
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    buckets: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"recall": [], "rr": [], "hit": []}
    )

    for question, relevant in zip(questions, judgements):
        retrieved = [c.id for c in retriever.search(question["question"], top_k=k)]
        found = set(retrieved) & relevant

        rr = 0.0
        for rank, chunk_id in enumerate(retrieved, start=1):
            if chunk_id in relevant:
                rr = 1.0 / rank
                break

        for bucket in ("all", question.get("type", "untyped")):
            buckets[bucket]["recall"].append(len(found) / len(relevant))
            buckets[bucket]["rr"].append(rr)
            buckets[bucket]["hit"].append(1.0 if found else 0.0)

    summary = {
        name: {metric: sum(vals) / len(vals) for metric, vals in metrics.items()}
        for name, metrics in buckets.items()
    }
    return summary.pop("all"), summary


def print_table(title: str, rows: list[tuple[str, dict]], k: int, n: int) -> None:
    label_width = max(len(label) for label, _ in rows)
    print(f"\n{title}  (k={k}, {n} questions)")
    print(f"{'':<{label_width}}  {'Recall@k':>9}  {'MRR@k':>7}  {'Hit@k':>7}")
    print("-" * (label_width + 29))
    for label, m in rows:
        print(f"{label:<{label_width}}  {m['recall']:>9.3f}  {m['rr']:>7.3f}  {m['hit']:>7.3f}")


def run_ablation(index: LoadedIndex, questions, judgements, k: int) -> None:
    """Score all four BM25 tokenizer settings in one run.

    The on-disk index holds only one of them, but BM25 is cheap to rebuild in
    memory, so this needs no re-indexing and no embeddings.
    """
    from rag.retriever import BM25Retriever

    rows = []
    for stem in (False, True):
        for remove_stopwords in (False, True):
            bm25 = BM25Index(stem=stem, remove_stopwords=remove_stopwords)
            for chunk in index.chunks:
                bm25.add(chunk["content"])
            bm25.finalize()
            variant = LoadedIndex(bm25=bm25, chunks=index.chunks, vectors=index.vectors)
            overall, _ = score_mode(BM25Retriever(variant), questions, judgements, k)
            label = f"stem={int(stem)} stopwords_removed={int(remove_stopwords)}"
            rows.append((label, overall))

    print_table("BM25 tokenizer ablation", rows, k, len(questions))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality.")
    parser.add_argument("--k", type=int, default=TOP_K, help="Cutoff for the metrics.")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["bm25", "vector", "hybrid"],
        choices=["bm25", "vector", "hybrid"],
    )
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--index-dir", type=Path, default=INDEX_DIR)
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="Also compare BM25 tokenizer settings (stemming, stopword removal).",
    )
    args = parser.parse_args()

    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    index = load_index(args.index_dir)
    print(f"Index: {len(index.chunks)} chunks from {args.index_dir}")
    print(f"BM25 tokenizer: stem={BM25_STEM}, remove_stopwords={BM25_REMOVE_STOPWORDS}")

    # A question whose `must_contain` matches nothing is a broken judgement, not
    # a retrieval failure. Drop it loudly rather than scoring a guaranteed zero.
    judgements, usable = [], []
    for question in questions:
        relevant = relevant_ids(index.chunks, question)
        if not relevant:
            print(
                f"  skipping (no chunk matches must_contain): {question['question']!r}",
                file=sys.stderr,
            )
            continue
        judgements.append(relevant)
        usable.append(question)

    if not usable:
        print("No usable questions. Check eval/questions.json against your corpus.")
        sys.exit(1)
    print(f"Questions: {len(usable)}/{len(questions)} usable")

    retrievers = build_retrievers(args.index_dir)
    rows, by_type = [], {}
    for mode in args.modes:
        overall, per_type = score_mode(retrievers.get(mode), usable, judgements, args.k)
        rows.append((mode, overall))
        by_type[mode] = per_type

    print_table("Retrieval mode", rows, args.k, len(usable))

    type_names = sorted({t for per_type in by_type.values() for t in per_type})
    for type_name in type_names:
        type_rows = [(mode, by_type[mode][type_name]) for mode in args.modes if type_name in by_type[mode]]
        count = sum(1 for q in usable if q.get("type", "untyped") == type_name)
        print_table(f"Question type: {type_name}", type_rows, args.k, count)

    if args.ablation:
        run_ablation(index, usable, judgements, args.k)


if __name__ == "__main__":
    main()