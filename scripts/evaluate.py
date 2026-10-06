"""
Prints Recall@k, MRR@k and Hit@k per retrieval mode, overall and per question type.
Usage:
    python -m scripts.evaluate
    python -m scripts.evaluate --k 3 --modes bm25 vector
    python -m scripts.evaluate --ablation      # BM25 tokenizer ablation
    python -m scripts.evaluate --grounding     # does the answer cite a relevant passage
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rag.bm25 import BM25Index
from rag.config import settings
from rag.models import Retriever, Retrievers
from rag.rag_agent import run_rag
from rag.retriever import BM25Retriever, LoadedIndex, build_retrievers, load_index

DEFAULT_QUESTIONS = Path(__file__).parent.parent / "eval" / "questions.json"


def _normalize(text: str) -> str:
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


def score_grounding(
    retrievers: Retrievers,
    mode: str,
    questions: list[dict],
    judgements: list[set[int]],
    k: int,
) -> tuple[dict[str, float], dict[str, dict[str, float]], int]:
    buckets: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"grounded": [], "retrievable": [], "cited": []}
    )
    failures = 0
    for question, relevant in zip(questions, judgements):
        try:
            result = run_rag(question["question"], mode, retrievers, top_k=k)
        except Exception as exc:  # noqa: BLE001 - one bad answer must not stop the run
            print(f"  generation failed on {question['question']!r}: {exc}", file=sys.stderr)
            failures += 1
            retrievable = grounded = cited = 0.0
        else:
            retrieved_ids = {c.id for c in result.chunks}
            cited_ids = {
                result.chunks[c - 1].id
                for c in result.citations
                if 1 <= c <= len(result.chunks)
            }
            retrievable = 1.0 if retrieved_ids & relevant else 0.0
            grounded = 1.0 if cited_ids & relevant else 0.0
            cited = 1.0 if result.citations else 0.0

        for bucket in ("all", question.get("type", "untyped")):
            buckets[bucket]["grounded"].append(grounded)
            buckets[bucket]["retrievable"].append(retrievable)
            buckets[bucket]["cited"].append(cited)

    summary = {
        name: {metric: sum(vals) / len(vals) for metric, vals in metrics.items()}
        for name, metrics in buckets.items()
    }
    return summary.pop("all"), summary, failures


def print_grounding(title: str, rows: list[tuple[str, dict]], k: int, n: int) -> None:
    label_width = max(len(label) for label, _ in rows)
    print(f"\n{title}  (k={k}, {n} questions)")
    print(
        f"{'':<{label_width}}  {'Grounded':>9}  {'Retr':>6}  {'G|Retr':>7}  {'Cited':>6}"
    )
    print("-" * (label_width + 36))
    for label, m in rows:
        over_retr = m["grounded"] / m["retrievable"] if m["retrievable"] else 0.0
        print(
            f"{label:<{label_width}}  {m['grounded']:>9.3f}  {m['retrievable']:>6.3f}  "
            f"{over_retr:>7.3f}  {m['cited']:>6.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval and answer grounding")
    parser.add_argument("--k", type=int, default=settings.top_k, help="Cutoff for the metrics.")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["bm25", "vector", "hybrid"],
        choices=["bm25", "vector", "hybrid"],
    )
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--index-dir", type=Path, default=settings.index_dir)
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="run BM25 with stemming and stopword removal switched on and off",
    )
    parser.add_argument(
        "--grounding",
        action="store_true",
        help="check whether generated answers cite a relevant passage",
    )
    args = parser.parse_args()

    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    index = load_index(args.index_dir)
    print(f"Index: {len(index.chunks)} chunks from {args.index_dir}")
    print(f"BM25 tokenizer: stem={settings.bm25_stem}, remove_stopwords={settings.bm25_remove_stopwords}")

    judgements, usable = [], []
    for question in questions:
        relevant = relevant_ids(index.chunks, question)
        if not relevant:
            print(
                f"  skipping, no chunk matches must_contain: {question['question']!r}",
                file=sys.stderr,
            )
            continue
        judgements.append(relevant)
        usable.append(question)

    if not usable:
        print("no usable questions. Check eval/questions.json")
        sys.exit(1)
    print(f"Questions: {len(usable)}/{len(questions)} usable")

    retrievers = build_retrievers(index=index)
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

    if args.grounding:
        g_rows, g_by_type, failures = [], {}, 0
        for mode in args.modes:
            overall, per_type, mode_failures = score_grounding(
                retrievers, mode, usable, judgements, args.k
            )
            g_rows.append((mode, overall))
            g_by_type[mode] = per_type
            failures += mode_failures

        print_grounding("Answer grounding", g_rows, args.k, len(usable))
        for type_name in type_names:
            type_rows = [
                (mode, g_by_type[mode][type_name])
                for mode in args.modes
                if type_name in g_by_type[mode]
            ]
            count = sum(1 for q in usable if q.get("type", "untyped") == type_name)
            print_grounding(f"Answer grounding ({type_name})", type_rows, args.k, count)

        if failures:
            print(
                f"\n{failures} of the generation calls failed and count as not grounded.",
                file=sys.stderr,
            )


if __name__ == "__main__":
    main()