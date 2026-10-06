"""Score aligners on the reviewed gold set.

    python -m automation.benchmarks.alignment.run_benchmark --project indonesia [--split test]
        [--aligners index,heuristic,bertalign,vecalign]

Prints a table of strict / lenient P/R/F1, evidence recall and per-bead-type
recall, and writes ``results_<split>.json`` next to the data. It also reports,
for the unannotated remainder, how often source and translation sentence counts
already match (the cases where index pairing needs no aligner at all).
"""

import argparse
import json
from pathlib import Path

from automation.benchmarks.alignment.aligners import BASELINE_ALIGNERS, Aligner
from automation.benchmarks.alignment.build_gold import BENCH_DIR
from automation.benchmarks.alignment.data import Unit, read_jsonl
from automation.benchmarks.alignment.metrics import normalize, score_document, summarize
from automation.src.constants import DEFAULT_PROJECT


def load_aligners(names: list[str]) -> dict[str, Aligner]:
    aligners: dict[str, Aligner] = {}
    for name in names:
        if name in BASELINE_ALIGNERS:
            aligners[name] = BASELINE_ALIGNERS[name]
        else:
            # Optional dependencies: imported only when asked for.
            from automation.benchmarks.alignment import embedding_aligners

            aligners[name] = embedding_aligners.get_aligner(name)
    return aligners


def evaluate(units: list[Unit], aligners: dict[str, Aligner]) -> dict:
    results = {}
    for name, align in aligners.items():
        docs = [score_document(align(u), normalize(u.alignment)) for u in units]
        results[name] = summarize(docs)
    return results


def remainder_stats(path: Path) -> dict:
    if not path.exists():
        return {}
    units = read_jsonl(path)
    matched = sum(1 for u in units if u.counts_match)
    return {
        "units": len(units),
        "counts_match": matched,
        "counts_match_rate": matched / len(units) if units else 0.0,
    }


def format_table(results: dict) -> str:
    header = f"{'aligner':<12}{'strict P':>9}{'strict R':>9}{'strict F1':>10}{'lenient F1':>11}{'evid. R':>9}"
    lines = [header, "-" * len(header)]
    for name, r in results.items():
        lines.append(
            f"{name:<12}{r['strict']['precision']:>9.3f}{r['strict']['recall']:>9.3f}"
            f"{r['strict']['f1']:>10.3f}{r['lenient']['f1']:>11.3f}{r['evidence_recall']:>9.3f}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--split", choices=["dev", "test", "all"], default="test")
    parser.add_argument("--aligners", default="index,heuristic")
    args = parser.parse_args(argv)

    base = BENCH_DIR / args.project
    gold_path = base / "gold.jsonl"
    if not gold_path.exists():
        raise SystemExit(f"{gold_path} not found: review gold_draft.jsonl and save it as gold.jsonl")

    units = [u for u in read_jsonl(gold_path) if u.status == "reviewed"]
    if args.split != "all":
        units = [u for u in units if u.split == args.split]
    if not units:
        raise SystemExit("No reviewed gold units for this split.")

    results = evaluate(units, load_aligners(args.aligners.split(",")))
    print(f"{len(units)} gold units ({args.split})\n")
    print(format_table(results))

    stats = remainder_stats(base / "remainder.jsonl")
    if stats:
        print(
            f"\nRemainder (unannotated): {stats['counts_match']}/{stats['units']} units "
            f"({stats['counts_match_rate']:.1%}) already have equal sentence counts."
        )

    out = base / f"results_{args.split}.json"
    out.write_text(
        json.dumps({"units": len(units), "results": results, "remainder": stats}, indent=2),
        encoding="utf-8",
    )
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
