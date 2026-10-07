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
from automation.benchmarks.alignment.build_gold import BENCH_DIR, validate_alignment
from automation.benchmarks.alignment.data import Unit, read_jsonl
from automation.benchmarks.alignment.metrics import normalize, score_document, summarize
from automation.src.constants import DEFAULT_PROJECT


ELIMINATED = "eliminate"


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


def predict(units: list[Unit], align: Aligner, cache: Path | None) -> dict[str, set]:
    """Predictions per unit id, from `cache` when it covers every unit."""
    if cache is not None and cache.exists():
        stored = json.loads(cache.read_text(encoding="utf-8"))
        if all(u.id in stored for u in units):
            return {u.id: normalize(stored[u.id]) for u in units}
    predictions = {u.id: align(u) for u in units}
    if cache is not None:
        cache.write_text(
            json.dumps({k: sorted(v) for k, v in predictions.items()}), encoding="utf-8"
        )
    return predictions


def evaluate(units: list[Unit], aligners: dict[str, Aligner], cache_dir: Path | None = None) -> dict:
    """Run each aligner once per unit (cached); summarize per split (dev / test / all)."""
    results: dict[str, dict] = {}
    for name, align in aligners.items():
        cache = cache_dir / f"predictions_{name}.json" if cache_dir else None
        predictions = predict(units, align, cache)
        scored = [(u.split, score_document(predictions[u.id], normalize(u.alignment))) for u in units]
        results[name] = {"all": summarize([d for _, d in scored])}
        for split in ("dev", "test"):
            docs = [d for sp, d in scored if sp == split]
            if docs:
                results[name][split] = summarize(docs)
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


def format_table(results: dict, split: str) -> str:
    header = f"{'aligner':<12}{'strict P':>9}{'strict R':>9}{'strict F1':>10}{'lenient F1':>11}{'evid. R':>9}"
    lines = [header, "-" * len(header)]
    for name, by_split in results.items():
        r = by_split[split]
        lines.append(
            f"{name:<12}{r['strict']['precision']:>9.3f}{r['strict']['recall']:>9.3f}"
            f"{r['strict']['f1']:>10.3f}{r['lenient']['f1']:>11.3f}{r['evidence_recall']:>9.3f}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--project", default=DEFAULT_PROJECT)
    parser.add_argument("--aligners", default="index,heuristic")
    args = parser.parse_args(argv)

    base = BENCH_DIR / args.project
    gold_path = base / "gold.jsonl"
    if not gold_path.exists():
        raise SystemExit(f"{gold_path} not found: review gold_draft.jsonl and save it as gold.jsonl")

    # gold.jsonl is the human-corrected copy of gold_draft.jsonl. Everything in it
    # counts as reviewed except units explicitly marked "eliminate".
    all_units = read_jsonl(gold_path)
    units = [u for u in all_units if u.status != ELIMINATED]
    invalid = {u.id: validate_alignment(u, u.alignment) for u in units}
    invalid = {k: v for k, v in invalid.items() if v}
    if invalid:
        for unit_id, problems in invalid.items():
            print(f"INVALID {unit_id}: {'; '.join(problems)}")
        raise SystemExit(f"{len(invalid)} gold unit(s) have invalid alignments; fix gold.jsonl first")
    print(f"{len(all_units) - len(units)} unit(s) eliminated")
    if not units:
        raise SystemExit("No gold units.")

    results = evaluate(units, load_aligners(args.aligners.split(",")), cache_dir=base)
    for split in ("test", "dev", "all"):
        n = sum(1 for u in units if split == "all" or u.split == split)
        print(f"\n== {split} ({n} units) ==")
        print(format_table(results, split))

    stats = remainder_stats(base / "remainder.jsonl")
    if stats:
        print(
            f"\nRemainder (unannotated): {stats['counts_match']}/{stats['units']} units "
            f"({stats['counts_match_rate']:.1%}) already have equal sentence counts."
        )

    out = base / "results.json"
    out.write_text(
        json.dumps({"units": len(units), "results": results, "remainder": stats}, indent=2),
        encoding="utf-8",
    )
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
