"""Metrics for sentence alignment.

An alignment is a set of *beads*: ``(src_indices, tgt_indices)`` tuples of
sorted sentence indices, e.g. ``((0,), (0, 1))`` for a 1-2 link. Empty sides
are allowed (a deletion or insertion).

* strict   -- a predicted bead counts only if an identical bead is in gold.
              This is the metric Bertalign and Vecalign papers report.
* lenient  -- a predicted bead counts if it shares at least one source AND one
              target sentence with some gold bead (right place, boundaries off).
* evidence -- the downstream use: for each source sentence, did the method
              return exactly the gold target sentences? (source-side recall)
"""

from collections import defaultdict
from collections.abc import Iterable

Bead = tuple[tuple[int, ...], tuple[int, ...]]


def normalize(beads: Iterable[Iterable]) -> set[Bead]:
    """Accept lists/tuples of (src, tgt) index lists and return a bead set."""
    return {
        (tuple(sorted(int(i) for i in src)), tuple(sorted(int(i) for i in tgt)))
        for src, tgt in beads
    }


def bead_type(bead: Bead) -> str:
    return f"{len(bead[0])}-{len(bead[1])}"


def _prf(correct: float, predicted: int, gold: int) -> dict:
    precision = correct / predicted if predicted else 0.0
    recall = correct / gold if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _prf2(correct_p: float, correct_r: float, predicted: int, gold: int) -> dict:
    precision = correct_p / predicted if predicted else 0.0
    recall = correct_r / gold if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def score_document(predicted: set[Bead], gold: set[Bead]) -> dict:
    """Raw counts for one document; sum them across documents, then `summarize`."""
    strict_correct = len(predicted & gold)
    def touches(a: Bead, b: Bead) -> bool:
        return bool(set(a[0]) & set(b[0]) and set(a[1]) & set(b[1]))

    # Lenient precision counts predicted beads that touch some gold bead;
    # lenient recall counts gold beads touched by some predicted bead. They are
    # counted separately because one gold bead can be covered by several
    # predicted beads (e.g. a gold 2-2 predicted as two 1-1).
    lenient_precise = sum(1 for p in predicted if any(touches(p, g) for g in gold))
    lenient_recalled = sum(1 for g in gold if any(touches(p, g) for p in predicted))

    gold_by_src: dict[int, tuple[int, ...]] = {}
    for src, tgt in gold:
        for i in src:
            gold_by_src[i] = tgt
    pred_by_src: dict[int, tuple[int, ...]] = {}
    for src, tgt in predicted:
        for i in src:
            pred_by_src[i] = tgt
    evidence_correct = sum(1 for i, tgt in gold_by_src.items() if pred_by_src.get(i) == tgt)

    by_type: dict[str, dict[str, int]] = defaultdict(lambda: {"gold": 0, "found": 0})
    for bead in gold:
        by_type[bead_type(bead)]["gold"] += 1
        if bead in predicted:
            by_type[bead_type(bead)]["found"] += 1

    return {
        "predicted": len(predicted),
        "gold": len(gold),
        "strict_correct": strict_correct,
        "lenient_precise": lenient_precise,
        "lenient_recalled": lenient_recalled,
        "evidence_total": len(gold_by_src),
        "evidence_correct": evidence_correct,
        "by_type": {k: dict(v) for k, v in by_type.items()},
    }


def summarize(per_document: list[dict]) -> dict:
    """Micro-averaged P/R/F1 over documents, plus per-bead-type recall."""
    totals = defaultdict(int)
    by_type: dict[str, dict[str, int]] = defaultdict(lambda: {"gold": 0, "found": 0})
    for doc in per_document:
        for key in ("predicted", "gold", "strict_correct", "lenient_precise", "lenient_recalled", "evidence_total", "evidence_correct"):
            totals[key] += doc[key]
        for kind, counts in doc["by_type"].items():
            by_type[kind]["gold"] += counts["gold"]
            by_type[kind]["found"] += counts["found"]

    return {
        "documents": len(per_document),
        "strict": _prf(totals["strict_correct"], totals["predicted"], totals["gold"]),
        "lenient": _prf2(totals["lenient_precise"], totals["lenient_recalled"], totals["predicted"], totals["gold"]),
        "evidence_recall": (
            totals["evidence_correct"] / totals["evidence_total"] if totals["evidence_total"] else 0.0
        ),
        "recall_by_type": {
            kind: {**counts, "recall": counts["found"] / counts["gold"] if counts["gold"] else 0.0}
            for kind, counts in sorted(by_type.items())
        },
    }
