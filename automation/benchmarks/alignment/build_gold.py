"""Build the alignment golden set.

    python -m automation.benchmarks.alignment.build_gold sample --project indonesia --n-gold 120
    python -m automation.benchmarks.alignment.build_gold draft  --project indonesia

`sample` writes (under ``automation/benchmarks/alignment/data/<project>/``):
    units_all.jsonl   every eligible section pair
    gold_draft.jsonl  the sampled gold candidates, alignment still empty
    remainder.jsonl   everything else (not annotated)

`draft` asks an LLM to propose an alignment for each gold candidate and writes
it back to ``gold_draft.jsonl`` with ``status: "draft"``. A human then corrects
the file and saves it as ``gold.jsonl`` with ``status: "reviewed"`` on each
line. Only reviewed units are used by `run_benchmark`.

Alignment format: a list of beads ``[[src_indices], [tgt_indices]]`` over the
0-based sentence lists ``src`` / ``tgt`` stored on each line; an empty side
means that sentence has no counterpart.
"""

import argparse
import json
import logging
from pathlib import Path

from pydantic import BaseModel

from automation.benchmarks.alignment.data import (
    Unit,
    build_units,
    read_jsonl,
    sample_gold,
    write_jsonl,
)
from automation.src.constants import DEFAULT_PROJECT
from automation.src.paths import project_dirs

logger = logging.getLogger(__name__)

BENCH_DIR = Path(__file__).resolve().parent / "data"

DRAFT_PROMPT = """You are building a gold standard for sentence alignment between an Indonesian legal text and its English translation.

Align the sentences. Output a list of "beads"; each bead links some Indonesian sentence indices to some English sentence indices that express the same content.
- Most beads are 1-1. Use 1-2 / 2-1 when the translation split or merged sentences.
- A sentence with no counterpart gets a bead with an empty list on the other side.
- Every Indonesian index and every English index must appear in exactly one bead.
- Beads must be in document order, and indices within a bead must be contiguous.

INDONESIAN:
{src}

ENGLISH:
{tgt}
"""


class DraftBead(BaseModel):
    src: list[int]
    tgt: list[int]


class DraftAlignment(BaseModel):
    beads: list[DraftBead]


def numbered(sentences: list[str]) -> str:
    return "\n".join(f"[{i}] {s}" for i, s in enumerate(sentences))


def validate_alignment(unit: Unit, beads: list[list[list[int]]]) -> list[str]:
    """Problems that make an alignment unusable as gold (empty list = fine)."""
    problems = []
    src_seen = sorted(i for b in beads for i in b[0])
    tgt_seen = sorted(i for b in beads for i in b[1])
    if src_seen != list(range(len(unit.src))):
        problems.append("source indices are not covered exactly once")
    if tgt_seen != list(range(len(unit.tgt))):
        problems.append("target indices are not covered exactly once")
    return problems


def cmd_sample(args: argparse.Namespace) -> None:
    dirs = project_dirs(args.project)
    units = build_units(dirs.processed_cleaned_markdown, dirs.processed_translation_markdown)
    if not units:
        raise SystemExit(f"No eligible section pairs under {dirs.processed}")
    gold, remainder = sample_gold(
        units, args.n_gold, seed=args.seed, mismatch_fraction=args.mismatch_fraction
    )
    out = BENCH_DIR / args.project
    write_jsonl(out / "units_all.jsonl", units)
    write_jsonl(out / "gold_draft.jsonl", gold)
    write_jsonl(out / "remainder.jsonl", remainder)
    matched = sum(1 for u in gold if u.counts_match)
    print(
        f"{len(units)} eligible units; gold candidates {len(gold)} "
        f"({matched} count-matched, {len(gold) - matched} mismatched; "
        f"{sum(u.split == 'dev' for u in gold)} dev / {sum(u.split == 'test' for u in gold)} test); "
        f"remainder {len(remainder)}. Written to {out}"
    )


def cmd_draft(args: argparse.Namespace) -> None:
    # Imported here so `sample` and the tests do not need LLM credentials.
    from automation.src.config_loader import load_pipeline_config
    from automation.src.llm.wrapper import AzureLLMWrapper

    path = BENCH_DIR / args.project / "gold_draft.jsonl"
    units = read_jsonl(path)
    config = load_pipeline_config()
    wrapper = AzureLLMWrapper.from_profile(config.evaluation_model)

    done = failed = 0
    for unit in units:
        if unit.alignment and not args.redo:
            continue
        prompt = DRAFT_PROMPT.format(src=numbered(unit.src), tgt=numbered(unit.tgt))
        try:
            result = wrapper.complete_structured(prompt, DraftAlignment)
            beads = [[b["src"], b["tgt"]] for b in result["beads"]]
        except Exception as exc:  # keep going; the unit stays un-drafted
            logger.error("[%s] draft failed: %s", unit.id, exc)
            failed += 1
            continue
        problems = validate_alignment(unit, beads)
        if problems:
            logger.warning("[%s] draft needs attention: %s", unit.id, "; ".join(problems))
        unit.alignment = beads
        unit.status = "draft"
        done += 1
        write_jsonl(path, units)  # persist as we go; calls are expensive

    print(f"drafted {done}, failed {failed}. Review {path}, then save as gold.jsonl")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    sample = sub.add_parser("sample")
    sample.add_argument("--project", default=DEFAULT_PROJECT)
    sample.add_argument("--n-gold", type=int, default=120)
    sample.add_argument("--seed", type=int, default=13)
    sample.add_argument("--mismatch-fraction", type=float, default=0.5)
    sample.set_defaults(func=cmd_sample)

    draft = sub.add_parser("draft")
    draft.add_argument("--project", default=DEFAULT_PROJECT)
    draft.add_argument("--redo", action="store_true", help="Re-draft units that already have an alignment.")
    draft.set_defaults(func=cmd_draft)

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    args.func(args)


if __name__ == "__main__":
    main()
