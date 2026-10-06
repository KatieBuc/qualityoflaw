"""Sampling section pairs for the alignment benchmark.

A *unit* is one heading section of a source document paired with the same
section of its translation (only documents whose headers validate, so the
pairing by position is sound -- see `markdown.validate`). Sentences come from
the same splitter the pipeline uses, so what is benchmarked is what the
pipeline would align.
"""

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

from automation.src.markdown.chunking import MD_HEADING_RE, parse_sections
from automation.src.markdown.policy_files import CLEANED_MD_SUFFIX, policy_stem
from automation.src.markdown.validate import validate_headers
from automation.src.rag.sentence_split import split_sentences

MIN_SENTENCES = 2
MAX_SENTENCES = 30


@dataclass
class Unit:
    id: str
    policy: str
    section_index: int
    heading: str | None
    src_text: str
    tgt_text: str
    src: list[str]
    tgt: list[str]
    # Filled for gold units only.
    alignment: list[list[list[int]]] = field(default_factory=list)
    split: str = ""  # "dev" | "test" (gold) | "" (remainder)
    status: str = ""  # "draft" | "reviewed"

    @property
    def counts_match(self) -> bool:
        return len(self.src) == len(self.tgt)


def body_text(section_text: str) -> str:
    """A section without its heading lines: headings are structure, not sentences."""
    return "\n".join(
        line for line in section_text.splitlines() if not MD_HEADING_RE.match(line)
    ).strip()


def build_units(
    source_dir: Path,
    translated_dir: Path,
    source_suffix: str = CLEANED_MD_SUFFIX,
) -> list[Unit]:
    sources = {policy_stem(p): p for p in sorted(source_dir.glob(f"*{source_suffix}"))}
    units: list[Unit] = []
    for stem, src_path in sources.items():
        tgt_path = translated_dir / f"{stem}.md"
        if not tgt_path.exists():
            continue
        src_md = src_path.read_text(encoding="utf-8")
        tgt_md = tgt_path.read_text(encoding="utf-8")
        if not validate_headers(stem, src_md, tgt_md).ok:
            continue
        for index, (s, t) in enumerate(zip(parse_sections(src_md), parse_sections(tgt_md))):
            src_text, tgt_text = body_text(s.text), body_text(t.text)
            src = split_sentences(src_text, protect_abbreviations=True)
            tgt = split_sentences(tgt_text)
            if not (MIN_SENTENCES <= len(src) <= MAX_SENTENCES):
                continue
            if not (MIN_SENTENCES <= len(tgt) <= MAX_SENTENCES):
                continue
            units.append(
                Unit(
                    id=f"{stem}#{index}",
                    policy=stem,
                    section_index=index,
                    heading=s.heading,
                    src_text=src_text,
                    tgt_text=tgt_text,
                    src=src,
                    tgt=tgt,
                )
            )
    return units


def sample_gold(
    units: list[Unit],
    n_gold: int,
    *,
    seed: int = 13,
    mismatch_fraction: float = 0.5,
    dev_fraction: float = 0.3,
    max_per_policy: int = 3,
) -> tuple[list[Unit], list[Unit]]:
    """Pick `n_gold` units to annotate; everything else is the remainder.

    Stratified: `mismatch_fraction` of the gold units have unequal sentence
    counts. Sampling uniformly would give mostly count-matched units, where the
    trivial index pairing already works and the aligners are not exercised.
    At most `max_per_policy` units per document keeps one policy from
    dominating. Gold units are tagged dev/test; dev is for tuning aligner
    settings, test is what gets reported.
    """
    rng = random.Random(seed)
    pool = list(units)
    rng.shuffle(pool)

    per_policy: dict[str, int] = {}
    chosen: list[Unit] = []

    def take(candidates: list[Unit], k: int) -> None:
        for unit in candidates:
            if k <= 0:
                return
            if unit in chosen or per_policy.get(unit.policy, 0) >= max_per_policy:
                continue
            chosen.append(unit)
            per_policy[unit.policy] = per_policy.get(unit.policy, 0) + 1
            k -= 1

    n_mismatch = round(n_gold * mismatch_fraction)
    take([u for u in pool if not u.counts_match], n_mismatch)
    take([u for u in pool if u.counts_match], n_gold - len(chosen))
    take(pool, n_gold - len(chosen))  # top up if a stratum ran short

    rng.shuffle(chosen)
    n_dev = round(len(chosen) * dev_fraction)
    for i, unit in enumerate(chosen):
        unit.split = "dev" if i < n_dev else "test"
        unit.status = "draft"

    chosen_ids = {u.id for u in chosen}
    remainder = [u for u in units if u.id not in chosen_ids]
    return chosen, remainder


def write_jsonl(path: Path, units: list[Unit]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for unit in units:
            f.write(json.dumps(asdict(unit), ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[Unit]:
    with path.open(encoding="utf-8") as f:
        return [Unit(**json.loads(line)) for line in f if line.strip()]
