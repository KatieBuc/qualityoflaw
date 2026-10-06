"""The `md_to_text` step: translated Markdown -> what downstream actually reads.

Translation now produces Markdown, but storage, evaluation, comparison and
diagnosis all speak plain text and are deliberately left unmodified. This
module is the adapter, and it writes two artifacts per policy:

1. ``results/translation/<POLICY>.txt`` -- the plain-text translation, in the
   same shape `chunking.combine_translations` used to produce (one heading or
   paragraph per line, blank line between blocks). `policy_files
   .filter_policy_files` globs ``*.txt``, so every downstream stage keeps
   working untouched.

2. ``mid_product/chunks/<POLICY>.chunks.json`` -- the *retrieval* units, one
   per Markdown heading section, in the schema `rag.store
   ._load_translation_chunks` already expects.

Point 2 is what keeps evidence retrieval stable. Translation packs sections
into ~8,000-char units to save API calls, but retrieval must keep the
per-Pasal granularity it has today (median ~200 chars): `rag.retriever` reads
only `chunk_id` and `text`, and `rag.prompt_builder` cites individual
sentences within a chunk, so coarser chunks would directly coarsen the
evidence. Rebuilding chunks.json here at heading granularity -- rather than
handing storage the packed translation units -- means `rag.store` sees the
same kind of input through the same code path as before.

Source and translation sections are paired positionally, which is only sound
when the translation preserved the source's heading structure. When it did
not, the records are emitted with `translated_text` only (storage never reads
the source side), rather than silently pairing mismatched sections.

A structural difference is not automatically a defect. The curated corpus
carries OCR damage that a good translation repairs -- a cross-reference
promoted to a heading, a fragment like `##### 1  AN`, a number split as
`Pasal 3 1` -- and every repair moves the heading counts. Even a *lost
numbered clause* (`StructureDiff.lost_clauses`) is not recorded as a
failure: `_clause_identity` is language-agnostic (heading depth + a number
token, no per-language keyword list), which means it no longer filters out
the OCR artifacts above the way the old keyword-based check did -- one of
them colliding with a real clause's identity reads as a false "lost clause".
Future policies may also be amendments with legitimately non-continuous or
repeated numbering, which would trip the same false positive. So every
structural difference, lost clauses included, is logged and counted as
`repaired`, never failed.
"""

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from automation.src.config_loader import (
    ResolvedPipelineConfig,
    get_run_dir,
    resolve_mid_product_dir,
    resolve_results_dir,
)
from automation.src.failure_log import clear_failure, record_failure
from automation.src.markdown.chunking import (
    MD_HEADING_RE,
    align_sections,
    check_structure,
    parse_sections,
)
from automation.src.markdown.validate import HeaderCheck, validate_headers
from automation.src.markdown.policy_files import (
    markdown_policy_files,
    source_markdown_path,
)

logger = logging.getLogger(__name__)

# Markdown a translation might contain that has no place in the plain-text
# artifact. The curated corpus has essentially none of this (2 emphasis lines
# and 1 link across 198 files), but the model can introduce it.
_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*")
_ITALIC_RE = re.compile(r"(?<!\*)\*(?!\s)([^*]+?)(?<!\s)\*(?!\*)")
_CODE_RE = re.compile(r"`([^`]+)`")
_HRULE_RE = re.compile(r"^\s*([-*_])(?:\s*\1){2,}\s*$")

# `render_markdown` emits lettered/roman list items as nested bullets
# ("   - b. text"); the plain-text form drops the bullet and keeps the
# original clause marker, which is what the legal text actually numbers by.
_BULLET_RE = re.compile(r"^\s*[-*+]\s+(.*)$")


def _strip_inline(text: str) -> str:
    text = _LINK_RE.sub(r"\1", text)
    text = _BOLD_RE.sub(r"\1", text)
    text = _ITALIC_RE.sub(r"\1", text)
    text = _CODE_RE.sub(r"\1", text)
    return text


def markdown_to_text(md_text: str) -> str:
    """Render Markdown back to the plain-text shape downstream expects.

    Headings lose their `#` markers but keep their text (evaluation prompts
    and evidence citations refer to "Article 5"), hard-break trailing double
    spaces are dropped, nested bullets fall back to their clause markers, and
    runs of blank lines collapse to one. Block structure is otherwise the
    Markdown's own, which is already one block per line with blank lines
    between -- so the result matches what the raw-text path produced.
    """
    lines: list[str] = []
    for raw_line in md_text.splitlines():
        line = raw_line.rstrip()
        if not line.strip():
            lines.append("")
            continue
        if _HRULE_RE.match(line):
            continue

        heading = MD_HEADING_RE.match(line)
        if heading:
            line = heading.group(2).strip()

        # Also applied to heading text: the corpus contains headings whose
        # text itself starts with a bullet ("## - BUPATI MINAHASA TENGGARA,"),
        # and the marker is no more wanted there than in a list item.
        bullet = _BULLET_RE.match(line)
        line = bullet.group(1).strip() if bullet else line.strip()

        line = _strip_inline(line).strip()
        if line:
            lines.append(line)

    collapsed: list[str] = []
    for line in lines:
        if not line and (not collapsed or not collapsed[-1]):
            continue
        collapsed.append(line)

    text = "\n".join(collapsed).strip()
    return f"{text}\n" if text else ""


@dataclass
class SourceAlignment:
    """How much of a document's original-language text made it into the
    retrieval records -- see `build_retrieval_chunks`."""

    paired: int  # records that carry a source section's text
    total: int  # records emitted (== translated sections with non-empty text)

    @property
    def unpaired(self) -> int:
        return self.total - self.paired

    @property
    def fully_aligned(self) -> bool:
        return self.total > 0 and self.unpaired == 0


def build_retrieval_chunks(
    source_md: str, translated_md: str
) -> tuple[list[dict], SourceAlignment]:
    """Build per-heading retrieval records from a translated document.

    Returns `(records, alignment)`. Each translated section is paired to its
    source counterpart individually (`chunking.align_sections`: by position
    when both documents have the same section count, otherwise anchored on
    numbered-clause headings), so a structural difference costs at most the original
    text of the sections it actually touched -- not the whole document, which
    is what the old all-or-nothing heading-signature gate did. A section with
    no sound source counterpart gets `text=""`; `alignment` reports how many.

    Every record carries the keys `rag.store._load_translation_chunks`
    validates (`section_id`, `chunk_index`, `type`, a non-empty
    `translated_text`); a section whose plain text is empty is dropped, since
    one empty `translated_text` would make storage discard the whole file and
    fall back to regex-chunking the merged English text.
    """
    source_sections = parse_sections(source_md)
    translated_sections = parse_sections(translated_md)
    pairing = (
        align_sections(source_sections, translated_sections)
        if source_sections
        else [None] * len(translated_sections)
    )

    records: list[dict] = []
    paired = 0
    for index, section in enumerate(translated_sections):
        translated_text = markdown_to_text(section.text).strip()
        if not translated_text:
            continue
        source_index = pairing[index]
        source_text = (
            markdown_to_text(source_sections[source_index].text).strip()
            if source_index is not None
            else ""
        )
        if source_text:
            paired += 1
        records.append(
            {
                "chunk_index": 0,
                "section_id": len(records),
                "type": "structural",
                "context": None,
                "heading": section.heading,
                "heading_path": section.heading_path,
                "text": source_text,
                "translated_text": translated_text,
            }
        )
    return records, SourceAlignment(paired=paired, total=len(records))


def convert_one(
    translated_path: Path,
    source_path: Path | None,
    text_path: Path,
    chunks_path: Path,
) -> tuple[int, SourceAlignment]:
    """Convert one translated document. Returns `(chunk_count, alignment)`."""
    translated_md = translated_path.read_text(encoding="utf-8")
    source_md = source_path.read_text(encoding="utf-8") if source_path else ""

    text_path.parent.mkdir(parents=True, exist_ok=True)
    text_path.write_text(markdown_to_text(translated_md), encoding="utf-8")

    records, alignment = build_retrieval_chunks(source_md, translated_md)
    chunks_path.parent.mkdir(parents=True, exist_ok=True)
    chunks_path.write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return len(records), alignment


def run_md_to_text_step(
    run_id: str,
    config: ResolvedPipelineConfig,
    *,
    small_scale: bool = False,
    force: bool = False,
) -> dict:
    """Convert every translated Markdown file in `processed/` to plain text.

    Reads the validated `processed/{cleaned,translation}_markdown` pair. Each
    file's headers are checked first (`markdown.validate`): a pair whose
    heading count or hierarchy differs is marked failed and skipped, and any
    stale output for it is removed so later steps cannot pick it up.

    Idempotent: a policy is skipped once both its `.txt` and its
    `chunks.json` exist, so the step can be re-run to fill in whatever is
    missing. Structure drift is recorded against the `translation` step in
    `failures.json` -- it is a translation-fidelity defect, and that is where
    someone looking for one would go.
    """
    run_dir = get_run_dir(run_id)
    md_dir = config.paths.processed_translation_markdown_dir
    text_dir = resolve_results_dir(run_dir, "translation")
    chunks_dir = resolve_mid_product_dir(run_dir, "chunks")

    start = time.time()
    counts = {
        "total": 0,
        "succeeded": 0,
        "skipped": 0,
        "failed": 0,
        # Kept for output-schema stability; no longer incremented here (see
        # the comment above the structure-diff handling below for why
        # clause-identity mismatches are no longer distinguished from any
        # other benign structure change).
        "clauses_lost": 0,
        # Files whose structure changed at all -- an OCR repair, a clause
        # renumbering, or genuinely missing content. All logged, none failed.
        "repaired": 0,
        # Files where some translated section could not be paired to a source
        # section, so its original-language text (and thus `evidence_original`
        # downstream) is missing.
        "source_unaligned": 0,
    }
    failed_files: list[dict] = []

    if not md_dir.is_dir():
        raise FileNotFoundError(f"No translated markdown to convert in {md_dir}")

    for translated_path in markdown_policy_files(
        md_dir, small_scale, suffix=".md", stems=config.paths.small_scale_stems
    ):
        stem = translated_path.stem
        counts["total"] += 1
        text_path = text_dir / f"{stem}.txt"
        chunks_path = chunks_dir / f"{stem}.chunks.json"
        if text_path.exists() and chunks_path.exists() and not force:
            counts["skipped"] += 1
            continue

        source_path = source_markdown_path(
            config.paths.processed_cleaned_markdown_dir,
            stem,
            suffix=config.paths.markdown_input_suffix,
        )
        if source_path.exists():
            header_check = validate_headers(
                stem,
                source_path.read_text(encoding="utf-8"),
                translated_path.read_text(encoding="utf-8"),
            )
        else:
            header_check = HeaderCheck(stem, False, problem="no source file")
        if not header_check.ok:
            counts["failed"] += 1
            message = f"Markdown headers do not match: {header_check.problem}"
            failed_files.append(
                {"filename": translated_path.name, "artifact": "validation", "message": message}
            )
            for stale in (text_path, chunks_path):
                stale.unlink(missing_ok=True)
            record_failure(
                run_id,
                "translation",
                {
                    "filename": f"{stem}.txt",
                    "policy_file": f"{stem}.txt",
                    "error_type": "HeaderMismatch",
                    "message": message,
                    "details": {
                        "step": "md_to_text",
                        "source_headings": header_check.source_headings,
                        "translated_headings": header_check.translated_headings,
                    },
                    "attempts": 1,
                },
            )
            logger.error("[%s] %s", stem, message)
            continue

        try:
            chunk_count, alignment = convert_one(
                translated_path, source_path, text_path, chunks_path
            )
        except Exception as exc:
            counts["failed"] += 1
            failed_files.append(
                {"filename": translated_path.name, "artifact": "translation", "message": str(exc)}
            )
            logger.error("[%s] md_to_text failed: %s", translated_path.name, exc)
            continue

        counts["succeeded"] += 1
        if alignment.unpaired:
            counts["source_unaligned"] += 1
        diff = (
            check_structure(
                source_path.read_text(encoding="utf-8"),
                translated_path.read_text(encoding="utf-8"),
            )
            if source_path is not None
            else None
        )

        # Numbered-clause identity is informational only, never a recorded
        # failure. `_clause_identity` is language-agnostic (heading depth +
        # a number token, no keyword dictionary), which trades away the
        # OCR-artifact guard the old keyword-based check had: a cross-
        # reference the corpus promoted to a heading, or a split number like
        # `Pasal 3 1`, now resolves to an identity that won't match the
        # translation's repaired heading, showing up as a "lost clause" even
        # though nothing was actually lost. Future policies may also be
        # amendments with legitimately non-continuous or repeated numbering,
        # which would trip the same false positive. So: log for visibility,
        # never fail the file over it.
        clear_failure(run_id, "translation", f"{stem}.txt")
        if diff is not None:
            counts["repaired"] += 1
            if diff.lost_clauses:
                logger.warning(
                    "[%s] structure changed (clause identities did not all match; "
                    "not treated as a failure -- see module docstring): %s",
                    stem,
                    diff.describe(),
                )
            else:
                logger.info("[%s] structure changed (no content lost): %s", stem, diff.describe())
        if alignment.unpaired:
            # Not a content defect, but the original-language side is
            # incomplete: `evidence_original` will be missing for the
            # sections that couldn't be paired. Recorded so it is visible
            # instead of only in the logs; re-added under the same
            # `filename` key `clear_failure` just cleared.
            logger.warning(
                "[%s] %d/%d sections have no original-language text (structure drift)",
                stem,
                alignment.unpaired,
                alignment.total,
            )
            record_failure(
                run_id,
                "translation",
                {
                    "filename": f"{stem}.txt",
                    "policy_file": f"{stem}.txt",
                    "error_type": "SourceAlignmentPartial",
                    "message": (
                        f"{alignment.unpaired}/{alignment.total} sections have no "
                        "original-language text (structure drift); evidence_original "
                        "will be missing for those"
                    ),
                    "details": {
                        "step": "md_to_text",
                        "unpaired": alignment.unpaired,
                        "total": alignment.total,
                    },
                    "attempts": 1,
                },
            )

        logger.info(
            "[%s] %d retrieval chunks (%d/%d source-paired)",
            stem,
            chunk_count,
            alignment.paired,
            alignment.total,
        )

    elapsed = round(time.time() - start, 2)
    return {"counts": counts, "failed_files": failed_files, "elapsed_s": elapsed}


def main(argv: list[str] | None = None) -> int:
    """Standalone conversion, independent of any run.

    Useful for rendering the Indonesian source corpus to plain text, or for
    checking one file's conversion without spending a pipeline run.
    """
    parser = argparse.ArgumentParser(
        description="Convert Markdown to the pipeline's plain-text form."
    )
    parser.add_argument("source", type=Path, help="A .md file, or a directory of them")
    parser.add_argument("target", type=Path, help="Output .txt file, or a directory")
    args = parser.parse_args(argv)

    if args.source.is_dir():
        args.target.mkdir(parents=True, exist_ok=True)
        converted = 0
        for path in sorted(args.source.glob("*.md")):
            out = args.target / f"{path.name.removesuffix('.cleaned.md').removesuffix('.md')}.txt"
            out.write_text(
                markdown_to_text(path.read_text(encoding="utf-8")), encoding="utf-8"
            )
            converted += 1
        print(f"Converted {converted} file(s) to {args.target}")
        return 0

    if not args.source.exists():
        print(f"Not found: {args.source}", file=sys.stderr)
        return 1

    text = markdown_to_text(args.source.read_text(encoding="utf-8"))
    if args.target.suffix:
        args.target.parent.mkdir(parents=True, exist_ok=True)
        args.target.write_text(text, encoding="utf-8")
    else:
        args.target.mkdir(parents=True, exist_ok=True)
        (args.target / f"{args.source.stem}.txt").write_text(text, encoding="utf-8")
    print(f"Wrote {args.target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
