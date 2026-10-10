"""Stage 1.1, step 3: quality-check, fix, re-check and sort every extracted text.

For each policy (stem) in ``extracted_txt/``:

    pick the best source  (__ocr > __txt > __html > __pdf)
    measure  ->  findings_before
    fix      ->  hygiene + the fixers the findings call for (limits enforced)
    measure again ->  findings_after, and the verdict is taken on THIS text
    PASS       -> cleaned_txt/<stem>.txt   (marker removed)
    NEEDS_OCR  -> listed in the report with the file name to supply
    ELIMINATE  -> listed in the report with the rules that fired

A run re-evaluates everything, rewrites the report, and makes ``cleaned_txt/``
mirror the latest result. To retry a PDF that needs OCR, put the OCR text in
``extracted_txt/<stem>__ocr.txt`` and run again.
"""

import json
import logging
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from automation.src.input_process.config import InputProcessConfig
from automation.src.input_process.extract import (
    SOURCE_FORMATS,
    cleaned_stems,
    marked_name,
    split_marked,
)
from automation.src.input_process.fix import apply_fixes
from automation.src.input_process.quality.criteria import (
    all_findings,
    decide,
    fix_acceptable,
    fix_plan,
    resolve_thresholds,
)
from automation.src.input_process.quality.metrics import (
    corpus_unique_word_ratio,
    pdf_metrics,
    profile_for,
    text_metrics,
    word_set,
)

logger = logging.getLogger(__name__)

#: Which extracted file wins when a policy has several (first match). A supplied
#: .txt outranks text extracted from a PDF/HTML: it is the text the translations
#: and golden labels were built on.
PRECEDENCE = ("ocr", "txt", "html", "pdf")

#: The corpus-relative unique-word metric needs this many OTHER documents.
MIN_REFERENCE_DOCS = 10

PASS, NEEDS_OCR, ELIMINATE = "PASS", "NEEDS_OCR", "ELIMINATE"

#: Metrics kept in the report (the full set minus bulky or redundant fields).
REPORT_METRICS = (
    "n_letters", "prose_letters", "n_lines", "n_tokens", "expected_script_ratio",
    "garbage_char_ratio", "noisy_token_ratio", "spaced_letter_runs_per_1k", "glued_tokens_per_1k",
    "mojibake_per_1k_lines", "repeated_line_ratio", "chrome_per_1k_lines", "url_per_1k_lines",
    "html_residue_count", "paywall_marker_count", "n_unit_numbers", "numbering_disorder",
    "corpus_unique_word_ratio", "pdf_pages", "pdf_low_text_page_ratio", "letters_per_pdf_page",
)


@dataclass
class FileOutcome:
    stem: str
    source_format: str
    language: str
    extracted_file: str
    status: str
    reason: str = ""
    failed: list[dict] = field(default_factory=list)
    fixes_applied: list[str] = field(default_factory=list)
    fix_rejected: str = ""
    findings_before: dict = field(default_factory=dict)
    findings_after: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    review: str = ""
    duplicate_of: str = ""
    superseded: list[str] = field(default_factory=list)
    cleaned_file: str = ""
    # NEEDS_OCR only
    pdf: str = ""
    expected_ocr_file: str = ""


def group_extracted(extracted_dir: Path) -> tuple[dict[str, dict[str, Path]], list[str]]:
    """{stem: {fmt: path}} for files carrying a known marker, plus the names that carry none."""
    groups: dict[str, dict[str, Path]] = {}
    unmarked: list[str] = []
    if not extracted_dir.is_dir():
        return groups, unmarked
    for path in sorted(p for p in extracted_dir.iterdir() if p.is_file()):
        parsed = split_marked(path.name)
        if parsed is None:
            unmarked.append(path.name)
            continue
        stem, fmt = parsed
        groups.setdefault(stem, {})[fmt] = path
    return groups, unmarked


def best_source(formats: dict[str, Path]) -> tuple[str, Path]:
    for fmt in PRECEDENCE:
        if fmt in formats:
            return fmt, formats[fmt]
    raise ValueError("no known source format")  # unreachable: group_extracted only keeps known ones


def find_original_pdf(config: InputProcessConfig, stem: str) -> Path | None:
    for suffix in (".pdf", ".PDF"):
        candidate = config.raw_files_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    if config.raw_files_dir.is_dir():  # any other casing, e.g. .Pdf
        for path in config.raw_files_dir.iterdir():
            if path.stem == stem and SOURCE_FORMATS.get(path.suffix.lower()) == "pdf":
                return path
    return None


class PdfMetricsCache:
    """pdf_metrics is slow; cache it by file name, size and mtime."""

    def __init__(self, path: Path):
        self.path = path
        try:
            self.data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.data = {}

    def get(self, pdf: Path) -> dict:
        stat = pdf.stat()
        key = f"{pdf.name}|{stat.st_size}|{int(stat.st_mtime)}"
        if key not in self.data:
            try:
                self.data[key] = pdf_metrics(pdf)
            except Exception as exc:  # damaged PDF
                logger.warning("[%s] PDF metrics failed: %s", pdf.name, exc)
                self.data[key] = {"error": str(exc)}
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.data, indent=1, sort_keys=True), encoding="utf-8")
        return self.data[key]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _measure(text: str, profile, doc_freq, n_docs: int, pdf_info: dict | None) -> dict:
    """`doc_freq` / `n_docs` are those of the text's own language."""
    m = text_metrics(text, profile)
    m["corpus_unique_word_ratio"] = (
        corpus_unique_word_ratio(text, profile, doc_freq) if n_docs - 1 >= MIN_REFERENCE_DOCS else None
    )
    if pdf_info and "error" not in pdf_info:
        for key in ("pdf_pages", "pdf_median_chars_per_page", "pdf_low_text_page_ratio"):
            m[key] = pdf_info.get(key)
        pages = pdf_info.get("pdf_pages")
        m["letters_per_pdf_page"] = round(m["n_letters"] / pages, 1) if pages else None
    return m


@dataclass
class Analysis:
    """Everything the quality check found, in full detail (the report keeps only a slice)."""

    outcomes: dict[str, FileOutcome]   # policies checked in this run (everything still in extracted_txt)
    cleaned: dict[str, str]            # texts that passed in this run (to be written to cleaned_txt)
    unmarked: list[str]
    missing: list[str]
    done: dict[str, Path] = field(default_factory=dict)   # policies already in cleaned_txt
    sources: dict[str, list[Path]] = field(default_factory=dict)  # extracted files per checked policy
    leftovers: list[str] = field(default_factory=list)    # extracted files of policies already done (ignored)
    adopted: list[str] = field(default_factory=list)      # files moved by hand into cleaned_txt, renamed
    conflicts: list[str] = field(default_factory=list)    # ... that could not be renamed


def analyze(config: InputProcessConfig, *, small_scale: bool = False) -> Analysis:
    """Check, fix and re-check every text in ``extracted_txt/``. Writes nothing.

    Policies already in ``cleaned_txt/`` are done: their extracted files (if any are
    left) are ignored, and their texts only serve as the reference corpus.
    """
    thresholds = resolve_thresholds(config.thresholds)
    done = cleaned_stems(config.cleaned_txt_dir)
    groups, unmarked = group_extracted(config.extracted_txt_dir)
    leftovers = sorted(p.name for stem, formats in groups.items() if stem in done for p in formats.values())
    groups = {stem: formats for stem, formats in groups.items() if stem not in done}
    if small_scale:
        groups = {s: f for s, f in groups.items() if s in set(config.small_scale_stems)}

    chosen = {stem: best_source(formats) for stem, formats in groups.items()}
    texts = {stem: _read(path) for stem, (_fmt, path) in chosen.items()}

    # Each policy is judged as its own language (from its name, else the configured one).
    languages = {stem: config.language_for(stem) for stem in texts}
    done_languages = {stem: config.language_for(stem) for stem in done}
    profiles = {lang: profile_for(lang) for lang in set(languages.values()) | set(done_languages.values())}

    # Corpus statistics for the unique-word metric (OCR text only), per language:
    # a Spanish file is compared with the other Spanish files, not with Arabic ones.
    # The reference corpus is the texts being checked plus everything already in
    # cleaned_txt: once passed files leave extracted_txt, a lone OCR file would
    # otherwise be compared with nothing.
    doc_freq: dict[str, dict[str, int]] = {}
    n_docs: dict[str, int] = {}

    def add_to_corpus(lang: str, text: str) -> None:
        counts = doc_freq.setdefault(lang, {})
        for word in word_set(text, profiles[lang]):
            counts[word] = counts.get(word, 0) + 1
        n_docs[lang] = n_docs.get(lang, 0) + 1

    for stem, text in texts.items():
        add_to_corpus(languages[stem], text)
    for stem, path in done.items():
        add_to_corpus(done_languages[stem], _read(path))

    cache = PdfMetricsCache(config.report_path.with_name("pdf_metrics_cache.json"))
    outcomes: dict[str, FileOutcome] = {}
    cleaned: dict[str, str] = {}
    hashes: dict[str, str] = {}

    for stem in sorted(chosen):
        fmt, path = chosen[stem]
        text0 = texts[stem]
        lang = languages[stem]
        profile, freq, n_lang = profiles[lang], doc_freq[lang], n_docs[lang]
        pdf = find_original_pdf(config, stem) if fmt == "pdf" else None
        pdf_info = cache.get(pdf) if pdf else None

        m0 = _measure(text0, profile, freq, n_lang, pdf_info)
        before = all_findings({**m0, "source_format": fmt}, thresholds, ocr_derived=fmt == "ocr")

        text1, applied = apply_fixes(text0, fix_plan(before))
        m1 = m0 if text1 == text0 else _measure(text1, profile, freq, n_lang, pdf_info)
        rejected = ""
        ok, why = fix_acceptable(m0, m1, thresholds)
        if not ok:
            # Too destructive: keep only the lossless normalization.
            rejected = why
            text1, applied = apply_fixes(text0, [])
            m1 = m0 if text1 == text0 else _measure(text1, profile, freq, n_lang, pdf_info)

        verdict = decide(m1, fmt, thresholds)
        outcome = FileOutcome(
            stem=stem,
            source_format=fmt,
            language=lang,
            extracted_file=path.name,
            status=verdict.status,
            reason=verdict.reason,
            failed=verdict.failed,
            fixes_applied=applied,
            fix_rejected=rejected,
            findings_before=before,
            findings_after=verdict.findings,
            metrics={k: m1.get(k) for k in REPORT_METRICS if m1.get(k) is not None},
            review=verdict.review,
            superseded=sorted(p.name for f, p in groups[stem].items() if f != fmt),
        )
        if verdict.status == NEEDS_OCR:
            outcome.pdf = pdf.name if pdf else ""
            outcome.expected_ocr_file = marked_name(stem, "ocr")
        if verdict.status == PASS:
            cleaned[stem] = text1
            hashes[stem] = m1["content_hash"]
            outcome.cleaned_file = f"{stem}.txt"
        outcomes[stem] = outcome

    # Exact duplicates are flagged, never removed: dropping a policy would also
    # drop its row in the golden dataset.
    first_seen: dict[str, str] = {}
    for stem in sorted(cleaned):
        digest = hashes[stem]
        if digest in first_seen:
            outcomes[stem].duplicate_of = first_seen[digest]
            note = f"duplicate_of={first_seen[digest]}"
            outcomes[stem].review = f"{outcomes[stem].review};{note}".strip(";")
        else:
            first_seen[digest] = stem

    missing = _missing_extraction(config, groups, small_scale, done)
    return Analysis(
        outcomes, cleaned, unmarked, missing,
        done=done,
        sources={stem: list(formats.values()) for stem, formats in groups.items()},
        leftovers=leftovers,
    )


def adopt_moved(config: InputProcessConfig, *, dry_run: bool = False) -> tuple[list[str], list[str]]:
    """Rename files the user moved by hand from ``extracted_txt/`` to ``cleaned_txt/``.

    A moved file still carries its marker (``<stem>__txt.txt``); the policy name
    is ``<stem>.txt``. The text is left exactly as the user put it. A file is
    never renamed over an existing ``<stem>.txt``; it is reported as a conflict.
    Returns ``(adopted, conflicts)`` as file names.
    """
    adopted: list[str] = []
    conflicts: list[str] = []
    if not config.cleaned_txt_dir.is_dir():
        return adopted, conflicts
    for path in sorted(config.cleaned_txt_dir.glob("*.txt")):
        parsed = split_marked(path.name)
        if parsed is None:
            continue
        target = path.with_name(f"{parsed[0]}.txt")
        if target.exists():
            conflicts.append(path.name)
            continue
        if not dry_run:
            path.rename(target)
        adopted.append(f"{path.name} -> {target.name}")
    return adopted, conflicts


def tidy_extracted(config: InputProcessConfig, *, dry_run: bool = False) -> list[str]:
    """Remove extracted files of policies that are already in ``cleaned_txt/``.

    One-off for data produced before ``extracted_txt/`` was reduced to the
    unresolved files. Returns the names it removed (or would remove).
    """
    done = cleaned_stems(config.cleaned_txt_dir)
    groups, _ = group_extracted(config.extracted_txt_dir)
    removed = []
    for stem, formats in groups.items():
        if stem in done:
            for path in formats.values():
                removed.append(path.name)
                if not dry_run:
                    path.unlink()
    return sorted(removed)


def check_all(
    config: InputProcessConfig,
    *,
    write_cleaned: bool = True,
    small_scale: bool = False,
) -> dict:
    """Run the quality check over ``extracted_txt/``.

    Files that pass are written to ``cleaned_txt/`` and removed from
    ``extracted_txt/``, which keeps only what still needs OCR or was eliminated.
    ``cleaned_txt/`` is never pruned. With ``write_cleaned=False`` only the report is
    written: nothing is renamed, moved or deleted.
    """
    adopted, conflicts = adopt_moved(config, dry_run=not write_cleaned)
    result = analyze(config, small_scale=small_scale)
    result.adopted, result.conflicts = adopted, conflicts
    report = build_report(config, result)

    if write_cleaned:
        _write_cleaned(config.cleaned_txt_dir, result.cleaned)
        for stem in result.cleaned:  # only after its cleaned text is on disk
            for path in result.sources.get(stem, []):
                path.unlink(missing_ok=True)
    config.report_path.parent.mkdir(parents=True, exist_ok=True)
    config.report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def _missing_extraction(
    config: InputProcessConfig, groups: dict, small_scale: bool, done: dict
) -> list[str]:
    """Originals in raw_files/ with no extracted text yet (extraction failed or was not run)."""
    if not config.raw_files_dir.is_dir():
        return []
    missing = []
    for path in sorted(p for p in config.raw_files_dir.iterdir() if p.is_file()):
        fmt = SOURCE_FORMATS.get(path.suffix.lower())
        if fmt is None:
            continue
        if small_scale and path.stem not in set(config.small_scale_stems):
            continue
        if path.stem in done:
            continue
        if fmt not in groups.get(path.stem, {}) and "ocr" not in groups.get(path.stem, {}):
            missing.append(path.name)
    return missing


def _write_cleaned(cleaned_dir: Path, cleaned: dict[str, str]) -> None:
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    for stem, text in cleaned.items():  # only ever adds or replaces; cleaned_txt/ is not pruned
        (cleaned_dir / f"{stem}.txt").write_text(text, encoding="utf-8", newline="")


def build_report(config: InputProcessConfig, result: Analysis) -> dict:
    """The report: counts, then only the files that did not pass and the indicators they failed.

    Passed files are not listed. Fixes, review flags and metrics stay in the
    full `Analysis`. ``summary.passed`` is the number of policies in
    ``cleaned_txt/`` after this run (earlier runs and files moved there by hand
    included), so ``files`` is every policy: passed + needs_ocr + eliminated.
    `warnings` appears only when something needs attention.
    """
    outcomes = result.outcomes
    by_status = Counter(o.status for o in outcomes.values())
    in_cleaned = set(result.done) | set(result.cleaned)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "project": config.project,
        "summary": {
            "files": len(in_cleaned) + by_status[NEEDS_OCR] + by_status[ELIMINATE],
            "passed": len(in_cleaned),
            "needs_ocr": by_status[NEEDS_OCR],
            "eliminated": by_status[ELIMINATE],
        },
        "needs_ocr": [
            {"stem": o.stem, "pdf": o.pdf, "expected_ocr_file": o.expected_ocr_file, "failed": o.failed}
            for o in outcomes.values() if o.status == NEEDS_OCR
        ],
        "eliminated": [
            {"stem": o.stem, "source_format": o.source_format, "failed": o.failed}
            for o in outcomes.values() if o.status == ELIMINATE
        ],
    }
    warnings = {}
    if result.missing:
        warnings["not_extracted"] = result.missing
    if result.unmarked:
        warnings["unrecognized_files"] = result.unmarked
    if result.leftovers:
        warnings["already_in_cleaned_txt"] = result.leftovers  # ignored; delete them or move the cleaned file back
    if result.conflicts:
        warnings["cleaned_txt_conflicts"] = result.conflicts  # <stem>.txt already exists; not renamed
    if warnings:
        report["warnings"] = warnings
    if result.adopted:
        report["adopted"] = result.adopted  # files moved into cleaned_txt by hand, now renamed
    return report
