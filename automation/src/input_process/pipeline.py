"""Stage 1.1, step 3: quality-check, fix, re-check and sort every waiting text.

Where a policy stands is read from its file name (see ``extract``):

    <stem>.txt in cleaned_txt/    done: skipped (its text only serves as reference corpus)
    <stem>.txt in extracted_txt/  measured, fixed, measured again, and sorted:
        PASS       -> cleaned_txt/<stem>.txt, and removed from extracted_txt/
        NEEDS_OCR  -> stays in extracted_txt/; WARNING in the log with the file to replace
        ELIMINATE  -> stays in extracted_txt/; WARNING in the log with the rules that fired

To retry a file, replace ``extracted_txt/<stem>.txt`` (with OCR text or your own fix) and run again.
Everything that happens is appended to ``raw/input_process.log``.
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
    ExtractionManifest,
    cleaned_stems,
    extracted_files,
    original_format,
)
from automation.src.input_process.fix import apply_fixes
from automation.src.input_process.runlog import run_log
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
    cleaned_file: str = ""
    # NEEDS_OCR only
    pdf: str = ""
    expected_ocr_file: str = ""


def source_format(config: InputProcessConfig, manifest: ExtractionManifest, stem: str, path: Path) -> str:
    """How the text in `path` came to be, which decides the rules it is judged by.

    * our own untouched extraction  -> the format it was extracted from (``pdf`` / ``html`` / ``txt``)
    * anything you supplied         -> ``ocr`` when the policy's original is a PDF (OCR text is what you
                                       replace a scanned PDF's text with), else ``txt``
    Only a machine-extracted PDF text can be judged "needs OCR"; your text never is.
    """
    machine = manifest.machine_format(stem, path)
    if machine:
        return machine
    return "ocr" if original_format(config, stem) == "pdf" else "txt"


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

    outcomes: dict[str, FileOutcome]   # policies checked in this run (everything waiting in extracted_txt)
    cleaned: dict[str, str]            # texts that passed in this run (to be written to cleaned_txt)
    unrecognized: list[str]            # files in extracted_txt that are not .txt
    missing: list[str]
    done: dict[str, Path] = field(default_factory=dict)   # policies already in cleaned_txt
    sources: dict[str, Path] = field(default_factory=dict)  # the waiting file of each checked policy
    leftovers: list[str] = field(default_factory=list)    # waiting files of policies already done (ignored)


def analyze(config: InputProcessConfig, *, small_scale: bool = False) -> Analysis:
    """Check, fix and re-check every text in ``extracted_txt/``. Writes nothing.

    Policies already in ``cleaned_txt/`` are done: their extracted files (if any are
    left) are ignored, and their texts only serve as the reference corpus.
    """
    thresholds = resolve_thresholds(config.thresholds)
    done = cleaned_stems(config.cleaned_txt_dir)
    waiting, unrecognized = extracted_files(config.extracted_txt_dir)
    leftovers = sorted(path.name for stem, path in waiting.items() if stem in done)
    waiting = {stem: path for stem, path in waiting.items() if stem not in done}
    if small_scale:
        waiting = {s: f for s, f in waiting.items() if s in set(config.small_scale_stems)}

    manifest = ExtractionManifest(config.manifest_path)
    chosen = {stem: (source_format(config, manifest, stem, path), path) for stem, path in waiting.items()}
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

    cache = PdfMetricsCache(config.log_path.with_name("pdf_metrics_cache.json"))
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
        )
        if verdict.status == NEEDS_OCR:
            outcome.pdf = pdf.name if pdf else ""
            outcome.expected_ocr_file = f"{stem}.txt"
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

    missing = _missing_extraction(config, set(waiting), small_scale, done)
    return Analysis(
        outcomes, cleaned, unrecognized, missing,
        done=done,
        sources=dict(waiting),
        leftovers=leftovers,
    )


def tidy_extracted(config: InputProcessConfig, *, dry_run: bool = False) -> list[str]:
    """Remove ``extracted_txt/<stem>.txt`` of policies that are already in ``cleaned_txt/``.

    Such a file is ignored by the quality check; this deletes it on request. Returns the names it
    removed (or would remove).
    """
    done = cleaned_stems(config.cleaned_txt_dir)
    waiting, _ = extracted_files(config.extracted_txt_dir)
    removed = []
    for stem, path in waiting.items():
        if stem in done:
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
    """Run the quality check over ``extracted_txt/`` and log the result.

    Files that pass are written to ``cleaned_txt/`` and removed from ``extracted_txt/``, which keeps only what
    still needs OCR or was eliminated. ``cleaned_txt/`` is never pruned. With ``write_cleaned=False`` only the
    log is written: nothing is moved or deleted. Returns the summary (counts and the failed files).
    """
    result = analyze(config, small_scale=small_scale)
    report = build_report(config, result)

    if write_cleaned:
        _write_cleaned(config.cleaned_txt_dir, result.cleaned)
        manifest = ExtractionManifest(config.manifest_path)
        for stem in result.cleaned:  # only after its cleaned text is on disk
            result.sources[stem].unlink(missing_ok=True)
            manifest.forget(stem)
        manifest.save()
    with run_log(config.log_path) as log:
        log_check(log, config, result, report, wrote=write_cleaned)
    return report


def log_check(log, config: InputProcessConfig, result: Analysis, report: dict, *, wrote: bool) -> None:
    """One log line per checked file, a WARNING for every file that failed, then the totals."""
    s = report["summary"]
    log.info("QUALITY CHECK started (project=%s%s)", config.project, "" if wrote else ", log only: nothing is moved")
    for stem in sorted(result.outcomes):
        o = result.outcomes[stem]
        failed = ", ".join(f"{f['rule']}={f['value']} (limit {f['limit']})" for f in o.failed)
        if o.status == PASS:
            fixes = f" [fixes: {', '.join(o.fixes_applied)}]" if o.fixes_applied else ""
            note = f" [{o.review}]" if o.review else ""
            log.info("PASS %s (%s) -> cleaned_txt/%s.txt%s%s", stem, o.source_format, stem, fixes, note)
        elif o.status == NEEDS_OCR:
            log.warning(
                "NEEDS_OCR %s: %s - OCR %s and replace extracted_txt/%s.txt with the text",
                stem, failed or o.reason, o.pdf or "the PDF", stem,
            )
        else:
            log.warning(
                "ELIMINATED %s (%s): %s - fix extracted_txt/%s.txt (it is checked again) or move it to cleaned_txt/",
                stem, o.source_format, failed or o.reason, stem,
            )
    for name in result.missing:
        log.warning("NOT EXTRACTED %s: no extracted text yet (extraction failed or was not run)", name)
    for name in result.leftovers:
        log.warning("IGNORED extracted_txt/%s: the policy is already in cleaned_txt (delete the file or run `tidy`)", name)
    for name in result.unrecognized:
        log.warning("IGNORED extracted_txt/%s: not a .txt file", name)
    log.info(
        "QUALITY CHECK finished: %d policies, %d in cleaned_txt, %d need OCR, %d eliminated (%d were already done)",
        s["files"], s["passed"], s["needs_ocr"], s["eliminated"], len(result.done),
    )


def _missing_extraction(config: InputProcessConfig, waiting: set, small_scale: bool, done: dict) -> list[str]:
    """Originals in raw_files/ whose policy has no text yet (extraction failed or was not run)."""
    if not config.raw_files_dir.is_dir():
        return []
    missing = []
    for path in sorted(p for p in config.raw_files_dir.iterdir() if p.is_file()):
        if SOURCE_FORMATS.get(path.suffix.lower()) is None:
            continue
        if small_scale and path.stem not in set(config.small_scale_stems):
            continue
        if path.stem not in done and path.stem not in waiting:
            missing.append(path.name)
    return missing


def _write_cleaned(cleaned_dir: Path, cleaned: dict[str, str]) -> None:
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    for stem, text in cleaned.items():  # only ever adds or replaces; cleaned_txt/ is not pruned
        (cleaned_dir / f"{stem}.txt").write_text(text, encoding="utf-8", newline="")


def build_report(config: InputProcessConfig, result: Analysis) -> dict:
    """The summary of a run: counts, then only the files that did not pass and the indicators they failed.

    ``summary.passed`` is the number of policies in ``cleaned_txt/`` after this run (earlier runs included),
    so ``files`` is every policy: passed + needs_ocr + eliminated.
    """
    outcomes = result.outcomes
    by_status = Counter(o.status for o in outcomes.values())
    in_cleaned = set(result.done) | set(result.cleaned)
    return {
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
        "warnings": {
            name: items
            for name, items in (
                ("not_extracted", result.missing),
                ("unrecognized_files", result.unrecognized),
                ("already_in_cleaned_txt", result.leftovers),
            )
            if items
        },
    }
