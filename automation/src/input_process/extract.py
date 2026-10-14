"""Stage 1.1, step 2: turn the originals in ``raw_files/`` into text.

PDF  -> pdfplumber, page by page.
HTML -> trafilatura, with a BeautifulSoup fallback when trafilatura drops
        headings or most of the text (the approach of the GlobalLaws
        ``extract_html.py``, with Indonesian headings).
TXT  -> copied as UTF-8.

Every result is ``extracted_txt/<stem>.txt``: the file name is the policy name and nothing else. Where a
policy stands is read from which folder holds its name:

    <stem>.txt in cleaned_txt/    done: nothing is extracted or checked again
    <stem>.txt in extracted_txt/  waiting: it is quality-checked (again after you replace it)
    neither                       extract the original, then quality-check it

A policy with several originals (the same name as .txt, .html and .pdf) is extracted from one of them:
``txt`` first, then ``html``, then ``pdf``. Text you get from OCR, or any fix you make, replaces
``extracted_txt/<stem>.txt``. To tell your file from our extraction, the hash of every file we write is
kept in ``raw/extraction_manifest.json``; a file that no longer matches was supplied by you.

Extraction only extracts: it never judges quality. A scanned PDF that yields no text is written as an empty
file, and the quality check turns that into "needs OCR".
"""

import hashlib
import html
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from automation.src.input_process.config import InputProcessConfig
from automation.src.input_process.runlog import run_log

logger = logging.getLogger(__name__)

# pdfminer / pdfplumber warn once per odd font or metadata key; hundreds of lines per run.
for _noisy in ("pdfminer", "pdfplumber"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

#: Extension (lower case) -> source format.
SOURCE_FORMATS = {".pdf": "pdf", ".html": "html", ".htm": "html", ".txt": "txt"}

#: Which original is extracted when a policy has several (first match).
ORIGINAL_PRECEDENCE = ("txt", "html", "pdf")

# Statute headings (Indonesian and English), used only as a red-flag detector
# for HTML extraction: trafilatura must not lose them.
HEADING_PAT = re.compile(
    r"\b(?:Pasal|Bab|Bagian|Paragraf|Ayat|Article|Section|Art)\.?\s*(?:\d{1,4}|[IVXLC]{1,6})\b|§",
    re.IGNORECASE,
)


@dataclass
class ExtractionResult:
    source: str
    fmt: str | None
    output: str | None
    status: str  # extracted | skipped_existing | skipped_done | skipped_duplicate | failed | unsupported
    detail: str = ""


# --------------------------------------------------------------------------- readers
def decode_text(raw: bytes) -> tuple[str, str]:
    """Decode a text file: UTF-8 (BOM allowed), then cp1252, then lossy UTF-8."""
    for encoding in ("utf-8-sig", "cp1252"):
        try:
            return raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace"), "utf-8(replace)"


def extract_txt(path: Path) -> tuple[str, str]:
    text, encoding = decode_text(path.read_bytes())
    return text, f"decoded as {encoding}"


def extract_pdf(path: Path) -> tuple[str, str]:
    import pdfplumber  # imported here: only PDF extraction needs it

    pages: list[str] = []
    with pdfplumber.open(path) as pdf:
        total = len(pdf.pages)
        for page in pdf.pages:
            try:
                text = page.extract_text() or ""
            except Exception as exc:  # one bad page must not lose the others
                logger.warning("[%s] page %s: %s", path.name, page.page_number, exc)
                text = ""
            pages.append(text.strip())
            page.close()  # release the parsed layout; large PDFs otherwise exhaust memory
    # One newline between pages: a sentence can run across a page break.
    return "\n".join(p for p in pages if p), f"{total} page(s)"


def safe_bs4_extract(markup: str) -> str:
    """Complete but noisy extraction: drops only script/style/noscript/comments."""
    from bs4 import BeautifulSoup, Comment

    soup = BeautifulSoup(markup, "html.parser")
    for tag in soup.find_all(["script", "style", "noscript"]):
        tag.decompose()
    for comment in soup.find_all(string=lambda s: isinstance(s, Comment)):
        comment.extract()
    lines = [line.strip() for line in soup.get_text("\n", strip=True).splitlines() if line.strip()]
    out: list[str] = []
    for line in lines:
        if out and line == out[-1]:  # only back-to-back exact duplicates
            continue
        out.append(line)
    return "\n".join(out)


def traf_extract(markup: str) -> str:
    try:
        import trafilatura

        return (
            trafilatura.extract(
                markup,
                favor_recall=True,
                include_tables=True,
                output_format="txt",
                include_comments=False,
            )
            or ""
        )
    except Exception as exc:
        logger.warning("trafilatura failed: %s", exc)
        return ""


def heading_set(text: str) -> set[str]:
    return {m.group(0).lower().replace(" ", "").replace(".", "") for m in HEADING_PAT.finditer(text)}


def clean_basic(text: str) -> str:
    """Decode entities and cap blank-line runs at one. No frequency-based
    line filtering: that is how real statute lines get lost."""
    out: list[str] = []
    blank = 0
    for line in html.unescape(text).split("\n"):
        if not line.strip():
            blank += 1
            if blank > 1:
                continue
        else:
            blank = 0
        out.append(line)
    return "\n".join(out).strip()


def decode_html(raw: bytes) -> str:
    from bs4 import UnicodeDammit

    return UnicodeDammit(raw).unicode_markup or raw.decode("utf-8", errors="replace")


def extract_html(path: Path) -> tuple[str, str]:
    markup = decode_html(path.read_bytes())
    bs4_out = safe_bs4_extract(markup)
    traf_out = traf_extract(markup)

    bs4_h, traf_h = heading_set(bs4_out), heading_set(traf_out)
    # Prefer trafilatura (less page furniture) only when it keeps at least 90 %
    # of the headings the complete bs4 extraction found and is not drastically
    # shorter -- both are proxies for silently dropped sections.
    heading_ok = len(bs4_h) < 5 or len(traf_h) >= 0.9 * len(bs4_h)
    length_ok = not bs4_out or len(traf_out) >= 0.5 * len(bs4_out)
    if traf_out and heading_ok and length_ok:
        return clean_basic(traf_out), "trafilatura"
    return clean_basic(bs4_out), "bs4"


READERS = {"pdf": extract_pdf, "html": extract_html, "txt": extract_txt}


# --------------------------------------------------------------------------- manifest
def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ExtractionManifest:
    """Which ``extracted_txt/<stem>.txt`` files are our own extraction: ``{stem: {"format", "sha256"}}``.

    A file whose hash still matches was machine-extracted from that format, so (for a PDF) it can be judged
    "needs OCR". A file that no longer matches, or has no entry, was supplied by the user.
    """

    def __init__(self, path: Path):
        self.path = path
        try:
            self.data: dict = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.data = {}

    def record(self, stem: str, fmt: str, written: Path) -> None:
        self.data[stem] = {"format": fmt, "sha256": file_hash(written)}

    def machine_format(self, stem: str, file: Path) -> str | None:
        """The format `file` was extracted from, or None when it is not our untouched extraction."""
        entry = self.data.get(stem)
        return entry["format"] if entry and entry.get("sha256") == file_hash(file) else None

    def forget(self, stem: str) -> None:
        self.data.pop(stem, None)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=1, sort_keys=True), encoding="utf-8")


# --------------------------------------------------------------------------- driver
def cleaned_stems(cleaned_dir: Path) -> dict[str, Path]:
    """Policies that are done: ``{stem: path}`` for every ``<stem>.txt`` in ``cleaned_txt/``."""
    if not cleaned_dir.is_dir():
        return {}
    return {p.stem: p for p in sorted(cleaned_dir.iterdir()) if p.is_file() and p.suffix.lower() == ".txt"}


def extracted_files(extracted_dir: Path) -> tuple[dict[str, Path], list[str]]:
    """``({stem: path}`` of ``extracted_txt/<stem>.txt``, names of files that are not .txt)``."""
    files: dict[str, Path] = {}
    other: list[str] = []
    if not extracted_dir.is_dir():
        return files, other
    for path in sorted(p for p in extracted_dir.iterdir() if p.is_file()):
        if path.suffix.lower() == ".txt":
            files[path.stem] = path
        else:
            other.append(path.name)
    return files, other


def list_sources(config: InputProcessConfig, *, small_scale: bool = False) -> list[Path]:
    if not config.raw_files_dir.is_dir():
        raise FileNotFoundError(
            f"Original files folder not found: {config.raw_files_dir} "
            "(move existing files in with `migrate-raw`)"
        )
    files = sorted(p for p in config.raw_files_dir.iterdir() if p.is_file())
    if small_scale:
        wanted = set(config.small_scale_stems)
        files = [p for p in files if p.stem in wanted]
    return files


def choose_originals(files: list[Path]) -> tuple[dict[str, tuple[str, Path]], list[Path]]:
    """``({stem: (format, path)}`` -- the one original each policy is extracted from --, unsupported files)``."""
    by_stem: dict[str, dict[str, Path]] = {}
    unsupported: list[Path] = []
    for path in files:
        fmt = SOURCE_FORMATS.get(path.suffix.lower())
        if fmt is None:
            unsupported.append(path)
        else:
            by_stem.setdefault(path.stem, {}).setdefault(fmt, path)
    chosen = {}
    for stem, formats in by_stem.items():
        fmt = next(f for f in ORIGINAL_PRECEDENCE if f in formats)
        chosen[stem] = (fmt, formats[fmt])
    return chosen, unsupported


def original_format(config: InputProcessConfig, stem: str) -> str | None:
    """Format of the original `extract_all` would use for `stem` (None when there is none)."""
    if not config.raw_files_dir.is_dir():
        return None
    originals, _ = choose_originals([p for p in config.raw_files_dir.iterdir() if p.is_file() and p.stem == stem])
    return originals[stem][0] if stem in originals else None


def extract_all(config: InputProcessConfig, *, small_scale: bool = False) -> list[ExtractionResult]:
    """Extract every policy that is in neither ``cleaned_txt/`` nor ``extracted_txt/``."""
    results: list[ExtractionResult] = []
    done = cleaned_stems(config.cleaned_txt_dir)
    waiting, _ = extracted_files(config.extracted_txt_dir)
    sources = list_sources(config, small_scale=small_scale)
    chosen, unsupported = choose_originals(sources)
    for path in unsupported:
        results.append(ExtractionResult(path.name, None, None, "unsupported", path.suffix or "no extension"))

    manifest = ExtractionManifest(config.manifest_path)
    for path in sources:
        fmt = SOURCE_FORMATS.get(path.suffix.lower())
        if fmt is None:
            continue
        stem = path.stem
        if chosen[stem][1] != path:
            results.append(ExtractionResult(path.name, fmt, None, "skipped_duplicate", f"{chosen[stem][1].name} is used"))
            continue
        if stem in done:  # already in cleaned_txt: nothing left to do
            results.append(ExtractionResult(path.name, fmt, None, "skipped_done"))
            continue
        if stem in waiting:  # already in extracted_txt: it is checked as it is (maybe you replaced it)
            results.append(ExtractionResult(path.name, fmt, waiting[stem].name, "skipped_existing"))
            continue
        try:
            text, detail = READERS[fmt](path)
        except Exception as exc:  # corrupt / encrypted file: report it, keep going
            logger.error("[%s] extraction failed: %s", path.name, exc)
            results.append(ExtractionResult(path.name, fmt, None, "failed", f"{type(exc).__name__}: {exc}"))
            continue

        target = config.extracted_txt_dir / f"{stem}.txt"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text.rstrip("\n") + "\n" if text.strip() else "", encoding="utf-8", newline="")
        manifest.record(stem, fmt, target)
        results.append(ExtractionResult(path.name, fmt, target.name, "extracted", detail))
    manifest.save()
    log_extraction(config, results)
    return results


def log_extraction(config: InputProcessConfig, results: list[ExtractionResult]) -> None:
    """Append one line per extracted or failed original to the stage 1 log; skipped ones are only counted."""
    with run_log(config.log_path) as log:
        skipped = {"skipped_done": 0, "skipped_existing": 0, "skipped_duplicate": 0}
        for r in results:
            if r.status == "extracted":
                log.info("EXTRACTED %s (%s, %s) -> extracted_txt/%s", r.source, r.fmt, r.detail, r.output)
            elif r.status == "failed":
                log.warning("EXTRACTION FAILED %s: %s", r.source, r.detail)
            elif r.status == "unsupported":
                log.warning("IGNORED %s in raw_files: unsupported file type (%s)", r.source, r.detail)
            else:
                skipped[r.status] += 1
        log.info(
            "EXTRACTION finished: %d extracted, %d already in cleaned_txt, %d already in extracted_txt, "
            "%d with another original used, %d failed",
            sum(r.status == "extracted" for r in results),
            skipped["skipped_done"], skipped["skipped_existing"], skipped["skipped_duplicate"],
            sum(r.status == "failed" for r in results),
        )
