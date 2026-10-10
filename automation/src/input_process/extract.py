"""Stage 1.1, step 2: turn the originals in ``raw_files/`` into text.

PDF  -> pdfplumber, page by page.
HTML -> trafilatura, with a BeautifulSoup fallback when trafilatura drops
        headings or most of the text (the approach of the GlobalLaws
        ``extract_html.py``, with Indonesian headings).
TXT  -> copied as UTF-8.

Each result is written to ``extracted_txt/`` as ``<stem>__<fmt>.txt`` where
``fmt`` is the source format (``pdf``, ``html``, ``txt``). Text the user gets
from OCR is dropped in the same folder as ``<stem>__ocr.txt``. The marker lets
the quality check treat each source differently (only a PDF can need OCR) and
is removed again when a file passes into ``cleaned_txt/``.

Extraction only extracts: it never judges quality. A scanned PDF that yields
no text is written as an empty file, and the quality check turns that into
"needs OCR".
"""

import html
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from automation.src.input_process.config import InputProcessConfig

logger = logging.getLogger(__name__)

# pdfminer / pdfplumber warn once per odd font or metadata key; hundreds of lines per run.
for _noisy in ("pdfminer", "pdfplumber"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

#: Extension (lower case) -> source format.
SOURCE_FORMATS = {".pdf": "pdf", ".html": "html", ".htm": "html", ".txt": "txt"}

#: Formats a file in ``extracted_txt/`` can carry. ``ocr`` is user-supplied.
EXTRACTED_FORMATS = ("pdf", "html", "txt", "ocr")

MARKER_SEP = "__"

# Statute headings (Indonesian and English), used only as a red-flag detector
# for HTML extraction: trafilatura must not lose them.
HEADING_PAT = re.compile(
    r"\b(?:Pasal|Bab|Bagian|Paragraf|Ayat|Article|Section|Art)\.?\s*(?:\d{1,4}|[IVXLC]{1,6})\b|§",
    re.IGNORECASE,
)


def marked_name(stem: str, fmt: str) -> str:
    return f"{stem}{MARKER_SEP}{fmt}.txt"


def split_marked(filename: str) -> tuple[str, str] | None:
    """``ACEH__pdf.txt`` -> ``("ACEH", "pdf")``; None when it carries no known marker."""
    name = Path(filename).name
    if not name.lower().endswith(".txt"):
        return None
    base = name[: -len(".txt")]
    stem, sep, fmt = base.rpartition(MARKER_SEP)
    if not sep or not stem or fmt not in EXTRACTED_FORMATS:
        return None
    return stem, fmt


@dataclass
class ExtractionResult:
    source: str
    fmt: str | None
    output: str | None
    status: str  # extracted | skipped_existing | skipped_done | failed | unsupported
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


# --------------------------------------------------------------------------- driver
def cleaned_stems(cleaned_dir: Path) -> dict[str, Path]:
    """Policies that are done: ``{stem: path}`` for every text in ``cleaned_txt/``.

    A plain ``<stem>.txt`` is what the pipeline writes. A file still carrying its
    marker (``<stem>__txt.txt``) is one the user moved over by hand from
    ``extracted_txt/``; it counts as done too and is renamed on the next check.
    """
    done: dict[str, Path] = {}
    if not cleaned_dir.is_dir():
        return done
    for path in sorted(p for p in cleaned_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"):
        parsed = split_marked(path.name)
        stem = parsed[0] if parsed else path.name[: -len(path.suffix)]
        done.setdefault(stem, path)  # sorted: a plain "<stem>.txt" comes before "<stem>__fmt.txt"
    return done


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


def extract_all(config: InputProcessConfig, *, small_scale: bool = False) -> list[ExtractionResult]:
    results: list[ExtractionResult] = []
    done = cleaned_stems(config.cleaned_txt_dir)
    for path in list_sources(config, small_scale=small_scale):
        fmt = SOURCE_FORMATS.get(path.suffix.lower())
        if fmt is None:
            results.append(ExtractionResult(path.name, None, None, "unsupported", path.suffix or "no extension"))
            continue

        if path.stem in done:  # already in cleaned_txt: nothing left to extract
            results.append(ExtractionResult(path.name, fmt, None, "skipped_done"))
            continue

        target = config.extracted_txt_dir / marked_name(path.stem, fmt)
        if target.exists():
            results.append(ExtractionResult(path.name, fmt, target.name, "skipped_existing"))
            continue
        try:
            text, detail = READERS[fmt](path)
        except Exception as exc:  # corrupt / encrypted file: report it, keep going
            logger.error("[%s] extraction failed: %s", path.name, exc)
            results.append(ExtractionResult(path.name, fmt, None, "failed", f"{type(exc).__name__}: {exc}"))
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text.rstrip("\n") + "\n" if text.strip() else "", encoding="utf-8", newline="")
        results.append(ExtractionResult(path.name, fmt, target.name, "extracted", detail))
    return results
