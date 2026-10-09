"""Quantitative text-quality metrics for stage 1.

Ported from the GlobalLaws ``quality_metrics.py`` and adapted for one language
per run (``input_process.language``, Indonesian by default): the filename-based
language lookup and the 50-language tables are gone, and the statute-heading and
web-chrome vocabularies know Indonesian (*Pasal*, *kebijakan privasi*, ...).

This module only MEASURES. Thresholds and decisions live in ``criteria.py``.
Every ratio is in [0, 1]; every ``*_per_1k`` rate is per 1,000 tokens (or lines,
where stated). ``None`` means "not applicable" and never fires a rule.
"""

import functools
import hashlib
import re
import statistics
import unicodedata
from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------- language
@dataclass(frozen=True)
class LanguageProfile:
    name: str
    #: Unicode scripts the letters are expected to be written in.
    scripts: frozenset = frozenset({"LATIN"})
    #: Words are separated by spaces (token-based metrics are meaningful).
    spaced: bool = True
    #: Normal orthography uses diacritics (enables the lost-diacritics check).
    diacritics: bool = False


#: Indonesian and Malay are written in plain Latin script without diacritics.
#: Add an entry here to support another language; unknown names fall back to
#: the default Latin / spaced profile.
PROFILES = {
    "Indonesian": LanguageProfile("Indonesian"),
    "Malay": LanguageProfile("Malay"),
    "English": LanguageProfile("English"),
}


def profile_for(language: str) -> LanguageProfile:
    return PROFILES.get(language, LanguageProfile(language or "unknown"))


# --------------------------------------------------------------------------- patterns
# Symbols (Unicode category So/Sk/Sm) that legitimately occur in statutes.
ALLOWED_SYMBOLS = set("°©®™№§¶+=<>|~^`´¨¸¯˚×÷±∙•·–—‐‑←→↑↓✓✔☐☑□■▪►▶◆◊○●√∞≤≥≠≈%$€£¥¢")

TOKEN_RE = re.compile(r"\S+")
EDGE_PUNCT = "\"'“”‘’«»„‚()[]{}<>.,;:!?¿¡*_-–—/\\|•·…"

# Heading-SHAPED lines (short, at most a short title after the number) used for
# the numbering-order check, so a cross-reference that happens to start a line
# ("Pasal 5 ayat (1) huruf f, meliputi:") is ignored. Roman-numeral Bab headings
# are left out on purpose: the check reads Arabic numbers.
UNIT_HEADING_RE = re.compile(
    r"^\s*(?:pasal|art(?:icle)?\.?|section|sec\.|§)\s*(\d{1,4})\s*(?:[.:\-–—)]|º|°)?\s*(.*)$",
    re.IGNORECASE,
)
# What follows a "heading" when it is really a wrapped cross-reference.
XREF_START_RE = re.compile(
    r"^(?:of|and|to|in|ayat|huruf|angka|butir|dan|atau|sampai|s\.d|dengan|tentang|"
    r"yang|ini|itu|pada|dalam|di|sebagaimana|dimaksud|jo|juncto|,)(?:\b|$)",
    re.IGNORECASE,
)
DECIMAL_NUM_RE = re.compile(r"^\s*(\d{1,3})\.(\d{1,3})\.?\s+[^\W\d_]")
BARE_NUM_RE = re.compile(r"^\s*(\d{1,4})\.\s+[^\W\d_]")
MAX_HEADING_LINE = 90
MAX_TITLE_WORDS = 10

PAYWALL_RE = re.compile(
    r"sign up or sign in|pro membership|full text is available|active subscription"
    r"|subscribe to (?:read|view|access)|login to (?:read|view)"
    r"|berlangganan untuk (?:membaca|melihat|mengakses)|khusus (?:pelanggan|member|anggota)"
    r"|konten berbayar|login untuk (?:membaca|melihat|mengunduh)",
    re.IGNORECASE,
)
PLACEHOLDER_LINE_RE = re.compile(r"^\s*(?:\.{3,}|…+)\s*$")

# Web / print chrome. Only unambiguous phrases: ordinary Indonesian words such
# as "masuk" (enter) and "daftar" (list/register) occur in real statutes.
CHROME_RE = re.compile(
    r"\b(?:cookies?|log ?in|sign ?in|sign ?up|subscribe|newsletter|privacy policy"
    r"|terms of use|all rights reserved|javascript|browser|click here|share on"
    r"|facebook|twitter|linkedin|whatsapp|home ?page|back to top|skip to"
    r"|kebijakan privasi|syarat (?:dan|&) ketentuan|hubungi kami|tentang kami"
    r"|peta situs|kembali ke atas|halaman utama|beranda|bagikan (?:ke|di|melalui)"
    r"|unduh (?:dokumen|file|pdf)|cetak halaman|hak cipta dilindungi)\b",
    re.IGNORECASE,
)
URL_RE = re.compile(r"https?://\S+|www\.\S+|\S+@\S+\.\w{2,}")
HTML_TAGS = ("a|abbr|b|body|br|center|div|em|font|h[1-6]|head|hr|html|i|img|li|link|meta|nav|ol|p|"
             "script|section|span|strong|style|sub|sup|table|tbody|td|th|thead|title|tr|u|ul")
HTML_RESIDUE_RE = re.compile(
    r"</?(?:" + HTML_TAGS + r")(?:\s[^<>]{0,120})?/?>|<[a-z]{2,15}>|&[a-zA-Z]{2,8};|&#\d{2,6};",
    re.IGNORECASE,
)
# UTF-8 bytes decoded as cp1252/latin-1 ("Ã©", "â€™", ...).
MOJIBAKE_RE = re.compile(
    r"Ã[\u0080-¿]|â€[\u0080-¿‘-„†-™]|Â[ -¿]"
    r"|Ð[\u0080-¿]|Ñ[\u0080-\u008F]"
)

LEGIT_ALNUM_RE = re.compile(
    r"^(?:\d+[^\W\d_]{1,4}|[^\W\d_]{1,3}\d+[^\W\d_]?|\d+[^\W\d_]\d+|[A-Z]\d{1,4}[^\W\d_]+)$"
)
INWORD_OK = "'’‘‛‟ʼ`-‐‑./;:,!?)("
DETACHED_DIACRITIC_RE = re.compile(r"[^\W\d_][´¨˜ˆ¸˚˝˘ˇ˛]")

PROSE_LINE_MIN_CHARS = 40
UNIQUE_WORD_RE = re.compile(r"[^\W\d_]{4,}")


# --------------------------------------------------------------------------- characters
@functools.lru_cache(maxsize=None)
def _script_of(ch: str) -> str:
    try:
        name = unicodedata.name(ch)
    except ValueError:
        return "UNKNOWN"
    for key in ("LATIN", "CYRILLIC", "GREEK", "ARABIC", "HEBREW", "GEORGIAN",
                "ARMENIAN", "THAI", "TIBETAN", "HANGUL", "HIRAGANA", "KATAKANA"):
        if name.startswith(key) or f" {key} " in f" {name} ":
            return key
    if "CJK" in name or "IDEOGRAPH" in name:
        return "CJK"
    return "OTHER"


def _is_symbol_font_char(ch: str) -> bool:
    """Private-use U+F000-U+F0FF: Symbol/Wingdings glyphs PDF extractors emit
    verbatim (bullets, dashes). Cosmetic, not corruption."""
    return 0xF000 <= ord(ch) <= 0xF0FF


@functools.lru_cache(maxsize=None)
def _is_garbage_char(ch: str) -> bool:
    if ch in "\n\t\r ":
        return False
    if ch == "�":
        return True
    if _is_symbol_font_char(ch):
        return False
    cat = unicodedata.category(ch)
    if cat in ("Co", "Cn", "Cs", "Cc"):
        return True
    return cat in ("So", "Sk") and ch not in ALLOWED_SYMBOLS


def _strip_token(tok: str) -> str:
    return tok.strip(EDGE_PUNCT)


def _is_noisy_token(core: str) -> bool:
    """True if a token shows a typical OCR / decoding corruption pattern."""
    if not any(c.isalpha() for c in core) or len(core) < 2:
        return False
    core = "".join(c for c in core if not unicodedata.category(c).startswith("M"))
    core = re.sub(r"[´¨˜ˆ¸˚˝˘ˇ˛]", "", core)
    # (a) digits inside a word ("Pur5uant", "Pemer1ntah")
    if any(c.isdigit() for c in core) and not LEGIT_ALNUM_RE.match(core):
        if re.search(r"[^\W\d_]\d+[^\W\d_]", core):
            return True
    # (b) symbol wedged between letters ("ARRAN€iEMENT")
    for a, b, c in zip(core, core[1:], core[2:]):
        if a.isalpha() and c.isalpha() and not b.isalnum() and not b.isspace() and b not in INWORD_OK:
            return True
    # (c) two scripts inside one word (Latin/Cyrillic confusion)
    scripts = {_script_of(c) for c in core if c.isalpha()}
    scripts.discard("OTHER")
    if len(scripts) > 1:
        return True
    # (d) irregular case: lower->upper flips inside a word that is not all-caps
    flips = sum(1 for a, b in zip(core, core[1:]) if a.islower() and b.isupper())
    if flips >= 1 and not core.isupper():
        if re.search(r"[a-zà-ÿ][A-ZÀ-Þ]", core):
            if flips + sum(1 for a, b in zip(core, core[1:]) if a.isupper() and b.islower()) >= 3:
                return True
    return False


# --------------------------------------------------------------------------- numbering
def unit_numbers(lines: list[str]) -> list[int]:
    """Sequence of statutory unit numbers taken from heading-shaped lines.

    Three numbering styles are collected separately (worded "Pasal 12", decimal
    "16.20.", bare "12.") and the style with the most hits is used.
    """
    worded, decimal, bare = [], [], []
    for line in lines:
        line = re.sub(r"^[\s|]+|[\s|]+$", "", line)
        if len(line) > MAX_HEADING_LINE:
            continue
        m = UNIT_HEADING_RE.match(line)
        first = line.lstrip()[:1]
        if (m and (first == "§" or first.isupper())
                and len(m.group(2).split()) <= MAX_TITLE_WORDS
                and not XREF_START_RE.match(m.group(2).strip())):
            worded.append(int(m.group(1)))
            continue
        m = DECIMAL_NUM_RE.match(line)
        if m:
            decimal.append(int(m.group(1)) * 1000 + int(m.group(2)))
            continue
        m = BARE_NUM_RE.match(line)
        if m:
            bare.append(int(m.group(1)))
    return max((worded, decimal, bare), key=len)


def numbering_disorder(nums: list[int]) -> float:
    """Share of consecutive unit-number steps that go the 'wrong way'.

    A table of contents followed by the body, an annex, or a second bundled act
    each cause ONE legitimate reset to a low number, so a step only counts when
    it goes backwards AND does not land on a restart (target <= 2).
    """
    if len(nums) < 2:
        return 0.0
    bad = sum(1 for a, b in zip(nums, nums[1:]) if b < a and b % 1000 > 2)
    return bad / (len(nums) - 1)


# --------------------------------------------------------------------------- text metrics
def text_metrics(text: str, profile: LanguageProfile) -> dict:
    """All per-document text metrics (see the module docstring)."""
    lines = [l for l in text.split("\n") if l.strip()]
    n_lines = len(lines)
    tokens = TOKEN_RE.findall(text)
    n_tokens = len(tokens) or 1
    non_ws = [c for c in text if not c.isspace()]
    n_chars = len(non_ws)
    letters = [c for c in non_ws if c.isalpha()]
    n_letters = len(letters)

    # Letters on "prose" lines (>= 40 chars): a page of navigation chrome has
    # plenty of letters but almost no prose letters.
    prose_letters = sum(
        sum(1 for c in l if c.isalpha()) for l in lines if len(l.strip()) >= PROSE_LINE_MIN_CHARS
    )

    expected_script_ratio = (
        sum(1 for c in letters if _script_of(c) in profile.scripts) / n_letters if n_letters else 0.0
    )
    garbage = sum(1 for c in non_ws if _is_garbage_char(c))
    garbage_char_ratio = garbage / n_chars if n_chars else 0.0
    detached_diacritics = len(DETACHED_DIACRITIC_RE.findall(text))

    noisy_token_ratio = spaced_runs_per_1k = glued_per_1k = None
    if profile.spaced:
        cores = [_strip_token(t) for t in tokens]
        word_cores = [c for c in cores if any(ch.isalpha() for ch in c)]
        noisy = sum(1 for c in word_cores if _is_noisy_token(c))
        noisy_token_ratio = round(noisy / (len(word_cores) or 1), 4)

        # Runs of >= 4 single-letter tokens ("j a n v i e r"): shattered words.
        spaced_runs, run = 0, 0
        for c in cores + [""]:
            if len(c) == 1 and c.isalpha():
                run += 1
            else:
                if run >= 4:
                    spaced_runs += 1
                run = 0
        spaced_runs_per_1k = round(spaced_runs * 1000 / n_tokens, 3)

        # Letter-only tokens of >= 30 letters = words glued together.
        glued = sum(1 for c in word_cores if len(c) >= 30 and c.isalpha())
        glued_per_1k = round(glued * 1000 / n_tokens, 3)

    diacritic_ratio = None
    if profile.diacritics:
        latin = [c for c in letters if _script_of(c) == "LATIN"]
        diacritic_ratio = round(sum(1 for c in latin if ord(c) > 127) / len(latin), 4) if latin else 0.0

    # Exact repetition of long lines (running headers, ads, nav menus).
    long_lines = [l.strip() for l in lines if len(l.strip()) >= 15]
    repeated_line_ratio = (1 - len(set(long_lines)) / len(long_lines)) if long_lines else 0.0

    per_1k_lines = 1000 / (n_lines or 1)
    nums = unit_numbers(lines)
    norm = re.sub(r"\s+", " ", text).strip().lower()
    return {
        "language": profile.name,
        "n_chars": n_chars,
        "n_letters": n_letters,
        "prose_letters": prose_letters,
        "n_lines": n_lines,
        "n_tokens": len(tokens),
        "expected_script_ratio": round(expected_script_ratio, 4),
        "garbage_char_ratio": round(garbage_char_ratio, 4),
        "noisy_token_ratio": noisy_token_ratio,
        "detached_diacritics_per_1k": round(detached_diacritics * 1000 / n_tokens, 2),
        "spaced_letter_runs_per_1k": spaced_runs_per_1k,
        "glued_tokens_per_1k": glued_per_1k,
        "diacritic_ratio": diacritic_ratio,
        "mojibake_per_1k_lines": round(len(MOJIBAKE_RE.findall(text)) * per_1k_lines, 2),
        "repeated_line_ratio": round(repeated_line_ratio, 4),
        "chrome_count": len(CHROME_RE.findall(text)),
        "chrome_per_1k_lines": round(len(CHROME_RE.findall(text)) * per_1k_lines, 2),
        "url_count": len(URL_RE.findall(text)),
        "url_per_1k_lines": round(len(URL_RE.findall(text)) * per_1k_lines, 2),
        "html_residue_count": len(HTML_RESIDUE_RE.findall(text)),
        "paywall_marker_count": len(PAYWALL_RE.findall(text)),
        "placeholder_line_count": sum(1 for l in lines if PLACEHOLDER_LINE_RE.match(l)),
        "n_unit_numbers": len(nums),
        "numbering_disorder": round(numbering_disorder(nums), 4),
        # distinct unit numbers, space-separated: fix safety limit K1 compares them
        "unit_numbers_distinct": " ".join(str(n) for n in sorted(set(nums))),
        "content_hash": hashlib.sha1(norm.encode("utf-8")).hexdigest(),
    }


# --------------------------------------------------------------------------- corpus-relative
def _words(text: str, profile: LanguageProfile) -> list[str]:
    """Lower-cased words of >= 4 letters in the language's own script(s)."""
    return [w.lower() for w in UNIQUE_WORD_RE.findall(text) if _script_of(w[0]) in profile.scripts]


def word_set(text: str, profile: LanguageProfile) -> set[str]:
    return set(_words(text, profile))


def corpus_unique_word_ratio(text: str, profile: LanguageProfile, doc_freq) -> float | None:
    """Share of word tokens that occur in NO other document of the corpus.

    OCR misreadings are almost always unique to one file, while legitimate legal
    vocabulary recurs across the corpus. `doc_freq` counts, per word, how many
    documents contain it (including this one).
    """
    toks = _words(text, profile)
    if not toks:
        return None
    unique = sum(1 for w in toks if doc_freq.get(w, 0) <= 1)
    return round(unique / len(toks), 4)


# --------------------------------------------------------------------------- source (PDF)
def pdf_metrics(pdf_path: Path | str, min_chars_per_page: int = 200, max_sample_pages: int = 60) -> dict:
    """Text-layer coverage of a source PDF.

    A page whose text layer has fewer than `min_chars_per_page` non-space
    characters counts as image-only (scanned). Long PDFs are measured on
    `max_sample_pages` evenly spaced pages (always including first and last),
    which is deterministic.
    """
    import pdfplumber  # local import: only source metrics need it

    per_page = []
    with pdfplumber.open(pdf_path) as pdf:
        total = len(pdf.pages)
        if total > max_sample_pages:
            step = (total - 1) / (max_sample_pages - 1)
            indices = sorted({round(i * step) for i in range(max_sample_pages)})
        else:
            indices = range(total)
        for i in indices:
            page = pdf.pages[i]
            try:
                n = sum(1 for ch in page.chars if not ch.get("text", " ").isspace())
            except Exception:
                n = 0
            per_page.append(n)
            page.close()  # large PDFs otherwise exhaust memory
    n_sampled = len(per_page) or 1
    low = sum(1 for n in per_page if n < min_chars_per_page)
    return {
        "pdf_pages": total,
        "pdf_pages_sampled": len(per_page),
        "pdf_median_chars_per_page": int(statistics.median(per_page)) if per_page else 0,
        "pdf_low_text_page_ratio": round(low / n_sampled, 4),
    }
