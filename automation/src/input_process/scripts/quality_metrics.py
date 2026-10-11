#!/usr/bin/env python3
"""
quality_metrics.py — Quantitative text-quality metrics for the GlobalLaws
cleaning pipeline.

Every OCR / routing / removal decision in CLEANING_CRITERIA.md is expressed
as a threshold on one of the metrics computed here, so the decisions can be
reproduced by anyone re-running the pipeline instead of depending on who (or
which model) reviewed the file. This module only MEASURES; the thresholds and
decision rules live in quality_criteria.py.

Two groups of metrics:

  * Text metrics      — computed from a .txt file (raw extraction in
                        globallaws/, or a cleaned file). Pure Python, fast.
  * Source metrics    — computed from the original PDF in Domestic_Violence/
                        (text-layer coverage per page). Slow (pdfplumber), so
                        results are cached in a JSON file next to the output.

All ratios are in [0, 1]; all "_per_1k" rates are per 1,000 tokens (or lines,
where stated).

USAGE (as a library)
--------------------
    from quality_metrics import text_metrics, pdf_metrics
    m = text_metrics(text, language="Spanish")
"""
import functools
import hashlib
import os
import re
import statistics
import unicodedata

# ---------------------------------------------------------------------------
# Language -> expected script(s). A file's language is the word after
# "DomViol - " in its filename.
# ---------------------------------------------------------------------------
LANG_SCRIPTS = {
    "Arabic": {"ARABIC"},
    "Armenian": {"ARMENIAN"},
    "Bhutanese": {"TIBETAN"},
    "Bulgarian": {"CYRILLIC"},
    "Chinese": {"CJK"},
    "Georgian": {"GEORGIAN"},
    "Greek": {"GREEK"},
    "Hebrew": {"HEBREW"},
    "Japanese": {"CJK", "HIRAGANA", "KATAKANA"},
    "Korean": {"HANGUL", "CJK"},
    "Macedonian": {"CYRILLIC"},
    "Russian": {"CYRILLIC"},
    "Thai": {"THAI"},
    "Ukrainian": {"CYRILLIC"},
    # Bosnian / Serbian / Montenegrin statutes are published in either
    # alphabet — both count as expected.
    "Bosnian": {"LATIN", "CYRILLIC"},
}
DEFAULT_SCRIPTS = {"LATIN"}

# Languages written without spaces between words. Token-based metrics
# (noisy_token_ratio, spaced/glued tokens, corpus_unique_word_ratio) are not
# meaningful for them and are reported as empty.
UNSPACED_LANGS = {"Chinese", "Japanese", "Thai", "Bhutanese"}

# Latin-script languages whose normal orthography uses diacritics. Used by
# diacritic_ratio: a text in one of these languages with (almost) no
# diacritics has lost them in extraction (wrong charset / dropped glyphs).
DIACRITIC_LANGS = {
    "Albanian", "Azerbaijani", "Bosnian", "Croatian", "Czech", "Danish",
    "Estonian", "French", "German", "Hungarian", "Icelandic", "Italian",
    "Latvian", "Lithuanian", "Norwegian", "Polish", "Portuguese",
    "Potuguese", "Romanian", "Slovak", "Slovenian", "Spanish", "Swedish",
    "Turkish",
}

# Symbols (Unicode category So/Sk/Sm) that legitimately occur in statutes and
# must not count as garbage.
ALLOWED_SYMBOLS = set("°©®™№§¶+=<>|~^`´¨¸¯˚×÷±∙•·–—‐‑←→↑↓✓✔☐☑□■▪►▶◆◊○●√∞≤≥≠≈%$€£¥¢")

TOKEN_RE = re.compile(r"\S+")
EDGE_PUNCT = "\"'“”‘’«»„‚()[]{}<>.,;:!?¿¡*_-–—/\\|•·…"

# Line-anchored statutory unit headings used for the numbering-order check.
# Only heading-SHAPED lines count (short line, at most a short title after
# the number), so that cross-references which happen to start a line
# ("Section 3624 of title 18 is amended ...") are ignored.
UNIT_HEADING_RE = re.compile(
    r"^\s*(?:art(?:icle|ículo|iculo|igo|icolo|ikel|ikkel|ykuł|ikla|ikulli|iklis)?\.?"
    r"|section|sec\.|§|članak|člen|clanak|статья|стаття|член|άρθρο|madde|straipsnis|pant)"
    r"\s*(\d{1,4})\s*(?:[.:\-–—)]|º|°)?\s*(.*)$",
    re.IGNORECASE,
)
# Words that show the "heading" is really a cross-reference.
XREF_START_RE = re.compile(r"^(?:of|de|del|du|des|da|do|van|von|and|et|y|e|to|in|,)(?:\b|$)", re.IGNORECASE)
# (unit_numbers() also requires the heading keyword to start with a capital
# letter or "§": a lowercase "article 5 of ..." at a line start is a
# cross-reference wrapped onto a new line.)
# Decimal numbering "16.20. Definitions" (chapter.section).
DECIMAL_NUM_RE = re.compile(r"^\s*(\d{1,3})\.(\d{1,3})\.?\s+[^\W\d_]")
# Common-law style "12. Heading" at line start.
BARE_NUM_RE = re.compile(r"^\s*(\d{1,4})\.\s+[^\W\d_]")
MAX_HEADING_LINE = 90
MAX_TITLE_WORDS = 10

# Paywall / withheld-content placeholders.
PAYWALL_RE = re.compile(
    r"sign up or sign in|pro membership|full text is available|active subscription"
    r"|subscribe to (?:read|view|access)|login to (?:read|view)|réservé aux abonnés"
    r"|contenido exclusivo|conteúdo exclusivo|nur für abonnenten",
    re.IGNORECASE,
)
PLACEHOLDER_LINE_RE = re.compile(r"^\s*(?:\.{3,}|…+)\s*$")

# Web / print chrome vocabulary (multilingual, deliberately generic).
CHROME_RE = re.compile(
    r"\b(?:cookies?|log ?in|sign ?in|sign ?up|subscribe|newsletter|privacy policy"
    r"|terms of use|all rights reserved|javascript|browser|click here|share on"
    r"|facebook|twitter|linkedin|whatsapp|home page|homepage|back to top|skip to"
    r"|impressum|datenschutz|abonnez|connexion|mentions légales|iniciar sesión"
    r"|política de privacidad|aviso legal|inscreva|entrar|zum seitenanfang"
    r"|shopping cart|košarica|prijava)\b",
    re.IGNORECASE,
)
URL_RE = re.compile(r"https?://\S+|www\.\S+|\S+@\S+\.\w{2,}")
# Raw HTML left in the text: known tag names (any case, with attributes), a bare
# lowercase pseudo-tag such as "<english>", or an entity. Angle-bracket notes
# that are part of the statute (Belgian "<L 2000-06-26/42, art. 2, En vigueur
# : 01-01-2002>") do not match.
HTML_TAGS = ("a|abbr|b|body|br|center|div|em|font|h[1-6]|head|hr|html|i|img|li|link|meta|nav|ol|p|"
             "script|section|span|strong|style|sub|sup|table|tbody|td|th|thead|title|tr|u|ul")
HTML_RESIDUE_RE = re.compile(
    r"</?(?:" + HTML_TAGS + r")(?:\s[^<>]{0,120})?/?>|<[a-z]{2,15}>|&[a-zA-Z]{2,8};|&#\d{2,6};",
    re.IGNORECASE,
)
# UTF-8 bytes decoded as cp1252/latin-1 ("Ã©", "â€™", "Ã±", ...).
MOJIBAKE_RE = re.compile(r"Ã[\u0080-\u00BF]|â€[\u0080-\u00BF\u2018-\u201E\u2020-\u2122]|Â[\u00A0-\u00BF]|Ð[\u0080-\u00BF]|Ñ[\u0080-\u008F]")


def language_of(filename):
    m = re.search(r"DomViol - ([A-Za-z]+)", filename)
    return m.group(1) if m else ""


@functools.lru_cache(maxsize=None)
def _script_of(ch):
    """Coarse script bucket for a letter, from its Unicode name."""
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


def _is_symbol_font_char(ch):
    """Private-use U+F000-U+F0FF: glyphs from Symbol/Wingdings fonts (bullets,
    dashes, check boxes) that PDF extractors emit verbatim. Cosmetic, not
    corruption — counted separately as symbol_font_chars."""
    return 0xF000 <= ord(ch) <= 0xF0FF


@functools.lru_cache(maxsize=None)
def _is_garbage_char(ch):
    if ch in "\n\t\r ":
        return False
    if ch == "\ufffd":
        return True
    if _is_symbol_font_char(ch):
        return False
    cat = unicodedata.category(ch)
    if cat in ("Co", "Cn", "Cs"):
        return True
    if cat == "Cc":
        return True
    if cat in ("So", "Sk") and ch not in ALLOWED_SYMBOLS:
        return True
    return False


def _strip_token(tok):
    return tok.strip(EDGE_PUNCT)


# Legit letter+digit tokens: ordinals / suffixes ("1st", "2e", "12bis", "4ª",
# "3A"), short prefixes ("S12", "No5", "A1"), and legislation.gov.uk
# annotation markers glued to the next word ("F102Part", "F46for").
LEGIT_ALNUM_RE = re.compile(
    r"^(?:\d+[^\W\d_]{1,4}|[^\W\d_]{1,3}\d+[^\W\d_]?|\d+[^\W\d_]\d+|[A-Z]\d{1,4}[^\W\d_]+)$"
)
# Characters that may legitimately sit between two letters: apostrophe
# variants, hyphens, dots, slashes, and sentence/list punctuation (a missing
# space after ";" or ":" is a layout issue, not character corruption).
INWORD_OK = "'’‘‛‟ʼ`-‐‑./;:,!?)("
# Spacing (non-combining) diacritics that PDF extractors sometimes emit AFTER
# the base letter instead of composing it: "socie´te´" for "société".
# (The backtick is excluded: Armenian uses it as a native punctuation mark.)
DETACHED_DIACRITIC_RE = re.compile(r"[^\W\d_][´¨˜ˆ¸˚˝˘ˇ˛]")


def _is_noisy_token(core):
    """True if a token shows a typical OCR / decoding corruption pattern."""
    has_alpha = any(c.isalpha() for c in core)
    if not has_alpha or len(core) < 2:
        return False
    # Combining marks (Arabic harakat, Hebrew points, etc.) and detached
    # spacing diacritics are measured elsewhere — drop them here.
    core = "".join(c for c in core if not unicodedata.category(c).startswith("M"))
    core = re.sub(r"[´¨˜ˆ¸˚˝˘ˇ˛]", "", core)
    # (a) digits inside a word ("Pur5uant", "Ve1du", "NATIt3N")
    if any(c.isdigit() for c in core) and not LEGIT_ALNUM_RE.match(core):
        letters_around = re.search(r"[^\W\d_]\d+[^\W\d_]", core)
        if letters_around:
            return True
    # (b) symbol wedged between letters ("ARRAN€iEMENT", "R¥RT")
    for a, b, c in zip(core, core[1:], core[2:]):
        if a.isalpha() and c.isalpha() and not b.isalnum() and not b.isspace() and b not in INWORD_OK:
            return True
    # (c) two scripts inside one word (Latin/Cyrillic confusion)
    scripts = {_script_of(c) for c in core if c.isalpha()}
    scripts.discard("OTHER")
    if len(scripts) > 1:
        return True
    # (d) irregular case: lower->upper inside a word that is not all-caps
    #     ("CoDIgo", "RTiGo", "PenaL"). Allows a single leading capital
    #     prefix like "McDonald" by requiring >= 2 case flips.
    flips = sum(1 for a, b in zip(core, core[1:]) if a.islower() and b.isupper())
    if flips >= 1 and not core.isupper():
        ups_after_low = re.search(r"[a-zà-ÿ][A-ZÀ-Þ]", core)
        if ups_after_low and flips + sum(1 for a, b in zip(core, core[1:]) if a.isupper() and b.islower()) >= 3:
            return True
    return False


def unit_numbers(lines):
    """Sequence of statutory unit numbers taken from heading-shaped lines.

    Table-cell pipes around a line are ignored. Three numbering styles are
    collected separately (worded "Article 12",
    decimal "16.20.", bare "12.") and the style with the most hits is used.
    """
    worded, decimal, bare = [], [], []
    for line in lines:
        # Headings rendered as table cells ("|  | § 1 |") count as headings.
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


def numbering_disorder(nums):
    """Share of consecutive unit-number steps that go the 'wrong way'.

    A statute's units run 1, 2, 3, ... A table of contents followed by the
    body, an annex, or a second bundled act each cause ONE legitimate reset
    back to a low number. Scrambled extraction (two columns / two documents
    interleaved) instead produces many small backward jumps (20 -> 26 -> 21
    -> 27 ...). So a step counts as disorder only if it goes backwards AND
    does not land on a restart (target <= 2), and the metric is the share of
    such steps among all steps.
    """
    if len(nums) < 2:
        return 0.0
    bad = 0
    for a, b in zip(nums, nums[1:]):
        if b < a and b % 1000 > 2:
            bad += 1
    return bad / (len(nums) - 1)


PROSE_LINE_MIN_CHARS = 40
UNIQUE_WORD_RE = re.compile(r"[^\W\d_]{4,}")


def text_metrics(text, language=""):
    """Compute all per-document text metrics.

    Token-based metrics are None for UNSPACED_LANGS; diacritic_ratio is None
    for languages outside DIACRITIC_LANGS. None means "not applicable" and
    never triggers a rule.
    """
    spaced = language not in UNSPACED_LANGS
    lines = [l for l in text.split("\n") if l.strip()]
    n_lines = len(lines)
    tokens = TOKEN_RE.findall(text)
    n_tokens = len(tokens) or 1
    non_ws = [c for c in text if not c.isspace()]
    n_chars = len(non_ws)
    letters = [c for c in non_ws if c.isalpha()]
    n_letters = len(letters)

    # Letters on "prose" lines (>= 40 chars). Navigation menus, link lists
    # and UI labels are made of short lines, so a page of page-chrome has
    # plenty of letters but almost no prose letters.
    prose_letters = sum(
        sum(1 for c in l if c.isalpha()) for l in lines if len(l.strip()) >= PROSE_LINE_MIN_CHARS
    )

    expected = LANG_SCRIPTS.get(language, DEFAULT_SCRIPTS)
    expected_script_ratio = (
        sum(1 for c in letters if _script_of(c) in expected) / n_letters if n_letters else 0.0
    )

    garbage = sum(1 for c in non_ws if _is_garbage_char(c))
    garbage_char_ratio = garbage / n_chars if n_chars else 0.0
    symbol_font_chars = sum(1 for c in non_ws if _is_symbol_font_char(c))
    detached_diacritics = len(DETACHED_DIACRITIC_RE.findall(text))

    noisy_token_ratio = spaced_runs_per_1k = glued_per_1k = None
    if spaced:
        cores = [_strip_token(t) for t in tokens]
        word_cores = [c for c in cores if any(ch.isalpha() for ch in c)]
        noisy = sum(1 for c in word_cores if _is_noisy_token(c))
        noisy_token_ratio = round(noisy / (len(word_cores) or 1), 4)

        # Runs of >= 4 consecutive single-letter tokens ("j a n v i e r",
        # "D O M E S T I C") — letter-spaced or shattered words.
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
    if language in DIACRITIC_LANGS:
        latin = [c for c in letters if _script_of(c) == "LATIN"]
        diacritic_ratio = round(sum(1 for c in latin if ord(c) > 127) / len(latin), 4) if latin else 0.0

    # Exact line repetition (running headers, ads, nav menus). Only lines of
    # >= 15 chars, so short legit repeats ("(a)", "Article", "§ 1") are ignored.
    long_lines = [l.strip() for l in lines if len(l.strip()) >= 15]
    repeated_line_ratio = (1 - len(set(long_lines)) / len(long_lines)) if long_lines else 0.0

    short_line_ratio = sum(1 for l in lines if len(l.split()) <= 3) / n_lines if n_lines else 0.0

    per_1k_lines = 1000 / (n_lines or 1)
    nums = unit_numbers(lines)

    norm = re.sub(r"\s+", " ", text).strip().lower()
    return {
        "language": language,
        "n_chars": n_chars,
        "n_letters": n_letters,
        "prose_letters": prose_letters,
        "n_lines": n_lines,
        "n_tokens": len(tokens),
        "expected_script_ratio": round(expected_script_ratio, 4),
        "garbage_char_ratio": round(garbage_char_ratio, 4),
        "symbol_font_chars": symbol_font_chars,
        "noisy_token_ratio": noisy_token_ratio,
        "detached_diacritics_per_1k": round(detached_diacritics * 1000 / n_tokens, 2),
        "spaced_letter_runs_per_1k": spaced_runs_per_1k,
        "glued_tokens_per_1k": glued_per_1k,
        "diacritic_ratio": diacritic_ratio,
        "mojibake_per_1k_lines": round(len(MOJIBAKE_RE.findall(text)) * per_1k_lines, 2),
        "repeated_line_ratio": round(repeated_line_ratio, 4),
        "short_line_ratio": round(short_line_ratio, 4),
        "chrome_count": len(CHROME_RE.findall(text)),
        "chrome_per_1k_lines": round(len(CHROME_RE.findall(text)) * per_1k_lines, 2),
        "url_count": len(URL_RE.findall(text)),
        "url_per_1k_lines": round(len(URL_RE.findall(text)) * per_1k_lines, 2),
        "html_residue_count": len(HTML_RESIDUE_RE.findall(text)),
        "paywall_marker_count": len(PAYWALL_RE.findall(text)),
        "placeholder_line_count": sum(1 for l in lines if PLACEHOLDER_LINE_RE.match(l)),
        "n_unit_numbers": len(nums),
        "numbering_disorder": round(numbering_disorder(nums), 4),
        # distinct unit numbers, space-separated — used by route acceptance
        # rule K1 to detect headings lost during trimming / re-extraction
        "unit_numbers_distinct": " ".join(str(n) for n in sorted(set(nums))),
        "content_hash": hashlib.sha1(norm.encode("utf-8")).hexdigest(),
    }


# ---------------------------------------------------------------------------
# Corpus-relative metric
# ---------------------------------------------------------------------------
def _words(text, language):
    """Lower-cased words of >= 4 letters written in the language's own
    script(s). Words in another script (the Arabic half of a French/Arabic
    gazette) are left out: they would all look corpus-unique."""
    expected = LANG_SCRIPTS.get(language, DEFAULT_SCRIPTS)
    return [w.lower() for w in UNIQUE_WORD_RE.findall(text) if _script_of(w[0]) in expected]


def word_set(text, language=""):
    return set(_words(text, language))


def corpus_unique_word_ratio(text, language, doc_freq):
    """Share of word tokens (>= 4 letters, in the language's own script) that
    occur in NO other document of the same language.

    `doc_freq` counts, for each word, how many documents of that language
    contain it (including this one). OCR misreadings ("Juterpretation",
    "DOMESTIC VIOLEXCE") are almost always unique to one file, while
    legitimate legal vocabulary recurs across the corpus.
    """
    toks = _words(text, language)
    if not toks:
        return None
    unique = sum(1 for w in toks if doc_freq.get(w, 0) <= 1)
    return round(unique / len(toks), 4)


# ---------------------------------------------------------------------------
# Source (PDF) metrics
# ---------------------------------------------------------------------------
def pdf_metrics(pdf_path, min_chars_per_page=200, max_sample_pages=60):
    """Text-layer coverage of a source PDF.

    A page whose text layer has fewer than `min_chars_per_page` non-space
    characters is counted as image-only (scanned) — running headers/footers
    alone ("LAWS OF GUYANA  Cap. 11:09  3") stay well below 200.

    PDFs longer than `max_sample_pages` are measured on that many evenly
    spaced pages (always including the first and last page); the result is
    deterministic and the ratio is what the criteria use.
    """
    import pdfplumber  # local import: only needed for source metrics

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
            page.close()  # release the parsed layout; large PDFs otherwise exhaust memory
    n_sampled = len(per_page) or 1
    low = sum(1 for n in per_page if n < min_chars_per_page)
    return {
        "pdf_pages": total,
        "pdf_pages_sampled": len(per_page),
        "pdf_median_chars_per_page": int(statistics.median(per_page)) if per_page else 0,
        "pdf_low_text_page_ratio": round(low / n_sampled, 4),
    }
