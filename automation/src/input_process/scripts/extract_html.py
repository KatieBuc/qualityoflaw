#!/usr/bin/env python3
"""
extract_html.py — Re-extract statute text from saved HTML pages
(Domestic_Violence/*.html) into clean plain text (globallaws_cleaned/*.txt).

This is the corrected, final version of the pipeline used for Route A1 and
the HTML-based files under Route B2/B3 of the globallaws cleaning project.
It supersedes several earlier, buggier versions — see "LESSONS LEARNED"
below before changing the extraction logic, since each point corresponds to
a real content-loss bug found (via a user-reported missing article) and
fixed during this project.

USAGE
-----
Process every A1/B2/B3 HTML file listed in cleaning_routing_plan.csv:
    python extract_html.py

Process a single file (for testing a fix to one source):
    python extract_html.py --file "23. Brazil - DomViol - Portuguese - html.txt"

Re-run the heading-parity audit only, without writing any output:
    python extract_html.py --audit-only

Paths (SRC_DIR / OUT_DIR / PLAN) are hard-coded for this project below;
adjust the constants at the top if you move the folders.

LESSONS LEARNED (read before modifying the extraction logic)
--------------------------------------------------------------------------
1. Do NOT add any frequency-based "drop short lines that repeat often"
   filter as a chrome-removal step. It looks appealing for killing things
   like a bookmark widget spamming "Add to favorites" down a page, but
   large codified-law documents legitimately repeat short citation
   fragments hundreds of times ("Pena -", "Vigência", "(Incluído", the bare
   ordinal mark "º", "§ 1", etc.). An earlier version of this pipeline used
   such a filter and it silently deleted Article 1 of the Brazilian Penal
   Code (and equivalent fragments elsewhere) corpus-wide. There is no safe
   frequency threshold for this — don't reintroduce it.

2. `trafilatura` can silently drop a heading or paragraph when it sits in
   an unusual nested-tag structure (e.g. a self-linking
   `<h4><a>Artículo 12</a></h4>`), even while the rest of the page extracts
   normally and the output length looks plausible. There is no way to
   detect this from length alone. This script guards against it with a
   heading-count parity check (`heading_set` + the `traf_h < 0.9 * bs4_h`
   fallback condition below) — keep that check when editing the decision
   logic, and prefer widening `HEADING_PAT` over removing the check if a
   new language's heading style isn't being detected.

3. Do NOT strip `<header>`, `<nav>`, or `<footer>` tags by tag name as a
   blanket "remove page chrome" step. At least one source site (Panama)
   reuses `<header>` *inside* each `<article>`/`<section>` to hold that
   article's own number — blanket-stripping `<header>` deleted every
   article number in that document even though the article bodies
   survived. Only `<script>`, `<style>`, and `<noscript>` are safe to
   strip unconditionally; everything else may carry real structure on some
   site, so leftover nav/footer text is tolerated as the lesser evil
   compared to losing content.

4. `FORCE_BS4` exists for files where the heading-parity check still
   doesn't catch a known bad trafilatura output — e.g. Taiwan's page
   embeds a `<script>` block that trafilatura pulls in as body text even
   though heading counts look fine (JS code doesn't affect the heading
   regex at all), and Brazil's Article 1 loss passed the 90% parity
   threshold in aggregate (it only lost 1 of ~360 headings) but was a
   confirmed real drop on manual check. If you find another such case,
   add the filename to FORCE_BS4 rather than trying to make the automatic
   heuristic catch every possible failure mode — some things need a human
   to notice via a content diff, not a metric.
"""
import argparse
import csv
import html
import os
import re

import trafilatura
from bs4 import BeautifulSoup, Comment

# ---------------------------------------------------------------------------
# Paths — adjust if the project layout changes.
# ---------------------------------------------------------------------------
PLAN = r"C:\myProject\Foundation_model\GlobalLaws\golden\cleaning\cleaning_routing_plan.csv"
SRC_DIR = r"C:\myProject\Foundation_model\Domestic_Violence"
OUT_DIR = r"C:\myProject\Foundation_model\globallaws_cleaned"

# ---------------------------------------------------------------------------
# Per-file overrides
# ---------------------------------------------------------------------------

# Some source pages declare the wrong charset in their <meta> tag (or none
# at all) while their actual bytes are in a legacy Windows codepage. Found
# by inspecting raw bytes around known accented words (see git history /
# session notes for the diagnostic method — decode candidate charsets and
# check whether known words like "Presidência" or "Promulgación" come out
# correctly).
CHARSET_OVERRIDE = {
    "182. Uruguay - DomViol - Spanish (a) - html.txt": "windows-1252",
    "182. Uruguay - DomViol - Spanish (b) - html.txt": "windows-1252",
    "23. Brazil - DomViol - Portuguese - html.txt": "windows-1252",
    "25. Bulgaria - DomViol - Bulgarian (a) - html.txt": "windows-1251",
    "25. Bulgaria - DomViol - Bulgarian (b) - html.txt": "windows-1251",
    "43. Cyprus - DomViol - Greek (a) - html.txt": "windows-1253",
    "44. Czechia - DomViol - Czech (c) - html.txt": "windows-1250",
}

# Files confirmed (by manual content diff) to lose real content via
# trafilatura even though the automatic heading-parity check doesn't catch
# it. Always use the safe bs4 extraction for these. See LESSONS LEARNED #4.
FORCE_BS4 = {
    "23. Brazil - DomViol - Portuguese - html.txt",   # drops Art. 1 (nested <font>/<b> tags)
    "167. Taiwan - DomViol - English - html.txt",     # pulls in a <script> block as body text
}

# Files confirmed (by direct inspection — the saved HTML's total plain text
# is a few hundred/thousand characters of page furniture only) to contain
# no recoverable statute text at all. Re-extraction can't fix these; they
# need a fresh crawl of a page that actually has the content (Route B3).
# Left untouched by this script.
NO_RECOVERABLE_CONTENT = {
    "89. Korea - DomViol - Korean - html.txt",
    "82. Italy - DomViol - Italian (a) - html.txt",
    "73. Hong Kong - DomViol - English - html.txt",
    "10. Azerbaijan - DomViol - Azerbaijani - html.txt",
}

# Generic, multi-language "Article N" / "Artículo N" / "§ N" heading pattern.
# Intentionally broad-recall — it's only used as a red-flag detector for the
# parity check, not for anything that has to be precise.
HEADING_PAT = re.compile(
    r"\b(?:Article|Articolo|Art[ií]culo|Artigo|Artikel|Stat[yi]a|"
    r"Član|Članak|Členok|Článek|Paragraf|§|Artikkel|Art)\.?\s*\d{1,4}\b",
    re.IGNORECASE,
)


def read_source(fn):
    """Read and decode the saved HTML for a given output filename."""
    src_path = os.path.join(SRC_DIR, fn[:-4] + ".html")
    with open(src_path, "rb") as f:
        raw = f.read()
    enc = CHARSET_OVERRIDE.get(fn, "utf-8")
    return raw, raw.decode(enc, errors="ignore")


def safe_bs4_extract(text_str):
    """Mechanical, complete extraction. Only strips tags that are never
    legitimately body content (script/style/noscript) — see LESSONS
    LEARNED #3 for why nothing else is stripped by tag name."""
    soup = BeautifulSoup(text_str, "html.parser")
    for tag in soup.find_all(["script", "style", "noscript"]):
        tag.decompose()
    for c in soup.find_all(string=lambda s: isinstance(s, Comment)):
        c.extract()
    text = soup.get_text("\n", strip=True)
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    out, prev = [], None
    for l in lines:
        if l == prev:  # collapse only back-to-back exact duplicate lines
            continue
        out.append(l)
        prev = l
    return "\n".join(out)


def traf_extract(source):
    """`source` may be raw bytes (utf-8 pages) or a decoded str (pages that
    needed a charset override) — trafilatura accepts either."""
    try:
        return trafilatura.extract(
            source, favor_recall=True, include_tables=True,
            output_format="txt", include_comments=False,
        ) or ""
    except Exception:
        return ""


def clean_basic(text):
    """Decode HTML entities and collapse runs of 2+ blank lines to 1.
    Deliberately does NOT do any frequency-based line filtering — see
    LESSONS LEARNED #1."""
    text = html.unescape(text)
    lines = text.split("\n")
    out, blank = [], 0
    for l in lines:
        if l.strip() == "":
            blank += 1
            if blank > 1:
                continue
        else:
            blank = 0
        out.append(l)
    return "\n".join(out).strip()


def heading_set(text):
    return {
        m.group(0).lower().replace(" ", "").replace(".", "")
        for m in HEADING_PAT.finditer(text)
    }


def extract_one(fn, write=True):
    """Run the full pipeline for one file. Returns (method, cleaned_text,
    bs4_heading_count, traf_heading_count)."""
    raw, text_str = read_source(fn)
    enc = CHARSET_OVERRIDE.get(fn)

    bs4_out = safe_bs4_extract(text_str)
    traf_out = traf_extract(text_str if enc else raw)

    bs4_h = heading_set(bs4_out)
    traf_h = heading_set(traf_out)

    if fn in FORCE_BS4:
        chosen, method = bs4_out, "bs4(forced)"
    else:
        # Prefer trafilatura only if it keeps at least 90% of the headings
        # the complete-but-noisier bs4 extraction found, and isn't
        # drastically shorter overall (a second proxy for dropped
        # sections). See LESSONS LEARNED #2.
        heading_ok = (len(bs4_h) < 5) or (len(traf_h) >= 0.9 * len(bs4_h))
        length_ok = len(bs4_out) == 0 or (len(traf_out) >= 0.5 * len(bs4_out))
        if traf_out and heading_ok and length_ok:
            chosen, method = traf_out, "trafilatura"
        else:
            chosen, method = bs4_out, "bs4(safe)"

    cleaned = clean_basic(chosen)

    if write:
        out_path = os.path.join(OUT_DIR, fn)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(cleaned + "\n")

    return method, cleaned, len(bs4_h), len(traf_h)


def audit_one(fn):
    """Compare heading counts between the raw source and the CURRENT
    globallaws_cleaned/ output (does not re-extract or write anything) —
    use this to check whether an already-generated output lost headings
    relative to its source."""
    raw, text_str = read_source(fn)
    soup = BeautifulSoup(text_str, "html.parser")
    for tag in soup.find_all(["script", "style", "noscript"]):
        tag.decompose()
    src_h = heading_set(soup.get_text(" "))

    out_path = os.path.join(OUT_DIR, fn)
    if not os.path.exists(out_path):
        return None
    with open(out_path, encoding="utf-8") as f:
        out_text = f.read()
    out_h = heading_set(out_text)

    missing = src_h - out_h
    return len(src_h), len(out_h), sorted(missing)


def load_target_files():
    with open(PLAN, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    files = [
        r["filename"] for r in rows
        if r["處置軌道"] in ("A1", "B2", "B3")
        and r["filename"] not in NO_RECOVERABLE_CONTENT
        and os.path.exists(os.path.join(SRC_DIR, r["filename"][:-4] + ".html"))
    ]
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--file", help="Process only this one output filename (e.g. '23. Brazil - DomViol - Portuguese - html.txt')")
    ap.add_argument("--audit-only", action="store_true", help="Only run the heading-parity audit against existing output; write nothing")
    args = ap.parse_args()

    targets = [args.file] if args.file else load_target_files()

    if args.audit_only:
        flagged = 0
        for fn in targets:
            result = audit_one(fn)
            if result is None:
                continue
            s, o, missing = result
            if s >= 5 and len(missing) > 0.15 * s:
                flagged += 1
                print(f"FLAGGED  {fn}\n  source_headings={s} output_headings={o} missing={len(missing)} sample={missing[:8]}")
        print(f"\nChecked {len(targets)} files, flagged {flagged}")
        return

    for fn in targets:
        method, cleaned, bs4_h, traf_h = extract_one(fn)
        print(f"{fn[:55]:55s} bs4_h={bs4_h:4d} traf_h={traf_h:4d} -> {method:14s} len={len(cleaned)}")

    print(f"\nProcessed {len(targets)} files -> {OUT_DIR}")


if __name__ == "__main__":
    main()
