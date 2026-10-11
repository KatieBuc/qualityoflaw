#!/usr/bin/env python3
"""
build_report.py — Regenerates CLEANING_REPORT.md from
scripts/file_processing_map.py (fix status per file) plus the short,
hand-written reason each removed file was quarantined (REMOVE_REASON
below), and SCOPE_NOTE for the content-scope-mismatch files that are
kept despite being flagged (a scope mismatch is not treated as a
text-quality defect, so it isn't grounds for quarantine here).

Reflects the corpus AFTER scripts/c1_quarantine_unusable.py has run:
globallaws_cleaned/ holds every file with clean text (issue fully fixed,
no issue found, or a content-scope note only); globallaws_removed/ holds
files with a confirmed unresolved text-quality problem, kept for the
record but not used downstream.

If you fix one of the removed files further, move it back into
globallaws_cleaned/, update FILE_STATUS in file_processing_map.py, remove
its entry from REMOVE_REASON below, then re-run this script.

USAGE
-----
    python build_report.py
"""
import sys
from collections import Counter

sys.path.insert(0, r"C:\myProject\Foundation_model\scripts")
from file_processing_map import FILE_ROUTE, FILE_STATUS

OUT_MD = r"C:\myProject\Foundation_model\GlobalLaws\reports\cleaning\CLEANING_REPORT.md"

DUPLICATE_NO_CONTENT = {"113. Mongolia - DomViol - English (b).txt"}

# ---- Method descriptions, by route code ----
METHOD = {
    "A1": "Re-extracted from the original `.html` (`extract_html.py`)",
    "A2": "Rule-based boilerplate trimming (`a2_trim_boilerplate.py`)",
    "A3": "Deduplication / isolated OCR-typo fixes (`a3_dedup_and_fixes.py`)",
    "A4": "Targeted OCR-typo find/replace after a full read (`a4_ocr_typo_fixes.py`)",
    "A5": "Removed a residual source marker (`a5_remove_markers.py`)",
    "B2": "Re-decoded with the correct charset, then re-extracted",
    "B3": "Re-ran the extraction pipeline on the saved HTML/PDF",
}

# ---- Short, one-line reason for every quarantined file ----
REMOVE_REASON = {
    # -- Source HTML has no recoverable content --
    "89. Korea - DomViol - Korean - html.txt": "Saved HTML is JS-rendered; no statute text was ever captured (~1,500 chars, all menu/UI text).",
    "82. Italy - DomViol - Italian (a) - html.txt": "Saved HTML is page furniture only (535 chars total); no article text captured.",
    "73. Hong Kong - DomViol - English - html.txt": "Saved HTML has no ordinance text (~3,900 chars, all nav/help text); likely AJAX-loaded.",
    "10. Azerbaijan - DomViol - Azerbaijani - html.txt": "Saved HTML has only 24 characters of real content; the scrape failed at capture time.",
    # -- Blocked by OCR tooling limitation (no OCR engine available in this environment) --
    "111. Micronesia - DomViol - English.txt": "Needs OCR re-run (English) — blocked, no OCR engine available.",
    "131. Papua New Guinea - DomViol - English.txt": "Needs OCR re-run (English) — blocked, no OCR engine available.",
    "172. Togo - DomViol - French.txt": "Needs OCR re-run (French) — blocked, no OCR engine available.",
    "47. Dominica - DomViol - English (a).txt": "Needs OCR re-run (English) — blocked, no OCR engine available.",
    "61. Gambia - DomViol - English.txt": "Needs OCR re-run (English) — blocked, no OCR engine available.",
    "83. Jamaica - DomViol - English (a).txt": "Needs OCR re-run (English) — blocked, no OCR engine available.",
    "84. Japan - DomViol - Japanese.txt": "Needs OCR re-run (Japanese) — blocked, no OCR engine available.",
    "183. Uzbekistan - DomViol - Russian (b).txt": "Needs OCR re-run (Russian) — blocked, no OCR engine available.",
    "19. Bhutan - DomViol - Bhutanese.txt": "Needs OCR re-run (Dzongkha) — blocked, no OCR engine available.",
    "62. Georgia - DomViol - Georgian.txt": "Needs OCR re-run (Georgian) — blocked, no OCR engine available.",
    "81. Israel - DomViol - Hebrew.txt": "Needs OCR re-run (Hebrew) — blocked, no OCR engine available.",
    "97. Liberia - DomViol - English.txt": "Needs OCR re-run (English) — blocked, no OCR engine available.",
    # -- Source PDF has no recoverable content / needs OCR --
    "168. Tajikistan - DomViol - English.txt": "Source PDF is a subscription-paywall placeholder page (0 extractable characters).",
    "50. Egypt - DomViol - Arabic.txt": "Non-standard Arabic font encoding; re-extraction produced unreadable output. Needs OCR.",
    "70. Guyana - DomViol - English.txt": "All 53 pages are scanned images with no text layer for the body. Needs OCR.",
    "115. Morocco - DomViol - Arabic.txt": "Extractable text is in reversed RTL order; a safe fix risks corrupting statute numbers. Needs OCR.",
    # -- Partially fixed, confirmed residual problem remains --
    "116. Mozambique - DomViol - Portuguese (a).txt": "Scanned-image PDF; article headings fixed but body text still has character-level OCR corruption.",
    "4. Angola - DomViol - Portuguese.txt": "Garbled TOC/ads removed, but a few paragraphs still have unrecoverable OCR dropouts.",
    "75. Iceland - DomViol - Icelandic (a).txt": "Systematic þ/b OCR confusion; only the unambiguous cases were fixed, ambiguous compounds remain.",
    "75. Iceland - DomViol - Icelandic (b).txt": "Same þ/b OCR issue and same partial fix as (a).",
    "88. Kiribati - DomViol - English.txt": "~26 high-confidence OCR fixes applied, but several section headings remain unrecoverable.",
    "33. Chad - DomViol - French (a).txt": "Masthead/date noise fixed, but a localized e/é↔ć font-encoding issue near Articles 14-16 remains.",
    "41. Cote dIvoire - DomViol - French.txt": "OCR typos fixed, but one sentence in Article 47 remains too fragmented to safely reconstruct.",
    "186. Viet Nam - DomViol - English (b).txt": "One typo fixed, but apparent content gaps around some article headers could not be verified.",
    # -- Duplicate, no independent content --
    "113. Mongolia - DomViol - English (b).txt": "Exact duplicate of (a); this file held only a pointer note, no unique text.",
}

# ---- Content-scope mismatch: text is clean, informational only (not excluded) ----
SCOPE_NOTE = {
    "109. Mauritius - DomViol - English (b).txt": "Wrong document: a parliamentary Order Paper, not the statute.",
    "46. Djibouti - DomViol - French.txt": "Wrong document: the Civil Code, not a dedicated domestic-violence statute.",
    "67. Guatemala - DomViol - Spanish (b).txt": "Wrong language: content is a Mayan-language (Poptí/Jakalteko) translation, not Spanish.",
    "9. Austria - DomViol - English.txt": "Off-topic: a legal report on the protection of vulnerable adults.",
    "136. Portugal - DomViol - Potuguese.txt": "Wrong content type: an index/list of related legislation, not a full law.",
    "16. Belgium - DomViol - French (a).txt": "Unrelated gazette miscellany (bankruptcy/inheritance notices); no statute text filtered out.",
    "181. United States - DomViol - English (a).txt": "Wrong content type: a section-by-section summary, not the verbatim law text.",
    "24. Brunei Darussalam - DomViol - English.txt": "Wrong document: a \"Married Women\" Act chapter, not the domestic-violence law.",
    "82. Italy - DomViol - English (c).txt": "Wrong content type: a summary article about a legislative amendment, not the law text.",
    "85. Jordan - DomViol - English.txt": "Wrong content type: a policy-framework report, not the statute articles.",
    "35. China - DomViol - English (a).txt": "Off-topic: the Women's Rights and Interests law, not the Anti-Domestic Violence law.",
    "35. China - DomViol - English (c).txt": "Wrong document: the full Civil Code (480k chars), not a dedicated statute.",
    "54. Estonia - DomViol - Estonian.txt": "Off-topic: the Gender Equality Act, not a domestic-violence statute.",
    "72. Honduras - DomViol - Spanish (c).txt": "Off-topic: the Equal Opportunities for Women Act, not a domestic-violence statute.",
}

with open(OUT_MD, "w", encoding="utf-8") as f:
    def w(line=""):
        f.write(line + "\n")

    total = len(FILE_STATUS)
    removed = {fn for fn, s in FILE_STATUS.items() if s in ("unfixable", "partial")} | DUPLICATE_NO_CONTENT
    kept = total - len(removed)
    counts = Counter(FILE_STATUS.values())

    w("# GlobalLaws Domestic Violence Corpus — Data Cleaning Report\n")
    w(f"**{total} source files** in `globallaws/` were reviewed. "
      f"**{kept} are cleaned and ready** in `globallaws_cleaned/` — this is the set the next "
      f"pipeline stage uses. **{len(removed)} are quarantined** in `globallaws_removed/` because "
      f"they have a confirmed, unresolved text-quality problem (garbled OCR, or no recoverable "
      f"statute text at all); nothing in that folder is used downstream. A wrong-document-for-the-"
      f"topic mismatch (e.g. the wrong law, wrong language, a summary instead of the statute) is "
      f"*not* treated as a defect here — that text is clean, so those files stay in "
      f"`globallaws_cleaned/` too (see the note below the route table).\n")

    w("## Summary\n")
    w("| | Count | |")
    w("|---|---|---|")
    w(f"| ✅ Ready (`globallaws_cleaned/`) | **{kept}** | {counts['fixed'] - 1} issues fixed + {counts['not_needed']} had no issue (minus 1 duplicate stub, see below) |")
    w(f"| 🗑️ Quarantined (`globallaws_removed/`) | **{len(removed)}** | {counts['unfixable']} unfixable + {counts['partial']} partially fixed + 1 duplicate stub |")
    w(f"| **Total** | **{total}** | |\n")

    w("## Why files are blocked\n")
    w("- **No OCR engine available** (16 files: 12 needing straight OCR + a few PDF/HTML variants): "
      "no admin rights to install Tesseract, a portable no-install build was blocked by the "
      "platform's own security layer, and Windows' built-in OCR API only has a Traditional Chinese "
      "language pack installed. None of this was routed around.\n"
      "- **No network access to re-crawl**: a handful of B3 (HTML/PDF) files would need a fresh "
      "crawl of the source site or an alternate source; direct access from this environment is "
      "restricted to a small domain allow-list that didn't cover these targets.\n")

    w("## Outcome by route\n")
    w("Every file was assigned a route in `cleaning_routing_plan.csv` based on its original "
      "problem; this is how each route actually resolved. \"Kept\" = ended up in "
      "`globallaws_cleaned/`; \"Removed\" = ended up in `globallaws_removed/`.\n")

    route_total, route_kept = Counter(), Counter()
    for fn, route in FILE_ROUTE.items():
        route_total[route] += 1
        if FILE_STATUS[fn] in ("fixed", "not_needed") and fn not in DUPLICATE_NO_CONTENT:
            route_kept[route] += 1

    ROUTE_METHOD = dict(METHOD)
    ROUTE_METHOD["B1"] = "Needs a full OCR re-run (`b1_mark_ocr_blocked.py` documents why it couldn't happen here)"
    ROUTE_METHOD["HOLD"] = "Content-scope mismatch — not a text-quality issue, kept as-is (see note below)"
    ROUTE_METHOD["OK"] = "No issue found; file left as-is"

    w("| Route | Method | Total | Kept | Removed |")
    w("|---|---|---|---|---|")
    for route in ["A1", "A2", "A3", "A4", "A5", "B1", "B2", "B3", "HOLD", "OK"]:
        tot = route_total.get(route, 0)
        if not tot:
            continue
        kept_n = route_kept.get(route, 0)
        removed_n = tot - kept_n
        w(f"| {route} | {ROUTE_METHOD[route]} | {tot} | {kept_n} | {removed_n} |")
    w(f"| **Total** | | **{sum(route_total.values())}** | **{sum(route_kept.values())}** | "
      f"**{sum(route_total.values()) - sum(route_kept.values())}** |\n")

    w("- **A1** (3 removed): source HTML confirmed to have no recoverable statute text (page "
      "chrome/UI only, nothing an extractor could recover).\n"
      "- **A3** (1 removed): the file itself was fixed correctly — an exact duplicate, properly "
      "identified and pointed at its canonical copy — but removed anyway because a pointer note "
      "isn't usable statute text on its own; the canonical copy stays in `globallaws_cleaned/`.\n"
      "- **A4** (8 removed): real, verified fixes were applied, but a full read found the original "
      "corruption was more pervasive than the initial sample suggested; the residual damage "
      "couldn't be fixed without guessing, so these stayed classified \"partial\" and were "
      "quarantined rather than shipped half-fixed.\n"
      "- **B1 / B2** (7 + 5 removed): needs OCR — no OCR engine was available in this environment.\n"
      "- **B3** (5 removed): source PDF/HTML confirmed to have no recoverable statute text, or "
      "needs OCR.\n"
      "- **HOLD** (0 removed): a content-scope mismatch (wrong law, wrong language, a summary "
      "instead of the statute) isn't a text-quality defect — the text itself is clean, so all 14 "
      "stay in `globallaws_cleaned/`. See the table below for what each one actually contains.\n")

    w("**Content-scope note** — these 14 files are in `globallaws_cleaned/`, flagged here only for "
      "awareness (e.g. if a later stage wants to filter by topic):\n")
    w("| Filename | What it actually is |")
    w("|---|---|")
    for fn, reason in SCOPE_NOTE.items():
        w(f"| {fn} | {reason} |")
    w("")

    w(f"## 🗑️ Quarantined — {len(removed)} files, not used in the next stage\n")

    groups = [
        ("Source HTML has no recoverable content", [
            "89. Korea - DomViol - Korean - html.txt", "82. Italy - DomViol - Italian (a) - html.txt",
            "73. Hong Kong - DomViol - English - html.txt", "10. Azerbaijan - DomViol - Azerbaijani - html.txt"]),
        ("Blocked by OCR tooling limitation", [
            "111. Micronesia - DomViol - English.txt", "131. Papua New Guinea - DomViol - English.txt",
            "172. Togo - DomViol - French.txt", "47. Dominica - DomViol - English (a).txt",
            "61. Gambia - DomViol - English.txt", "83. Jamaica - DomViol - English (a).txt",
            "84. Japan - DomViol - Japanese.txt", "183. Uzbekistan - DomViol - Russian (b).txt",
            "19. Bhutan - DomViol - Bhutanese.txt", "62. Georgia - DomViol - Georgian.txt",
            "81. Israel - DomViol - Hebrew.txt", "97. Liberia - DomViol - English.txt"]),
        ("Source PDF has no recoverable content / needs OCR", [
            "168. Tajikistan - DomViol - English.txt", "50. Egypt - DomViol - Arabic.txt",
            "70. Guyana - DomViol - English.txt", "115. Morocco - DomViol - Arabic.txt"]),
        ("Partially fixed — a confirmed residual problem remains", [
            "116. Mozambique - DomViol - Portuguese (a).txt", "4. Angola - DomViol - Portuguese.txt",
            "75. Iceland - DomViol - Icelandic (a).txt", "75. Iceland - DomViol - Icelandic (b).txt",
            "88. Kiribati - DomViol - English.txt", "33. Chad - DomViol - French (a).txt",
            "41. Cote dIvoire - DomViol - French.txt", "186. Viet Nam - DomViol - English (b).txt"]),
        ("Duplicate — no independent content", ["113. Mongolia - DomViol - English (b).txt"]),
    ]

    seen = set()
    for title, fns in groups:
        w(f"### {title} ({len(fns)})\n")
        w("| Filename | Reason |")
        w("|---|---|")
        for fn in fns:
            w(f"| {fn} | {REMOVE_REASON[fn]} |")
            seen.add(fn)
        w("")

    missing = removed - seen
    if missing:
        raise SystemExit(f"BUG: {len(missing)} removed file(s) not listed in any group: {missing}")

print("wrote", OUT_MD)
