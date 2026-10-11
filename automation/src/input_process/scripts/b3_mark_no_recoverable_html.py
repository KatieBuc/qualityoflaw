#!/usr/bin/env python3
"""
b3_mark_no_recoverable_html.py — Route B3 (HTML subset): document why 4
HTML-sourced files have no recoverable statute text in the saved source
page (page chrome / UI text only, or a failed-scrape placeholder). These
are exactly the 4 filenames in extract_html.py's NO_RECOVERABLE_CONTENT
set, which extract_html.py deliberately skips re-extracting for this
reason.

The body kept below the note is the existing best-effort chrome-stripped
text already present in globallaws_cleaned/ (there is no more content to
recover from the saved HTML — see the note for each file). This script
only replaces the explanatory note; it does not touch the body.

Reads from globallaws_cleaned/ (current best-effort body only, note
stripped), writes back to globallaws_cleaned/.

USAGE
-----
    python b3_mark_no_recoverable_html.py
"""
import os
import re

OUT_DIR = r"C:\myProject\Foundation_model\globallaws_cleaned"

NOTES = {
    "89. Korea - DomViol - Korean - html.txt": (
        "The saved static HTML source is a JavaScript-rendered page; the statute "
        "text itself is not present in the saved HTML (the whole page's plain "
        "text is only about 1,500 characters, all menu/help text, with no article "
        "content at all). Neither trafilatura nor other parsing methods can "
        "recover the body text from the existing file — this can't be resolved "
        "with rule-based cleaning. Recommend escalating to Route B3 (re-capture a "
        "version of the page that includes the body text, using a method that "
        "supports JavaScript rendering)."
    ),
    "82. Italy - DomViol - Italian (a) - html.txt": (
        "The saved static HTML source contains only the official-gazette page "
        "frame (Gazzetta Ufficiale header/footer navigation); the statute "
        "articles themselves are not present in the saved HTML (the whole page's "
        "plain text is only 535 characters). The body text cannot be recovered "
        "from the existing file. Recommend escalating to Route B3 (re-crawl a "
        "version of the page that includes the body text, e.g. by switching to "
        "the \"Testo vigente\" or full-article view)."
    ),
    "73. Hong Kong - DomViol - English - html.txt": (
        "The saved static HTML source contains only the ordinance-lookup "
        "interface and print/search help text; the ordinance body text itself is "
        "not present in the saved HTML (the whole page's plain text is only about "
        "3,900 characters, closely matching the noise already described in the "
        "original-issue note). The body is likely loaded dynamically via AJAX and "
        "cannot be recovered from the existing file. Recommend escalating to "
        "Route B3 (re-capture a version of the page that includes the body "
        "text)."
    ),
    "10. Azerbaijan - DomViol - Azerbaijani - html.txt": (
        "Re-checked the original HTML file (Domestic_Violence folder) and "
        "confirmed the whole page's plain-text content is only 24 characters "
        "(matching the existing file) — the source website's page was itself "
        "blank, or the scrape failed at capture time, and no statute text was "
        "ever captured. The body text cannot be recovered by re-parsing or "
        "cleaning the existing file, and this environment cannot reconnect to "
        "the source website to re-crawl it. Recommend marking this as "
        "\"unavailable\" until network access or an alternate source becomes "
        "available."
    ),
}

NOTE_PAT = re.compile(r"^\[[^\]]*\][^\n]*\n\n", re.S)


def run():
    for fn, note_body in NOTES.items():
        path = os.path.join(OUT_DIR, fn)
        with open(path, encoding="utf-8") as f:
            t = f.read()
        body = NOTE_PAT.sub("", t, count=1)
        note = f"[Data cleaning note] {note_body}\n\n"
        with open(path, "w", encoding="utf-8") as f:
            f.write(note + body.strip() + "\n")
        print(fn, "-> marked no-recoverable-content, existing best-effort body kept, note replaced")


if __name__ == "__main__":
    run()
