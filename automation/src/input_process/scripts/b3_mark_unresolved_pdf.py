#!/usr/bin/env python3
"""
b3_mark_unresolved_pdf.py — Route B3 (PDF subset): document why 4 files
could NOT be resolved from the files already in Domestic_Violence/ (per
instruction: no external re-crawling, only use what's already saved).

This does not need an OCR engine — it's the diagnostic write-up of *why*
each file needs one (or an alternate source), based on direct inspection
via pdfplumber. The 4 HTML-based B3 files (Romania, Slovenia (c), Czechia
(d), Azerbaijan) are handled by extract_html.py instead, since 3 of the 4
turned out to be recoverable by re-running the extraction pipeline against
the already-saved HTML.

Reads from globallaws/, writes to globallaws_cleaned/.

USAGE
-----
    python b3_mark_unresolved_pdf.py
"""
import os

SRC = r"C:\myProject\Foundation_model\globallaws"
OUT_DIR = r"C:\myProject\Foundation_model\globallaws_cleaned"
os.makedirs(OUT_DIR, exist_ok=True)

NOTES = {
    "168. Tajikistan - DomViol - English.txt": (
        "Testing the original PDF with pdfplumber found no extractable text at all "
        "(0 characters), confirming that the source file itself is a subscription "
        "paywall page (it only displays a \"Full text is available with an active "
        "Subscription\" message) and does not contain the actual statute text. The "
        "body text cannot be recovered by re-parsing the existing file, and this "
        "environment cannot connect to an alternate free source to re-crawl it. "
        "Recommend marking this as \"unavailable\" until an alternate source or "
        "network access becomes available."
    ),
    "50. Egypt - DomViol - Arabic.txt": (
        "Attempted to re-extract the original PDF with pdfplumber, but because the "
        "PDF uses a non-standard font encoding (no valid ToUnicode mapping), the "
        "re-extracted result was completely unreadable garbage (worse than the "
        "existing version), so it was not adopted. The original PDF is also very "
        "short (only 3 pages) and may only be a subsidiary-regulation fragment "
        "rather than the full statute. This issue would need OCR (converting the "
        "PDF to an image and re-recognizing it) to resolve, but no usable OCR engine "
        "is available in this environment — recommend marking this as \"unavailable\"."
    ),
    "70. Guyana - DomViol - English.txt": (
        "Re-checked the original PDF (53 pages) with pdfplumber and confirmed every "
        "page is a scanned image (6 embedded images per page); only the repeated "
        "running header/footer title text has an extractable text layer (e.g. "
        "\"LAWS OF GUYANA\"), while the statute body itself is entirely images with "
        "no text layer — matching the same problem as the existing txt file. This "
        "issue would need OCR to resolve, but no usable OCR engine is available in "
        "this environment — recommend marking this as \"unavailable\"."
    ),
    "115. Morocco - DomViol - Arabic.txt": (
        "Attempted to re-extract the original PDF with pdfplumber and found the text "
        "comes out in reversed order (Arabic RTL text was extracted in visual order "
        "rather than logical order). Testing showed that a simple character-order "
        "reversal makes the text roughly readable but also scrambles embedded digits "
        "(e.g. statute number 103.13 could become 31.301), risking misreported "
        "reference numbers and other key information, so this unreliable approach "
        "was not adopted. The original PDF is also very short (only 2 pages). This "
        "issue would need OCR to resolve reliably, but no usable OCR engine is "
        "available in this environment — recommend marking this as \"unavailable\"."
    ),
}


def run():
    for fn, note_body in NOTES.items():
        src_path = os.path.join(SRC, fn)
        with open(src_path, encoding="utf-8") as f:
            t = f.read()
        note = f"[Data cleaning note] {note_body}\n\n"
        with open(os.path.join(OUT_DIR, fn), "w", encoding="utf-8") as f:
            f.write(note + t.strip() + "\n")
        print(fn, "-> marked unresolved, original content kept, note prepended")


if __name__ == "__main__":
    run()
