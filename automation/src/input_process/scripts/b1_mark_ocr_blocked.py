#!/usr/bin/env python3
"""
b1_mark_ocr_blocked.py — Route B1: document why 12 files could not be
OCR'd in this environment (scanned-image PDFs, or PDFs/HTML with broken
custom font encodings, that need a real OCR pass to recover).

No OCR engine was available:
  1. No administrator rights — could not install Tesseract via a system
     package manager (choco).
  2. A portable, no-install Tesseract executable was located and a
     download was attempted with approval, but the platform's own safety
     layer blocked downloading and executing an unverified external .exe.
  3. Windows' built-in OCR API (Windows.Media.Ocr) was checked as a
     no-download alternative, but only the Traditional Chinese language
     pack is installed on this machine; adding further language packs
     also requires administrator rights.

Reads from globallaws/, writes to globallaws_cleaned/. Content is left
unchanged aside from the prepended note.

USAGE
-----
    python b1_mark_ocr_blocked.py
"""
import os

SRC = r"C:\myProject\Foundation_model\globallaws"
OUT_DIR = r"C:\myProject\Foundation_model\globallaws_cleaned"
os.makedirs(OUT_DIR, exist_ok=True)

# filename -> language needed for OCR (name + ISO 639-2/T code used by Tesseract)
FILES_AND_LANGUAGES = {
    "111. Micronesia - DomViol - English.txt": "English (eng)",
    "131. Papua New Guinea - DomViol - English.txt": "English (eng)",
    "172. Togo - DomViol - French.txt": "French (fra)",
    "47. Dominica - DomViol - English (a).txt": "English (eng)",
    "61. Gambia - DomViol - English.txt": "English (eng)",
    "83. Jamaica - DomViol - English (a).txt": "English (eng)",
    "84. Japan - DomViol - Japanese.txt": "Japanese (jpn)",
    "183. Uzbekistan - DomViol - Russian (b).txt": "Russian (rus)",
    "19. Bhutan - DomViol - Bhutanese.txt": (
        "Dzongkha / Bhutanese (dzo — Tesseract's official language-pack "
        "coverage for this script is uncertain and would need checking)"
    ),
    "62. Georgia - DomViol - Georgian.txt": "Georgian (kat)",
    "81. Israel - DomViol - Hebrew.txt": "Hebrew (heb)",
    "97. Liberia - DomViol - English.txt": "English (eng)",
}

NOTE_TEMPLATE = (
    "[Data cleaning note] This file's original problem requires re-recognition "
    "with an OCR engine (the source PDF is a scanned image, or the text uses a "
    "non-standard font encoding), but no usable OCR engine is available in this "
    "environment:\n"
    "(1) No administrator rights on this machine — could not install Tesseract "
    "via choco or a system package manager;\n"
    "(2) An attempt to download a portable, no-install version of the Tesseract "
    "executable was blocked by the system's security controls (downloading and "
    "running an executable from an unverified external source is not permitted);\n"
    "(3) Windows' built-in OCR API (Windows.Media.Ocr) only has the Traditional "
    "Chinese language pack installed, not the language this file needs "
    "({language}), and installing an additional language pack likewise requires "
    "administrator rights.\n"
    "This file's content is left unchanged; recommend re-running OCR with the "
    "correct language pack once an environment with OCR tooling (or "
    "administrator rights) is available.\n\n"
)


def run():
    for fn, language in FILES_AND_LANGUAGES.items():
        src_path = os.path.join(SRC, fn)
        with open(src_path, encoding="utf-8") as f:
            t = f.read()
        note = NOTE_TEMPLATE.format(language=language)
        with open(os.path.join(OUT_DIR, fn), "w", encoding="utf-8") as f:
            f.write(note + t.strip() + "\n")
        print(fn, "-> marked OCR-blocked, original content kept, note prepended")


if __name__ == "__main__":
    run()
