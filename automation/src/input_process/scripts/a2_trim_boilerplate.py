#!/usr/bin/env python3
"""
a2_trim_boilerplate.py — Route A2: rule-based removal of boilerplate from
PDF-derived text files that DON'T need re-extraction from source (subscription
price tables, unrelated file-name indexes, print timestamps, distribution
stamps, decorative dividers, leftover website chrome in a print-to-PDF
capture, etc). Each rule below was written after reading the specific file
in full and finding an exact, unambiguous boundary — none of this is
frequency-based or fuzzy (see extract_html.py's "LESSONS LEARNED" for why
that's dangerous for legal text).

Reads from globallaws/, writes to globallaws_cleaned/.

USAGE
-----
    python a2_trim_boilerplate.py

Each block below is independent; comment one out if you want to skip it.
"""
import os
import re

import pdfplumber

SRC = r"C:\myProject\Foundation_model\globallaws"
PDF_SRC = r"C:\myProject\Foundation_model\Domestic_Violence"
OUT = r"C:\myProject\Foundation_model\globallaws_cleaned"
os.makedirs(OUT, exist_ok=True)


def read(fn):
    with open(os.path.join(SRC, fn), encoding="utf-8") as f:
        return f.read()


def write(fn, text):
    with open(os.path.join(OUT, fn), "w", encoding="utf-8") as f:
        f.write(text.strip() + "\n")


def collapse_blank_lines(text, max_consecutive=1):
    lines = text.split("\n")
    out, blank = [], 0
    for l in lines:
        if l.strip() == "":
            blank += 1
            if blank > max_consecutive:
                continue
        else:
            blank = 0
        out.append(l)
    return "\n".join(out)


def normalize_tabs(text):
    """Collapse runs of 2+ tabs/spaces (crude table padding) to a single
    tab. Used only on files where the flagged issue was formatting, not
    content to remove (see Malawi/Spain below) — never deletes text."""
    return re.sub(r"[ \t]{2,}", "\t", text)


def run():
    log = []

    def done(fn, note):
        log.append((fn, note))
        print(fn, "->", note)

    # 1. Panama (a): cut trailing unrelated PDF-filename index
    fn = "130. Panama - DomViol - Spanish (a).txt"
    t = read(fn)
    idx = t.find("PROYECTO DE LEY: 2000_P_106.PDF")
    idx2 = t.rfind("ASAMBLEA NACIONAL", 0, idx)
    t2 = t[:idx2].rstrip()
    write(fn, t2)
    done(fn, f"removed trailing unrelated PDF-filename list ({len(t)}->{len(t2)} chars)")

    # 2. El Salvador (a): strip vLex site chrome front/end
    fn = "51. El Salvador - DomViol - Spanish (a).txt"
    t = read(fn)
    start = t.find("LA ASAMBLEA LEGISLATIVA DE LA REPUBLICA DE EL SALVADOR, CONSIDERANDO:")
    end = t.find("Índice de navegación Contenidos")
    t2 = t[start:end].rstrip()
    write(fn, t2)
    done(fn, f"removed vLex site nav/footer chrome ({len(t)}->{len(t2)} chars)")

    # 3. Gabon: strip site nav front + subscribe footer
    fn = "60. Gabon - DomViol - French.txt"
    t = read(fn)
    start = t.find("JOURNAL OFFICIEL N°131 DU 16 SEPTEMBRE 2021")
    end = t.find("Abonnez-vous au Journal Officiel")
    t2 = t[start:end].rstrip()
    write(fn, t2)
    done(fn, f"removed site nav front / subscription footer ({len(t)}->{len(t2)} chars)")

    # 4. Hungary (a): strip print-timestamp/garbled-language-menu front + cookie footer
    fn = "74. Hungary - DomViol - Hungarian (a).txt"
    t = read(fn)
    start = t.find("2009. évi LXXII. tàrvény", t.find("billentyüket."))
    end = t.find("Másolás a vágólapra")
    t2 = t[start:end].rstrip()
    write(fn, t2)
    done(fn, f"removed print-timestamp/garbled nav front, cookie footer ({len(t)}->{len(t2)} chars)")

    # 5. Malawi: legitimate annexed statutory forms — normalize tab padding only, no deletion
    fn = "102. Malawi - DomViol - English.txt"
    t = read(fn)
    t2 = normalize_tabs(t)
    write(fn, t2)
    done(fn, f"normalized tab spacing in annexed forms (content kept) ({len(t)}->{len(t2)} chars)")

    # 6. Spain (a): annex court-list table is genuine statutory content — normalize tabs only
    fn = "156. Spain - DomViol - Spanish (a).txt"
    t = read(fn)
    t2 = normalize_tabs(t)
    write(fn, t2)
    done(fn, f"normalized tab spacing in annex court table (content kept) ({len(t)}->{len(t2)} chars)")

    # 7. Benin: cut distribution-code stamp + UNDP footer credit
    fn = "18. Benin - DomViol - French.txt"
    t = read(fn)
    idx = t.find("AMPLIATIONS")
    t2 = t[:idx].rstrip()
    write(fn, t2)
    done(fn, f"removed institutional distribution stamp + UNDP credit ({len(t)}->{len(t2)} chars)")

    # 8. Venezuela (b): targeted accent/OCR fixes (verified single occurrences each)
    fn = "185. Venezuela - DomViol - Spanish (b).txt"
    t = read(fn)
    fixes = {
        "tendrãn": "tendrán",
        "Pííblicos": "Públicos",
        "REPÚBLICABOLIVARlANA": "REPÚBLICA BOLIVARIANA",
        "Parágrafo Cínico": "Parágrafo Único",
    }
    t2 = t
    for bad, good in fixes.items():
        t2 = t2.replace(bad, good)
    write(fn, t2)
    done(fn, f"fixed {len(fixes)} accent-mark misreadings")

    # 9. Cabo Verde: collapse blank-line padding, strip InDesign export stamp
    fn = "28. Cabo Verde - DomViol - Portuguese.txt"
    t = read(fn)
    t2 = re.sub(r"LEI VBG Final\.indd\s*\d+\s*[\d\-: ]+", "", t)
    t2 = collapse_blank_lines(t2, max_consecutive=1)
    write(fn, t2)
    done(fn, f"removed InDesign export stamp, collapsed excess blank lines ({len(t)}->{len(t2)} chars)")

    # 10. Algeria: cut front subscription/masthead block only (leave rest —
    # the trailing "statistics table" belongs to a different regulation
    # bundled in the same gazette issue, out of scope for this rule)
    fn = "3. Algeria - DomViol - French.txt"
    t = read(fn)
    start = t.find("2\tJOURNAL OFFICIEL DE LA REPUBLIQUE ALGERIENNE N° 71")
    if start == -1:
        start = t.find("JOURNAL OFFICIEL DE LA REPUBLIQUE ALGERIENNE N° 71")
    t2 = t[start:].strip()
    write(fn, t2)
    done(fn, f"removed subscription/masthead front matter ({len(t)}->{len(t2)} chars); "
             f"NOTE: trailing statistics table belongs to a different regulation "
             f"bundled in the same gazette issue and was deliberately left alone")

    # 11. Central African Republic: cut front masthead, cut trailing asterisk decoration
    fn = "32. Central African Republic - DomViol - French.txt"
    t = read(fn)
    start = t.find("LOI N°06. 032 DU 27 DECEMBRE 2006 PORTANT PROTECTION DE LA FEMME CONTRE LES VIOLENCES")
    end = t.find("***")
    t2 = t[start:end].rstrip()
    write(fn, t2)
    done(fn, f"removed masthead front / decorative asterisks end ({len(t)}->{len(t2)} chars)")

    # 12. Chad (b): re-extract from source PDF with pdfplumber — the root
    # cause of the missing-inter-word-spaces problem was the ORIGINAL
    # extractor's handling of justified text, not something regex can fix
    # reliably after the fact. pdfplumber's own extraction is clean here.
    fn = "33. Chad - DomViol - French (b).txt"
    pdf_path = os.path.join(PDF_SRC, "33. Chad - DomViol - French (b).pdf")
    with pdfplumber.open(pdf_path) as pdf:
        pages_text = [p.extract_text() or "" for p in pdf.pages]
    t2 = "\n\n".join(pages_text)
    t2 = collapse_blank_lines(t2, max_consecutive=1)
    write(fn, t2)
    done(fn, f"re-extracted from source PDF via pdfplumber (fixes missing inter-word spaces); "
             f"new length {len(t2)}")

    # 13. Congo, Rep: cut front masthead, cut trailing printer's imprint
    fn = "39. Congo, Rep - DomViol - French.txt"
    t = read(fn)
    start = t.find("SOMMAIRE")
    end = t.rfind("Imprimé dans les ateliers")
    t2 = t[start:end].rstrip()
    t2 = collapse_blank_lines(t2, max_consecutive=1)
    write(fn, t2)
    done(fn, f"removed masthead front / printer imprint end ({len(t)}->{len(t2)} chars); "
             f"NOTE: this gazette issue bundles several unrelated laws/notices — only "
             f"layout noise was removed, no topic-scoping was attempted here")

    # 14. Finland: cut nav front chrome, cut footer/site-widget chrome
    fn = "58. Finland - DomViol - Swedish.txt"
    t = read(fn)
    start = t.find("Lag om besöksförbud\n")
    end = t.find("Uppdaterad lagstiftning\nBeaktats t.o.m. FörfS 66/2024 (publicerad")
    t2 = t[start:end].rstrip()
    write(fn, t2)
    done(fn, f"removed site nav front / footer widget chrome ({len(t)}->{len(t2)} chars)")

    # 15. Germany: cut footer chrome
    fn = "63. Germany - DomViol - German.txt"
    t = read(fn)
    end = t.find("•\tzum Seitenanfang")
    t2 = t[:end].rstrip()
    write(fn, t2)
    done(fn, f"removed site footer chrome ({len(t)}->{len(t2)} chars)")

    # 16. Grenada: on full-file review, the flagged "dot leaders" turned out
    # to be legitimate blank-line placeholders in annexed statutory Forms
    # (e.g. "Dated this ...... day of ......"), not PDF TOC noise. No
    # change — copied as-is for completeness of the pipeline.
    fn = "66. Grenada - DomViol - English.txt"
    t = read(fn)
    write(fn, t)
    done(fn, "no change — confirmed on full read that the flagged dot-leaders are "
             "legitimate blank signature/date fields in an annexed Form, not noise")

    print(f"\nProcessed {len(log)} files")


if __name__ == "__main__":
    run()
