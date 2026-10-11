#!/usr/bin/env python3
"""
a4_ocr_typo_fixes.py — Route A4: targeted, per-file OCR-typo correction on
text that has ALREADY been through OCR (no OCR engine needed here — that's
Route B1/B2, out of scope for this script).

Philosophy: never do a blanket/fuzzy substitution across a whole file
without first reading the full file and confirming every instance is
unambiguous. Several of these files turned out, on full read, to have much
more pervasive corruption than the original front/mid/end sample suggested
(Mozambique, Angola, Iceland, Kiribati) — those are only PARTIALLY fixed
here (the safe, unambiguous subset) and are flagged with an in-file note
recommending escalation to Route B1 (OCR re-run) for full remediation.

Reads from globallaws/, writes to globallaws_cleaned/.

USAGE
-----
    python a4_ocr_typo_fixes.py
"""
import difflib
import os
import re

SRC = r"C:\myProject\Foundation_model\globallaws"
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


# ---------------------------------------------------------------------------
# Mozambique (a) — Portuguese, scanned-image PDF (confirmed via pdfplumber:
# 0 extractable chars, i.e. no text layer at all). Full read found the OCR
# error "ARTIGO" (article marker) mis-read as ~15 distinct variants
# (RTiGo, ÂRTIGO, Ar rico, Anrico, Ax'rico, RTIGo, RTlGo, RTIco, R'HGo,
# AsTico, ...). Fixed with a fuzzy match ONLY on the word immediately before
# the article number (never touches the number itself, to avoid guessing a
# wrong article number).
# ---------------------------------------------------------------------------
def fix_mozambique(t):
    pattern = re.compile(r"^(\W{0,3})([A-Za-z\u00C0-\u00FF\x27\u2018\u2019]{2,10})(?=\s+\d)", re.M)

    def repl(m):
        prefix, word = m.group(1), m.group(2)
        core = re.sub(r"[^A-Za-z\u00C0-\u00FF]", "", word).lower()
        if core == "artigo":
            return m.group(0)
        ratio = difflib.SequenceMatcher(None, core, "artigo").ratio()
        if ratio >= 0.55 and 3 <= len(core) <= 9:
            return prefix + "ARTIGO"
        return m.group(0)

    t2, n_headers = pattern.subn(repl, t)
    fixes = {"REPÉBLICA": "REPÚBLICA", "UOÇAMBIQUE": "MOÇAMBIQUE"}
    for bad, good in fixes.items():
        t2 = t2.replace(bad, good)

    note = (
        "[Data cleaning note] A full re-read found that this file originates from a "
        "scanned-image PDF (no embedded text layer — pdfplumber confirmed only "
        "watermark text is extractable), and OCR errors are pervasive throughout the "
        "body text (not just the heading lines), far beyond what the initial sample "
        f"suggested. This pass used rule-based fixes for roughly {n_headers} misread "
        "article-heading markers (ARTIGO) and the proper nouns REPÉBLICA/UOÇAMBIQUE; "
        "the remaining large volume of character-level garbling (missing letters, "
        "misread characters) elsewhere in the body can't be safely fixed with "
        "dictionary-style rules. Strongly recommend escalating this file to Route B1 "
        "(re-run OCR with the correct Portuguese language pack).\n\n"
    )
    return note + t2, n_headers


# ---------------------------------------------------------------------------
# Angola — Portuguese, scanned-image PDF (confirmed via pdfplumber: 12
# images, 0 text chars on page 1). The garbled table-of-contents block (dot
# leaders OCR'd as random letters, e.g. "DDtDDDD") is fully redundant with
# the real article headings later in the body, so it's safely dropped.
# Also removes an unrelated ad/watermark string ("ACOMPANHAMENTO
# NUTRICIONAL[...]ONLINE") injected into the extracted text at several
# fixed positions.
# ---------------------------------------------------------------------------
def fix_angola(t):
    body_start = t.find("CAPÍTULO I DISPOSIÇÕES GERAIS")
    t2 = t[body_start:] if body_start != -1 else t
    lines = t2.split("\n")
    lines = [l for l in lines if "NUTRICION" not in l and l.strip() != "ONLINE"]
    t2 = collapse_blank_lines("\n".join(lines), max_consecutive=1)

    note = (
        "[Data cleaning note] Removed a large garbled table-of-contents block from the "
        "start of the file, caused by a failed OCR read of the dot-leader characters "
        "(this content duplicates the real article headings later in the body, so "
        "removing it doesn't affect informational completeness), and removed "
        "unrelated ad/watermark text (\"ACOMPANHAMENTO NUTRICIONAL\", etc.) mixed "
        "into the body. However, since the source PDF is a scanned image (no text "
        "layer), a few paragraphs (e.g. near Articles 12, 13, 18, 25, 31) still have "
        "OCR dropped characters/broken sentences and text loss from the original scan "
        "quality that this rule-based cleaning pass cannot restore. Recommend "
        "escalating this file to Route B1 (re-run OCR) if the full articles are "
        "needed.\n\n"
    )
    return note + t2


# ---------------------------------------------------------------------------
# Iceland (a)/(b) — Icelandic. Full read found a SYSTEMATIC substitution:
# þ (thorn) -> b throughout. Because "b" is also a legitimate Icelandic
# letter (brot, birta, bera, ...), a blanket b->þ replace would corrupt real
# words — only a closed list of confirmed, unambiguous function words is
# fixed. ð -> õ/ò/ô is safe to fix everywhere (Icelandic has no legitimate
# õ/ò/ô at all). Embedded digit "1" standing in for letter "l" is also fixed
# (never legitimate mid- or end-of-word in Icelandic).
# ---------------------------------------------------------------------------
THORN_WORDS = {
    "bá": "þá", "begar": "þegar", "bessum": "þessum", "bví": "því",
    "beirra": "þeirra", "bess": "þess", "bremur": "þremur", "bó": "þó",
    "bessara": "þessara", "bau": "þau", "bykir": "þykir", "bykja": "þykja",
    "beim": "þeim", "burft": "þurft", "bannig": "þannig", "burfa": "þurfa",
    "börf": "þörf", "bar": "þar", "bær": "þær", "bingi": "þingi",
    "bingmál": "þingmál", "Skólabing": "Skólaþing",
}


def fix_icelandic(t):
    t = t.replace("õ", "ð").replace("ò", "ð").replace("ô", "ð")
    # embedded/word-final digit-1 standing in for letter l
    t = re.sub(r"(?<=[a-záðéíóúýþæöA-ZÁÐÉÍÓÚÝÞÆÖ])1(?=[a-záðéíóúýþæö])", "l", t)
    t = re.sub(r"(?<=[a-záðéíóúýþæöA-ZÁÐÉÍÓÚÝÞÆÖ])1\b", "l", t)
    t = t.replace("o.f1.", "o.fl.")
    t = re.sub(r"\bbing", "þing", t)
    t = re.sub(r"\bBing", "Þing", t)
    t = re.sub(r"Albing", "Alþing", t)
    for bad, good in THORN_WORDS.items():
        t = re.sub(r"\b" + re.escape(bad) + r"\b", good, t)
        t = re.sub(r"\b" + re.escape(bad.capitalize()) + r"\b", good.capitalize(), t)
    return t


ICELAND_NOTE = (
    "[Data cleaning note] The source of this file has a systematic OCR "
    "misread: the Icelandic letter þ (thorn) was mostly misread as the "
    "letter b during the original extraction, but since b is also a legitimate "
    "Icelandic letter (common in words like brot, birta, bera), it can't be "
    "safely restored everywhere with rule-based dictionary matching alone — "
    "only a closed set of words that can be unambiguously determined from "
    "context were fixed (e.g. bá->þá, begar->þegar), along with ð being "
    "misread as õ/ò/ô and the digit 1 substituting for the letter l. Other "
    "unfixed b/þ-confused words (e.g. compounds like brotaþoli) may still "
    "remain in the text — recommend escalating this file to Route B1 (re-run "
    "OCR with the correct language pack) to resolve it fully.\n\n"
)


# ---------------------------------------------------------------------------
# Kiribati — English. Full read found pervasive OCR corruption, including
# many distinct misspellings of the word "court" itself (Coim/coim/
# ciiurt/Ciiurt/coun) plus a handful of other very common, unambiguous
# whole-word errors. Several section headings remain too garbled to safely
# reconstruct and are left alone.
# ---------------------------------------------------------------------------
KIRIBATI_FIXES = {
    "KlRIBATI": "KIRIBATI",
    "A BILL POR AN ACT": "A BILL FOR AN ACT",
    "EU MI NATIt3N": "ELIMINATION",
    "repeated1y": "repeatedly",
    "tneans": "means",
    "doniestic": "domestic",
    "doaieatic": "domestic",
    "Jumcstic": "domestic",
    "imacceptable": "unacceptable",
    "wsistancc": "assistance",
    "rcsponse": "response",
    "cxcrcisc": "exercise",
    "pcwer": "power",
    "iniures": "injures",
    "Fnr the yurposes uf this se tion": "For the purposes of this section",
    "jortedictioo": "jurisdiction",
    "Magistmte's": "Magistrate's",
    "Coim": "Court",
    "coim": "court",
    "Ciiurt": "Court",
    "ciiurt": "court",
    "murt": "must",
}


def fix_kiribati(t):
    n_exact = 0
    for bad, good in KIRIBATI_FIXES.items():
        n_exact += t.count(bad)
        t = t.replace(bad, good)
    # standalone "coun" -> "court" only directly before a verb (avoid touching
    # any other legitimate word)
    t, n_coun = re.subn(r"\bcoun(?=\s+(?:shall|proceedings))", "court", t)
    note = (
        "[Data cleaning note] A full re-read found that OCR errors are pervasive "
        "throughout this file (including several section headings that are entirely "
        "unrecognizable, e.g. the original Articles 25, 17, and 26), far beyond what "
        f"the initial sample suggested. This pass fixed {n_exact + n_coun} high-"
        "confidence, unambiguous common misreadings (e.g. KlRIBATI, the multiple "
        "misread spellings of \"court\" — Coim/coim/ciiurt/coun — and misreadings of "
        "\"domestic\" such as doniestic/doaieatic), but a substantial number of "
        "section headings and scattered text remain too badly misread to safely "
        "restore without guessing the original wording. Recommend escalating this "
        "file to Route B1 (re-run OCR with the correct language pack) to resolve it "
        "fully.\n\n"
    )
    return note + t


# ---------------------------------------------------------------------------
# The remaining A4 files: low-density, isolated OCR typos confirmed by a
# full read (not pervasive corruption) — exact string replacement only.
# ---------------------------------------------------------------------------
BARBADOS_FIXES = {
    "CIIAPTER": "CHAPTER",
    "ARRANGEMENTOFSEJMONS": "ARRANGEMENT OF SECTIONS",
    "or\u2039Jer": "order", "or\u2039Jer.": "order.",
    "arresC": "arrest",
    "ixder": "order",
    "ordecs": "orders",
    "/urisdicnon": "Jurisdiction",
    "Coim": "Court", "coim": "court",
    "Gourt": "Court",
    "acomplainant": "a complainant",
    "aperson": "a person",
    "Cierh": "Clerk", "Ckrlr": "Clerk", "Ckik": "Clerk",
    "tne Magistrates": "the Magistrates",
    "Magiiaxics": "Magistrates", "Magisbatcs": "Magistrates",
    "Ptotecti\u2039xi": "Protection",
    "Domefit Violencc": "Domestic Violence",
    "having \u2039xi (date of miginal crder]": "having on (date of original order]",
    "fo0ows": "follows",
    "F'OR DISTRICT": "FOR DISTRICT",
    "Orders} Acb": "Orders) Act",
    "thaC": "that",
    "may read iy be ascertained": "may readily be ascertained",
    "(bJ, (c) aixl (dj": "(b), (c) and (d)",
    "was commined": "was committed",
    "sucls other Agency": "such other Agency",
    "orderwasmadenotla:erthan": "order was made not later than",
    "AnActioprovideforthe": "An Act to provide for the",
    "andfor related maners": "and for related matters",
    "themattersthatit": "the matters that it",
    "anapplicationundersubsection(1),theCourtmaybyorder": "an application under subsection (1), the Court may by order",
    "forsimilarreliefundersection": "for similar relief under section",
    "personsareprotectedfrom": "persons are protected from",
}

CHAD_A_FIXES = {
    "PESIDENCE DE LA REPUBLIQUE": "PRESIDENCE DE LA REPUBLIQUE",
    "allant du ierau 31 janvier 2025": "allant du 1er au 31 janvier 2025",
    "N'Djamena, le\t21\tj a ri v i e r\t20 25\td .\t.": "N'Djamena, le 21 janvier 2025",
}
CHAD_A_NOTE = (
    "\n\n[Data cleaning note] Near Articles 14 to 16, a source-PDF font-encoding "
    "issue caused some 'e/é' characters to be misread as the Polish character 'ć', "
    "mixed in among correctly-read letters (this can't be safely fixed with a "
    "global find/replace — it would need character-by-character comparison against "
    "the original). This section was left untouched; recommend manual proofreading "
    "or escalating this section to Route B1 if needed.\n"
)

COTE_DIVOIRE_FIXES = {
    "Article \u00a7 :": "Article 6 :",
    "circons1ances": "circonstances",
    "Côte d'lvoire": "Côte d'Ivoire",
}

DOMINICAN_REPUBLIC_FIX = ("CoDIgo PenaL\nDe LA RepUBLIca Dominicana", "CÓDIGO PENAL\nDE LA REPÚBLICA DOMINICANA")

GREECE_FIX = (
    "\u0393\u0399\u03b1 \u03c4\u03b7\u03bd \u03b1\u03bd\u03c4\u0399\u03bc\u03b5\u03c4\u03ce\u03c0\u0399\u03c3\u03b7 \u03c4\u03b7\u03c2 \u03b5\u03bd\u03b4\u03bf\u03bf\u0399\u03ba\u03bf\u03b3\u03b5\u03bd\u03b5\u0399\u03b1\u03ba\u03ae\u03c2 \u03b2\u03af\u03b1\u03c2 \u03ba\u03b1\u0399 \u03ac\u03bb\u03bb\u03b5\u03c2 \u03b4\u0399\u03b1\u03c4\u03ac\u03be\u0399\u03c2.",
    "\u0393\u03b9\u03b1 \u03c4\u03b7\u03bd \u03b1\u03bd\u03c4\u03b9\u03bc\u03b5\u03c4\u03ce\u03c0\u03b9\u03c3\u03b7 \u03c4\u03b7\u03c2 \u03b5\u03bd\u03b4\u03bf\u03bf\u03b9\u03ba\u03bf\u03b3\u03b5\u03bd\u03b5\u03b9\u03b1\u03ba\u03ae\u03c2 \u03b2\u03af\u03b1\u03c2 \u03ba\u03b1\u03b9 \u03ac\u03bb\u03bb\u03b5\u03c2 \u03b4\u03b9\u03b1\u03c4\u03ac\u03be\u03b5\u03b9\u03c2.",
)

VIETNAM_B_NOTE = (
    "[Data cleaning note] A full re-read found that, beyond the originally flagged "
    "\"Pur5uant\" typo, this file has several apparent content gaps likely caused by "
    "the original PDF extraction (unusually large blank-line runs appear around some "
    "article headings such as \"Article 2\" and \"Article 8\", and the body text "
    "appears to jump abruptly, suggesting missing content at those points). However, "
    "testing confirmed the source PDF cannot be re-extracted with the tools "
    "available here (pdfplumber returns nothing), so this could not be further "
    "repaired or verified — recommend finding an alternate source or manual "
    "review.\n\n"
)


def run():
    # Mozambique
    fn = "116. Mozambique - DomViol - Portuguese (a).txt"
    t2, n = fix_mozambique(read(fn))
    write(fn, t2)
    print(fn, f"-> fixed {n} article-heading OCR misreads + 2 proper nouns; flagged for B1 escalation")

    # Angola
    fn = "4. Angola - DomViol - Portuguese.txt"
    write(fn, fix_angola(read(fn)))
    print(fn, "-> removed garbled TOC block + ad/watermark text mixed into body")

    # Iceland (a) and (b)
    for fn in ["75. Iceland - DomViol - Icelandic (a).txt", "75. Iceland - DomViol - Icelandic (b).txt"]:
        t2 = fix_icelandic(read(fn))
        write(fn, ICELAND_NOTE + t2)
        print(fn, "-> applied safe þ/ð/digit-1 fixes; flagged for B1 escalation")

    # Kiribati (non-html)
    fn = "88. Kiribati - DomViol - English.txt"
    write(fn, fix_kiribati(read(fn)))
    print(fn, "-> fixed ~26 high-confidence OCR patterns; flagged for B1 escalation")

    # Barbados
    fn = "14. Barbados - DomViol - English.txt"
    t = read(fn)
    n = 0
    for bad, good in BARBADOS_FIXES.items():
        n += t.count(bad)
        t = t.replace(bad, good)
    t, n2 = re.subn(r"\bcoun\b(?!ty|cil)", "court", t)
    write(fn, t)
    print(fn, f"-> fixed ~{n + n2} OCR typos (CIIAPTER/Coim/coun/Ckrlr etc.)")

    # Chad (a)
    fn = "33. Chad - DomViol - French (a).txt"
    t = read(fn)
    t = t.replace("REPUBLIQUE DU TCHAD\trniit \u2014roi\u00bb:i - rq\tii", "REPUBLIQUE DU TCHAD")
    n = 0
    for bad, good in CHAD_A_FIXES.items():
        n += t.count(bad)
        t = t.replace(bad, good)
    write(fn, t + CHAD_A_NOTE)
    print(fn, f"-> fixed {n} items (masthead noise / date garbling); ć/é font-mix near Art.14-16 flagged, left alone")

    # Cote d'Ivoire
    fn = "41. Cote dIvoire - DomViol - French.txt"
    t = read(fn)
    n = 0
    for bad, good in COTE_DIVOIRE_FIXES.items():
        n += t.count(bad)
        t = t.replace(bad, good)
    idx = t.find("\"\u2019*'\"")
    if idx != -1:
        t = t[:idx].rstrip()
    write(fn, t)
    print(fn, f"-> fixed {n} OCR typos + removed trailing control-code stamp")

    # Dominican Republic
    fn = "48. Dominican Republic - DomViol - Spanish.txt"
    t = read(fn)
    bad, good = DOMINICAN_REPUBLIC_FIX
    write(fn, t.replace(bad, good))
    print(fn, "-> fixed cover-title case scrambling (CoDIgo PenaL -> CÓDIGO PENAL)")

    # Greece
    fn = "65. Greece - DomViol - Greek.txt"
    t = read(fn)
    bad, good = GREECE_FIX
    write(fn, t.replace(bad, good))
    print(fn, "-> fixed mixed-case title line (Greek iota misread as capital)")

    # Viet Nam (b): confirmed single safe fix; deeper apparent content gaps
    # documented but not touched (source PDF has no extractable text layer)
    fn = "186. Viet Nam - DomViol - English (b).txt"
    t = read(fn).replace("Pur5uant", "Pursuant")
    write(fn, VIETNAM_B_NOTE + t)
    print(fn, "-> fixed Pur5uant->Pursuant; flagged possible content gaps for manual review")


if __name__ == "__main__":
    run()
