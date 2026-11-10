#!/usr/bin/env python3
"""
d1_ocr_rerun_resolution.py — Resolves the 20 files that were re-OCR'd/
re-extracted into globallaw_OCR_txt/ (the 12 "needs OCR re-run" + 8
"partially fixed" files originally quarantined in globallaws_removed/,
per CLEANING_REPORT.md). This is a second pipeline round, separate from
the original A1-A5/B1-B3/OK/HOLD routes in file_processing_map.py.

What this script does, per file:

  1. CONFIRMED CLEAN (11 files) — the re-OCR fully fixed the original
     problem. Copied as-is from globallaw_OCR_txt/ to globallaws_cleaned/.

  2. WRONG DOCUMENT SUBSTITUTED, ACCEPTED (3 files) — the re-extraction
     pulled in an entire unrelated statute (Togo: full Penal Code;
     Uzbekistan (b): Code on Administrative Responsibility; Iceland (b):
     General Penal Code) instead of a domestic-violence-specific law.
     Per product decision, this is treated like the existing HOLD route
     (content-scope mismatch is not a text-quality defect) and accepted
     as-is — copied to globallaws_cleaned/. Re-sourcing the correct
     document is out of scope for this round.

  3. IRRELEVANT CONTENT REMOVED (3 files) — mechanical removal of
     verbatim junk with no attempt to recover or reconstruct anything:
       - Iceland (a) and (b): both end with several hundred lines of
         Althingi (Icelandic parliament) website navigation menu/footer
         chrome, appended after the actual law text. Cut everything from
         the "Þú ert hér: Forsíða > Lagasafn > Lög" breadcrumb onward.
       - Angola: a Portuguese nutrition-consultation ad banner
         ("ACOMPANHAMENTO NUTRICIONAL(X) ONLINE" / "AC0MPANHAMENTO..." /
         "ACCOMPANHAMENTO..." + "Cansado(a) de dietas restritivas?...
         AGENDE SUA CONSULTA AGORA") is injected 15 times through the
         body. Removed by exact line match; no other content touched.
       - Viet Nam (b): the source is a "Pro Membership" paywall page;
         106 standalone "..." lines and 43 "Please sign up or sign in
         to your Pro Membership to see English documents." lines mark
         where real statute text was withheld. Those marker lines are
         removed — the missing content itself is NOT reconstructed, so
         this file stays "partial" and in globallaws_removed/ with an
         updated reason.

  4. STILL UNRESOLVED, UNTOUCHED (4 files) — left exactly as they are in
     globallaws_removed/ from the original round; only CLEANING_REPORT.md
     prose changes for these:
       - Israel: still a comma/number soup with no recoverable Hebrew
         text (confirmed after this OCR re-run attempt too) — the source
         PDF itself is damaged. No text-quality fix is possible here
         without a better source PDF.
       - Uzbekistan (b) ALSO has 70 literal stray "None" lines (an
         extraction-script artifact) in addition to being the wrong
         document; left as-is along with the wrong-document acceptance
         per item 2 above — not cleaned this round.
       - Mozambique (a), Liberia: confirmed to have scrambled
         article/section order (interleaved multi-document or
         multi-column extraction). Not addressed this round.

Reads from globallaw_OCR_txt/. Writes resolved files to globallaws_cleaned/
and removes their stale entries from globallaws_removed/. Viet Nam (b)'s
de-junked-but-still-partial output replaces its stale entry in
globallaws_removed/ (it does not move to globallaws_cleaned/).

USAGE
-----
    python d1_ocr_rerun_resolution.py
"""
import os
import re

SRC = r"C:\myProject\Foundation_model\GlobalLaws\globallaw_OCR_txt"
CLEANED = r"C:\myProject\Foundation_model\GlobalLaws\globallaws_cleaned"
REMOVED = r"C:\myProject\Foundation_model\GlobalLaws\globallaws_removed"
os.makedirs(CLEANED, exist_ok=True)
os.makedirs(REMOVED, exist_ok=True)


def read(fn):
    with open(os.path.join(SRC, fn), encoding="utf-8") as f:
        return f.read()


def write(dest_dir, fn, text):
    with open(os.path.join(dest_dir, fn), "w", encoding="utf-8") as f:
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


ICELAND_NAV_MARKER = "Þú ert hér: Forsíða > Lagasafn > Lög"

ANGOLA_AD_TITLE_VARIANTS = {
    "ACOMPANHAMENTO NUTRICIONALX ONLINE",
    "ACOMPANHAMENTO NUTRICIONAL ONLINE",
    "AC0MPANHAMENTO NUTRICIONAL ONLINE",
    "ACCOMPANHAMENTO NUTRICIONAL ONLINE",
}
ANGOLA_AD_TAGLINE = (
    "Cansado(a) de dietas restritivas? Recupere peso saudável sem culpa. "
    "AGENDE SUA CONSULTA AGORA"
)

VIETNAM_PAYWALL_LINES = {
    "...",
    "Please sign up or sign in to your Pro Membership to see English documents.",
}


def strip_iceland_nav_chrome(text):
    idx = text.find(ICELAND_NAV_MARKER)
    if idx == -1:
        return text, False
    return text[:idx].rstrip(), True


def strip_angola_ads(text):
    lines = text.split("\n")
    kept, removed_count = [], 0
    for l in lines:
        stripped = l.strip()
        if stripped in ANGOLA_AD_TITLE_VARIANTS or stripped == ANGOLA_AD_TAGLINE:
            removed_count += 1
            continue
        kept.append(l)
    t2 = collapse_blank_lines("\n".join(kept), max_consecutive=1)
    return t2, removed_count


def strip_vietnam_paywall_markers(text):
    lines = text.split("\n")
    kept, removed_count = [], 0
    for l in lines:
        if l.strip() in VIETNAM_PAYWALL_LINES:
            removed_count += 1
            continue
        kept.append(l)
    t2 = collapse_blank_lines("\n".join(kept), max_consecutive=1)
    return t2, removed_count


def remove_from_removed(fn):
    p = os.path.join(REMOVED, fn)
    if os.path.exists(p):
        os.remove(p)
        return True
    return False


CONFIRMED_CLEAN = [
    "111. Micronesia - DomViol - English.txt",
    "131. Papua New Guinea - DomViol - English.txt",
    "19. Bhutan - DomViol - Bhutanese.txt",
    "33. Chad - DomViol - French (a).txt",
    "41. Cote dIvoire - DomViol - French.txt",
    "47. Dominica - DomViol - English (a).txt",
    "61. Gambia - DomViol - English.txt",
    "62. Georgia - DomViol - Georgian.txt",
    "83. Jamaica - DomViol - English (a).txt",
    "84. Japan - DomViol - Japanese.txt",
    "88. Kiribati - DomViol - English.txt",
]

WRONG_DOCUMENT_ACCEPTED_NO_CHANGE = [
    "172. Togo - DomViol - French.txt",
    "183. Uzbekistan - DomViol - Russian (b).txt",
]


def run():
    log = []

    # 1. Confirmed-clean files: copy through unchanged
    for fn in CONFIRMED_CLEAN:
        t = read(fn)
        write(CLEANED, fn, t)
        remove_from_removed(fn)
        log.append((fn, "confirmed clean -> globallaws_cleaned/ (copied as-is)"))

    # 2a. Wrong document, accepted as-is (Togo, Uzbekistan (b))
    for fn in WRONG_DOCUMENT_ACCEPTED_NO_CHANGE:
        t = read(fn)
        write(CLEANED, fn, t)
        remove_from_removed(fn)
        log.append((fn, "wrong document substituted; accepted as-is (content-scope "
                        "note only, per HOLD-route convention) -> globallaws_cleaned/"))

    # 2b. Iceland (b): wrong document (general Penal Code) AND has the same nav-chrome
    # pollution as (a). Accept the wrong-document scope per item 2, but still strip the
    # mechanical nav-chrome junk since it costs nothing and the result is objectively cleaner.
    fn = "75. Iceland - DomViol - Icelandic (b).txt"
    t = read(fn)
    t2, cut = strip_iceland_nav_chrome(t)
    write(CLEANED, fn, t2)
    remove_from_removed(fn)
    log.append((fn, f"wrong document (General Penal Code, not DV-specific); accepted "
                    f"as-is for scope + nav-chrome stripped ({'cut' if cut else 'no marker found'}, "
                    f"{len(t)}->{len(t2)} chars) -> globallaws_cleaned/"))

    # 3a. Iceland (a): strip trailing Althingi site nav chrome
    fn = "75. Iceland - DomViol - Icelandic (a).txt"
    t = read(fn)
    t2, cut = strip_iceland_nav_chrome(t)
    if not cut:
        raise SystemExit(f"BUG: nav marker not found in {fn}")
    write(CLEANED, fn, t2)
    remove_from_removed(fn)
    log.append((fn, f"removed trailing Althingi site nav chrome ({len(t)}->{len(t2)} chars) "
                    f"-> globallaws_cleaned/"))

    # 3b. Angola: remove injected nutrition-ad text blocks
    fn = "4. Angola - DomViol - Portuguese.txt"
    t = read(fn)
    t2, n = strip_angola_ads(t)
    if n == 0:
        raise SystemExit(f"BUG: no ad lines matched in {fn}")
    write(CLEANED, fn, t2)
    remove_from_removed(fn)
    log.append((fn, f"removed {n} injected ad lines (title+tagline pairs) "
                    f"({len(t)}->{len(t2)} chars) -> globallaws_cleaned/"))

    # 3c. Viet Nam (b): remove paywall placeholder marker lines; missing content is
    # NOT reconstructed. Stays quarantined (partial) — de-junked version replaces the
    # stale entry in globallaws_removed/, does NOT move to globallaws_cleaned/.
    fn = "186. Viet Nam - DomViol - English (b).txt"
    t = read(fn)
    t2, n = strip_vietnam_paywall_markers(t)
    if n == 0:
        raise SystemExit(f"BUG: no paywall marker lines matched in {fn}")
    write(REMOVED, fn, t2)
    log.append((fn, f"removed {n} paywall placeholder marker lines ('...' / 'Please sign "
                    f"up...') ({len(t)}->{len(t2)} chars); missing statute text NOT "
                    f"reconstructed, stays quarantined in globallaws_removed/ (partial)"))

    # 4. Still unresolved, untouched: Israel, Mozambique (a), Liberia — no action here.
    # (Uzbekistan (b)'s 70 stray "None" lines are also left untouched, bundled with its
    # wrong-document acceptance above.)

    print(f"Processed {len(log)} files:\n")
    for fn, note in log:
        print(f"  {fn}\n    -> {note}\n")


if __name__ == "__main__":
    run()
