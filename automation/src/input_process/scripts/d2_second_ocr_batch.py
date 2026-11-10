#!/usr/bin/env python3
"""
d2_second_ocr_batch.py — Resolves a second batch of re-OCR'd files added
to globallaw_OCR_txt/ after the first round (see d1_ocr_rerun_resolution.py
and CLEANING_REPORT.md's "OCR re-run round" section): Morocco, Egypt, and
Guyana, all originally quarantined under "Source PDF has no recoverable
content / needs OCR" (route B3).

Findings after re-review:

  - Morocco: OCR problem fully fixed — the reversed-RTL-order issue is
    gone, text reads correctly. But the content itself is a "ورقة تقنية"
    (technical fact sheet) *summarizing* Law 103.13, not the law's
    verbatim articles. This is a content-scope mismatch (wrong content
    type), not a text-quality defect — same treatment as the existing
    HOLD route (e.g. the US/Italy "summary instead of statute" entries):
    accepted as-is, moved to globallaws_cleaned/.

  - Guyana: OCR problem fixed — the source is Guyana's Family Violence
    Act 2024 (which repeals the old Domestic Violence Act, see its
    s.57), all 58 substantive sections plus 6 annexed forms are present
    and read cleanly. One confined defect: Form 4's body text (an
    application to vary/revoke/extend a protection order) degenerates
    into a garbled, nonsensically-repeated run of "(name and address of
    the person against whom the order was made)" — clearly OCR
    corruption, not real content, and not attempted to be reconstructed
    here. Since this is confined to one paragraph of one of six annexed
    procedural forms and the entire substantive Act (Parts I-VI,
    ss.1-58) is complete and correct, this is accepted as clean and
    moved to globallaws_cleaned/ with the defect flagged in
    CLEANING_REPORT.md for awareness (not fixed).

  - Egypt: OCR meaningfully improved — no longer "unreadable output" as
    originally quarantined — but a full read finds confirmed residual
    corruption: several numbered section headers alternate between the
    correct "( المادة ... )" ("Article...") and a garbled "( اللغة ... )"
    ("Language...") for what are clearly the same kind of section
    marker; the year in the opening line ("لسنة ٢٠١٢") is inconsistent
    with the 2023 Gregorian date and the sitting Prime Minister named at
    the end of the document; and the very first line is unintelligible.
    This stays "partial" in globallaws_removed/, with the improved (but
    still confirmed-corrupt) text replacing the old, worse version, and
    an updated reason.

Reads from globallaw_OCR_txt/. Writes Morocco and Guyana to
globallaws_cleaned/ (and removes their stale entries from
globallaws_removed/). Writes Egypt's improved-but-still-partial text
back into globallaws_removed/, replacing its stale entry.

USAGE
-----
    python d2_second_ocr_batch.py
"""
import os

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


def remove_from_removed(fn):
    p = os.path.join(REMOVED, fn)
    if os.path.exists(p):
        os.remove(p)
        return True
    return False


def run():
    log = []

    # Morocco: OCR fixed, but wrong content type (summary, not statute) —
    # accepted as content-scope mismatch, same as the HOLD route.
    fn = "115. Morocco - DomViol - Arabic.txt"
    t = read(fn)
    write(CLEANED, fn, t)
    remove_from_removed(fn)
    log.append((fn, "OCR fixed (RTL order issue resolved); content is a technical "
                    "fact sheet summarizing Law 103.13, not the statute itself — "
                    "content-scope mismatch, accepted as-is -> globallaws_cleaned/"))

    # Guyana: OCR fixed; substantive Act complete and clean, one confined
    # defect in an annexed form (not fixed, flagged in report only).
    fn = "70. Guyana - DomViol - English.txt"
    t = read(fn)
    write(CLEANED, fn, t)
    remove_from_removed(fn)
    log.append((fn, "OCR fixed; Family Violence Act 2024 complete and clean "
                    "(ss.1-58 + Forms 1-3, 5-6), one confined OCR-garbled run-on "
                    "in Form 4's body text left unfixed and flagged in report "
                    "-> globallaws_cleaned/"))

    # Egypt: OCR improved but confirmed residual corruption remains — stays
    # quarantined, improved text replaces the old (worse) quarantined copy.
    fn = "50. Egypt - DomViol - Arabic.txt"
    t = read(fn)
    write(REMOVED, fn, t)
    log.append((fn, "OCR meaningfully improved (previously fully unreadable), but "
                    "confirmed residual corruption remains: numbered section headers "
                    "inconsistently read as \"Language\" (اللغة) instead of \"Article\" "
                    "(المادة), the opening line is unintelligible, and the stated year "
                    "conflicts with the document's own later date/signatory -- stays "
                    "quarantined in globallaws_removed/ (partial)"))

    print(f"Processed {len(log)} files:\n")
    for fn, note in log:
        print(f"  {fn}\n    -> {note}\n")


if __name__ == "__main__":
    run()
