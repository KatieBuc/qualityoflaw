#!/usr/bin/env python3
"""
c1_quarantine_unusable.py — Finalize globallaws_cleaned/ by applying the
quantitative removal decision (CLEANING_CRITERIA.md §3) to every file.

The decision is read from quality_reports/removal_decisions.csv, produced by

    python scripts/assess_quality.py --stage raw
    python scripts/assess_quality.py --stage final

For each file:
  * final_decision == REMOVE -> moved to globallaws_removed/ (quarantine;
    nothing in that folder feeds the next stage). The rule IDs that fired
    (R1-R5, or an R6 review verdict) are listed in the output.
  * final_decision == KEEP   -> stays in (or, with --restore, is moved back
    to) globallaws_cleaned/, with any leading "[Data cleaning note] ..."
    placeholder text stripped.
  * final_decision == REVIEW -> an unresolved numbering-order review (R6).
    The script refuses to apply anything until every REVIEW has a verdict in
    criteria_overrides.csv.

Earlier versions of this script quarantined files from a hand-maintained
status list (file_processing_map.FILE_STATUS). That list is now only a
historical record; the decision comes from the measured metrics.

USAGE
-----
    python scripts/c1_quarantine_unusable.py            # dry run: print the plan
    python scripts/c1_quarantine_unusable.py --apply    # move files
    python scripts/c1_quarantine_unusable.py --apply --restore
        # also move files that are in globallaws_removed/ but now pass the
        # criteria back into globallaws_cleaned/
"""
import argparse
import csv
import os
import re
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLEANED = os.path.join(ROOT, "globallaws_cleaned")
REMOVED = os.path.join(ROOT, "globallaws_removed")
DECISIONS = os.path.join(ROOT, "reports", "cleaning", "quality_reports", "removal_decisions.csv")

NOTE_PAT = re.compile(
    r"^\[(?:Data cleaning note|This file is an exact content duplicate)\][^\n]*(?:\n(?!\n)[^\n]*)*\n*"
)


def strip_note(text):
    return NOTE_PAT.sub("", text, count=1)


def run(apply, restore):
    if not os.path.exists(DECISIONS):
        sys.exit(f"Missing {DECISIONS} — run assess_quality.py --stage raw and --stage final first.")
    with open(DECISIONS, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    unresolved = [r["filename"] for r in rows if r["final_decision"] == "REVIEW"]
    if unresolved:
        print(f"{len(unresolved)} file(s) still need a REVIEW verdict in criteria_overrides.csv:")
        for fn in unresolved:
            print("  -", fn)
        if apply:
            sys.exit("Nothing applied.")

    os.makedirs(REMOVED, exist_ok=True)
    to_remove, to_restore, kept, note_stripped = [], [], [], []
    for r in rows:
        fn, decision = r["filename"], r["final_decision"]
        in_cleaned = os.path.exists(os.path.join(CLEANED, fn))
        in_removed = os.path.exists(os.path.join(REMOVED, fn))
        why = r["reason"] or r["override_reason"]
        if decision == "REMOVE" and in_cleaned:
            to_remove.append((fn, why))
        elif decision == "KEEP" and in_removed and not in_cleaned and restore:
            to_restore.append(fn)
        elif decision == "KEEP" and in_cleaned:
            kept.append(fn)

    print(f"Quarantine to globallaws_removed/: {len(to_remove)}")
    for fn, why in to_remove:
        print(f"  - {fn}  [{why}]")
    if restore:
        print(f"Restore to globallaws_cleaned/:    {len(to_restore)}")
        for fn in to_restore:
            print(f"  - {fn}")
    if not apply:
        print("\nDry run — nothing moved. Re-run with --apply to move files.")
        return

    for fn, _ in to_remove:
        shutil.move(os.path.join(CLEANED, fn), os.path.join(REMOVED, fn))
    for fn in to_restore:
        src = os.path.join(REMOVED, fn)
        with open(src, encoding="utf-8") as f:
            text = strip_note(f.read())
        with open(os.path.join(CLEANED, fn), "w", encoding="utf-8", newline="\n") as f:
            f.write(text.strip() + "\n")
        os.remove(src)
        kept.append(fn)
    for fn in kept:
        path = os.path.join(CLEANED, fn)
        with open(path, encoding="utf-8") as f:
            t = f.read()
        stripped = strip_note(t)
        if stripped != t:
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write(stripped.strip() + "\n")
            note_stripped.append(fn)
    print(f"Kept in globallaws_cleaned/: {len(kept)}")
    if note_stripped:
        print(f"Placeholder notes stripped from kept files: {len(note_stripped)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true", help="move files (default: dry run)")
    ap.add_argument("--restore", action="store_true",
                    help="also move KEEP files out of globallaws_removed/")
    a = ap.parse_args()
    run(a.apply, a.restore)
