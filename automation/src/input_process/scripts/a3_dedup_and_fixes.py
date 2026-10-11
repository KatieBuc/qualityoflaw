#!/usr/bin/env python3
"""
a3_dedup_and_fixes.py — Route A3: deduplicate exact-duplicate file pairs and
fix isolated OCR typos found after a full read.

Reads from globallaws/, writes to globallaws_cleaned/.

USAGE
-----
    python a3_dedup_and_fixes.py
"""
import os

SRC = r"C:\myProject\Foundation_model\globallaws"
OUT = r"C:\myProject\Foundation_model\globallaws_cleaned"
os.makedirs(OUT, exist_ok=True)


def read(fn):
    with open(os.path.join(SRC, fn), encoding="utf-8") as f:
        return f.read()


def write(fn, text):
    with open(os.path.join(OUT, fn), "w", encoding="utf-8") as f:
        f.write(text.strip() + "\n")


def run():
    # --- Mongolia (a)/(b): confirmed byte-identical duplicates ---
    fn_a = "113. Mongolia - DomViol - English (a).txt"
    fn_b = "113. Mongolia - DomViol - English (b).txt"
    content_a = read(fn_a)
    content_b = read(fn_b)
    assert content_a == content_b, "Mongolia files are no longer identical — re-check before dedup!"
    write(fn_a, content_a)
    note = (
        "[This file is an exact content duplicate of 113. Mongolia - DomViol - "
        "English (a).txt and has been deduplicated during data cleaning. "
        "Please refer to the (a) file for the full statute text.]\n"
    )
    write(fn_b, note)
    print(fn_a, "-> kept full content (canonical copy after dedup)")
    print(fn_b, "-> identical to (a), deduplicated to a pointer note")

    # --- St. Vincent: fix confirmed OCR typos found on full read. Filename
    # keeps its original duplicated "and and" spelling on purpose, so it
    # still matches cleaning_routing_plan.csv / route_A_verification.csv for
    # traceability — see the final report for the filename note. ---
    fn = "160. St. Vincent and and the Grenadines - DomViol - English.txt"
    t = read(fn)
    before = t
    fixes = {
        "R¥RT II": "PART II",
        "rcspondent": "respondent",
    }
    for bad, good in fixes.items():
        t = t.replace(bad, good)
    write(fn, t)
    changed = sum(before.count(k) for k in fixes)
    print(fn, f"-> fixed {changed} OCR typos (R¥RT II->PART II, rcspondent->respondent)")


if __name__ == "__main__":
    run()
