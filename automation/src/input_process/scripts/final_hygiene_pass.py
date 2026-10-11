#!/usr/bin/env python3
"""
final_hygiene_pass.py — Final, corpus-wide hygiene pass: Unicode
normalization (NFC), whitespace/newline cleanup, and a residual-HTML-entity
re-check/fix. Applied to ALL 228 files in the corpus, not just the 98
already touched by Route A/B — this is also what completes
globallaws_cleaned/ into a full parallel copy of globallaws/ (the 130
untouched OK/HOLD files had never been copied over before).

For each file:
  - source = globallaws_cleaned/<file> if it already exists there (keeps
    every prior content fix), else globallaws/<file> (first-time copy for
    untouched OK/HOLD files — no content changes, hygiene only).
  - Unicode NFC normalization (canonical composition only — safe, doesn't
    change what any character means, unlike NFKC which can e.g. turn
    typographic ordinal marks into plain digits).
  - Line endings normalized to bare \\n.
  - Trailing whitespace stripped from every line.
  - Runs of 2+ blank lines collapsed to a single blank line.
  - Leading/trailing whitespace stripped from the whole file.
  - html.unescape() re-applied as a safety net, then verified with a regex
    that no `&name;` / `&#123;` entity remains.

Does NOT touch inline/inter-word spacing (no collapsing of multiple spaces
within a line) — that's a formatting judgment call already made file-by-file
in the A2 script where relevant (see LESSONS LEARNED in extract_html.py for
why blanket whitespace/line rewriting on legal text needs to stay narrow).

USAGE
-----
    python final_hygiene_pass.py
"""
import csv
import html
import os
import re
import unicodedata

PLAN = r"C:\myProject\Foundation_model\GlobalLaws\golden\cleaning\cleaning_routing_plan.csv"
SRC_ORIGINAL = r"C:\myProject\Foundation_model\globallaws"
SRC_CLEANED = r"C:\myProject\Foundation_model\globallaws_cleaned"
OUT_DIR = SRC_CLEANED

ENTITY_PAT = re.compile(r"&[a-zA-Z]+;|&#\d+;")


def hygiene(text):
    # normalize line endings first so blank-line/trailing-space logic below
    # only has to deal with \n
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = unicodedata.normalize("NFC", text)
    text = html.unescape(text)

    lines = [l.rstrip(" \t") for l in text.split("\n")]
    out, blank = [], 0
    for l in lines:
        if l.strip() == "":
            blank += 1
            if blank > 1:
                continue
        else:
            blank = 0
        out.append(l)
    return "\n".join(out).strip()


def run():
    with open(PLAN, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    files = [r["filename"] for r in rows]

    stats = {
        "total": 0, "newly_copied": 0, "changed": 0, "unchanged": 0,
        "entities_fixed": 0, "entities_remaining": [],
    }

    for fn in files:
        stats["total"] += 1
        cleaned_path = os.path.join(SRC_CLEANED, fn)
        original_path = os.path.join(SRC_ORIGINAL, fn)

        if os.path.exists(cleaned_path):
            with open(cleaned_path, encoding="utf-8") as f:
                before = f.read()
            source_desc = "globallaws_cleaned"
        else:
            with open(original_path, encoding="utf-8") as f:
                before = f.read()
            source_desc = "globallaws (first copy)"
            stats["newly_copied"] += 1

        n_entities_before = len(ENTITY_PAT.findall(before))
        after = hygiene(before)
        n_entities_after = len(ENTITY_PAT.findall(after))
        stats["entities_fixed"] += max(0, n_entities_before - n_entities_after)
        if n_entities_after:
            stats["entities_remaining"].append((fn, n_entities_after))

        if after.strip() + "\n" != before.strip() + "\n" and source_desc != "globallaws (first copy)":
            stats["changed"] += 1
        elif source_desc == "globallaws (first copy)":
            pass  # counted separately above
        else:
            stats["unchanged"] += 1

        with open(os.path.join(OUT_DIR, fn), "w", encoding="utf-8", newline="\n") as f:
            f.write(after + "\n")

    print(f"Total files processed:      {stats['total']}")
    print(f"Newly copied into cleaned/: {stats['newly_copied']} (previously untouched OK/HOLD files)")
    print(f"Changed by hygiene pass:    {stats['changed']}")
    print(f"Already clean, no change:   {stats['unchanged']}")
    print(f"HTML entities fixed:        {stats['entities_fixed']}")
    if stats["entities_remaining"]:
        print(f"Files still containing entity-like text after unescape ({len(stats['entities_remaining'])}):")
        for fn, n in stats["entities_remaining"]:
            print(f"  {fn}: {n} (likely literal & in a URL fragment, not a real HTML entity — check manually)")
    else:
        print("No residual HTML entities anywhere in the corpus.")


if __name__ == "__main__":
    run()
