#!/usr/bin/env python3
"""
assess_quality.py — Apply the quantitative criteria (quality_criteria.py)
to the corpus and write the decision tables to quality_reports/.

Two stages, matching the two points in the pipeline where a decision is
needed:

  --stage raw     Input: globallaws/ (raw extraction) + Domestic_Violence/
                  (source PDF/HTML). Decides, per file, whether OCR is needed
                  and which fix routes must run.
                  -> quality_reports/raw_metrics.csv
                  -> quality_reports/routing_decisions.csv

  --stage final   Input: the best available text of every file after all
                  routes ran (globallaws_cleaned/ + globallaws_removed/;
                  "[Data cleaning note] ..." headers are ignored). Decides
                  KEEP / REVIEW / REMOVE, and checks each triggered route's
                  acceptance rule against the raw text.
                  -> quality_reports/final_metrics.csv
                  -> quality_reports/removal_decisions.csv

A REVIEW decision is resolved by a recorded human verdict in
criteria_overrides.csv (columns: filename, rule, verdict, reason), where
verdict is KEEP or REMOVE. Unresolved REVIEW rows are listed at the end of
the run.

Both stages also copy the previous (manual) decision into the output
(prev_route from cleaning_routing_plan.csv, prev_folder) so disagreements
between the old judgement and the quantitative rules are easy to list.

USAGE
-----
    python scripts/assess_quality.py --stage raw
    python scripts/assess_quality.py --stage final
    python scripts/assess_quality.py --stage raw --no-pdf   # skip PDF scan
"""
import argparse
import csv
import json
import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from quality_metrics import (  # noqa: E402
    corpus_unique_word_ratio, language_of, pdf_metrics, text_metrics, word_set,
)
from quality_criteria import (  # noqa: E402
    ocr_decision, removal_decision, route_acceptance, route_triggers,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "globallaws")
SOURCE = os.path.join(ROOT, "Domestic_Violence")
CLEANED = os.path.join(ROOT, "globallaws_cleaned")
REMOVED = os.path.join(ROOT, "globallaws_removed")
# OCR re-run output (pre-fix text of the re-OCR'd files). Older layouts kept
# it at the project root, the current one inside globallaws/.
OCR_DIRS = [os.path.join(RAW, "globallaw_OCR_txt"), os.path.join(ROOT, "globallaw_OCR_txt")]
PLAN = os.path.join(ROOT, "golden", "cleaning", "cleaning_routing_plan.csv")
OVERRIDES = os.path.join(ROOT, "criteria_overrides.csv")
OUT = os.path.join(ROOT, "reports", "cleaning", "quality_reports")
PDF_CACHE = os.path.join(OUT, "pdf_metrics_cache.json")

# The corpus-unique-word metric needs enough other documents in the same
# language to be meaningful (English, Spanish, French in this corpus).
MIN_REFERENCE_DOCS = 10
LANG_ALIASES = {"Potuguese": "Portuguese"}  # filename typo in the source set

NOTE_PAT = re.compile(
    r"^\[(?:Data cleaning note|This file is an exact content duplicate)\][^\n]*(?:\n(?!\n)[^\n]*)*\n*"
)


def read_text(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return NOTE_PAT.sub("", f.read(), count=1)


def source_path(txt_name):
    """globallaws/<name>.txt -> Domestic_Violence/<name>.pdf|.html"""
    stem = txt_name[:-4]
    for ext in (".pdf", ".html", ".htm"):
        p = os.path.join(SOURCE, stem + ext)
        if os.path.exists(p):
            return p
    return None


def load_pdf_cache():
    if os.path.exists(PDF_CACHE):
        with open(PDF_CACHE, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_pdf_cache(cache):
    with open(PDF_CACHE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=1, sort_keys=True)


def lang_key(fn):
    l = language_of(fn)
    return LANG_ALIASES.get(l, l)


def corpus_metrics(texts, pdf_cache=None, scan_pdf=True):
    """Text + corpus-relative + source metrics for every file in `texts`."""
    words = {fn: word_set(t, language_of(fn)) for fn, t in texts.items()}
    doc_freq, n_docs = {}, Counter()
    for fn, ws in words.items():
        doc_freq.setdefault(lang_key(fn), Counter()).update(ws)
        n_docs[lang_key(fn)] += 1

    hashes = {}
    rows = {}
    for fn in sorted(texts, key=_sort_key):
        t = texts[fn]
        m = text_metrics(t, language_of(fn))
        m["corpus_unique_word_ratio"] = (
            corpus_unique_word_ratio(t, language_of(fn), doc_freq[lang_key(fn)])
            if n_docs[lang_key(fn)] - 1 >= MIN_REFERENCE_DOCS else None
        )
        src = source_path(fn)
        m["source_format"] = (
            "PDF" if src and src.lower().endswith(".pdf")
            else "HTML" if src else ""
        )
        if m["source_format"] == "PDF" and pdf_cache is not None:
            key = os.path.basename(src)
            if key not in pdf_cache and scan_pdf:
                print(f"  scanning PDF text layer: {key}", flush=True)
                try:
                    pdf_cache[key] = pdf_metrics(src)
                except Exception as e:  # damaged PDF
                    pdf_cache[key] = {"error": str(e)}
                save_pdf_cache(pdf_cache)
            pm = pdf_cache.get(key, {})
            for k in ("pdf_pages", "pdf_median_chars_per_page", "pdf_low_text_page_ratio"):
                m[k] = pm.get(k)
            m["letters_per_pdf_page"] = (
                round(m["n_letters"] / pm["pdf_pages"], 1) if pm.get("pdf_pages") else None
            )
        # exact duplicate of an earlier file (sorted by number, then suffix)
        m["duplicate_of"] = hashes.get(m["content_hash"]) if m["n_letters"] else None
        hashes.setdefault(m["content_hash"], fn)
        rows[fn] = m
    return rows


def _sort_key(fn):
    m = re.match(r"(\d+)\.", fn)
    return (int(m.group(1)) if m else 10**6, fn)


def previous_decisions():
    prev = {}
    if os.path.exists(PLAN):
        with open(PLAN, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                prev[r["filename"]] = r["Route"]
    return prev


def write_csv(path, rows, first_cols):
    cols = list(first_cols)
    for r in rows:
        for k in r:
            if k not in cols:
                cols.append(k)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in cols})


def stage_raw(scan_pdf):
    texts = {fn: read_text(os.path.join(RAW, fn)) for fn in os.listdir(RAW) if fn.endswith(".txt")}
    cache = load_pdf_cache()
    metrics = corpus_metrics(texts, cache, scan_pdf)
    prev = previous_decisions()

    metric_rows, decision_rows = [], []
    for fn, m in metrics.items():
        metric_rows.append({"filename": fn, **m})
        needs_ocr, ocr_reason = ocr_decision(m)
        routes = route_triggers(m, m.get("duplicate_of"))
        decision_rows.append({
            "filename": fn,
            "source_format": m["source_format"],
            "needs_ocr": "YES" if needs_ocr else "NO",
            "ocr_reason": ocr_reason,
            "routes": "+".join(sorted(routes)) or "OK",
            "route_reasons": " | ".join(f"{k}: {v}" for k, v in sorted(routes.items())),
            "prev_route": prev.get(fn, ""),
        })
    write_csv(os.path.join(OUT, "raw_metrics.csv"), metric_rows, ["filename"])
    write_csv(os.path.join(OUT, "routing_decisions.csv"), decision_rows, ["filename"])

    n_ocr = sum(r["needs_ocr"] == "YES" for r in decision_rows)
    route_count = Counter(x for r in decision_rows for x in r["routes"].split("+"))
    print(f"Files assessed:  {len(decision_rows)}")
    print(f"Needs OCR:       {n_ocr}")
    print("Route triggers:  " + ", ".join(f"{k}={v}" for k, v in sorted(route_count.items())))


def load_overrides():
    ov = {}
    if os.path.exists(OVERRIDES):
        with open(OVERRIDES, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                ov[r["filename"]] = r
    return ov


def stage_final():
    texts, folder = {}, {}
    for d, name in ((CLEANED, "cleaned"), (REMOVED, "removed")):
        for fn in os.listdir(d):
            if fn.endswith(".txt"):
                texts[fn] = read_text(os.path.join(d, fn))
                folder[fn] = name
    cache = load_pdf_cache()
    metrics = corpus_metrics(texts, cache, scan_pdf=False)

    raw_rows = {}
    raw_path = os.path.join(OUT, "raw_metrics.csv")
    routing_path = os.path.join(OUT, "routing_decisions.csv")
    routing = {}
    if os.path.exists(raw_path) and os.path.exists(routing_path):
        with open(raw_path, encoding="utf-8") as f:
            raw_rows = {r["filename"]: _num(r) for r in csv.DictReader(f)}
        with open(routing_path, encoding="utf-8") as f:
            routing = {r["filename"]: r for r in csv.DictReader(f)}
    else:
        print("NOTE: run --stage raw first to get route-acceptance checks.")

    # Pre-fix texts for R3 (paywall markers survive only in the pre-fix text).
    pre_fix = {fn: [r] for fn, r in raw_rows.items()}
    for d in OCR_DIRS:
        if os.path.isdir(d):
            for fn in os.listdir(d):
                if fn.endswith(".txt"):
                    t = read_text(os.path.join(d, fn))
                    pre_fix.setdefault(fn, []).append(text_metrics(t, language_of(fn)))

    overrides = load_overrides()
    metric_rows, decision_rows = [], []
    for fn, m in metrics.items():
        metric_rows.append({"filename": fn, **m})
        decision, reason = removal_decision(m, pre_fix.get(fn, []), m.get("duplicate_of"))
        final = decision
        ov = overrides.get(fn)
        if decision == "REVIEW" and ov and ov.get("verdict") in ("KEEP", "REMOVE"):
            final = ov["verdict"]
        acceptance = []
        if fn in routing and fn in raw_rows:
            for route in filter(None, routing[fn]["routes"].split("+")):
                if route == "OK":
                    continue
                ok, fails = route_acceptance(route, raw_rows[fn], m)
                acceptance.append(f"{route}:{'PASS' if ok else 'FAIL(' + fails + ')'}")
        decision_rows.append({
            "filename": fn,
            "decision": decision,
            "reason": reason,
            "override": ov.get("verdict", "") if ov else "",
            "override_reason": ov.get("reason", "") if ov else "",
            "final_decision": final,
            "route_acceptance": " ".join(acceptance),
            "prev_folder": folder[fn],
        })
    write_csv(os.path.join(OUT, "final_metrics.csv"), metric_rows, ["filename"])
    write_csv(os.path.join(OUT, "removal_decisions.csv"), decision_rows, ["filename"])

    c = Counter(r["final_decision"] for r in decision_rows)
    print(f"Files assessed: {len(decision_rows)}")
    print("Final decision: " + ", ".join(f"{k}={v}" for k, v in sorted(c.items())))
    unresolved = [r["filename"] for r in decision_rows if r["final_decision"] == "REVIEW"]
    if unresolved:
        print(f"Unresolved REVIEW (add a verdict to criteria_overrides.csv): {len(unresolved)}")
        for fn in unresolved:
            print("  -", fn)
    changed = [r for r in decision_rows
               if (r["final_decision"] == "REMOVE") != (r["prev_folder"] == "removed")]
    if changed:
        print(f"Differs from current folder placement: {len(changed)}")
        for r in changed:
            print(f"  - {r['filename']}: now {r['prev_folder']}, rule says {r['final_decision']} {r['reason']}")


def _num(row):
    out = {}
    for k, v in row.items():
        if v == "":
            out[k] = None
            continue
        try:
            out[k] = float(v) if any(c in v for c in ".e") else int(v)
        except ValueError:
            out[k] = v
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--stage", choices=("raw", "final"), required=True)
    ap.add_argument("--no-pdf", action="store_true", help="use cached PDF metrics only")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if args.stage == "raw":
        stage_raw(scan_pdf=not args.no_pdf)
    else:
        stage_final()


if __name__ == "__main__":
    main()
