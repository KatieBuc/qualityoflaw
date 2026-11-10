#!/usr/bin/env python3
"""
quality_criteria.py — The quantitative decision rules of the GlobalLaws
cleaning pipeline. CLEANING_CRITERIA.md is the human-readable version of this
file; the two must be kept in sync (every rule ID below appears there).

Three decisions, each a pure function of the metrics in quality_metrics.py:

  1. ocr_decision(m)                    -> does the file need an OCR (re-)run?
  2. route_triggers(m)                  -> which fix routes (A1..B3) must run?
     route_acceptance(route, before, after) -> did that route's output pass?
  3. removal_decision(m, pre_fix)       -> KEEP / REVIEW / REMOVE after all fixes

Every function returns the IDs of the rules that fired together with the
measured values, so each decision in the output CSVs traces back to a number
and a threshold.

Conventions
-----------
* A metric value of None means "not applicable" (e.g. token metrics for
  languages written without spaces) and never fires a rule.
* `m` is the metrics dict of ONE file: text metrics, plus the source metrics
  (source_format, pdf_low_text_page_ratio, letters_per_pdf_page) when known.
"""

# ---------------------------------------------------------------------------
# Thresholds. Change values ONLY here, and update CLEANING_CRITERIA.md.
# Calibration notes (corpus of 228 files, see CLEANING_CRITERIA.md §5) are
# given next to each value.
# ---------------------------------------------------------------------------
T = {
    # --- source: PDF text layer ---------------------------------------------
    # A page counts as image-only when its text layer has < 200 non-space
    # characters (quality_metrics.pdf_metrics).
    "scanned_page_ratio": 0.50,        # >= : the PDF is a scanned document
    "min_letters_per_pdf_page": 300,   # <  : the text is missing whole pages

    # --- text: encoding / OCR corruption ------------------------------------
    "min_expected_script_ratio": 0.30,  # < : wrong glyph mapping / decoding
                                        #     (bilingual statutes go down to 0.40)
    "max_garbage_char_ratio": 0.003,    # > : U+FFFD, control, stray symbols/emoji
    "noisy_token_minor": 0.0025,        # > : minor OCR noise   (clean corpus max 0.0045 after fixes)
    "noisy_token_severe": 0.0060,       # > : severe OCR noise  (all re-OCR'd files >= 0.0067)
    "unique_word_minor": 0.06,          # > : minor OCR noise   (OCR text only)
    "unique_word_severe": 0.11,         # > : severe OCR noise  (OCR text only;
                                        #     clean max 0.101, lowest re-OCR'd 0.122)
    "max_spaced_letter_runs_per_1k": 1.0,  # > : shattered words ("j a n v i e r")
    "max_glued_tokens_per_1k": 1.0,     # > : missing inter-word spaces
    "max_mojibake_per_1k_lines": 1.0,   # > : UTF-8 read as cp1252 ("Ã©")
    "min_diacritic_ratio": 0.004,       # < : accents dropped in decoding
                                        #     (languages that use diacritics only)
    "max_detached_diacritics_per_1k": 5.0,  # > : "socie´te´" instead of "société"

    # --- text: boilerplate / residue -----------------------------------------
    "max_chrome_per_1k_lines": 3.0,     # > : web/print chrome left in ...
    "max_url_per_1k_lines": 3.0,        # > : link lists / URL footers left in ...
    "min_boilerplate_hits": 5,          # ... and at least 5 hits (one contact line in a
                                        #     short file is not boilerplate)
    "max_repeated_line_ratio": 0.30,    # > : repeated headers / ads / nav blocks
    "max_html_residue": 0,              # > : raw tags / entities left in

    # --- removal -------------------------------------------------------------
    "min_letters": 1000,                # < : not enough text to be a statute
    "min_prose_letters": 1500,          # < : page chrome only, no prose
    "max_paywall_markers": 0,           # > : content withheld by a paywall
    "order_min_units": 10,              # numbering check needs >= this many headings
    "max_numbering_disorder": 0.10,     # >= : unit order may be scrambled (REVIEW)

    # --- route acceptance safety limits ---------------------------------------
    "max_trim_letter_loss": 0.30,       # A2/A5 may remove at most 30 % of letters
    "max_typo_fix_letter_change": 0.02, # A4 may change the letter count by at most 2 %
    "max_heading_loss": 0.05,           # A1/A2/A5 may lose at most 5 % of the distinct
                                        # unit numbers (Article/§/Section N) of the input
}

ROUTES = {
    "A1": "Re-extract text from the saved HTML source",
    "A2": "Rule-based trimming of boilerplate (web/print chrome, repeated blocks)",
    "A3": "Deduplicate an exact-duplicate file",
    "A4": "Targeted OCR-typo correction (minor noise only)",
    "A5": "Remove residual source markers (HTML tags / entities)",
    "B1": "Full OCR re-run from the source PDF images",
    "B2": "Fix character decoding (charset, broken font map, detached accents)",
    "B3": "Re-extract text from the source to recover missing / mangled text",
}

SEVERE, MINOR = "severe", "minor"


def _gt(v, t):
    return v is not None and v > t


def _lt(v, t):
    return v is not None and v < t


def is_scanned(m):
    v = m.get("pdf_low_text_page_ratio")
    return v is not None and v >= T["scanned_page_ratio"]


# ---------------------------------------------------------------------------
# Rule groups
# ---------------------------------------------------------------------------
def corruption_findings(m):
    """Encoding / OCR corruption rules C1-C9.

    Returns {rule_id: (value, severity)}. 'severe' = the text is not usable
    as-is (needs re-extraction / OCR, and is a removal reason if it survives
    all fixes). 'minor' = fixable in place (targeted typo fix or
    normalization); never a removal reason.
    """
    f = {}
    if _lt(m.get("expected_script_ratio"), T["min_expected_script_ratio"]):
        f["C1_wrong_script"] = (m["expected_script_ratio"], SEVERE)
    if _gt(m.get("garbage_char_ratio"), T["max_garbage_char_ratio"]):
        f["C2_garbage_chars"] = (m["garbage_char_ratio"], SEVERE)
    v = m.get("noisy_token_ratio")
    if _gt(v, T["noisy_token_severe"]):
        f["C3_noisy_tokens"] = (v, SEVERE)
    elif _gt(v, T["noisy_token_minor"]):
        f["C3_noisy_tokens"] = (v, MINOR)
    # C4 compares a file's vocabulary with the rest of the corpus. Clean text
    # in an unexpected language (e.g. a Dutch file labelled French) also
    # scores high, so C4 is only applied to OCR-derived text (scanned PDF).
    v = m.get("corpus_unique_word_ratio")
    if is_scanned(m):
        if _gt(v, T["unique_word_severe"]):
            f["C4_unique_words"] = (v, SEVERE)
        elif _gt(v, T["unique_word_minor"]):
            f["C4_unique_words"] = (v, MINOR)
    if _gt(m.get("spaced_letter_runs_per_1k"), T["max_spaced_letter_runs_per_1k"]):
        # Letter-spaced titles ("Z A K O N") are legitimate typography, so
        # this only ever triggers a fix, never removal.
        f["C5_shattered_words"] = (m["spaced_letter_runs_per_1k"], MINOR)
    if _gt(m.get("glued_tokens_per_1k"), T["max_glued_tokens_per_1k"]):
        f["C6_glued_words"] = (m["glued_tokens_per_1k"], SEVERE)
    if _gt(m.get("mojibake_per_1k_lines"), T["max_mojibake_per_1k_lines"]):
        f["C7_mojibake"] = (m["mojibake_per_1k_lines"], SEVERE)
    if _lt(m.get("diacritic_ratio"), T["min_diacritic_ratio"]):
        f["C8_lost_diacritics"] = (m["diacritic_ratio"], SEVERE)
    if _gt(m.get("detached_diacritics_per_1k"), T["max_detached_diacritics_per_1k"]):
        f["C9_detached_diacritics"] = (m["detached_diacritics_per_1k"], MINOR)
    return f


def boilerplate_findings(m):
    """Boilerplate / residue rules P1-P4. Always fixable; never a removal
    reason on their own."""
    f = {}
    if (_gt(m.get("chrome_per_1k_lines"), T["max_chrome_per_1k_lines"])
            and (m.get("chrome_count") or 0) >= T["min_boilerplate_hits"]):
        f["P1_web_chrome"] = m["chrome_per_1k_lines"]
    if (_gt(m.get("url_per_1k_lines"), T["max_url_per_1k_lines"])
            and (m.get("url_count") or 0) >= T["min_boilerplate_hits"]):
        f["P2_urls"] = m["url_per_1k_lines"]
    if _gt(m.get("repeated_line_ratio"), T["max_repeated_line_ratio"]):
        f["P3_repeated_lines"] = m["repeated_line_ratio"]
    if _gt(m.get("html_residue_count"), T["max_html_residue"]):
        f["P4_html_residue"] = m["html_residue_count"]
    return f


def coverage_findings(m):
    """Source coverage rules S1-S2 (PDF sources only)."""
    f = {}
    if is_scanned(m):
        f["S1_scanned_pdf"] = m["pdf_low_text_page_ratio"]
    if _lt(m.get("letters_per_pdf_page"), T["min_letters_per_pdf_page"]):
        f["S2_missing_pages"] = m["letters_per_pdf_page"]
    return f


def _only(findings, severity):
    return {k: v for k, v in findings.items() if v[1] == severity}


def _fmt(findings):
    out = []
    for k, v in findings.items():
        out.append(f"{k}={v[0]}({v[1]})" if isinstance(v, tuple) else f"{k}={v}")
    return ";".join(out)


ENCODING_RULES = ("C1_wrong_script", "C2_garbage_chars", "C7_mojibake", "C8_lost_diacritics")


# ---------------------------------------------------------------------------
# 1. OCR decision
# ---------------------------------------------------------------------------
def ocr_decision(m):
    """Return (needs_ocr, reason). Only PDF sources can need OCR.

      O1  scanned PDF (S1) whose current text has SEVERE corruption
          (C1-C4, C6-C8) or is missing pages (S2)           -> OCR re-run
      O2  PDF with a text layer that decodes to the wrong script or to
          garbage (C1/C2), or whose characters are severely OCR-noisy (C3
          severe — the text layer is itself a bad OCR layer embedded in the
          PDF). Re-extracting the same text layer returns the same
          characters                                        -> OCR from images
      O3  PDF (any) whose extracted text covers too few letters per page
          (S2): some pages exist only as images             -> OCR

    Minor noise (C3/C4 'minor') on a scanned PDF does NOT need OCR: it is
    fixed in place by route A4.
    """
    if m.get("source_format") != "PDF":
        return False, ""
    cov = coverage_findings(m)
    severe = _only(corruption_findings(m), SEVERE)
    if "S1_scanned_pdf" in cov and (severe or "S2_missing_pages" in cov):
        return True, "O1:" + _fmt({**cov, **severe})
    bad_layer = {k: v for k, v in severe.items()
                 if k in ("C1_wrong_script", "C2_garbage_chars", "C3_noisy_tokens")}
    if bad_layer:
        return True, "O2:" + _fmt(bad_layer)
    if "S2_missing_pages" in cov:
        return True, "O3:" + _fmt(cov)
    return False, ""


# ---------------------------------------------------------------------------
# 2. Route triggers and acceptance
# ---------------------------------------------------------------------------
def route_triggers(m, duplicate_of=None):
    """Return {route: reason} for every route this file must go through.
    Routes are independent of each other; a file can trigger several."""
    r = {}
    fmt = m.get("source_format")
    cor = corruption_findings(m)
    bp = boilerplate_findings(m)

    if duplicate_of:
        r["A3"] = f"D1_exact_duplicate_of={duplicate_of}"

    # Decoding problems (both formats) -> B2
    enc = {k: v for k, v in cor.items() if k in ENCODING_RULES or k == "C9_detached_diacritics"}

    if fmt == "HTML":
        if enc:
            r["B2"] = _fmt(enc)
        # Every other HTML extraction problem, including leftover markup,
        # is fixed by re-extracting from the saved page.
        other = {k: v for k, v in cor.items() if k not in enc}
        if bp or other:
            r["A1"] = _fmt({**bp, **other})
        if _lt(m.get("prose_letters"), T["min_prose_letters"]):
            r["B3"] = f"R2_prose_letters={m['prose_letters']}"
        return r

    needs_ocr, ocr_reason = ocr_decision(m)
    if needs_ocr:
        # O2 with a broken font map (C1/C2) is a decoding problem -> B2
        # (fixed via OCR); everything else is an OCR re-run -> B1.
        decoding = ocr_reason.startswith("O2") and ("C1_" in ocr_reason or "C2_" in ocr_reason)
        r["B2" if decoding else "B1"] = ocr_reason
    else:
        if enc:
            r["B2"] = _fmt(enc)
        layout = {k: v for k, v in cor.items() if k in ("C5_shattered_words", "C6_glued_words")}
        if layout:
            r["B3"] = _fmt(layout)  # re-extract the text layer with another extractor
        # Severe C3 always leads to OCR (O2) and severe C4 only exists on
        # scanned PDFs (O1), so what reaches this point is minor noise.
        typo = {k: v for k, v in cor.items() if k in ("C3_noisy_tokens", "C4_unique_words")}
        if typo:
            r["A4"] = _fmt(typo)
    trim = {k: v for k, v in bp.items() if k != "P4_html_residue"}
    if trim:
        r["A2"] = _fmt(trim)
    if "P4_html_residue" in bp:
        r["A5"] = _fmt({"P4_html_residue": bp["P4_html_residue"]})
    return r


# Which rule families each route is responsible for clearing.
ROUTE_TARGETS = {
    "A1": ("P1_web_chrome", "P2_urls", "P3_repeated_lines", "P4_html_residue", "C3_noisy_tokens",
           "C5_shattered_words", "C6_glued_words"),
    "A2": ("P1_web_chrome", "P2_urls", "P3_repeated_lines"),
    "A4": ("C3_noisy_tokens", "C4_unique_words"),
    "A5": ("P4_html_residue",),
    "B1": ("C1_wrong_script", "C2_garbage_chars", "C3_noisy_tokens", "C4_unique_words",
           "C6_glued_words", "S2_missing_pages"),
    "B2": ENCODING_RULES + ("C9_detached_diacritics",),
    "B3": ("C3_noisy_tokens", "C5_shattered_words", "C6_glued_words", "S2_missing_pages",
           "R2_prose_letters"),
}


ROUTE_TOLERATES_MINOR = {
    "A1": ("C3_noisy_tokens", "C5_shattered_words"),
    "B1": ("C3_noisy_tokens", "C4_unique_words", "C5_shattered_words"),
    "B3": ("C3_noisy_tokens", "C4_unique_words"),
}


def route_acceptance(route, before, after):
    """Did `route` produce an acceptable result?

    before / after: metric dicts of the file before and after the route.
    Returns (passed, failures). Pass = none of the rules the route is
    responsible for (ROUTE_TARGETS) still fires on `after`, plus the
    route-specific safety limits K1-K4 hold. A failed route leaves the file
    'partial': removal_decision() then decides whether what is left is
    still usable.
    """
    fails = {}
    lb, la = before.get("n_letters") or 0, after.get("n_letters") or 0
    change = (lb - la) / lb if lb else 0.0

    if route == "A3":
        if after.get("duplicate_of"):
            fails["K3_duplicate_remains"] = after["duplicate_of"]
        return (not fails), _fmt(fails)

    still = {**corruption_findings(after), **boilerplate_findings(after), **coverage_findings(after)}
    # A re-extraction / OCR run may leave MINOR residue of these rules behind;
    # it is then handled by A4 (typos) rather than failing the route.
    tolerated = ROUTE_TOLERATES_MINOR.get(route, ())
    still = {k: v for k, v in still.items()
             if not (k in tolerated and isinstance(v, tuple) and v[1] == MINOR)}
    if _lt(after.get("prose_letters"), T["min_prose_letters"]):
        still["R2_prose_letters"] = after["prose_letters"]
    # S1 describes the source, not the result — a scanned PDF stays scanned.
    still.pop("S1_scanned_pdf", None)
    fails.update({k: v for k, v in still.items() if k in ROUTE_TARGETS.get(route, ())})

    if route in ("A1", "A2", "A5"):
        # K1: removing boilerplate must not remove statute headings. Compared
        # on DISTINCT unit numbers, so dropping a duplicated table of
        # contents is fine but losing "§ 6" from the body is not.
        nb = set(str(before.get("unit_numbers_distinct") or "").split())
        na = set(str(after.get("unit_numbers_distinct") or "").split())
        if nb and len(nb - na) / len(nb) > T["max_heading_loss"]:
            fails["K1_headings_lost"] = f"{len(nb - na)}/{len(nb)}"
    if route in ("A2", "A5") and change > T["max_trim_letter_loss"]:
        fails["K2_letter_loss"] = round(change, 3)
    if route == "A4" and abs(change) > T["max_typo_fix_letter_change"]:
        fails["K4_letter_change"] = round(change, 3)
    return (not fails), _fmt(fails)


# ---------------------------------------------------------------------------
# 3. Removal decision
# ---------------------------------------------------------------------------
def removal_decision(m, pre_fix=(), duplicate_of=None):
    """Return (decision, reason), decision in KEEP / REVIEW / REMOVE.

    m        metrics of the BEST available text (after every triggered route,
             including OCR, has run).
    pre_fix  metrics of the same file before fixing (raw extraction and, if
             one was made, the OCR output). Used only for R3: stripping
             paywall placeholders from a text does not bring back the
             withheld statute text.

    Removal reasons are unrecoverable problems only. Leftover boilerplate
    (P1-P4) and minor noise are fixable and never remove a file, and a
    content-scope mismatch (wrong law / language / a summary instead of the
    statute) is not a text-quality defect: clean text stays in the corpus.
    """
    remove = {}
    if duplicate_of:
        remove["R5_duplicate_of"] = duplicate_of
    if _lt(m.get("n_letters"), T["min_letters"]):
        remove["R1_letters"] = m["n_letters"]
    if _lt(m.get("prose_letters"), T["min_prose_letters"]):
        remove["R2_prose_letters"] = m["prose_letters"]
    paywall = max([m.get("paywall_marker_count") or 0] + [p.get("paywall_marker_count") or 0 for p in pre_fix])
    if paywall > T["max_paywall_markers"]:
        remove["R3_paywall_markers"] = paywall
    for k, v in _only(corruption_findings(m), SEVERE).items():
        remove["R4_" + k] = v
    if remove:
        return "REMOVE", _fmt(remove)

    if (m.get("n_unit_numbers") or 0) >= T["order_min_units"] and (
        m.get("numbering_disorder") or 0
    ) >= T["max_numbering_disorder"]:
        return "REVIEW", f"R6_numbering_disorder={m['numbering_disorder']}(n={m['n_unit_numbers']})"
    return "KEEP", ""
