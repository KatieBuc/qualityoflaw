"""Decision rules for stage 1: which texts pass, which need OCR, which are dropped.

Ported from the GlobalLaws ``quality_criteria.py``. Every decision is a pure
function of the metrics in ``metrics.py`` and returns the IDs of the rules that
fired together with the measured values, so each line of the report traces back
to a number and a threshold.

Rule families
    C1-C9  encoding / OCR corruption (``severe`` = unusable, ``minor`` = tolerable)
    P1-P4  boilerplate / residue (fixable, never a reason to drop a file)
    S1-S2  PDF text-layer coverage
    O1-O3  the PDF needs an OCR run
    R1-R3  reasons to drop a file;  R6 numbering disorder (flag only)

Outcomes (``decide``)
    PASS        usable (possibly with a ``review`` flag or leftover minor noise)
    NEEDS_OCR   a PDF whose text layer cannot be used; OCR it and put the result in
                ``extracted_txt/<stem>.txt`` (replace the extraction)
    ELIMINATE   unrecoverable (too little text, corrupt, paywalled, ...)

Exact duplicates (the old R5) are flagged by the pipeline, never removed:
dropping a policy would also drop its row in the golden dataset.

Thresholds were calibrated on the GlobalLaws corpus. They are the defaults here;
override any of them under ``input_process.thresholds`` in the pipeline config.
"""

from dataclasses import dataclass, field
from typing import Mapping

THRESHOLDS: dict[str, float] = {
    # --- source: PDF text layer
    "scanned_page_ratio": 0.50,        # >= : the PDF is a scanned document
    "min_letters_per_pdf_page": 300,   # <  : whole pages missing from the text
    # --- text: encoding / OCR corruption
    "min_expected_script_ratio": 0.30,  # <  : wrong glyph mapping / decoding
    "max_garbage_char_ratio": 0.003,    # >  : U+FFFD, control, stray symbols
    "noisy_token_minor": 0.0025,        # >  : minor OCR noise
    "noisy_token_severe": 0.0060,       # >  : severe OCR noise
    "unique_word_minor": 0.06,          # >  : minor OCR noise (OCR text only)
    "unique_word_severe": 0.11,         # >  : severe OCR noise (OCR text only)
    "max_spaced_letter_runs_per_1k": 1.0,  # > : shattered words ("j a n v i e r")
    "max_glued_tokens_per_1k": 1.0,     # >  : missing inter-word spaces
    "max_mojibake_per_1k_lines": 1.0,   # >  : UTF-8 read as cp1252
    "min_diacritic_ratio": 0.004,       # <  : accents dropped (diacritic languages only)
    "max_detached_diacritics_per_1k": 5.0,
    # --- text: boilerplate / residue
    "max_chrome_per_1k_lines": 3.0,
    "max_url_per_1k_lines": 3.0,
    "min_boilerplate_hits": 5,          # one contact line in a short file is not boilerplate
    "max_repeated_line_ratio": 0.30,
    "max_html_residue": 0,
    # --- removal
    "min_letters": 1000,                # <  : not enough text to be a statute
    "min_prose_letters": 1500,          # <  : page chrome only, no prose
    "max_paywall_markers": 0,
    "order_min_units": 10,              # numbering check needs >= this many headings
    "max_numbering_disorder": 0.10,     # >= : unit order may be scrambled (flag only)
    # --- fix safety limits
    "max_trim_letter_loss": 0.30,       # a fix may remove at most 30 % of the letters
    "max_heading_loss": 0.05,           # ... and lose at most 5 % of the distinct unit numbers
}

SEVERE, MINOR = "severe", "minor"

#: Which fixer clears which finding (see ``fix.py``).
FIXERS_FOR = {
    "P1_web_chrome": "strip_boilerplate",
    "P2_urls": "strip_boilerplate",
    "P3_repeated_lines": "strip_boilerplate",
    "P4_html_residue": "strip_html_residue",
    "C5_shattered_words": "join_spaced_letters",
}

ENCODING_RULES = ("C1_wrong_script", "C2_garbage_chars", "C7_mojibake", "C8_lost_diacritics")


def resolve_thresholds(overrides: Mapping[str, float] | None = None) -> dict[str, float]:
    t = dict(THRESHOLDS)
    for key, value in (overrides or {}).items():
        if key not in t:
            raise ValueError(
                f"Unknown quality threshold '{key}'. Known: {', '.join(sorted(t))}"
            )
        t[key] = value
    return t


def _gt(v, t):
    return v is not None and v > t


def _lt(v, t):
    return v is not None and v < t


def is_scanned(m: Mapping, t: Mapping = THRESHOLDS) -> bool:
    v = m.get("pdf_low_text_page_ratio")
    return v is not None and v >= t["scanned_page_ratio"]


# --------------------------------------------------------------------------- findings
def corruption_findings(m: Mapping, t: Mapping = THRESHOLDS, *, ocr_derived: bool = False) -> dict:
    """Rules C1-C9 -> {rule_id: (value, severity)}.

    `ocr_derived`: the text is OCR output, so the corpus-relative C4 applies.
    """
    f = {}
    if _lt(m.get("expected_script_ratio"), t["min_expected_script_ratio"]):
        f["C1_wrong_script"] = (m["expected_script_ratio"], SEVERE)
    if _gt(m.get("garbage_char_ratio"), t["max_garbage_char_ratio"]):
        f["C2_garbage_chars"] = (m["garbage_char_ratio"], SEVERE)
    v = m.get("noisy_token_ratio")
    if _gt(v, t["noisy_token_severe"]):
        f["C3_noisy_tokens"] = (v, SEVERE)
    elif _gt(v, t["noisy_token_minor"]):
        f["C3_noisy_tokens"] = (v, MINOR)
    # C4 compares a file's vocabulary with the rest of the corpus; clean text in
    # an unexpected language scores high too, so it only applies to OCR text.
    v = m.get("corpus_unique_word_ratio")
    if ocr_derived or is_scanned(m, t):
        if _gt(v, t["unique_word_severe"]):
            f["C4_unique_words"] = (v, SEVERE)
        elif _gt(v, t["unique_word_minor"]):
            f["C4_unique_words"] = (v, MINOR)
    if _gt(m.get("spaced_letter_runs_per_1k"), t["max_spaced_letter_runs_per_1k"]):
        f["C5_shattered_words"] = (m["spaced_letter_runs_per_1k"], MINOR)
    if _gt(m.get("glued_tokens_per_1k"), t["max_glued_tokens_per_1k"]):
        f["C6_glued_words"] = (m["glued_tokens_per_1k"], SEVERE)
    if _gt(m.get("mojibake_per_1k_lines"), t["max_mojibake_per_1k_lines"]):
        f["C7_mojibake"] = (m["mojibake_per_1k_lines"], SEVERE)
    if _lt(m.get("diacritic_ratio"), t["min_diacritic_ratio"]):
        f["C8_lost_diacritics"] = (m["diacritic_ratio"], SEVERE)
    if _gt(m.get("detached_diacritics_per_1k"), t["max_detached_diacritics_per_1k"]):
        f["C9_detached_diacritics"] = (m["detached_diacritics_per_1k"], MINOR)
    return f


def boilerplate_findings(m: Mapping, t: Mapping = THRESHOLDS) -> dict:
    """Rules P1-P4 -> {rule_id: value}. Always fixable."""
    f = {}
    if (_gt(m.get("chrome_per_1k_lines"), t["max_chrome_per_1k_lines"])
            and (m.get("chrome_count") or 0) >= t["min_boilerplate_hits"]):
        f["P1_web_chrome"] = m["chrome_per_1k_lines"]
    if (_gt(m.get("url_per_1k_lines"), t["max_url_per_1k_lines"])
            and (m.get("url_count") or 0) >= t["min_boilerplate_hits"]):
        f["P2_urls"] = m["url_per_1k_lines"]
    if _gt(m.get("repeated_line_ratio"), t["max_repeated_line_ratio"]):
        f["P3_repeated_lines"] = m["repeated_line_ratio"]
    if _gt(m.get("html_residue_count"), t["max_html_residue"]):
        f["P4_html_residue"] = m["html_residue_count"]
    return f


def coverage_findings(m: Mapping, t: Mapping = THRESHOLDS) -> dict:
    """Rules S1-S2 (PDF sources only)."""
    f = {}
    if is_scanned(m, t):
        f["S1_scanned_pdf"] = m["pdf_low_text_page_ratio"]
    if _lt(m.get("letters_per_pdf_page"), t["min_letters_per_pdf_page"]):
        f["S2_missing_pages"] = m["letters_per_pdf_page"]
    return f


def _only(findings: Mapping, severity: str) -> dict:
    return {k: v for k, v in findings.items() if v[1] == severity}


def _fmt(findings: Mapping) -> str:
    return ";".join(
        f"{k}={v[0]}({v[1]})" if isinstance(v, tuple) else f"{k}={v}" for k, v in findings.items()
    )


def all_findings(m: Mapping, t: Mapping = THRESHOLDS, *, ocr_derived: bool = False) -> dict:
    """Every fired C/P/S rule as {rule_id: {"value": v, "severity": s | None}} (JSON-friendly)."""
    out = {}
    for rule, (value, severity) in corruption_findings(m, t, ocr_derived=ocr_derived).items():
        out[rule] = {"value": value, "severity": severity}
    for rule, value in {**boilerplate_findings(m, t), **coverage_findings(m, t)}.items():
        out[rule] = {"value": value, "severity": None}
    return out


def fix_plan(findings: Mapping) -> list[str]:
    """Names of the fixers the fired findings call for (hygiene always runs separately)."""
    wanted = []
    for rule in findings:
        fixer = FIXERS_FOR.get(rule)
        if fixer and fixer not in wanted:
            wanted.append(fixer)
    return wanted


# --------------------------------------------------------------------------- decisions
def ocr_findings(m: Mapping, t: Mapping = THRESHOLDS) -> tuple[str, dict]:
    """(rule code, the findings behind it); ``("", {})`` when no OCR is needed.
    Only a PDF-sourced text can need OCR.

      O1  scanned PDF whose text has SEVERE corruption or is missing pages
      O2  PDF with a text layer that decodes to the wrong script / garbage /
          severe noise (re-extracting returns the same characters)
      O3  too few letters per page: some pages exist only as images

    Minor noise on a scanned PDF does not need OCR.
    """
    if m.get("source_format") != "pdf":
        return "", {}
    cov = coverage_findings(m, t)
    severe = _only(corruption_findings(m, t), SEVERE)
    if "S1_scanned_pdf" in cov and (severe or "S2_missing_pages" in cov):
        return "O1", {**cov, **severe}
    bad_layer = {k: v for k, v in severe.items()
                 if k in ("C1_wrong_script", "C2_garbage_chars", "C3_noisy_tokens")}
    if bad_layer:
        return "O2", bad_layer
    if "S2_missing_pages" in cov:
        return "O3", cov
    return "", {}


def ocr_decision(m: Mapping, t: Mapping = THRESHOLDS) -> tuple[bool, str]:
    """(needs_ocr, reason string), see `ocr_findings`."""
    code, found = ocr_findings(m, t)
    return (True, f"{code}:{_fmt(found)}") if code else (False, "")


def removal_reasons(m: Mapping, t: Mapping = THRESHOLDS, *, ocr_derived: bool = False) -> dict:
    """Unrecoverable problems only. Leftover boilerplate and minor noise never drop a file."""
    remove = {}
    if _lt(m.get("n_letters"), t["min_letters"]):
        remove["R1_letters"] = m["n_letters"]
    if _lt(m.get("prose_letters"), t["min_prose_letters"]):
        remove["R2_prose_letters"] = m["prose_letters"]
    if (m.get("paywall_marker_count") or 0) > t["max_paywall_markers"]:
        remove["R3_paywall_markers"] = m["paywall_marker_count"]
    for k, v in _only(corruption_findings(m, t, ocr_derived=ocr_derived), SEVERE).items():
        remove["R4_" + k] = v
    return remove


#: The threshold each rule is compared with, for the report.
RULE_LIMITS = {
    "S1_scanned_pdf": "scanned_page_ratio",
    "S2_missing_pages": "min_letters_per_pdf_page",
    "C1_wrong_script": "min_expected_script_ratio",
    "C2_garbage_chars": "max_garbage_char_ratio",
    "C3_noisy_tokens": "noisy_token_severe",
    "C4_unique_words": "unique_word_severe",
    "C6_glued_words": "max_glued_tokens_per_1k",
    "C7_mojibake": "max_mojibake_per_1k_lines",
    "C8_lost_diacritics": "min_diacritic_ratio",
    "R1_letters": "min_letters",
    "R2_prose_letters": "min_prose_letters",
    "R3_paywall_markers": "max_paywall_markers",
}


def indicators(found: Mapping, t: Mapping = THRESHOLDS) -> list[dict]:
    """The failed indicators as ``[{"rule", "value", "limit"}]``.

    ``R4_`` (a severe corruption rule that survived fixing) is shown under the
    corruption rule's own name; `limit` is the threshold the value was compared with.
    """
    out = []
    for rule, raw in found.items():
        value = raw[0] if isinstance(raw, tuple) else raw
        name = rule[3:] if rule.startswith("R4_") else rule
        out.append({"rule": name, "value": value, "limit": t.get(RULE_LIMITS.get(name, ""))})
    return out


@dataclass
class Verdict:
    status: str                      # PASS | NEEDS_OCR | ELIMINATE
    reason: str = ""                 # rule IDs with values; empty for a clean PASS
    findings: dict = field(default_factory=dict)  # every fired C/P/S rule
    review: str = ""                 # non-blocking flag (R6 numbering disorder)
    failed: list = field(default_factory=list)  # indicators behind NEEDS_OCR / ELIMINATE


def decide(m: Mapping, source_format: str, t: Mapping = THRESHOLDS) -> Verdict:
    """Final verdict for one text, from its metrics (+ PDF metrics when it came from a PDF).

    `source_format` is ``pdf`` / ``html`` / ``txt`` / ``ocr``.
    """
    m = {**m, "source_format": source_format}
    ocr_derived = source_format == "ocr"
    findings = all_findings(m, t, ocr_derived=ocr_derived)

    code, ocr_found = ocr_findings(m, t)
    if code:
        return Verdict("NEEDS_OCR", f"{code}:{_fmt(ocr_found)}", findings, failed=indicators(ocr_found, t))

    reasons = removal_reasons(m, t, ocr_derived=ocr_derived)
    if reasons:
        return Verdict("ELIMINATE", _fmt(reasons), findings, failed=indicators(reasons, t))

    review = ""
    if (m.get("n_unit_numbers") or 0) >= t["order_min_units"] and (
        m.get("numbering_disorder") or 0
    ) >= t["max_numbering_disorder"]:
        review = f"R6_numbering_disorder={m['numbering_disorder']}(n={m['n_unit_numbers']})"
    return Verdict("PASS", "", findings, review)


def fix_acceptable(before: Mapping, after: Mapping, t: Mapping = THRESHOLDS) -> tuple[bool, str]:
    """Did a fix stay within the safety limits? (K1 headings lost, K2 letters lost)."""
    fails = {}
    lb, la = before.get("n_letters") or 0, after.get("n_letters") or 0
    loss = (lb - la) / lb if lb else 0.0
    if loss > t["max_trim_letter_loss"]:
        fails["K2_letter_loss"] = round(loss, 3)
    nb = set(str(before.get("unit_numbers_distinct") or "").split())
    na = set(str(after.get("unit_numbers_distinct") or "").split())
    if nb and len(nb - na) / len(nb) > t["max_heading_loss"]:
        fails["K1_headings_lost"] = f"{len(nb - na)}/{len(nb)}"
    return (not fails), _fmt(fails)
