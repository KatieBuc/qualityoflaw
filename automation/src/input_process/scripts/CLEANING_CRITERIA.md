# GlobalLaws Cleaning — Quantitative Criteria

This document defines **when a file needs OCR**, **which fix route a file must go through and when that
route's result is acceptable**, and **when a file is removed** from the corpus. Every decision is a
threshold on a measured metric, so re-running the pipeline gives the same answer no matter who (or which
model) runs it.

Scope: the cleaning stage only (`globallaws/` → `globallaws_cleaned/` / `globallaws_removed/`). The
Markdown conversion stage is out of scope.

| Where | What |
|---|---|
| `scripts/quality_metrics.py` | Computes every metric (§1). Measures only; makes no decisions. |
| `scripts/quality_criteria.py` | All thresholds (`T = {...}`) and decision rules (§2–§4). **Single source of truth** — this document mirrors it. |
| `scripts/assess_quality.py` | Runs metrics + rules over the corpus and writes the decision tables to `reports/cleaning/quality_reports/`. |
| `golden/cleaning/` | Golden labels (AI review, routing, verification) the criteria are calibrated against. |
| `criteria_overrides.csv` | Recorded human verdicts for the one rule that asks for review (R6). |
| `scripts/c1_quarantine_unusable.py` | Moves files according to `reports/cleaning/quality_reports/removal_decisions.csv`. |

---

## 1. Metrics

All metrics are computed per file. A metric that does not apply to a file (e.g. word-level metrics for
languages written without spaces: Chinese, Japanese, Thai, Dzongkha) is left empty and never fires a rule.

### 1.1 Source metrics (PDF only)

| Metric | Definition |
|---|---|
| `pdf_low_text_page_ratio` | Share of pages whose PDF text layer has < 200 non-space characters (image-only / scanned page). PDFs over 60 pages are measured on 60 evenly spaced pages. |
| `letters_per_pdf_page` | Letters in the extracted text ÷ number of PDF pages. |

### 1.2 Text metrics

| Metric | Definition | Detects |
|---|---|---|
| `n_letters` | Number of letters (Unicode category L). | Empty / near-empty files |
| `prose_letters` | Letters on lines of ≥ 40 characters. Menus, link lists and UI labels are short lines. | Pages that are only web chrome |
| `expected_script_ratio` | Share of letters in the script(s) of the file's language (from the filename; Bosnian accepts Latin and Cyrillic). | Broken font maps (Georgian/Hebrew decoded as Latin junk), lost Cyrillic/Greek |
| `garbage_char_ratio` | Share of non-space characters that are U+FFFD, control, unassigned, private-use (except the Symbol-font block U+F000–F0FF, counted separately as `symbol_font_chars`), or stray symbols/emoji. | Undecodable bytes, glyph soup |
| `noisy_token_ratio` | Share of words showing an OCR/decoding corruption pattern: a digit inside a word (`Pur5uant`), a symbol inside a word (`ARRAN€iEMENT`), two scripts in one word (Latin/Cyrillic), or chaotic case (`CoDIgo`, `RTiGo`). Legitimate forms are excluded (`1st`, `12bis`, `F102Part` annotation markers, apostrophes, hyphens, combining marks). | OCR errors |
| `corpus_unique_word_ratio` | Share of words (≥ 4 letters) that occur in **no other file** of the same language. Only computed for languages with ≥ 11 files (English, Spanish, French). | OCR misreadings (almost always unique misspellings) |
| `spaced_letter_runs_per_1k` | Runs of ≥ 4 single-letter tokens (`j a n v i e r`) per 1,000 tokens. | Shattered words |
| `glued_tokens_per_1k` | Letter-only tokens of ≥ 30 letters per 1,000 tokens. | Missing inter-word spaces |
| `diacritic_ratio` | Share of Latin letters that are non-ASCII. Only for Latin-script languages that use diacritics (French, Spanish, Czech, …). | Accents dropped by a wrong charset (`Promulgaci n`) |
| `detached_diacritics_per_1k` | Letters followed by a spacing accent (`socie´te´`) per 1,000 tokens. | Accents emitted separately by the PDF extractor |
| `mojibake_per_1k_lines` | UTF-8-read-as-cp1252 sequences (`Ã©`, `â€™`) per 1,000 lines. | Double-decoding |
| `chrome_per_1k_lines` | Web/print chrome words (cookie, login, subscribe, privacy policy, Impressum, …) per 1,000 lines. | Website chrome |
| `url_per_1k_lines` | URLs and e-mail addresses per 1,000 lines. | Link lists, URL footers |
| `repeated_line_ratio` | Share of lines (≥ 15 chars) that repeat an earlier line exactly. | Repeated headers, injected ads, nav blocks |
| `html_residue_count` | Raw HTML tags and entities (`<p>`, `&nbsp;`). | Incomplete HTML extraction |
| `paywall_marker_count` | Paywall phrases ("sign up or sign in", "Pro Membership", "Full text is available with an active Subscription", …). | Withheld content |
| `n_unit_numbers`, `numbering_disorder` | Sequence of unit numbers on heading-shaped lines (`Article 12`, `§ 3`, `16.20.`, `12.`; capitalised keyword, ≤ 90 chars, not followed by "of/de/…"), and the share of steps that go backwards without restarting at 1–2. | Scrambled unit order (multi-column pages extracted out of order) |
| `content_hash` / `duplicate_of` | SHA-1 of the whitespace-normalised, lower-cased text; the first file with the same hash. | Exact duplicates |

---

## 2. When OCR is needed

OCR applies to **PDF sources only**. A file needs OCR when any of these holds:

| Rule | Condition | Meaning |
|---|---|---|
| **O1** | `pdf_low_text_page_ratio` ≥ **0.50** (scanned PDF) **and** (a *severe* corruption rule fires on the current text, **or** `letters_per_pdf_page` < **300**) | The existing OCR text is bad → re-run OCR |
| **O2** | `expected_script_ratio` < **0.30**, **or** `garbage_char_ratio` > **0.003**, **or** `noisy_token_ratio` > **0.0060** | The PDF text layer itself is unusable: a broken font map (wrong script / garbage) or a bad OCR layer embedded in the PDF. Re-extracting the same text layer returns the same characters → OCR the page images |
| **O3** | `letters_per_pdf_page` < **300** | Whole pages exist only as images → OCR |

Severe corruption rules (used by O1 and by removal, §4):

| Rule | Metric | Severe when |
|---|---|---|
| C1 | `expected_script_ratio` | < 0.30 |
| C2 | `garbage_char_ratio` | > 0.003 |
| C3 | `noisy_token_ratio` | > 0.0060 (minor: > 0.0025) |
| C4 | `corpus_unique_word_ratio` — scanned PDFs only | > 0.11 (minor: > 0.06) |
| C6 | `glued_tokens_per_1k` | > 1.0 |
| C7 | `mojibake_per_1k_lines` | > 1.0 |
| C8 | `diacritic_ratio` | < 0.004 |

Minor-only rules (fixable in place, never OCR or removal by themselves): C3/C4 in their minor band,
**C5** `spaced_letter_runs_per_1k` > 1.0, **C9** `detached_diacritics_per_1k` > 5.0.

A scanned PDF with only *minor* noise does **not** need OCR; the typos are fixed in place by route A4.

C4 is restricted to scanned PDFs because clean text in an unexpected language (e.g. a Dutch file labelled
French) also has many corpus-unique words. On OCR text a high score means misreadings; on a text-layer
PDF it would just mean "different language".

---

## 3. Fix routes — trigger and acceptance

Each route is an independent script. A file enters a route when the route's **trigger** fires on its raw
extraction (`globallaws/`), and the route's result is **accepted** when the rules that route is
responsible for no longer fire on its output, and the route-specific safety limits (K1–K4) hold. A file
can trigger several routes.

| Route | What it does | Trigger (on raw text) | Acceptance (on output) |
|---|---|---|---|
| **A1** | Re-extract text from the saved HTML | HTML source and any of P1–P4, C3, C5, C6 | P1–P4, C6 no longer fire; C3 at most minor (C5 tolerated); **K1** |
| **A2** | Rule-based trimming of boilerplate | PDF source and any of **P1** `chrome_per_1k_lines` > 3.0 (and ≥ 5 hits), **P2** `url_per_1k_lines` > 3.0 (and ≥ 5 hits), **P3** `repeated_line_ratio` > 0.30 | P1–P3 no longer fire; **K1**; **K2** |
| **A3** | Deduplicate | **D1** identical `content_hash` to an earlier file | No duplicate left (**K3**) |
| **A4** | Targeted OCR-typo correction | PDF, no OCR needed, and C3 / C4 in the **minor** band | C3/C4 no longer fire; **K4** letter count changes by ≤ 2 % |
| **A5** | Remove residual markup | PDF source and **P4** `html_residue_count` > 0 (HTML sources go to A1) | P4 no longer fires; **K1**; **K2** |
| **B1** | Full OCR re-run | OCR rule O1 or O3, or O2 caused by C3 (§2) | C1, C2, C6 and S2 (`letters_per_pdf_page` < 300) no longer fire; C3/C4 at most minor |
| **B2** | Fix decoding (charset, broken font map, detached accents) | HTML with C1/C2/C7/C8/C9; PDF with O2 caused by C1/C2 (fixed via OCR), or C7/C8/C9 | C1, C2, C7, C8, C9 no longer fire |
| **B3** | Re-extract from source to recover missing / mangled text | HTML with `prose_letters` < 1,500; text-layer PDF with C5/C6 | C5, C6, S2 no longer fire; C3 at most minor; `prose_letters` ≥ 1,500 |

Safety limits:

| Limit | Applies to | Condition to pass |
|---|---|---|
| **K1** | A1, A2, A5 | At most **5 %** of the distinct unit numbers (`Article N`, `§ N`, `Section N`, `N.`) of the input are missing from the output. Compared on distinct numbers, so dropping a duplicated table of contents is fine but losing `§ 6` from the body is not |
| **K2** | A2, A5 | At most **30 %** of the letters removed |
| **K3** | A3 | The output is no longer an exact duplicate |
| **K4** | A4 | The letter count changes by at most **2 %** |

"At most minor" means the route may leave minor residue of that rule behind; A4 then handles it. Any
other rule listed must no longer fire at all.

A route whose acceptance fails leaves the file *partial*: it goes to the next applicable route, or to the
removal decision (§4), which judges whether what is left is usable.

---

## 4. When a file is removed

The removal decision is applied to the **best available text** of each file, i.e. after every triggered
route (including OCR) has run. A file is removed when **any** of these holds:

| Rule | Condition | Meaning |
|---|---|---|
| **R1** | `n_letters` < **1,000** | Not enough text to be a statute |
| **R2** | `prose_letters` < **1,500** | Page chrome only — no statute prose |
| **R3** | `paywall_marker_count` > **0** in the final text **or in any pre-fix text** (raw extraction, OCR output) | Content withheld by a paywall. Deleting the placeholders does not bring the text back |
| **R4** | Any **severe** corruption rule (C1, C2, C3, C4, C6, C7, C8) still fires after all fixes | Corruption that no route could repair |
| **R5** | Exact duplicate of an earlier file | No independent content |
| **R6** | `numbering_disorder` ≥ **0.10** with ≥ **10** unit headings → **REVIEW** | Unit order may be scrambled. A reviewer records KEEP/REMOVE with the evidence in `criteria_overrides.csv`; the pipeline refuses to finalise while any REVIEW is open |

R6 is the only rule that needs a human verdict. Legitimate structures produce the same pattern as real
scrambling (bilingual and trilingual parallel text, amending acts citing other articles, gazettes bundling
several acts, interleaved chapter numbers), so the metric raises the question and the reviewer answers it
from the heading sequence, which is printed in the reason.

**Not removal reasons:**
- Leftover boilerplate (P1–P4) and minor noise (C3/C4 minor band, C5, C9). These are fixable, so they are
  reported as failed route acceptance, not removed.
- A content-scope mismatch: the wrong law, the wrong language, or a summary instead of the statute. Clean
  text stays in the corpus. This is unchanged from the original cleaning policy.

---

## 5. Running it

```bash
python scripts/assess_quality.py --stage raw     # OCR + routing decisions  -> reports/cleaning/quality_reports/routing_decisions.csv
# ... run the triggered route scripts / OCR ...
python scripts/assess_quality.py --stage final   # removal decisions       -> reports/cleaning/quality_reports/removal_decisions.csv
python scripts/c1_quarantine_unusable.py         # dry run: shows what would move
python scripts/c1_quarantine_unusable.py --apply # move files
```

Outputs in `reports/cleaning/quality_reports/`:

| File | Content |
|---|---|
| `raw_metrics.csv` | Every metric for every raw file |
| `routing_decisions.csv` | `needs_ocr`, OCR rule, triggered routes with the firing rules and values, and the previous manual route (`prev_route`) |
| `final_metrics.csv` | Every metric for the best available text |
| `removal_decisions.csv` | KEEP / REVIEW / REMOVE with firing rules, override verdict, `final_decision`, per-route acceptance (PASS / FAIL with the rules still firing), and the current folder (`prev_folder`) |
| `pdf_metrics_cache.json` | Cached PDF text-layer measurements (slow to compute) |

To change a threshold, edit `T` in `scripts/quality_criteria.py`, update the tables above, and re-run both
stages.

---

## 6. Calibration

Thresholds sit in the gap between files that needed an action and files that did not, using the golden
cleaning labels (`golden/cleaning/`) and the 11 files in `globallaws_removed/` as reference. Result on the
228 raw files:

- **OCR (§2):** 12 files flagged, all among the 23 files that were actually re-OCR'd.
- **Removal (§4):** 13 files removed; 10 agree with the historical quarantine. Differences: Egypt (kept by
  rule — defect not measurable, see §7), Italy (d) and Serbia (a) (rule removes pages that are navigation
  or article pickers), Italy (e) (accents lost to a wrong charset; recoverable via route B2).
- **Routes (§3):** 154 of 228 files trigger no route. Small historical fixes (single boilerplate blocks,
  isolated typos) are now below threshold; a few files trigger a route for the first time (repeated running
  headers → A2, detached accents → B2).
- **Route acceptance:** the K1 (headings lost) check flags Czechia (b) (413 of 420 section numbers missing
  after HTML re-extraction) and, for re-check, Kazakhstan, Canada (b), Ukraine (a)/(b), Croatia (b),
  Bolivia (b). Details: `route_acceptance` column of `reports/cleaning/quality_reports/removal_decisions.csv`.

The margins on C3, C4 and C8 are narrow; re-check them when files are added.

---

## 7. Known limitations

1. Substitutions that produce valid letters are invisible to character metrics (Morocco font map, Egypt's
   wrong header word). Catching them needs the perplexity pass in `sanity_check/`.
2. C4 needs at least 11 files in the language (English, Spanish, French only).
3. A scanned PDF with an embedded noisy OCR layer counts as a text-layer PDF; only O2 / the C3 minor band applies.
4. R6 (numbering order) needs a human verdict, recorded in `criteria_overrides.csv`.
5. Language comes from the filename; mislabelled files are judged against the wrong script expectations.
6. PDFs over 60 pages are sampled; results are cached in `reports/cleaning/quality_reports/pdf_metrics_cache.json`.
