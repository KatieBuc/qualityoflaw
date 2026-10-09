# Stage 1.1 — input processing

Turns the originals (PDF, HTML, TXT) into quality-checked plain text, and reports which files
were dropped and which PDFs need OCR. Code: `automation/src/input_process/`. TXT→Markdown
(stage 1.2) is a separate, not yet implemented step (`automation/src/ingest/`).

## Folders

All under `data/<project>/raw/`; each can be moved in `pipeline_config.yaml` (`input_process:`).

```
raw/
  raw_files/        originals: PDF, HTML, TXT
  extracted_txt/    <stem>__pdf.txt | <stem>__html.txt | <stem>__txt.txt   (+ your <stem>__ocr.txt)
  cleaned_txt/      <stem>.txt  — passed the quality check
  input_report.json
```

The `__pdf` / `__html` / `__txt` / `__ocr` marker records the source format so each file is judged
as what it is (only a PDF can need OCR). It is removed when a file passes into `cleaned_txt/`, so
downstream steps keep the plain policy name.

## Commands

```bash
python -m automation.src.input_process migrate-raw      # one-time: move raw/*.pdf|txt|html into raw/raw_files/
python -m automation.src.input_process extract          # raw_files/ -> extracted_txt/ (existing outputs are kept)
python -m automation.src.input_process check            # quality check + fix + re-check -> cleaned_txt/ + report
python -m automation.src.input_process check --report-only   # report only, cleaned_txt/ untouched
python -m automation.src.input_process run              # extract, then check
```

`--small-scale` limits any command to `paths.small_scale_stems`; `--project` overrides the config.

## What happens

1. **Extract.** PDF → `pdfplumber`, page by page (pages joined by a single newline, since a sentence
   can cross a page break). HTML → `trafilatura`, falling back to BeautifulSoup when trafilatura
   drops headings or more than half the text. TXT → copied as UTF-8 (BOM removed, cp1252 accepted).
   A scanned PDF yields an empty file; the quality check, not the extractor, decides it needs OCR.
   An existing output is never overwritten — delete it to extract again.
2. **Pick the best source** per policy: `__ocr` › `__txt` › `__html` › `__pdf`. A `.txt` you supplied outranks
   text extracted from a PDF/HTML of the same name; the ignored files are listed under `superseded`.
3. **Check, fix, check again.** Measure the text → apply `hygiene` plus the fixers the findings call
   for → measure the fixed text → decide on *that* text. A fix that removes more than 30 % of the
   letters (K2) or loses more than 5 % of the distinct *Pasal* numbers (K1) is discarded and only the
   lossless normalization is kept.
4. **Sort.** `PASS` → `cleaned_txt/<stem>.txt`. `NEEDS_OCR` → report entry naming the file to
   supply. `ELIMINATE` → report entry with the rules that fired.

Every run re-evaluates all of `extracted_txt/`, rewrites the report, and makes `cleaned_txt/` mirror
the latest result (a file that no longer passes is removed from it).

## OCR round trip

1. Run `run` (or `check`). The report's `needs_ocr` lists each PDF with its `expected_ocr_file`.
2. OCR the PDF yourself and save the text as `extracted_txt/<stem>__ocr.txt`.
3. Run `check` again. The OCR text replaces the PDF text; if it is still unusable the file is
   `ELIMINATE`d (it is not sent back to OCR).

## Rules

Thresholds live in `quality/criteria.py` (`THRESHOLDS`); override any of them under
`input_process.thresholds`. Each report line carries the rule IDs that fired and their values.

| Rule | Meaning | Effect |
|---|---|---|
| C1–C9 | corruption: wrong script, garbage characters, noisy tokens (C3), corpus-unique words (C4, OCR text only), shattered words (C5), glued words (C6), mojibake (C7), lost / detached diacritics (C8–C9, not used for Indonesian) | `severe` → eliminate; `minor` → kept, C5 is fixed |
| P1–P4 | boilerplate: web chrome, URLs, repeated lines, HTML residue | fixed, never eliminates |
| S1–S2 | PDF text layer: scanned pages, too few letters per page | feed O1–O3 |
| O1–O3 | the PDF needs OCR | `NEEDS_OCR` |
| R1–R3 | too few letters, too little prose, paywall text | eliminate |
| R4 | a severe C rule that survives fixing | eliminate |
| R6 | unit numbering looks scrambled | flag for review only |
| duplicate | identical text in two policies | flag for review only (dropping one would also drop its golden row) |

Adapted for Indonesian from the GlobalLaws pipeline: one language per run (`input_process.language`);
`Pasal` headings drive the numbering check; wrapped cross-references ("Pasal 5 ayat (1) huruf f")
are not headings; the web-chrome vocabulary uses unambiguous Indonesian phrases only (ordinary words
like *masuk* or *daftar* appear in statutes). The thresholds were calibrated on the other corpus —
check them against your files before relying on them.

## Report (`input_report.json`)

| Key | Content |
|---|---|
| `summary` | counts: files, passed, needs_ocr, eliminated, fixed, flagged_for_review, by_source_format, not_extracted |
| `eliminated` | `{stem, source_format, extracted_file, reason}` |
| `needs_ocr` | `{stem, pdf, extracted_file, reason, expected_ocr_file}` |
| `not_extracted` | originals in `raw_files/` with no extracted text yet |
| `unrecognized_files` | files in `extracted_txt/` without a known marker (ignored) |
| `files` | per policy: `status`, `source_format`, `extracted_file`, `fixes_applied`, `fix_rejected`, `findings_before`, `findings_after`, `metrics`, `review`, `duplicate_of`, `superseded`, `cleaned_file` |
| `thresholds` | the thresholds used (defaults plus your overrides) |

## Supporting another language

Add a `LanguageProfile` to `PROFILES` in `quality/metrics.py` (expected scripts, whether words are
space-separated, whether diacritics are expected) and set `input_process.language`. Unknown names
fall back to Latin script, space-separated words.

## The `scripts/` folder

`automation/src/input_process/scripts/` is the finished cleaning pipeline of another project
(GlobalLaws): per-file fixes with hardcoded paths. Nothing here imports it; the reusable parts
(metrics, rules, hygiene, HTML extraction) were ported into this package.
