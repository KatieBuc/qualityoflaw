# Stage 1.1 — input processing

Turns the originals (PDF, HTML, TXT) into quality-checked plain text, and reports which files
were dropped and which PDFs need OCR. Code: `automation/src/input_process/`. TXT→Markdown
(stage 1.2) is a separate, not yet implemented step (`automation/src/ingest/`).

## Folders

All under `data/<project>/raw/`; each can be moved in `pipeline_config.yaml` (`input_process:`).

```
raw/
  raw_files/        originals: PDF, HTML, TXT
  extracted_txt/    only what is NOT resolved yet: files that need OCR or were eliminated
                    <stem>__pdf.txt | <stem>__html.txt | <stem>__txt.txt   (+ your <stem>__ocr.txt)
  cleaned_txt/      <stem>.txt  — passed the quality check (or accepted by you); kept across runs
  gold/             <stem>.txt  — hand-cleaned texts (optional input of stage 1.2)
  input_report.json
```

The `__pdf` / `__html` / `__txt` / `__ocr` marker records the source format so each file is judged
as what it is (only a PDF can need OCR). It is removed when a file passes into `cleaned_txt/`, so
downstream steps keep the plain policy name.

**A file that passes is moved, not copied.** Its text is written to `cleaned_txt/<stem>.txt` and every
extracted file of that policy is deleted from `extracted_txt/`. After a run, `extracted_txt/` holds only
the work that is left, and `cleaned_txt/` is never pruned: a policy in it is done.

## Commands

```bash
python -m automation.src.input_process migrate-raw      # one-time: move raw/*.pdf|txt|html into raw/raw_files/
python -m automation.src.input_process extract          # raw_files/ -> extracted_txt/ (skips policies already extracted or in cleaned_txt)
python -m automation.src.input_process check            # quality check + fix + re-check; passed files move to cleaned_txt/; writes the report
python -m automation.src.input_process check --report-only   # report only: nothing is moved, renamed or deleted
python -m automation.src.input_process run              # extract, then check
python -m automation.src.input_process tidy [--dry-run] # one-off: delete extracted files of policies already in cleaned_txt
```

`--small-scale` limits any command to `paths.small_scale_stems`; `--project` overrides the config.

## What happens

1. **Extract.** PDF → `pdfplumber`, page by page (pages joined by a single newline, since a sentence
   can cross a page break). HTML → `trafilatura`, falling back to BeautifulSoup when trafilatura
   drops headings or more than half the text. TXT → copied as UTF-8 (BOM removed, cp1252 accepted).
   A scanned PDF yields an empty file; the quality check, not the extractor, decides it needs OCR.
   An existing output is never overwritten, and a policy already in `cleaned_txt/` is skipped, so
   passed files are not extracted again.
2. **Pick the best source** per policy: `__ocr` › `__txt` › `__html` › `__pdf`. A `.txt` you supplied outranks
   text extracted from a PDF/HTML of the same name.
3. **Check, fix, check again.** Measure the text → apply `hygiene` plus the fixers the findings call
   for → measure the fixed text → decide on *that* text. A fix that removes more than 30 % of the
   letters (K2) or loses more than 5 % of the distinct *Pasal* numbers (K1) is discarded and only the
   lossless normalization is kept. The unique-word rule compares an OCR text with the other texts of
   its language, **including those already in `cleaned_txt/`**.
4. **Sort.** `PASS` → written to `cleaned_txt/<stem>.txt`, and all its extracted files are removed.
   `NEEDS_OCR` → report entry naming the file to supply. `ELIMINATE` → report entry with the
   indicators that failed. Both stay in `extracted_txt/`.

Each `check` re-evaluates what is in `extracted_txt/`. Policies already in `cleaned_txt/` are not
checked again; to re-check one, move its file back to `extracted_txt/` as `<stem>__txt.txt`.

## Accepting a file by hand (per-file exception)

Move the file from `extracted_txt/` to `cleaned_txt/`. That is all.

- The name may keep its marker (`<stem>__txt.txt`); the next `check` renames it to `<stem>.txt` and
  lists it under `adopted` in the report. The text is left exactly as you put it: it is not checked or
  fixed.
- Use this for an eliminated file you have read and accept, or one you corrected by hand.
- If `cleaned_txt/<stem>.txt` already exists it is never overwritten; the report lists the moved file
  under `warnings.cleaned_txt_conflicts` and leaves both.
- Files left in `extracted_txt/` for a policy that is already in `cleaned_txt/` are ignored and listed
  under `warnings.already_in_cleaned_txt`. They are never deleted for you (`tidy` does that on request).

## OCR round trip

1. Run `run` (or `check`). The report's `needs_ocr` lists each PDF with its `expected_ocr_file` and the indicators that failed.
2. OCR the PDF yourself and save the text as `extracted_txt/<stem>__ocr.txt`.
3. Run `check` again. The OCR text replaces the PDF text; if it passes it moves to `cleaned_txt/`
   (taking the PDF text with it), and if it is still unusable the file is `ELIMINATE`d (it is not
   sent back to OCR).

## Gold texts and the source of stage 1.2

`raw/gold/` can hold hand-cleaned texts (`<stem>.txt`). Which folder stage 1.2 (TXT → Markdown)
converts is a setting:

```yaml
input_process:
  markdown_source: cleaned_txt   # or: gold
  # gold_dir: data/global/raw/gold
```

`cleaned_txt` (the default) is this stage's output. The chosen folder is
`InputProcessConfig.markdown_source_dir`.

## Stage 1.2: TXT -> Markdown

```bash
python -m automation.src.input_process markdown [--force] [--small-scale]
```

Converts every `.txt` of the chosen source into `data/<project>/preprocessed/cleaned_markdown/<stem>.cleaned.md`
with `txt_to_md_v7.py`; existing outputs are kept unless `--force`. Texts with no recognisable heading
structure are not written; they are listed under `skipped` in `raw/markdown_report.json`.

- **Vocabulary**: every language's heading words, ordinals, cross-reference phrases, TOC titles and page words are
  in `heading_keywords.py`. To support a language, add its words there.
- **Depths**: a language with a profile in `heading_keywords.PROFILES` (Indonesian) uses fixed depths
  (`BAB #`, `Bagian ##`, `Paragraf ###`, `Pasal ####`, no generated title) and counts a level from 2 line-anchored
  hits. The language of each policy comes from `input_process.language_pattern` / `language`. Other languages
  use depths that follow the levels the file uses.

## Rules

Thresholds live in `quality/criteria.py` (`THRESHOLDS`); override any of them under
`input_process.thresholds`. Each report entry carries the failed indicators with their value and limit.

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

Ported from the GlobalLaws pipeline with its multilingual support kept, plus Indonesian: `Pasal`
headings next to `Article` / `Artículo` / `Artigo` / `Artikel` / `Madde` ... drive the numbering check;
wrapped cross-references ("Pasal 5 ayat (1) huruf f") are not headings; the web-chrome and paywall
vocabularies cover Indonesian, English, Spanish, French, German, Portuguese and Croatian (only
unambiguous Indonesian phrases: ordinary words like *masuk* or *daftar* appear in statutes). The
thresholds were calibrated on GlobalLaws — check them against your own files before relying on them.

## Report (`input_report.json`)

Short on purpose: counts, then only the files that did **not** pass and the indicators they failed.
Passed files are not listed.

```json
{
  "generated_at": "...", "project": "global",
  "summary": {"files": 228, "passed": 174, "needs_ocr": 43, "eliminated": 11},
  "needs_ocr": [
    {"stem": "54. Estonia - DomViol - Estonian", "pdf": "54. Estonia - DomViol - Estonian.pdf",
     "expected_ocr_file": "54. Estonia - DomViol - Estonian__ocr.txt",
     "failed": [{"rule": "S1_scanned_pdf", "value": 1.0, "limit": 0.5}]}
  ],
  "eliminated": [
    {"stem": "2. Albania - DomViol - Albanian (b)", "source_format": "pdf",
     "failed": [{"rule": "C6_glued_words", "value": 25.198, "limit": 1.0}]}
  ]
}
```

| Field | Meaning |
|---|---|
| `summary.passed` | policies in `cleaned_txt/` after this run, including earlier runs and files you moved there; `files` = `passed` + `needs_ocr` + `eliminated` |
| `failed[].rule` | the indicator (table above); a severe corruption rule that survives fixing is shown under its own name, e.g. `C6_glued_words`, not `R4_C6_...` |
| `failed[].value` / `limit` | what was measured, and the threshold it was compared with (so it is clear how far off it is) |
| `needs_ocr[].expected_ocr_file` | the name to give the OCR text in `extracted_txt/` |
| `warnings` | present **only** when needed: `not_extracted` (originals with no extracted text yet), `unrecognized_files` (files in `extracted_txt/` without a known marker), `already_in_cleaned_txt` and `cleaned_txt_conflicts` (see above) |
| `adopted` | present only when you moved files into `cleaned_txt/` by hand: `["<stem>__txt.txt -> <stem>.txt"]` |

Details that used to be in the report — fixes applied, findings before / after, metrics, review
flags, ignored sources — are still computed (`pipeline.analyze()` returns them) but are not written.

## Languages

Every policy is judged as its own language:

- `input_process.language` is the default (Indonesian).
- `input_process.language_pattern` is an optional regex with **one group** that picks the language out
  of a policy name; names it does not match use the default. For GlobalLaws,
  `DomViol - ([A-Za-z]+)` reads `Azerbaijani` out of `10. Azerbaijan - DomViol - Azerbaijani - html`, while
  `ACEH_BIREUEN` falls back to Indonesian, so one config serves both projects.
- The language decides the expected script (`LANG_SCRIPTS`: Arabic, Armenian, Bhutanese, Bulgarian,
  Chinese, Georgian, Greek, Hebrew, Japanese, Korean, Macedonian, Russian, Thai, Ukrainian, Bosnian;
  everything else is Latin), whether words are space-separated (`UNSPACED_LANGS`: Chinese, Japanese,
  Thai, Bhutanese — token metrics are skipped), and whether lost diacritics are checked
  (`DIACRITIC_LANGS`: Spanish, French, German, ...; not Indonesian, Malay or English). Name typos are
  corrected through `LANG_ALIASES` (`Potuguese`).
- The corpus-relative unique-word metric (OCR text only) is computed per language and needs more than
  10 other files of the same language.

The tables live in `quality/metrics.py`; add a language there to give it its own script or rules.

## The `scripts/` folder

`automation/src/input_process/scripts/` is the finished cleaning pipeline of another project
(GlobalLaws): per-file fixes with hardcoded paths. Nothing here imports it; the reusable parts
(metrics, rules, hygiene, HTML extraction) were ported into this package.
