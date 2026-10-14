# Stage 1.1 — input processing

Turns the originals (PDF, HTML, TXT) into quality-checked plain text and logs what happened to every
file. Code: `automation/src/input_process/`. TXT→Markdown is stage 1.2 (below).

## Folders

```
data/<project>/
  raw/
    raw_files/            originals: PDF, HTML, TXT
    extracted_txt/        <stem>.txt of the policies that are NOT resolved: files that need OCR or were eliminated
    gold/                 <stem>.txt hand-cleaned texts (optional source of stage 1.2)
    input_process.log     what stage 1 did to each file (appended to; WARNING = needs your attention)
    extraction_manifest.json   hashes of the texts we extracted, to tell them from yours (do not edit)
  preprocessed/
    cleaned_txt/          <stem>.txt that passed the quality check (or that you accepted); kept across runs
```

Each folder can be moved in `pipeline_config.yaml` (`input_process:`). **There are no markers in file names.**
A policy is the same name everywhere (`ACEH_BIREUEN.pdf` → `extracted_txt/ACEH_BIREUEN.txt` →
`cleaned_txt/ACEH_BIREUEN.txt`) and the folder that holds the name says where it stands.

## What happens, in this order

For every policy name in `raw_files/`:

1. **`preprocessed/cleaned_txt/<name>.txt` exists** → done. Skipped (its text only serves as reference corpus).
2. **else `extracted_txt/<name>.txt` exists** → quality-check it. This is how a file you fixed or OCRed is picked up.
3. **else** → extract the original into `extracted_txt/<name>.txt`, then quality-check it.

A policy with several originals (`A.txt`, `A.html`, `A.pdf`) is extracted from one: `txt`, then `html`, then `pdf`.

The quality check measures the text, applies `hygiene` plus the fixers the findings call for, measures it again, and decides
on *that* text. A fix that removes more than 30 % of the letters (K2) or loses more than 5 % of the distinct
*Pasal* numbers (K1) is discarded and only the lossless normalization is kept. Then:

| Result | What happens | Log |
|---|---|---|
| `PASS` | written to `cleaned_txt/<name>.txt` and removed from `extracted_txt/` | `INFO PASS ...` |
| `NEEDS_OCR` (a scanned PDF) | stays in `extracted_txt/` | `WARNING NEEDS_OCR ...` with the failed indicators |
| `ELIMINATE` | stays in `extracted_txt/` | `WARNING ELIMINATED ...` with the failed indicators |

Policies already in `cleaned_txt/` are not checked again; to re-check one, move its file back to `extracted_txt/`.

## Commands

```bash
python -m automation.src.input_process extract          # raw_files/ -> extracted_txt/ (only policies in neither folder)
python -m automation.src.input_process check            # quality check; passed files go to cleaned_txt/; logs everything
python -m automation.src.input_process check --report-only   # only log: nothing is moved or deleted
python -m automation.src.input_process run              # extract, then check (this is what the pipeline step does)
python -m automation.src.input_process tidy [--dry-run] # delete extracted_txt/<name>.txt of policies already in cleaned_txt
python -m automation.src.input_process migrate-layout [--dry-run]   # one-off, data made by the earlier layout
python -m automation.src.input_process markdown [--force] [--small-scale]   # stage 1.2
```

`--small-scale` limits any command to `paths.small_scale_stems`; `--project` overrides the config.

## Fixing a file or doing OCR

Both work the same way, with nothing but file names:

- **A PDF that needs OCR**: OCR it and save the text over `extracted_txt/<name>.txt` (the empty or garbled extraction).
- **An eliminated file**: edit `extracted_txt/<name>.txt` in place, or replace it with your corrected text.
- Run `run` (or the pipeline) again. The file is checked again. Text you supplied is never judged "needs OCR": if it is
  still unusable it is eliminated.
- **To accept a file as it is**: move `extracted_txt/<name>.txt` into `preprocessed/cleaned_txt/`. It is not checked or
  fixed again.

How is your text told from our extraction? `raw/extraction_manifest.json` holds the hash of every file we wrote; a file
that no longer matches (or has no entry) is yours. A PDF-sourced text that is yours is judged with the OCR rules.

If `extracted_txt/<name>.txt` is left for a policy that is already in `cleaned_txt/`, it is ignored and logged as a
WARNING; it is never deleted for you (`tidy` does that on request).

## The log (`raw/input_process.log`)

One line per event: time, level, result and file. It is appended to, so the history of every run stays.

```
2026-10-14 12:03:13,495 INFO    EXTRACTED 3. Algeria - DomViol - French.pdf (pdf, 41 page(s)) -> extracted_txt/3. Algeria - DomViol - French.txt
2026-10-14 12:03:13,496 INFO    PASS ACEH_BIREUEN (txt) -> cleaned_txt/ACEH_BIREUEN.txt [fixes: hygiene]
2026-10-14 12:03:13,497 WARNING NEEDS_OCR 54. Estonia - DomViol - Estonian: S1_scanned_pdf=1.0 (limit 0.5) - OCR 54. Estonia - DomViol - Estonian.pdf and replace extracted_txt/54. Estonia - DomViol - Estonian.txt with the text
2026-10-14 12:03:13,496 WARNING ELIMINATED BALI_BULELENG (txt): C6_glued_words=1.181 (limit 1.0) - fix extracted_txt/BALI_BULELENG.txt (it is checked again) or move it to cleaned_txt/
2026-10-14 12:03:13,499 INFO    QUALITY CHECK finished: 198 policies, 174 in cleaned_txt, 0 need OCR, 24 eliminated (174 were already done)
```

`failed indicators` are written as `rule=value (limit N)` so it is clear how far off a file is. Other WARNING lines: an
original that could not be extracted, an unsupported file, a file left in `extracted_txt/` for a finished policy, and (stage 1.2)
a text with no heading structure.

## Gold texts and the source of stage 1.2

`raw/gold/` can hold hand-cleaned texts (`<stem>.txt`). Which folder stage 1.2 (TXT → Markdown)
converts is a setting:

```yaml
input_process:
  markdown_source: cleaned_txt   # or: gold
  # gold_dir: data/global/raw/gold
```

`cleaned_txt` (the default, `preprocessed/cleaned_txt`) is this stage's output. The chosen folder is
`InputProcessConfig.markdown_source_dir`.

## Stage 1.2: TXT -> Markdown

```bash
python -m automation.src.input_process markdown [--force] [--small-scale]
```

Converts every `.txt` of the chosen source into `data/<project>/preprocessed/cleaned_markdown/<stem>.cleaned.md`
with `txt_to_md_v7.py`; existing outputs are kept unless `--force`. Texts with no recognisable heading
structure are not written; they are logged as a WARNING in `raw/input_process.log`.

- **Vocabulary**: every language's heading words, ordinals, cross-reference phrases, TOC titles and page words are
  in `heading_keywords.py`. To support a language, add its words there.
- **Depths**: a language with a profile in `heading_keywords.PROFILES` (Indonesian) uses fixed depths
  (`BAB #`, `Bagian ##`, `Paragraf ###`, `Pasal ####`, no generated title) and counts a level from 2 line-anchored
  hits. The language of each policy comes from `input_process.language_pattern` / `language`. Other languages
  use depths that follow the levels the file uses.

## Rules

Thresholds live in `quality/criteria.py` (`THRESHOLDS`); override any of them under
`input_process.thresholds`. Each WARNING in the log carries the failed indicators with their value and limit.

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
