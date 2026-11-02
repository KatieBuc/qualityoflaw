# Stage 1.1 — input processing

Turns the originals (PDF, HTML, TXT) into quality-checked plain text and logs what happened to every
file. Code: `automation/src/input_process/`. TXT→Markdown is stage 1.2 (below).

## Folders

```
data/<project>/
  raw/
    raw_files/            originals: PDF, HTML, TXT
    gold/                 <stem>.txt hand-cleaned texts (optional source of stage 1.2)
    input_process.log     one line per checked file (appended to; WARNING = failed the quality check)
    extraction_manifest.json   hashes of the texts we extracted, to tell them from yours (do not edit)
  preprocessed/
    cleaned_txt/          <stem>.txt: EVERY extracted text, the one folder every later step reads
```

Each folder can be moved in `pipeline_config.yaml` (`input_process:`). **There is one path.** Extraction writes every
text to `preprocessed/cleaned_txt/<name>.txt`; the quality check judges and fixes those same files; stage 1.2 and the
rest read that folder. Nothing is moved between folders and there are no markers in file names: a policy is the same
name everywhere (`ACEH_BIREUEN.pdf` → `cleaned_txt/ACEH_BIREUEN.txt`).

## What happens

**`extract`**: for every policy name in `raw_files/` that has no `cleaned_txt/<name>.txt` yet, extract the original into
that file. A policy that already has its file is not touched, so OCR text or a fix you saved over it is kept. A policy
with several originals (`A.txt`, `A.html`, `A.pdf`) is extracted from one: `txt`, then `html`, then `pdf`.

**`quality_check`**: for every file in `cleaned_txt/`, every run: measure the text, apply `hygiene` plus the fixers the
findings call for, measure it again, and decide on *that* text. A fix that removes more than 30 % of the letters (K2)
or loses more than 5 % of the distinct *Pasal* numbers (K1) is discarded and only the lossless normalization is kept.

| Result | The file | Log |
|---|---|---|
| `PASS` | replaced by the fixed text (only when the fixes changed something) | `INFO PASS <name> (<format>)` |
| `NEEDS_OCR` (a scanned PDF) | left as it is (an empty file) | `WARNING NEEDS_OCR ...` with the failed indicators |
| `ELIMINATE` | left as it is | `WARNING ELIMINATED ...` with the failed indicators |

The failed files stay in `cleaned_txt/`. The pipeline pauses at them (see [pipeline.md](pipeline.md)); with
`--skip-blocked` it goes on and the next steps read them too.

## Commands

```bash
python -m automation.src.input_process extract          # raw_files/ -> cleaned_txt/ (only policies without a file there)
python -m automation.src.input_process check            # quality check of cleaned_txt/; passing texts are fixed in place
python -m automation.src.input_process check --report-only   # only log: no file is rewritten
python -m automation.src.input_process run              # extract, then check (what the two pipeline steps do)
python -m automation.src.input_process migrate-layout [--dry-run]   # one-off, data of an earlier layout
python -m automation.src.input_process markdown [--force] [--small-scale]   # stage 1.2
```

`--small-scale` limits any command to `paths.small_scale_stems`; `--project` overrides the config.

## Fixing a file or doing OCR

Both work the same way, with nothing but file names:

- **A PDF that needs OCR**: OCR it and save the text over `cleaned_txt/<name>.txt` (the empty extraction).
- **An eliminated file**: edit `cleaned_txt/<name>.txt` in place, or replace it with your corrected text.
- Run `run` (or the pipeline) again. `extract` leaves your file alone and `quality_check` judges it again. Text you
  supplied is never judged "needs OCR": if it is still unusable it is eliminated.
- **A policy you do not want**: remove its original from `raw_files/` and its file from `cleaned_txt/` (otherwise
  `extract` would extract it again).

How is your text told from our extraction? `raw/extraction_manifest.json` holds the hash of every file we wrote (and
re-wrote after fixing); a file that no longer matches, or has no entry, is yours. A PDF-sourced text that is yours is
judged with the OCR rules.

## The log (`raw/input_process.log`)

It records only the outcome of the quality check, one line per file: time, level, result and file. It is appended to,
so the history of every run stays.

```
2026-10-14 12:03:13,496 INFO    PASS ACEH_BIREUEN (txt) [fixed: hygiene]
2026-10-14 12:03:13,497 WARNING NEEDS_OCR 54. Estonia - DomViol - Estonian: S1_scanned_pdf=1.0 (limit 0.5) - OCR 54. Estonia - DomViol - Estonian.pdf and replace cleaned_txt/54. Estonia - DomViol - Estonian.txt with the text
2026-10-14 12:03:13,496 WARNING ELIMINATED BALI_BULELENG (txt): C6_glued_words=1.181 (limit 1.0) - fix cleaned_txt/BALI_BULELENG.txt and run the quality check again
2026-10-14 12:03:13,499 INFO    QUALITY CHECK: 198 policies, 174 passed, 0 need OCR, 24 eliminated
```

Failed indicators are written as `rule=value (limit N)` so it is clear how far off a file is. An original that could not
be extracted at all is also an `ELIMINATED` WARNING (it has no file to check). Stage 1.2 adds a WARNING for a text with
no heading structure.

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
