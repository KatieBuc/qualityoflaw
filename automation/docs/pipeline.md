# Pipeline and breakpoints

One command runs the whole pipeline. Where a person has to act, the run **pauses** (a breakpoint): it
says what happened, what to do next and which docs to read, then stops without an error (exit code 0,
`metadata.json` status `paused`). Run the **same command again** to continue; every step skips finished work.

```bash
python -m automation.src.run_pipeline                 # all stages
python -m automation.src.run_pipeline --small-scale   # only paths.small_scale_stems
```

## Stages and steps

| Stage | Step | What it does | Breakpoint |
|---|---|---|---|
| 1.1 | `input_process` | `raw/raw_files` -> text -> quality check and fix -> `preprocessed/cleaned_txt`; log `raw/input_process.log` | **yes**: files that need OCR or were eliminated |
| 1.2 | `txt_to_md` | `cleaned_txt` (or `gold`, see `input_process.markdown_source`) -> `preprocessed/cleaned_markdown` | no (policies without heading structure are listed) |
| 2 | `translation_md` | `cleaned_markdown` -> `preprocessed/translation_markdown` | no |
| 2 | `translation_qa_md` | optional, run with `--steps` | no |
| 3 | `human_review` | optional pause; off unless `human_review.pause: true` | **yes**, only when enabled |
| 4 | `validate` | header check of every pair in `preprocessed/`; mismatches go to `preprocessed/header_check.log`. **Moves nothing** | **yes**: while `processed/` is empty |
| 4 | `md_to_text`, `storage`, `evaluation` | section mapping, per-section embeddings, RAG evaluation | no |
| 4 | `comparison`, `discrepancy_diagnosis` | optional | no |

## Breakpoints

### `input_process`: files that need your attention
The log (`raw/input_process.log`) has a WARNING line for every file that did not pass, with the indicators that failed.

1. **Needs OCR** (a scanned PDF): OCR it and save the text over `raw/extracted_txt/<name>.txt`.
2. **Eliminated**: fix `raw/extracted_txt/<name>.txt` (it is checked again). If you accept the file as it is, move it into
   `preprocessed/cleaned_txt/` instead.
3. Files you do not want can stay where they are.
4. Run the command again. To continue with the files that passed without fixing the others: `--skip-blocked`.

No markers in file names: the file is always `<name>.txt`. Details: [input_process.md](input_process.md).

### `human_review` (optional)
Set `human_review.pause: true` in `automation/config/pipeline_config.yaml` to review before evaluation.
After translation the run stops. Edit the files in `preprocessed/cleaned_markdown` and
`preprocessed/translation_markdown` if you want to. **Rule: the headings of the two files of a policy must
stay identical** (same number of headings, same level at every position); the text is free to change.
Then run the command again with `--reviewed`.

### `validate`: you move the pairs to `processed/`
The pipeline **never moves files into `processed/`**. The `validate` step checks the headings of every pair in
`preprocessed/` and writes one line per pair to `preprocessed/header_check.log` (`INFO OK ...`, `WARNING MISMATCH ... <why>`),
and prints the mismatches. While `processed/` holds no translation yet, the run pauses:

1. Fix each mismatch (make the headings of the two files match).
2. Move each pair you accept: the file from `preprocessed/cleaned_markdown` to `processed/cleaned_markdown` and the file
   from `preprocessed/translation_markdown` to `processed/translation_markdown` (same file names).
3. Run the command again. Policies still in `preprocessed/` are not evaluated; they are checked and reported on every run.

`md_to_text` checks the headings of the pairs in `processed/` once more, so a mismatching pair you moved is reported there.

## Flags added for breakpoints

| Flag | Meaning |
|------|---------|
| `--skip-blocked` | At the `input_process` breakpoint, go on with the files that passed |
| `--reviewed` | Confirm the optional human review and continue |

`--force` cleans only `data/<project>/automation/`; `raw/`, `preprocessed/` and `processed/` are not touched.

## Folders

```
data/<project>/
  raw/raw_files/             originals (PDF, HTML, TXT)
  raw/extracted_txt/         <name>.txt of unresolved files only (OCR needed / eliminated)
  raw/input_process.log      stage 1 log
  raw/gold/                  hand-cleaned texts (optional source of txt_to_md)
  preprocessed/cleaned_txt/  passed texts
  preprocessed/cleaned_markdown/, translation_markdown/    open to human edits; header_check.log
  processed/cleaned_markdown/, translation_markdown/       pairs you moved here, the input of evaluation
  automation/                final output of the run
```
