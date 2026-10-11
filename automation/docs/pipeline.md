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
| 1.1 | `input_process` | `raw/raw_files` -> text -> quality check and fix -> `raw/cleaned_txt`; report `raw/input_report.json` | **yes**: files that need OCR or were eliminated |
| 1.2 | `txt_to_md` | `cleaned_txt` (or `gold`, see `input_process.markdown_source`) -> `preprocessed/cleaned_markdown` | no (policies without heading structure are listed) |
| 2 | `translation_md` | `cleaned_markdown` -> `preprocessed/translation_markdown` | no |
| 2 | `translation_qa_md` | optional, run with `--steps` | no |
| 3 | `human_review` | optional pause; off unless `human_review.pause: true` | **yes**, only when enabled |
| 4 | `promote` | header check of each pair, then `preprocessed/` -> `processed/` | **yes**: pairs whose headings differ |
| 4 | `md_to_text`, `storage`, `evaluation` | section mapping, per-section embeddings, RAG evaluation | no |
| 4 | `comparison`, `discrepancy_diagnosis` | optional | no |

## Breakpoints

### `input_process`: files that need your attention
The report lists, for every file that did not pass, the indicators that failed.

1. **Needs OCR** (a scanned PDF): OCR it and save the text as `raw/extracted_txt/<name>__ocr.txt`.
2. **Eliminated**: fix the text and save it as `raw/extracted_txt/<name>__txt.txt`; it goes through the
   quality check again. If you accept the file as it is, move it into `raw/cleaned_txt/` instead.
3. Files you do not want can stay where they are.
4. Run the command again. To continue with the files that passed without fixing the others: `--skip-blocked`.

Details: [input_process.md](input_process.md).

### `human_review` (optional)
Set `human_review.pause: true` in `automation/config/pipeline_config.yaml` to review before evaluation.
After translation the run stops. Edit the files in `preprocessed/cleaned_markdown` and
`preprocessed/translation_markdown` if you want to. **Rule: the headings of the two files of a policy must
stay identical** (same number of headings, same level at every position); the text is free to change.
Then run the command again with `--reviewed`.

### `promote`: headings differ
Every pair is checked; pairs that match move to `processed/` right away, the others stay in `preprocessed/`
and are listed with the first difference (for example "heading hierarchy differs at heading 12"). Make the
headings of the two files match and run again; a blocked policy is left out of the evaluation until it
passes. `--skip-blocked` evaluates the policies that are ready.

`promote` never overwrites a file already in `processed/`. To redo a policy, delete its files in
`processed/` and `preprocessed/` first.

## Flags added for breakpoints

| Flag | Meaning |
|------|---------|
| `--skip-blocked` | At a breakpoint caused by files that are not ready (`input_process`, `promote`), go on with the ones that are |
| `--reviewed` | Confirm the optional human review and continue |

`--force` cleans only `data/<project>/automation/`; `raw/`, `preprocessed/` and `processed/` are not touched.

## Folders

```
data/<project>/
  raw/raw_files/             originals (PDF, HTML, TXT)
  raw/extracted_txt/         only unresolved files (OCR needed / eliminated)
  raw/cleaned_txt/           passed texts
  raw/gold/                  hand-cleaned texts (optional source of txt_to_md)
  preprocessed/cleaned_markdown/, translation_markdown/    open to human edits
  processed/cleaned_markdown/, translation_markdown/       validated pairs, the input of evaluation
  automation/                final output of the run
```
