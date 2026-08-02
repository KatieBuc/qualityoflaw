# Automation Pipeline

Config-driven orchestration for policy translation, LLM quality evaluation, golden-dataset comparison, and discrepancy diagnosis.

Golden-dataset comparison logic lives in [`src/evaluate_accuracy.py`](src/evaluate_accuracy.py).

## Layout

```
automation/
├── config/
│   ├── model_config.yaml      # Model deployment parameters
│   └── pipeline_config.yaml   # Experiment combination
├── prompts/
│   ├── translation/v1/
│   ├── quality_eval/v1/
│   └── discrepancy_diagnosis/v1/
└── src/
    ├── llm/                   # Shared Azure LLM wrapper
    ├── evaluate_accuracy.py   # Golden-dataset comparison
    ├── run_pipeline.py        # Main entry point
    ├── translate.py
    ├── run_eval.py
    ├── diagnose.py            # Discrepancy diagnosis (final pipeline step)
    ├── diagnose_prompt_builder.py
    └── compare.py             # Thin wrapper around evaluate_accuracy
```

## Run output (`data/automation/<run_id>/`)

Each pipeline execution writes a self-contained run folder under `data/automation/`. The run ID format is `YYYYMMDD_HHMMSS` (UTC).

```
data/automation/<run_id>/
├── metadata.json              # Run summary (steps, timing, config, counts)
├── failures.json              # Persistent log of translation/evaluation/diagnosis failures (when present)
├── config/                    # Snapshot of configs used for this run
│   ├── model_config.yaml
│   └── pipeline_config.yaml
├── results/                   # Main deliverables of the pipeline
│   ├── cleaned_text/          # OCR-noise-stripped, reflowed original-language text, rendered as Markdown (one per input policy)
│   │   └── <policy>.cleaned.md
│   ├── translation/           # English policy .txt files (one per input policy) — plain text, read by storage/RAG/evaluation
│   ├── translation_markdown/  # Markdown-rendered copy of the translation output, for human reading only
│   │   └── <policy>.md
│   ├── evaluation/            # LLM JSON reports (one per policy)
│   ├── comparison/            # Golden-dataset comparison outputs
│   │   ├── metrics.csv
│   │   ├── error_analysis.csv
│   │   └── unmatched_indicators.csv   # Only when join pairs are missing on one side
│   └── diagnosis/             # Discrepancy diagnosis reports (one per policy with mismatches)
└── mid_product/                # Intermediate artifacts consumed by later stages
    ├── chunks/                # Per-policy chunk boundaries + per-chunk translations
    │   └── <policy>.chunks.json
    ├── rag_store/              # Per-policy embedded chunk store
    └── rag_candidates/         # Full RAG candidate list per indicator (one per policy, written by evaluation)
```

`results/` holds the pipeline's final deliverables; `mid_product/` holds intermediate artifacts that only exist to feed later stages.

### `metadata.json`

| Field | Description |
|-------|-------------|
| `run_id` | Unique run identifier |
| `experiment_name` | From `pipeline_config.yaml` |
| `status` | `running`, `completed`, `completed_with_errors`, or `failed` |
| `timestamps.created_at` / `timestamps.updated_at` | UTC timestamps |
| `config` | Model and prompt versions used |
| `execution_scope.steps_executed` | List of completed steps: `translation`, `storage`, `evaluation`, `comparison`, `discrepancy_diagnosis` |
| `execution_scope.small_scale` | Whether only the 5 benchmark policies were processed |
| `execution_scope.evaluated_policy_files` | Policy filenames compared during the comparison step |
| `file_counts` | Per-step (`translation`, `storage`, `evaluation`, `discrepancy_diagnosis`) success/failure counts |
| `failures` | Count of logged failures per step (see `failures.json`) |
| `token_usage` | Aggregated LLM token usage (translation, evaluation, and discrepancy_diagnosis steps) |
| `timing_seconds` | Elapsed seconds per step (`translation`, `evaluation`, `comparison`, `discrepancy_diagnosis`) and `total` |

Example:

```json
{
  "run_id": "20260704_062818",
  "experiment_name": "first-full-run",
  "status": "completed",
  "timestamps": {
    "created_at": "2026-07-04T06:28:18.354534+00:00",
    "updated_at": "2026-07-05T04:01:01.378106+00:00"
  },
  "config": {
    "translation_model": "claude-sonnet-4-5",
    "evaluation_model": "gpt-5.2",
    "discrepancy_diagnosis_model": "gpt-5.2-diagnosis",
    "translation_prompt_version": "v2",
    "evaluation_prompt_version": "v1",
    "discrepancy_diagnosis_prompt_version": "v1",
    "concurrency": { "enabled": true, "max_workers": 5 },
    "chunking": { "enabled": true, "safe_limit": 32000 }
  },
  "execution_scope": {
    "steps_executed": ["translation", "evaluation", "comparison"],
    "small_scale": false,
    "evaluated_policy_files": ["ACEH_BIREUEN.txt", "ACEH_SUBULUSSALAM_KOTA.txt"]
  },
  "file_counts": {
    "translation": { "total": 190, "succeeded": 1, "skipped": 189, "failed": 0 },
    "evaluation": { "total": 190, "succeeded": 1, "skipped": 189, "failed": 0, "saved_reports": 1 }
  },
  "failures": { "translation": 0, "evaluation": 0, "discrepancy_diagnosis": 0 },
  "token_usage": {
    "translation": { "prompt_tokens": 15458, "completion_tokens": 7397, "total_tokens": 22855 },
    "evaluation": { "prompt_tokens": 50772, "completion_tokens": 4309, "total_tokens": 55081 }
  },
  "timing_seconds": { "translation": 257.79, "evaluation": 13.13, "comparison": 0.21, "total": 271.13 }
}
```

### `translation/`

UTF-8 `.txt` files with the same basenames as the Indonesian source policies (e.g. `ACEH_BIREUEN.txt`). Filenames must match the `filename` column in the golden CSV for comparison to work.

### `evaluation/`

Per-policy JSON reports named `<timestamp>-<model>-<POLICY_BASENAME>.json`. Each report contains:

- `policy_file` — basename used to join with golden data
- `evaluation_results` — 56 indicators keyed by `indicator_id`, each with `included` (`"Yes"` / `"No"`)
- `model`, `evaluated_at`, `prompt_version`

If multiple reports exist for the same policy, comparison keeps the **latest** by timestamp.

### `rag_candidates/`

Written by the `evaluation` step alongside each policy's report. One JSON file per policy (no timestamp prefix, unlike `evaluation/`): `<stem>.json`, matching the policy's basename minus `.txt`. Contains the **full** RAG candidate list retrieved for every indicator — not just the single candidate the judge cited as `evidence` — so the `discrepancy_diagnosis` step can inspect what evidence was available without recomputing retrieval:

```json
{
  "policy_file": "ACEH_BIREUEN.txt",
  "candidates": {
    "4.2": [{ "chunk_id": 7, "text": "...", "score": 0.81 }, ...]
  }
}
```

Only written for policies whose evaluation report was actually saved (matches `finalize_and_save_report`'s success/`--allow-partial` conditions). Runs whose `evaluation` step executed before this directory existed do not have it — re-run `evaluation` with `--force` (evaluation skips policies with an existing report by default) to backfill it before running `discrepancy_diagnosis`.

### `failures.json`

Written when translation, storage, evaluation, or discrepancy_diagnosis failures occur, or when a step fails outright (e.g. missing prerequisite files). Each step maintains its own list of per-item failures; re-running a successful item removes it from the log. Whole-step failures (the step raised before producing any per-item results) are recorded separately under `step_failures` and printed to the CLI as `Step '<name>' failed: <message>`.

| Field (`step_failures` entry) | Description |
|--------------------------------|-------------|
| `step` | Name of the step that failed (`translation`, `storage`, `evaluation`, `comparison`, `discrepancy_diagnosis`) |
| `message` | Human-readable error message |
| `at` | UTC timestamp |

| Field (translation entry) | Description |
|---------------------------|-------------|
| `filename` | Source policy basename |
| `error_type` | Exception class (e.g. `RateLimitError`) |
| `message` | Human-readable error message |
| `details` | `status_code`, `request_id`, and other API metadata |
| `attempts` | Number of retry attempts before failure |
| `at` | UTC timestamp |

| Field (evaluation entry) | Description |
|--------------------------|-------------|
| `policy_file` | Policy basename |
| `failed_dimensions` | Criteria files that failed |
| `errors` | Per-dimension error messages |
| `error_type` / `message` / `details` / `attempts` / `at` | Same as translation |

| Field (discrepancy_diagnosis entry) | Description |
|--------------------------------------|-------------|
| `policy_file` | Policy basename |
| `error_type` | Exception class, or `DiagnosisIncomplete` when some discrepancies couldn't be diagnosed and `--allow-partial` wasn't set |
| `message` / `details` / `attempts` / `at` | Same as translation |

### `comparison/` — golden-dataset reports

Comparison joins LLM results to [`data/processed/long_policy_encoding.csv`](../data/processed/long_policy_encoding.csv) on **`(filename, indicator_id)`**. Golden rows are filtered to evaluated policies only; when a `year` column exists, the latest year per policy is used.

#### `metrics.csv`

One row per evaluation dimension, plus a leading `overall` row:

| Column | Description |
|--------|-------------|
| `Dimension` | `overall`, or an evaluation dimension name (e.g. `budget_funding_sources`) |
| `Accuracy` | Fraction of matched indicator pairs where LLM agrees with golden, to 6 decimal places |
| `Matched_Pairs` | Number of `(filename, indicator_id)` pairs compared |
| `Correct_Pairs` | Matched pairs where golden and LLM agree |
| `Error_Count` | Matched pairs where golden and LLM disagree |
| `Notes` | Only set on the `overall` row: confusion-matrix counts as `TP:.., TN:.., FP:.., FN:..` |

Example:

```csv
Dimension,Accuracy,Matched_Pairs,Correct_Pairs,Error_Count,Notes
overall,0.898853,10638,9562,1076,"TP:5575, TN:3987, FP:838, FN:238"
budget_funding_sources,0.956842,950,909,41,
institutional_mechanism,0.863916,1139,984,155,
```

Policy filenames evaluated in the run are recorded in `metadata.json` under `execution_scope.evaluated_policy_files` instead of in this file.

#### `error_analysis.csv`

One row per **value mismatch** (golden ≠ LLM prediction):

| Column | Description |
|--------|-------------|
| `fullname` | Human-readable policy name from golden CSV |
| `filename` | Policy file basename |
| `indicator_id` | Criterion ID (e.g. `1.10`) |
| `indicator_value` | Criterion name |
| `dimension` | Evaluation dimension |
| `golden_label` / `pred_label` | `Yes` or `No` |
| `value` / `pred_value` | Numeric labels (1.0 = Yes, 0.0 = No) |
| `error_type` | `false_positive` (golden No, LLM Yes) or `false_negative` (golden Yes, LLM No) |
| `evidence` | The judge's cited evidence snippet for this indicator, from `evaluation/` |
| `rationale` | The judge's rationale for this indicator, from `evaluation/` |
| `discrepancy_root_cause` | Always `Reference only` — root causes are produced by the `discrepancy_diagnosis` step (see `diagnosis/<stem>.json`), which runs after this file is written and doesn't write back into it |

#### `unmatched_indicators.csv`

Written only when some `(filename, indicator_id)` pairs cannot be joined:

| `match_status` | Meaning |
|----------------|---------|
| `golden_only` | In golden CSV, missing from LLM report |
| `llm_only` | In LLM report, missing from golden for that policy |

These rows are **excluded from accuracy**. An empty or absent file means all pairs joined successfully.

### `diagnosis/` — discrepancy diagnosis reports

Written by the `discrepancy_diagnosis` step, the final stage of the pipeline. For every policy with at least one mismatch in `comparison/error_analysis.csv`, one JSON report is written to `<stem>.json` (no timestamp prefix, matching `rag_candidates/`'s convention). Policies with zero mismatches get no file. If `error_analysis.csv` doesn't exist at all (comparison found zero mismatches, or hasn't run), the whole step is a no-op and the `diagnosis/` directory isn't created.

For each mismatched indicator, the step gathers the original Indonesian policy text, the golden and predicted labels, the RAG evidence candidates persisted in `rag_candidates/` during evaluation, and the judge's cited evidence snippet + rationale from `evaluation/`, then asks an LLM to diagnose the root cause.

```json
{
  "policy_file": "ACEH_BIREUEN.txt",
  "model": "gpt-5.2-diagnosis",
  "diagnosed_at": "2026-07-14T05:00:00+00:00",
  "prompt_version": "v1",
  "discrepancy_count": 4,
  "diagnosed_count": 4,
  "missing_diagnoses": [],
  "unresolved_indicators": [],
  "diagnoses": {
    "4.2": {
      "indicator": "...",
      "dimension": "...",
      "golden_label": "Yes",
      "pred_label": "No",
      "error_type": "false_negative",
      "translated_snippet": "...",
      "evaluator_rationale": "...",
      "evidence_candidates": [{ "chunk_id": 7, "text": "...", "score": 0.81 }],
      "root_causes": ["rag_candidate_issue"],
      "rationale": "...",
      "diagnosis_status": "ok"
    }
  }
}
```

| Field | Description |
|-------|-------------|
| `discrepancy_count` | Number of mismatched indicators for this policy |
| `diagnosed_count` | `discrepancy_count` minus `missing_diagnoses` and `unresolved_indicators` |
| `missing_diagnoses` | Indicator IDs the diagnosis LLM omitted from its response |
| `unresolved_indicators` | Indicator IDs with no entry in `rag_candidates/` (excluded from the LLM call) |
| `diagnoses.<indicator_id>.root_causes` | One or more of `translation_quality`, `evaluation_failure`, `rag_candidate_issue`, `golden_label_issue` |
| `diagnoses.<indicator_id>.diagnosis_status` | `ok`, `missing_from_llm_response`, or `unresolved_indicator` |

A policy with any `missing_diagnoses` or `unresolved_indicators` is treated as **partial**; its report is only written if `--allow-partial` is set (otherwise it's recorded as a failure in `failures.json` and no file is written).

## Configuration

### `config/model_config.yaml`

Named model profiles (credentials stay in `.env`):

```yaml
models:
  gpt-5.2-translate:
    deployment: gpt-5.2
    temperature: 0.2
    max_tokens: null
    max_retries: 3

  gpt-5.2-diagnosis:
    deployment: gpt-5.2
    temperature: 0.2
    max_tokens: null
    max_retries: 3
```

### `config/pipeline_config.yaml`

Selects which models and prompt versions to use:

```yaml
experiment_name: baseline_v1
concurrency:
  enabled: true
  max_workers: 5
translation:
  model: gpt-5.2-translate
  prompt_version: v1
evaluation:
  model: gpt-4o-eval
  prompt_version: v1
discrepancy_diagnosis:
  model: gpt-5.2-diagnosis
  prompt_version: v1
```

`discrepancy_diagnosis.model`/`prompt_version` are required just like `translation`/`evaluation`'s (config is validated eagerly on every invocation, regardless of which `--steps` are selected). The template lives at `automation/prompts/discrepancy_diagnosis/<prompt_version>/prompt_template.txt`.

#### Markdown rendering (`markdown` step)

`results/cleaned_text/<policy>.cleaned.md` and `results/translation_markdown/<policy>.md`
are written by their own pipeline step, `markdown` — independent of `translation`,
and idempotent: it only (re)renders a policy whose `.md` doesn't exist yet,
so it's safe to run anytime (`--steps markdown`) to backfill whatever is
missing, without re-running translation.

- `results/cleaned_text/<policy>.cleaned.md` is rendered straight from the raw
  original-language input (`data/raw/localpolicies/<policy>.txt`) — it has no
  dependency on translation having run.
- `results/translation_markdown/<policy>.md` is rendered from
  `results/translation/<policy>.txt` (the translated English output), so a
  given policy is only rendered once its translation exists; policies not
  yet translated are silently skipped, not treated as failures.
- `results/translation/<policy>.txt` itself always stays plain text — it's
  what the storage/RAG and evaluation steps read, and reformatting it as
  Markdown in place would change both.

Both artifacts are rendered by `chunking.render_markdown` on top of
`chunking.clean_text` (OCR-noise removal, paragraph reflow, structural-marker
detection — `automation/src/chunking.py:clean_text`): structural markers
become headings nested by keyword (`BAB`/`Chapter` → `#`,
`Bagian`/`Part`/`Section` → `##`, `Paragraf`/`Paragraph` → `###`,
`Pasal`/`Article` → `####`, with an unrecognized/foreign-language marker
nested one level under the most recent keyword heading), and
numbered/lettered/roman-numeral list starts become proper
ordered/nested-bullet Markdown lists.

`markdown` runs by default as part of the full pipeline (right after
`translation`). See `run_markdown_step` in `automation/src/translate.py`.

#### Chunked translation (`translation.chunking`)

Every chunked translation run cleans and structurally splits each source file
first — strips OCR noise lines (including stray lone-punctuation residue like
a leftover `.` on its own line), reflows broken lines back into paragraphs,
and keeps structural markers (`BAB` / `Pasal` / `Bagian` / `Paragraf` /
all-caps titles) on their own line (`automation/src/chunking.py:clean_text`) —
this is internal to chunking (it decides the chunk boundaries) and isn't
persisted on its own; the `markdown` step above re-derives the same cleaning
from the raw input when it renders `results/cleaned_text/*.cleaned.md`.

Some source policy files are large enough that a single translation call risks
a timeout. When `translation.chunking.enabled` is `true`, `translate.py` runs
each file through the rest of the chunk -> translate -> combine pipeline
instead of a single API call:

1. **Cleaning** — see above.
2. **Structural split** — cuts the cleaned text into sections at structural
   markers (no overlap; these are the document's natural boundaries).
   Consecutive markers with no body text between them (e.g. a `BAB` line
   followed immediately by its all-caps title and then `Pasal 1`) merge into
   the same section instead of each becoming a one-line section — a new
   section only starts once a marker follows actual body content.
3. **Fallback split** — only for a section that still exceeds `safe_limit`
   characters (e.g. a Lampiran/appendix): splits further on sentence
   boundaries (`.` / `;`), carrying the previous sub-chunk's trailing text as
   non-translated context so the model can resolve continuations.
4. **Translate** — structural chunks are translated directly with the normal
   translation prompt; fallback chunks use
   `automation/prompts/translation/chunking/fallback_prompt.txt`, which
   instructs the model to translate only the marked target text.
5. **Combine** — sub-chunks within a section are concatenated directly;
   sections are joined with a blank line.

```yaml
translation:
  model: gpt-5.2-translate
  prompt_version: v1
  chunking:
    enabled: true
    safe_limit: 32000   # chars; default 32000, well under the ~50,000 timeout ceiling
```

Chunking is disabled by default — existing runs are unaffected unless `enabled: true` is set.

When chunking is enabled, `mid_product/chunks/<policy>.chunks.json` is also always written — the chunk list (section 2 output): `chunk_index`, `section_id`, `type` (`structural`/`fallback`), `context`, `text`, `translated_text`, in translation order. The RAG storage step reuses it instead of re-chunking the merged English output.

##### Embedding chunk size (storage step)

`translation.chunking.safe_limit` sizes chunks in characters against the *translation* LLM's context window — it's much larger than the *embedding* model's hard 8192-token input cap, and legal/citation-heavy text (dense with digits and punctuation) can tokenize less efficiently than prose, so a chunk under `safe_limit` can still exceed 8192 tokens at embedding time. The storage step sub-splits any chunk over `EMBEDDING_SAFE_CHAR_LIMIT` (6000 chars, `automation/src/rag/store.py`) before calling the embedding API, independent of the translation chunk boundaries — this only affects `mid_product/rag_store/`'s retrieval granularity, not `mid_product/chunks/` or the translated output.

#### Golden-label corrections (`data/corrections/manual_overwrites.yaml`)

Before comparing predictions against the golden dataset, the comparison step applies any corrections from `data/corrections/manual_overwrites.yaml` — `filename: {indicator_id: corrected_value}` — directly to the golden `value`s in memory, so `results/comparison/` always reflects the latest corrections even if `data/processed/long_policy_encoding.csv` hasn't been regenerated yet. A correction with no matching (filename, indicator_id) in the golden CSV is skipped with a warning rather than failing the run; a missing/absent corrections file is treated as no corrections. See `evaluate_accuracy.py:apply_manual_overwrites`.

#### Concurrency

Translation and evaluation issue Azure OpenAI requests in parallel, capped by a shared `max_workers` semaphore. Evaluation parallelizes both across policies and across the 7 criteria dimensions per policy. The comparison step remains local (no API calls).

| Key | Default | Description |
|-----|---------|-------------|
| `concurrency.enabled` | `true` | When `false`, all API calls run serially |
| `concurrency.max_workers` | `5` | Maximum simultaneous in-flight API requests |

Start with `max_workers: 5` and increase gradually while monitoring for rate-limit (`429`) errors. Real speedup depends on deployment TPM/RPM limits and prompt size.

Override config paths with `--pipeline-config` and `--model-config`.

## Usage

Run from the project root:

```bash
# Full small-scale pipeline (5 benchmark policies)
python -m automation.src.run_pipeline --small-scale

# Translation only
python -m automation.src.run_pipeline --steps translation --small-scale

# Evaluation + comparison on an existing run
python -m automation.src.run_pipeline --run-id 20250630_143022 --steps evaluation,comparison --small-scale

# Resume a run — steps are idempotent by default and skip items that already have output
python -m automation.src.run_pipeline --run-id 20250630_143022 --steps translation,evaluation

# Force re-run of a step even though output already exists
python -m automation.src.run_pipeline --run-id 20250630_143022 --steps evaluation --force

# Discrepancy diagnosis on an existing run that already has evaluation + comparison output
python -m automation.src.run_pipeline --run-id 20250630_143022 --steps discrepancy_diagnosis --small-scale
```

### CLI flags

| Flag | Description |
|------|-------------|
| `--small-scale` | Process only 5 benchmark policy files |
| `--steps` | `translation`, `storage`, `evaluation`, `comparison`, `discrepancy_diagnosis` (default: all) |
| `--run-id` | Existing run ID (required for eval/comparison/diagnosis without translation) |
| `--force` | Re-run a step even if its output already exists (all steps are idempotent by default — they skip items that already have output) |
| `--allow-partial` | Save incomplete evaluation or discrepancy diagnosis reports |
| `--max-workers` | Override `concurrency.max_workers` from pipeline config |
| `--no-concurrency` | Disable parallel API calls (serial mode) |
| `--pipeline-config` | Path to experiment YAML (default: [`automation\config\pipeline_config.yaml`](.\config\pipeline_config.yaml))|
| `--model-config` | Path to model parameters YAML (default: [`automation\config\model_config.yaml`](.\config\model_config.yaml)) |

### Step rules

- **New run** (no `--run-id`): auto-generates `run_id`, runs all steps by default (`translation`, `storage`, `evaluation`, `comparison`, `discrepancy_diagnosis`)
- **Eval only**: requires `--run-id` and existing `translation/` outputs
- **Comparison only**: requires `--run-id` and existing `evaluation/` JSON reports
- **Discrepancy diagnosis**: requires `--run-id` and existing `rag_candidates/`, `evaluation/*.json`, and `comparison/` outputs. Runs whose `evaluation` step predates `rag_candidates/` need `evaluation` re-run with `--force` first. Produces no report files (not an error) when `comparison/error_analysis.csv` doesn't exist, i.e. zero mismatches
- **Idempotent by default**: translation, storage, evaluation, and discrepancy_diagnosis each skip an item (file/policy) that already has output — translation and storage skip an input whose output file exists, evaluation skips a policy that already has an eval report, discrepancy_diagnosis skips a policy that already has a `diagnosis/<stem>.json`. Comparison always re-runs (it recomputes `metrics.csv`/`error_analysis.csv` from whatever `evaluation/` reports currently exist). Pass `--force` to re-run a step's items regardless of existing output
- **Step failures**: if a step fails outright (e.g. a required input directory is missing), the pipeline records the step name and error message under `step_failures` in `failures.json`, prints `Step '<name>' failed: <message>` to stderr, and stops before running any later steps

API retry messages include error type, message, HTTP status, and request ID (SDK-level httpx retry noise is suppressed).
