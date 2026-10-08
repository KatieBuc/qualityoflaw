# Run output reference

`run_pipeline` writes its results to one folder per project, `data/<project>/automation/`. There are no run IDs: this folder is the project's final result, steps add to it, and `--force` cleans it before a full rerun.

## Folder layout

```
data/<project>/automation/
├── metadata.json              # Run summary (steps, timing, config, counts)
├── failures.json              # Persistent log of translation/evaluation/diagnosis failures (when present)
├── config/                    # Snapshot of configs used for this run
│   ├── model_config.yaml
│   └── pipeline_config.yaml
├── results/                   # Main deliverables of the pipeline
│   ├── translation/           # English policy .txt files — plain text, read by storage/RAG/evaluation.
│   │   │                      # Derived from translation_markdown by the `md_to_text` step.
│   │   └── <policy>.txt
│   ├── translation_qa/        # QA audit reports (one per policy) — action_required/issues per chunk, not the text itself
│   │   └── <policy>.json
│   ├── evaluation/            # LLM JSON reports (one per policy)
│   ├── evaluation_sliding_window/   # same, when evaluation.method is sliding_window
│   ├── comparison/            # Golden-dataset comparison outputs
│   │   ├── metrics.csv
│   │   ├── error_analysis.csv
│   │   ├── confidence_report.json     # Low-confidence coverage of the errors, calibration, breakdowns
│   │   ├── low_confidence.csv         # Only when the primary threshold flags something
│   │   └── unmatched_indicators.csv   # Only when join pairs are missing on one side
│   ├── comparison_sliding_window/   # same as comparison/, for the sliding_window method
│   └── diagnosis/             # Discrepancy diagnosis reports (one per policy with mismatches)
└── mid_product/                # Intermediate artifacts consumed by later stages
    ├── chunks/                # Per-heading retrieval units (~77 per policy) + per-chunk translations.
    │   │                      # Rebuilt by `md_to_text`; read by RAG storage.
    │   └── <policy>.chunks.json
    ├── rag_store/              # Per-policy embedded chunk store
    ├── rag_candidates/         # Full RAG candidate list per indicator (one per policy, written by evaluation)
    └── sliding_window_candidates/   # Same idea for the sliding_window method
```

`results/` holds the pipeline's final deliverables; `mid_product/` holds intermediate artifacts that only exist to feed later stages.

### `metadata.json`

| Field | Description |
|-------|-------------|
| `experiment_name` | From `pipeline_config.yaml` |
| `status` | `running`, `completed`, `completed_with_errors`, or `failed` |
| `timestamps.created_at` / `timestamps.updated_at` | UTC timestamps |
| `config` | Model and prompt versions used |
| `execution_scope.steps_executed` | List of completed steps: `translation_md`, `translation_qa_md`, `md_to_text`, `storage`, `evaluation`, `comparison`, `discrepancy_diagnosis` |
| `execution_scope.small_scale` | Whether only the `paths.small_scale_stems` policies were processed |
| `execution_scope.evaluated_policy_files` | Policy filenames compared during the comparison step |
| `file_counts` | Per-step counts keyed by step name (e.g. `translation_md`, `md_to_text`, `storage`, `evaluation`) |
| `failures` | Count of logged failures per step (see `failures.json`) |
| `token_usage` | Aggregated LLM token usage (the LLM steps: `translation_md`, `translation_qa_md`, `evaluation`, `discrepancy_diagnosis`) |
| `timing_seconds` | Elapsed seconds per step, plus `total` |

Example (abridged, from a real run):

```json
{
  "experiment_name": "first-full-run",
  "status": "completed_with_errors",
  "timestamps": {
    "created_at": "2026-09-28T09:50:10.297379+00:00",
    "updated_at": "2026-09-28T15:54:05.355949+00:00"
  },
  "execution_scope": {
    "steps_executed": [
      "translation_md",
      "md_to_text",
      "storage",
      "evaluation",
      "comparison"
    ],
    "small_scale": false,
    "evaluated_policy_files": [
      "ACEH_BIREUEN.txt",
      "ACEH_SUBULUSSALAM_KOTA.txt",
      "..."
    ]
  },
  "file_counts": {
    "translation_md": {
      "total": 198,
      "succeeded": 0,
      "skipped": 198,
      "failed": 0
    },
    "md_to_text": {
      "total": 198,
      "succeeded": 0,
      "skipped": 198,
      "failed": 0,
      "clauses_lost": 0,
      "repaired": 0,
      "source_unaligned": 0
    },
    "storage": {
      "total": 198,
      "succeeded": 0,
      "skipped": 198,
      "failed": 0
    },
    "evaluation": {
      "total": 198,
      "succeeded": 197,
      "failed": 1,
      "skipped": 0,
      "saved_reports": 197
    }
  },
  "failures": {
    "translation": 1,
    "translation_qa": 0,
    "storage": 0,
    "evaluation": 1,
    "discrepancy_diagnosis": 0,
    "step_failures": 0
  },
  "token_usage": {
    "translation_md": {
      "prompt_tokens": 0,
      "completion_tokens": 0,
      "total_tokens": 0
    },
    "evaluation": {
      "prompt_tokens": 87784139,
      "completion_tokens": 936370,
      "total_tokens": 88720509
    }
  },
  "timing_seconds": {
    "translation_md": 0.04,
    "md_to_text": 0.02,
    "storage": 0.04,
    "evaluation": 9260.29,
    "comparison": 4.12,
    "total": 9265.8
  }
}
```

### `translation/`

UTF-8 `.txt` files with the same basenames as the Indonesian source policies (e.g. `ACEH_BIREUEN.txt`). Filenames must match the `filename` column in the golden CSV for comparison to work.

### `translation_qa/`

Written by `translation_qa_md` (Markdown path: per packed unit, audited repeatedly up to `qa_max_passes`; the report also carries `qa_passes_total` and per-chunk `passes`/`converged`) or by the legacy `translation_qa` step, which runs right after `translation` and is described below. For each policy, an LLM checks the translated text against the original (per chunk, when `mid_product/chunks/<stem>.chunks.json` exists — otherwise once for the whole file) for three things: leftover LLM notes/preambles, meaning drift from the original, and numbered/lettered/roman list markers scrambled by OCR layout errors. If a check finds a problem, the LLM's corrected text replaces the translation in place (both `results/translation/<policy>.txt` and, when chunked, `mid_product/chunks/<stem>.chunks.json`'s `translated_text` fields — the latter is what RAG storage actually reads, so corrections must land there too); otherwise the translation is left untouched.

One audit-only JSON report per policy at `<stem>.json` (the corrected text itself isn't duplicated here — it already lives in `translation/`/`chunks.json`):

```json
{
  "policy_file": "ACEH_BIREUEN.txt",
  "model": "gpt-5.2",
  "checked_at": "2026-08-10T12:00:00+00:00",
  "prompt_version": "v1",
  "granularity": "chunked",
  "chunks_checked": 12,
  "chunks_corrected": 2,
  "chunks_response_incomplete": 0,
  "items": [{ "chunk_index": 0, "section_id": 0, "action_required": false, "issues": [], "response_incomplete": false }]
}
```

If any chunk's QA call **raises** (a real API/LLM-call failure, after the wrapper's own retries are exhausted), the whole policy's correction is aborted atomically — nothing is written for that policy (`translation/`, `chunks.json`, and `translation_qa/<stem>.json` are all left exactly as they were), and the failure is recorded the same way as a translation failure.

A different, more common case is handled separately: the LLM sometimes sets `action_required: true` but doesn't actually return `corrected_text` (a content-quality slip, not an API error). Since one policy can be split into dozens or hundreds of chunks, treating this as a hard failure would mean even a small per-chunk chance of it happening discards every other chunk's already-good result. Instead, that one unit is retried once; if it's still missing, the unit is degraded to "no action" (its original text is kept) and flagged with `response_incomplete: true` in the report for manual review, rather than failing the whole policy.

### `evaluation/`

Per-policy JSON reports named `<timestamp>-<model>-<POLICY_BASENAME>.json`. Each report contains:

- `policy_file` — basename used to join with golden data
- `evaluation_results` — 56 indicators keyed by `indicator_id`, each with `included` (`"Yes"` / `"No"`)
- `model`, `evaluated_at`, `prompt_version`

If multiple reports exist for the same policy, comparison keeps the **latest** by timestamp.

Confidence fields are described in [design.md](design.md#per-indicator-confidence).

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

Comparison joins LLM results to [`data/processed/long_policy_encoding.csv`](../../data/processed/long_policy_encoding.csv) on **`(filename, indicator_id)`**. Golden rows are filtered to evaluated policies only; when a `year` column exists, the latest year per policy is used.

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
| `confidence` | The judge's confidence in the wrong answer, from `evaluation/`; empty for runs without confidence data |
| `evidence` | The judge's cited evidence snippet for this indicator, from `evaluation/` |
| `rationale` | The judge's rationale for this indicator, from `evaluation/` |
| `discrepancy_root_cause` | Always `Reference only` — root causes are produced by the `discrepancy_diagnosis` step (see `diagnosis/<stem>.json`), which runs after this file is written and doesn't write back into it |

#### `confidence_report.json`

Answers the operational question the accuracy numbers can't: **if the least-confident predictions were sent for human review, what share of the actual errors would that catch?** That share — `errors_captured / total_errors` — is the `coverage_rate`.

A prediction counts as flagged when its `confidence` is strictly below a cutoff. The report sweeps both kinds of cutoff, because confidence saturates near 1.0 and absolute thresholds alone can flag almost nothing:

| Section | Contents |
|---------|----------|
| `method_comparison` | **Read this first when choosing `primary`.** One compact row per captured method — availability, AUROC, coverage, precision, ECE, Brier — ranked best error-detector first |
| `methods` | The full analysis repeated independently for each captured method, plus per-method `notes` (e.g. how often margin observed its counterpart) |
| `primary_method` | Which method the top-level sections below describe — the method actually scored, which is not necessarily the one config asked for (see `primary_method_requested` / `primary_method_warning`) |
| `confidence_source` | Model, temperature, methods present, and how many pairs actually carry a score (`availability`, `source_counts`) |
| `baseline` | Accuracy, error count, and confusion matrix over **all** matched pairs — the denominator the coverage rate is measured against |
| `coverage.by_absolute_threshold` | One row per configured probability cutoff |
| `coverage.by_quantile` | One row per configured quantile, with the confidence value that quantile resolves to |
| `coverage.primary` | The row for `comparison.confidence.primary_threshold` |
| `calibration` | Confidence-bucket accuracy table, plus ECE, max calibration error, and Brier score |
| `discrimination` | `auroc_error_detection` (AUROC of `1 - confidence` for predicting an error; 0.5 means no signal), AURC, and a risk–coverage curve |
| `breakdowns` | Coverage by dimension, and split by `false_positive` vs `false_negative` |
| `flagged_sample` | The N least-confident predictions with their labels and whether each was actually wrong |

Each coverage row carries `flagged`, `flagged_share`, `errors_captured`, `coverage_rate`, `precision`, `lift` (precision ÷ baseline error rate), `remaining_errors`, and `accuracy_on_unflagged`. Rates over an empty population are `null`, never `0` — a cutoff that flags nothing has undefined precision, not perfect precision.

Coverage statistics are computed over the pairs that carry a score, while `baseline` covers all of them; compare `pairs_with_confidence` against `pairs_total` before reading too much into a partially-scored run. Methods can differ in availability — `verbalized` is absent from indicators the model declined to rate, `margin` from answers that couldn't be aligned — so each method block reports its own `pairs_scored` and `errors_in_scope`. The file is written on every comparison, including for runs whose evaluation predates this feature — those get `"status": "no_confidence_data"` with the analysis sections `null`, so a missing file always means comparison didn't run.

#### `low_confidence.csv`

Every prediction below `primary_threshold`, least confident first, with `correct` and `error_type` alongside the judge's rationale — the manual-review worklist behind the coverage number. Deleted rather than left stale when nothing is flagged.

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

Translated Markdown now lives beside its source in `data/<project>/preprocessed/translation_markdown/` (outside the automation folder, so `--force` never touches it); see `automation/README.md`.

Default evaluation reports now carry the simplified core per indicator: `included` (Answer), `evidence` (Translated Evidence), `evidence_original` (Source Evidence), `rationale` and `confidence` (verbalized). Token-probability fields (`answer_logprob`, `p_yes`, `margin`, ...) appear only when `evaluation.confidence.methods` includes `logprobs` or `margin`.
