# Automation Pipeline

Config-driven orchestration for policy translation, LLM quality evaluation, and golden-dataset comparison.

Legacy CLIs in `translation/` and `quality_eval/` remain available independently. Golden-dataset comparison logic lives in [`src/evaluate_accuracy.py`](src/evaluate_accuracy.py); `quality_eval/evaluate_accuracy.py` re-exports it for backward compatibility.

## Layout

```
automation/
├── config/
│   ├── model_config.yaml      # Model deployment parameters
│   └── pipeline_config.yaml   # Experiment combination
├── prompts/
│   ├── translation/v1/
│   └── quality_eval/v1/
└── src/
    ├── llm/                   # Shared Azure LLM wrapper
    ├── evaluate_accuracy.py   # Golden-dataset comparison
    ├── run_pipeline.py        # Main entry point
    ├── translate.py
    ├── run_eval.py
    └── compare.py             # Thin wrapper around evaluate_accuracy
```

## Run output (`data/automation/<run_id>/`)

Each pipeline execution writes a self-contained run folder under `data/automation/`. The run ID format is `YYYYMMDD_HHMMSS` (UTC).

```
data/automation/<run_id>/
├── metadata.json              # Run summary (steps, timing, config, counts)
├── config/                    # Snapshot of configs used for this run
│   ├── model_config.yaml
│   └── pipeline_config.yaml
├── translation/               # English policy .txt files (one per input policy)
├── evaluation/                # LLM JSON reports (one per policy)
└── comparison/                # Golden-dataset comparison outputs
    ├── metrics.json
    ├── error_analysis.csv
    └── unmatched_indicators.csv   # Only when join pairs are missing on one side
```

### `metadata.json`

| Field | Description |
|-------|-------------|
| `run_id` | Unique run identifier |
| `experiment_name` | From `pipeline_config.yaml` |
| `created_at` / `updated_at` | UTC timestamps |
| `status` | `running`, `completed`, `completed_with_errors`, or `failed` |
| `steps_executed` | List of completed steps: `translation`, `evaluation`, `comparison` |
| `small_scale` | Whether only the 5 benchmark policies were processed |
| `file_counts` | Per-step success/failure counts |
| `timing` | Elapsed seconds per step and total |
| `token_usage` | Aggregated LLM token usage (translation step) |
| `config` | Model and prompt versions used |

### `translation/`

UTF-8 `.txt` files with the same basenames as the Indonesian source policies (e.g. `ACEH_BIREUEN.txt`). Filenames must match the `filename` column in the golden CSV for comparison to work.

### `evaluation/`

Per-policy JSON reports named `<timestamp>-<model>-<POLICY_BASENAME>.json`. Each report contains:

- `policy_file` — basename used to join with golden data
- `evaluation_results` — 56 indicators keyed by `indicator_id`, each with `included` (`"Yes"` / `"No"`)
- `model`, `evaluated_at`, `prompt_version`

If multiple reports exist for the same policy, comparison keeps the **latest** by timestamp.

### `comparison/` — golden-dataset reports

Comparison joins LLM results to [`data/processed/long_policy_encoding.csv`](../data/processed/long_policy_encoding.csv) on **`(filename, indicator_id)`**. Golden rows are filtered to evaluated policies only; when a `year` column exists, the latest year per policy is used.

#### `metrics.json`

Overall accuracy summary:

| Section | Contents |
|---------|----------|
| `overall.accuracy` | Fraction of matched indicator pairs where LLM agrees with golden |
| `overall.matched_pairs` | Number of `(filename, indicator_id)` pairs compared |
| `overall.evaluated_policies` | Policies in this run with LLM reports |
| `overall.golden_policies_total` | Total unique policies in the full golden CSV (e.g. 198) |
| `overall.evaluated_policy_files` | List of policy filenames compared |
| `overall.value_mismatch_count` | Rows where golden and LLM disagree |
| `overall.unmatched_pair_count` | Pairs present on only one side (see below) |
| `confusion_matrix` | TN / FP / FN / TP counts |
| `by_dimension` | Per-dimension accuracy, matched count, correct count |
| `error_summary` | Error counts by type, dimension, and indicator |

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

#### `unmatched_indicators.csv`

Written only when some `(filename, indicator_id)` pairs cannot be joined:

| `match_status` | Meaning |
|----------------|---------|
| `golden_only` | In golden CSV, missing from LLM report |
| `llm_only` | In LLM report, missing from golden for that policy |

These rows are **excluded from accuracy**. An empty or absent file means all pairs joined successfully.

### Standalone comparison CLI

```bash
python -m automation.src.evaluate_accuracy \
  -g data/processed/long_policy_encoding.csv \
  -l data/automation/<run_id>/evaluation \
  -e data/automation/<run_id>/comparison/error_analysis.csv \
  -m data/automation/<run_id>/comparison/metrics.json
```

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
```

### `config/pipeline_config.yaml`

Selects which models and prompt versions to use:

```yaml
experiment_name: baseline_v1
translation:
  model: gpt-5.2-translate
  prompt_version: v1
evaluation:
  model: gpt-4o-eval
  prompt_version: v1
```

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
```

### CLI flags

| Flag | Description |
|------|-------------|
| `--small-scale` | Process only 5 benchmark policy files |
| `--steps` | `translation`, `evaluation`, `comparison` (default: all) |
| `--run-id` | Existing run ID (required for eval/comparison without translation) |
| `--force` | Re-translate files even if output exists |
| `--allow-partial` | Save incomplete evaluation reports |
| `--pipeline-config` | Path to experiment YAML |
| `--model-config` | Path to model parameters YAML |

### Step rules

- **New run** (no `--run-id`): auto-generates `run_id`, runs all steps by default
- **Eval only**: requires `--run-id` and existing `translation/` outputs
- **Comparison only**: requires `--run-id` and existing `evaluation/` JSON reports

## Tests

```bash
python -m pytest automation/tests/ -v
```
