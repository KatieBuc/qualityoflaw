# Policy Quality Evaluation

This folder contains the LLM-based pipeline for evaluating Indonesian women's protection policies against a structured indicator framework. The goal is to automate the binary coding of policy text (whether each indicator is present or absent) and to measure how well an LLM judge agrees with a human-coded golden dataset.

## Overview

The evaluation process has three stages:

1. **Prepare policy text** — Policies are translated from Indonesian into English and stored as plain-text files.
2. **Run LLM evaluation** — An Azure OpenAI model scores each policy against 56 indicators across 7 thematic dimensions, producing a structured JSON report per policy.
3. **Measure accuracy** — LLM outputs are compared against the human-coded golden dataset to compute accuracy, classification metrics, and per-indicator error analysis.

```
Translated policy (.txt)
        │
        ▼
  run_eval_new.py  ──►  Azure OpenAI (7 API calls per policy, one per dimension)
        │                  accepts a single .txt file or a folder of .txt files
        ▼
  v1/result/<model>/*.json
        │
        ▼
  evaluate_accuracy.py  ──►  error_analysis.csv + metrics summary JSON
        │
        ▼
  data/processed/long_policy_encoding.csv  (golden / human-coded labels)

Or run both steps at once:

  run_pipeline.py  ──►  LLM eval + accuracy report
```

## Scripts

| Script | Purpose |
|--------|---------|
| [`run_pipeline.py`](run_pipeline.py) | **Recommended entry point.** Runs LLM evaluation and accuracy comparison in one command with sensible defaults. |
| [`run_eval_new.py`](run_eval_new.py) | LLM evaluation only. Accepts one policy file or a folder of `.txt` files. |
| [`evaluate_accuracy.py`](evaluate_accuracy.py) | Accuracy comparison only. Reads saved JSON reports and compares them to the golden CSV. |
| [`v1/criteria.py`](v1/criteria.py) | Loads the 56 indicator IDs from `index_schema.yaml` for validation and per-dimension metrics. |
| [`v1/prompts/prompt_loader.py`](v1/prompts/prompt_loader.py) | Assembles the judge prompt from criteria files, template, and policy text. |

Run all commands from the **project root** so Python package imports resolve correctly.

## Data Sources

Three related but distinct sources define the evaluation framework:

| Source | Location | Used for |
|--------|----------|----------|
| Indicator schema | [`data/mapping/index_schema.yaml`](../data/mapping/index_schema.yaml) | Canonical list of 56 indicator IDs, names, and dimensions. Loaded by `v1/criteria.py` for completeness checks and accuracy breakdowns. |
| Criteria prompts | [`v1/prompts/01~07_*.txt`](v1/prompts/) | Pipe-delimited questions sent to the LLM. Used by `prompt_loader.py` to build prompts. |
| Golden labels | [`data/processed/long_policy_encoding.csv`](../data/processed/long_policy_encoding.csv) | Human-coded ground truth (198 policies, multiple years). Used by `evaluate_accuracy.py`. |

**Important:** Prompt assembly reads from `v1/prompts/*.txt`, not directly from `index_schema.yaml`. The schema is used to verify that criteria files and LLM output cover all 56 indicators.

## Evaluation Framework

Indicators are organised into seven dimensions (56 indicators total). Each indicator has a unique ID (`X.Y`), a short name, and a yes/no question.

| Dimension | File | Indicators | ID range |
|-----------|------|------------|----------|
| Scope of violence | `01_scope_of_violence.txt` | 10 | 1.1 – 1.10 |
| Institutional mechanism | `02_institutional_mechanism.txt` | 6 | 2.1 – 2.6 |
| Specialised support services | `03_specialised_support_services.txt` | 10 | 3.1 – 3.10 |
| Primary prevention | `04_primary_prevention.txt` | 4 | 4.1 – 4.4 |
| Stakeholder engagement | `05_stakeholder_engagement.txt` | 8 | 5.1 – 5.8 |
| Intersectional approach | `06_intersectional_approach.txt` | 13 | 6.1 – 6.13 |
| Budget / funding sources | `07_budget_funding_sources.txt` | 5 | 7.1 – 7.5 |

Each criteria file uses a pipe-delimited format:

```
<id> | <indicator name> | <evaluation question>
```

Example:

```
1.1 | Domestic violence | does the policy include Domestic violence?
```

## How the LLM Judge Works

### Prompt assembly

For each of the seven dimension files, `generate_judge_prompt()`:

1. Reads the criteria file and formats each line into a bullet list (`- 1.1 (Domestic violence): does the policy include...`).
2. Reads the full translated policy text.
3. Injects both into [`prompt_template.txt`](v1/prompts/prompt_template.txt), replacing `{{CRITERIA_LIST}}` and `{{POLICY_TEXT}}`.

The template instructs the model to act as a policy analyst, decide **Yes** or **No** for each indicator (explicit or implicit coverage), quote evidence when Yes, and explain the rationale.

### API calls

- **7 API calls per policy** — one call per dimension, not one call per indicator.
- The full policy text is included in every call (once per dimension).
- Uses Azure OpenAI `client.beta.chat.completions.parse` with a Pydantic schema.
- Temperature is set to `0.1` for reproducibility.
- Failed calls retry up to **3 times** with exponential backoff.

### Structured output fields

| Field | Description |
|-------|-------------|
| `id` | Indicator ID (e.g. `1.1`) |
| `indicator` | Indicator name |
| `included` | `"Yes"` or `"No"` |
| `evidence` | Exact quote from the policy when `included` is Yes; otherwise `null` |
| `rationale` | Explanation of the judgment |

Results from all seven batches are merged into a single JSON report keyed by indicator ID.

### Report metadata and validation

Each saved JSON report includes:

| Field | Description |
|-------|-------------|
| `policy_file` | Source filename (e.g. `ACEH_BIREUEN.txt`) |
| `model` | Azure deployment name |
| `evaluated_at` | ISO 8601 UTC timestamp |
| `prompt_version` | Currently `"v1"` |
| `indicator_count` | Number of indicators in the report |
| `completed_dimensions` | Criteria files processed successfully |
| `failed_dimensions` | Criteria files that raised an error |
| `errors` | Error messages for failed dimensions |

Before saving, the runner validates that all **56** expected indicator IDs are present. Incomplete evaluations are **not saved** unless `--allow-partial` is passed. The process still exits with code `1` when any policy fails.

### Output naming

Reports are saved as:

```
<DDMMYYYYHHMMSS>-<model>-<POLICY_NAME>.json
```

Example: `27062026223404-gpt-4o-ACEH_BIREUEN.json`

## Accuracy Validation

After running evaluations, compare LLM judgments against the human-coded golden dataset.

### Golden dataset

- **File:** `data/processed/long_policy_encoding.csv`
- **Columns used:** `filename`, `indicator_id`, `value` (1.0 = present, 0.0 = absent), `year`
- **Scope:** 198 policies; the accuracy script only compares policies that appear in the LLM results.

When a policy has multiple years in the golden CSV, only the **latest year** is used for comparison.

### What the accuracy script does

1. Loads one or more LLM JSON reports (single file or folder).
2. **Deduplicates** multiple runs per policy — keeps the **latest** report per `policy_file` (by `evaluated_at`, or filename timestamp prefix).
3. Converts each `included` field to `1.0` (Yes) or `0.0` (No).
4. Filters the golden CSV to evaluated policies and latest year.
5. Inner-joins on `filename` + `indicator_id`.
6. Prints overall accuracy, classification report, confusion matrix, and per-dimension accuracy.
7. Exports mismatches to a CSV for manual review.
8. Writes a metrics summary JSON.

### Output files

| File | Description |
|------|-------------|
| `v1/error_analysis_<model>.csv` | Rows where LLM prediction differs from golden label |
| `v1/error_analysis_<model>.metrics.json` | Overall accuracy, confusion matrix, per-dimension breakdown |

## Folder Structure

```
quality_eval/
├── README.md
├── run_pipeline.py          # Run LLM evaluation + accuracy in one command
├── run_eval_new.py          # LLM evaluation only (file or folder input)
├── evaluate_accuracy.py     # Accuracy comparison only
├── requirements.txt         # Python dependencies
├── tests/                   # pytest suite
└── v1/
    ├── criteria.py          # 56 indicator IDs from index_schema.yaml
    ├── prompts/
    │   ├── prompt_template.txt
    │   ├── prompt_loader.py
    │   └── 01_scope_of_violence.txt … 07_budget_funding_sources.txt
    ├── translated_policy/   # English policy texts (evaluation input)
    ├── result/
    │   ├── gpt-4o/          # LLM JSON reports by model
    │   └── gpt-5.2/
    ├── error_analysis_gpt-4o.csv
    ├── error_analysis_gpt-4o.metrics.json
    └── error_analysis_gpt-5.2.csv
```

## Environment Setup

Create a `.env` file in the project root:

```
AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com/openai/v1/
AZURE_OPENAI_API_KEY=<your-key>
AZURE_OPENAI_MODEL=gpt-4o
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## How to Run

### 1. Run both steps at once (recommended)

Evaluate all policies in the default test folder and produce accuracy reports:

```bash
python -m quality_eval.run_pipeline
```

Custom paths:

```bash
# Windows
python -m quality_eval.run_pipeline ^
  -p quality_eval\v1\translated_policy ^
  -o quality_eval\v1\result\gpt-4o\ ^
  -g data\processed\long_policy_encoding.csv ^
  -e quality_eval\v1\error_analysis_gpt-4o.csv

# macOS / Linux
python -m quality_eval.run_pipeline \
  -p quality_eval/v1/translated_policy \
  -o quality_eval/v1/result/gpt-4o/ \
  -g data/processed/long_policy_encoding.csv \
  -e quality_eval/v1/error_analysis_gpt-4o.csv
```

| Flag | Description |
|------|-------------|
| `-p` / `--policy` | Policy `.txt` file or folder (default: `v1/translated_policy`) |
| `-o` / `--output_folder` | LLM JSON output folder (default: `v1/result/<model>/`) |
| `-c` / `--criteria_folder` | Criteria prompts folder (default: `v1/prompts`) |
| `-t` / `--template` | Prompt template file (default: `v1/prompts/prompt_template.txt`) |
| `-g` / `--golden_csv` | Golden dataset CSV (default: `data/processed/long_policy_encoding.csv`) |
| `-e` / `--export_errors` | Error analysis CSV (default: `v1/error_analysis_<model>.csv`) |
| `-m` / `--export_metrics` | Metrics summary JSON (default: `<export_errors>.metrics.json`) |
| `--allow-partial` | Save incomplete LLM reports when some dimensions fail |
| `--skip-eval` | Skip LLM evaluation; only run accuracy on existing JSON reports |
| `--skip-accuracy` | Run LLM evaluation only; skip accuracy comparison |

Re-run accuracy on existing results without calling the API:

```bash
python -m quality_eval.run_pipeline --skip-eval
```

### 2. LLM evaluation only

Single policy file:

```bash
python -m quality_eval.run_eval_new ^
  -p quality_eval\v1\translated_policy\ACEH_BIREUEN.txt ^
  -o quality_eval\v1\result\gpt-4o\ ^
  -c quality_eval\v1\prompts ^
  -t quality_eval\v1\prompts\prompt_template.txt
```

Entire folder (all `.txt` files):

```bash
python -m quality_eval.run_eval_new ^
  -p quality_eval\v1\translated_policy ^
  -o quality_eval\v1\result\gpt-4o\ ^
  -c quality_eval\v1\prompts ^
  -t quality_eval\v1\prompts\prompt_template.txt
```

| Flag | Description |
|------|-------------|
| `-p` / `--policy` | Path to a `.txt` policy file, or a folder of `.txt` files |
| `-o` / `--output_folder` | Directory for output JSON reports |
| `-c` / `--criteria_folder` | Folder containing the seven criteria files |
| `-t` / `--template` | Path to `prompt_template.txt` |
| `--allow-partial` | Save incomplete reports when some dimensions fail (exits non-zero) |

When evaluating a folder, each policy is processed sequentially. A batch summary is printed at the end. Exit code `0` only if every policy completes successfully.

### 3. Accuracy comparison only

Single report:

```bash
python -m quality_eval.evaluate_accuracy ^
  -g data\processed\long_policy_encoding.csv ^
  -l quality_eval\v1\result\gpt-4o\27062026223404-gpt-4o-ACEH_BIREUEN.json ^
  -e quality_eval\v1\error_analysis_gpt-4o.csv
```

All reports in a folder (latest per policy is used automatically):

```bash
python -m quality_eval.evaluate_accuracy ^
  -g data\processed\long_policy_encoding.csv ^
  -l quality_eval\v1\result\gpt-4o\ ^
  -e quality_eval\v1\error_analysis_gpt-4o.csv
```

| Flag | Description |
|------|-------------|
| `-g` / `--golden_csv` | Path to the human-coded golden dataset CSV |
| `-l` / `--llm_input` | A single JSON report or a folder of reports |
| `-e` / `--export_errors` | Output path for the mismatch CSV (default: `error_analysis.csv`) |
| `-m` / `--export_metrics` | Output path for metrics summary JSON (default: `<export_errors>.metrics.json`) |

### 4. Run tests

```bash
python -m pytest quality_eval/tests/ -v
```

Tests cover criteria validation, prompt assembly, completeness checks, report deduplication, and golden-year filtering.

### 5. Preview a generated prompt (optional)

```bash
python -m quality_eval.v1.prompts.prompt_loader
```

Prints the assembled prompt for a sample policy and criteria file (useful for debugging prompt wording).

## Small-Scale Test Set

Five policies were randomly selected for the initial evaluation run. Translated texts live in `v1/translated_policy/`; Claude translations used in early tests are also under `translation/small_scale_test/v1/claude/`.

| Policy file | Region |
|-------------|--------|
| `ACEH_BIREUEN.txt` | Aceh — Bireuen Regency |
| `LAMPUNG_LAMPUNG_TIMUR.txt` | Lampung — East Lampung Regency |
| `NUSA_TENGGARA_TIMUR_TIMOR_TENGAH_UTARA.txt` | East Nusa Tenggara — North Central Timor Regency |
| `SUMATERA_BARAT_PADANG_PARIAMAN.txt` | West Sumatra — Padang Pariaman Regency |
| `JAWA_TENGAH_SEMARANG.txt` | Central Java — Semarang City |

## Example Output

### LLM evaluation JSON

```json
{
  "policy_file": "ACEH_BIREUEN.txt",
  "model": "gpt-4o",
  "evaluated_at": "2026-06-30T05:10:41.560282+00:00",
  "prompt_version": "v1",
  "indicator_count": 56,
  "completed_dimensions": ["01_scope_of_violence.txt", "..."],
  "failed_dimensions": [],
  "errors": [],
  "evaluation_results": {
    "1.1": {
      "id": "1.1",
      "indicator": "Domestic violence",
      "included": "Yes",
      "evidence": "Article 25 mentions 'violence against women'.",
      "rationale": "The policy addresses domestic violence through protection measures."
    }
  }
}
```

### Error analysis CSV

Rows where the LLM disagrees with human coding:

```csv
fullname,filename,indicator_id,indicator_value,value,pred_value
ACEH BIREUEN,ACEH_BIREUEN.txt,5.1,NGO collaboration,0.0,1.0
```

- `value` — golden label (0.0 = absent, 1.0 = present)
- `pred_value` — LLM prediction

### Metrics summary JSON

```json
{
  "overall": {
    "accuracy": 0.81,
    "matched_pairs": 212,
    "evaluated_policies": 5,
    "golden_policies_total": 198,
    "evaluated_policy_files": ["ACEH_BIREUEN.txt", "..."]
  },
  "confusion_matrix": {
    "true_negative": 47,
    "false_positive": 40,
    "false_negative": 1,
    "true_positive": 124
  },
  "by_dimension": {
    "scope_of_violence": { "accuracy": 0.83, "matched": 40, "correct": 33 },
    "primary_prevention": { "accuracy": 0.31, "matched": 16, "correct": 5 }
  }
}
```

## Model Comparison

Store evaluation reports under separate model folders:

```
v1/result/gpt-4o/
v1/result/gpt-5.2/
```

Switch `AZURE_OPENAI_MODEL` in `.env`, re-run evaluation, then compare accuracy:

```bash
python -m quality_eval.evaluate_accuracy -g data/processed/long_policy_encoding.csv -l quality_eval/v1/result/gpt-4o/ -e quality_eval/v1/error_analysis_gpt-4o.csv
python -m quality_eval.evaluate_accuracy -g data/processed/long_policy_encoding.csv -l quality_eval/v1/result/gpt-5.2/ -e quality_eval/v1/error_analysis_gpt-5.2.csv
```

## Exit Codes

| Script | Code | Meaning |
|--------|------|---------|
| `run_eval_new.py` | `0` | All policies evaluated and saved successfully |
| `run_eval_new.py` | `1` | Missing API key, invalid input, or one or more policies failed |
| `run_pipeline.py` | `0` | Both steps completed without policy failures |
| `run_pipeline.py` | `1` | Evaluation or accuracy step failed |

## Troubleshooting

| Problem | Likely cause | What to check |
|---------|--------------|---------------|
| `Error: API key not found` | Missing `.env` | `AZURE_OPENAI_API_KEY` in project root |
| `Evaluation incomplete; report not saved` | A dimension API call failed or indicators missing | Re-run with `--allow-partial` to inspect partial JSON; check Azure quota/errors |
| `No matching data` in accuracy step | Filename mismatch between JSON and golden CSV | Ensure `policy_file` in JSON matches `filename` in golden CSV exactly |
| Lower matched pairs than expected | Golden CSV has multiple years per policy | Script uses latest year only; some policies may lack golden rows for all 56 indicators |
| Duplicate runs inflating metrics | Old behaviour before dedup | Current code keeps latest report per policy when loading a folder |
| `ModuleNotFoundError: quality_eval` | Wrong working directory | Run commands from project root |

## Related Files Outside This Folder

| Path | Role |
|------|------|
| `data/mapping/index_schema.yaml` | Canonical indicator schema |
| `data/processed/long_policy_encoding.csv` | Human-coded golden labels |
| `translation/small_scale_test/v1/claude/` | Source translations for the test set |
| `prompts/prompt_v1.yaml` | Legacy per-indicator prompts (not used by this pipeline) |
