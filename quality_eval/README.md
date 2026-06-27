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
  run_eval_new.py  ──►  Azure OpenAI (structured JSON per indicator)
        │
        ▼
  result/<model>/*.json
        │
        ▼
  evaluate_accuracy.py  ──►  metrics + error_analysis.csv
        │
        ▼
  data/processed/long_policy_encoding.csv  (golden / human-coded labels)
```

## Evaluation Framework

Indicators are defined in `v1/prompts/` and organised into seven dimensions (56 indicators total). Each indicator has a unique ID (`X.Y`), a short name, and a yes/no question.

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

The full indicator schema (names and descriptions) is also documented in `data/mapping/index_schema.yaml`.

## How the LLM Judge Works

### Prompt assembly (`v1/prompts/prompt_loader.py`)

For each of the seven dimension files, `generate_judge_prompt()`:

1. Reads the criteria file and formats each line into a bullet list (`- 1.1 (Domestic violence): does the policy include...`).
2. Reads the full translated policy text.
3. Injects both into `prompt_template.txt`, replacing `{{CRITERIA_LIST}}` and `{{POLICY_TEXT}}`.

The template instructs the model to act as a policy analyst, decide **Yes** or **No** for each indicator (explicit or implicit coverage), quote evidence when Yes, and explain the rationale.

### Structured output (`run_eval_new.py`)

`run_eval_new.py` calls Azure OpenAI once per dimension (7 API calls per policy) using `client.beta.chat.completions.parse` with a Pydantic schema:

| Field | Description |
|-------|-------------|
| `id` | Indicator ID (e.g. `1.1`) |
| `indicator` | Indicator name |
| `included` | `"Yes"` or `"No"` |
| `evidence` | Exact quote from the policy when `included` is Yes; otherwise `null` |
| `rationale` | Explanation of the judgment |

Results from all seven batches are merged into a single JSON report keyed by indicator ID. Temperature is set to `0.1` for reproducibility.

### Output naming

Reports are saved as:

```
<DDMMYYYYHHMMSS>-<model>-<POLICY_NAME>.json
```

For example: `27062026223404-gpt-4o-ACEH_BIREUEN.json`

## Accuracy Validation (`evaluate_accuracy.py`)

After running evaluations, compare LLM judgments against the human-coded golden dataset:

- **Golden dataset:** `data/processed/long_policy_encoding.csv`
- **Columns used:** `filename`, `indicator_id`, `value` (1.0 = present, 0.0 = absent)

The script:

1. Loads one or more LLM JSON reports (single file or a folder of `.json` files).
2. Converts each `included` field to `1.0` (Yes) or `0.0` (No).
3. Inner-joins on `filename` + `indicator_id` with the golden CSV.
4. Reports accuracy, a sklearn classification report, and a confusion matrix.
5. Exports mismatches to a CSV for manual review (e.g. `v1/error_analysis_gpt-4o.csv`).

## Folder Structure

```
quality_eval/
├── README.md
├── run_eval_new.py          # Run LLM evaluation on a policy
├── evaluate_accuracy.py     # Compare LLM output to golden labels
└── v1/
    ├── prompts/
    │   ├── prompt_template.txt
    │   ├── prompt_loader.py
    │   └── 01_scope_of_violence.txt … 07_budget_funding_sources.txt
    ├── translated_policy/   # English policy texts used as evaluation input
    ├── result/
    │   ├── gpt-4o/          # Evaluation reports by model
    │   └── gpt-5.2/
    ├── error_analysis_gpt-4o.csv
    └── error_analysis_gpt-5.2.csv
```

## Environment Setup

Copy the `.env` file in the project root with your Azure OpenAI credentials:

```
AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com/openai/v1/
AZURE_OPENAI_API_KEY=<your-key>
AZURE_OPENAI_MODEL=gpt-4o | gpt-5.2
```

Install dependencies:

```bash
pip install python-dotenv pydantic openai scikit-learn pandas
```

## How to Run

### 1. Evaluate a single policy

```bash
python -m quality_eval.run_eval_new ^
  -p quality_eval\v1\translated_policy\ACEH_BIREUEN.txt ^
  -o quality_eval\v1\result\gpt-4o\ ^
  -c quality_eval\v1\prompts ^
  -t quality_eval\v1\prompts\prompt_template.txt
```

| Flag | Description |
|------|-------------|
| `-p` / `--policy` | Path to the translated policy `.txt` file |
| `-o` / `--output_folder` | Directory for the output JSON report |
| `-c` / `--criteria_folder` | Folder containing the seven criteria files |
| `-t` / `--template` | Path to `prompt_template.txt` |

Run from the project root so the `quality_eval` package imports resolve correctly.

For full CLI options:

```bash
python -m quality_eval.run_eval_new --help
```

### 2. Check accuracy against the golden dataset

Single report:

```bash
python -m quality_eval.evaluate_accuracy ^
  -g data\processed\long_policy_encoding.csv ^
  -l quality_eval\v1\result\gpt-4o\27062026223404-gpt-4o-ACEH_BIREUEN.json ^
  -e quality_eval\v1\error_analysis_gpt-4o.csv
```

All reports in a folder (e.g. after evaluating multiple policies with the same model):

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

### 3. Preview a generated prompt (optional)

```bash
python -m quality_eval.v1.prompts.prompt_loader
```

This prints the assembled prompt for a sample policy and criteria file (useful for debugging prompt wording).

## Small-Scale Test Set

The following five policies were randomly selected for the initial evaluation run. Translated texts live in `v1/translated_policy/`; source translations are also under `translation/small_scale_test/v1/claude/`.

| Policy file | 
|-------------|
| `ACEH_BIREUEN.txt` |
| `LAMPUNG_LAMPUNG_TIMUR.txt` | 
| `NUSA_TENGGARA_TIMUR_TIMOR_TENGAH_UTARA.txt` | 
| `SUMATERA_BARAT_PADANG_PARIAMAN.txt` | 
| `JAWA_TENGAH_SEMARANG.txt` | 

## Example Output

Each evaluation JSON has this structure:

```json
{
  "policy_file": "ACEH_BIREUEN.txt",
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

Error analysis CSV rows show where the LLM disagrees with human coding:

```csv
fullname,filename,indicator_id,indicator_value,value,pred_value
ACEH BIREUEN,ACEH_BIREUEN.txt,5.1,NGO collaboration,0.0,1.0
```

Here `value` is the golden label and `pred_value` is the LLM prediction.

## Model Comparison

Evaluation reports are stored under `v1/result/<model>/` so results from different deployments (e.g. `gpt-4o`, `gpt-5.2`) can be compared side by side using `evaluate_accuracy.py` with the same golden CSV.
