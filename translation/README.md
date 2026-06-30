# Translation

This folder contains translation utilities and experiment artifacts for policy documents.

## Folder Overview

- `llm/`  
  Shared LLM components:
  - `azure_client.py`: loads Azure/OpenAI config from `.env` and returns the correct client type.
  - `translator.py`: core translation logic (prompt injection, API call, retries, output cleanup).

- `azure_openai/`  
  Azure OpenAI translation runner and generated outputs:
  - `translate.py`: CLI for single-file and batch translation.
  - `<model_name>/`: default output folder per model (for example `gpt-5.2/`).

- `small_scale_test/`  
  Benchmark/evaluation artifacts and prompts used for small-scale translation comparison.

- `api_test.py`  
  Smoke test for Azure OpenAI connectivity.

## Environment Variables

Configured in project root `.env`:

- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_MODEL`
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_VERSION` (optional; used for non-`/openai/v1` endpoints)

## Usage

Run from project root:

```bash
# Smoke test
python translation/api_test.py

# Translate 5 benchmark files
python translation/azure_openai/translate.py --small-scale

# Translate full corpus from data/raw/localpolicies
python translation/azure_openai/translate.py

# Translate a single file
python translation/azure_openai/translate.py \
  -i data/raw/localpolicies/ACEH_BIREUEN.txt \
  -o translation/azure_openai/gpt-5.2/ACEH_BIREUEN.txt
```

## Default Input and Output

- Input directory (batch): `data/raw/localpolicies/`
- Prompt template: `translation/small_scale_test/v1/prompt.txt`
- Output directory (default): `translation/azure_openai/<AZURE_OPENAI_MODEL>/`

Use `--output-dir` to override the default output path.
