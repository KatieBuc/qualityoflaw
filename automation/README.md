# qualityoflaw

Pipeline that translates Indonesian local policies to English, has an LLM code each translation against a
56-indicator rubric, and compares the result with a hand-coded golden dataset.

```
extract -> quality_check -> txt_to_md -> translation_md -> human_review -> validate -> md_to_text -> storage -> evaluation -> comparison
```

The first steps need a person now and then (OCR, fixing a text, checking headings). Where that happens the run
**pauses**, prints what to do, and continues when you run the same command again: see [docs/pipeline.md](docs/pipeline.md).

`comparison` and `discrepancy_diagnosis` (LLM root-cause report for every mismatch, after `comparison`) run only when you ask
for them. **Legacy steps**, also only on request: `translation_qa_md` (audits and repairs each translated chunk, after
`translation_md`) and the raw-OCR-text path `translation`, `translation_qa`, `markdown`.

## Setup

Python 3.13 and an Azure OpenAI deployment are required.

```bash
python -m venv .venv
.venv/Scripts/activate            # Windows; on macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt   # add requirements-dev.txt for notebooks and tests
cp .env.example .env              # then fill in your keys; .env is gitignored
```

## Run

All commands run from the repository root.

```bash
# Small run on the benchmark policies (paths.small_scale_stems)
python -m automation.src.run_pipeline --small-scale

# Full run on the whole corpus
python -m automation.src.run_pipeline

# Run later steps on what is already in the output folder; finished items are skipped
python -m automation.src.run_pipeline --steps evaluation,comparison

# Start over: clean the output folder, then run (rebuilds everything the folder held)
python -m automation.src.run_pipeline --force
```

| Flag | Meaning |
|------|---------|
| `--steps a,b,c` | Steps to run. Default: the five in the chain above. Steps always execute in pipeline order |
| `--small-scale` | Only the policies listed in `paths.small_scale_stems` |
| `--force` | Clean `data/<project>/automation/`, then run. Checked first: if the requested steps cannot rebuild what they need, nothing is deleted. Does not touch `preprocessed/` or `processed/`, so existing translations are kept (delete those files to retranslate) |
| `--skip-blocked` | At the `quality_check` breakpoint, continue with the files that passed |
| `--reviewed` | Confirm the optional human review and continue |
| `--allow-partial` | Save incomplete evaluation / diagnosis reports instead of failing them |
| `--max-workers N`, `--no-concurrency` | Override or disable parallel API calls |
| `--pipeline-config`, `--model-config` | Use other YAML files than the ones in `automation/config/` |

## Steps

| Step | Reads | Writes (under `data/<project>/automation/` unless noted) |
|------|-------|-----------|
| `extract` | `raw/raw_files/` | `raw/extracted_txt/<name>.txt`, `raw/input_process.log` |
| `quality_check` | `raw/extracted_txt/` | `preprocessed/cleaned_txt/`, `raw/input_process.log` (pauses for OCR / eliminated files) |
| `txt_to_md` | `preprocessed/cleaned_txt/` or `raw/gold/` | `preprocessed/cleaned_markdown/` |
| `translation_md` | `data/<project>/preprocessed/cleaned_markdown/*<suffix>` | `data/<project>/preprocessed/translation_markdown/` (no chunks saved) |
| `translation_qa_md` *(legacy)* | source + translated markdown | rewrites the translated markdown in place; `results/translation_qa/` |
| `human_review` *(optional pause)* | - | - (pauses when `human_review.pause: true`) |
| `validate` | `preprocessed/{cleaned,translation}_markdown` | `preprocessed/header_check.log`; moves nothing, you move accepted pairs to `processed/` (pauses while it is empty) |
| `md_to_text` | translated + source markdown (`preprocessed/`) | `results/translation/*.txt`, `mid_product/chunks/` |
| `storage` | `results/translation/`, `mid_product/chunks/` | `mid_product/rag_store/` (embeddings) |
| `evaluation` | translation, RAG store, `prompts/quality_eval/` | `results/evaluation/`, `mid_product/rag_candidates/` (sliding-window method: `evaluation_sliding_window/`, `sliding_window_candidates/`) |
| `comparison` *(optional, not in the default chain)* | evaluation, golden CSV, `data/<project>/corrections/manual_overwrites.yaml` | `results/comparison/` (`metrics.csv`, `error_analysis.csv`, `confidence_report.json`, ...) |
| `discrepancy_diagnosis` *(optional)* | comparison, evaluation, RAG candidates, raw text | `results/diagnosis/` |

A failed step stops the pipeline; the error is printed and logged in `failures.json`. Each run also has
`metadata.json` (steps, counts, timing, tokens) and a `config/` snapshot of the YAML files it used.

The older raw-OCR-text steps `translation`, `translation_qa` and `markdown` still work with `--steps` but are not part
of the default chain; their code is in `automation/src/legacy/` (see [legacy.md](automation/docs/legacy.md)).

## Configuration

| File | What it controls |
|------|------------------|
| `automation/config/pipeline_config.yaml` | Which model and prompt version each stage uses, chunk sizes, evaluation method (`rag` or `sliding_window`), retrieval settings, confidence, input paths, small-scale policy list |
| `automation/config/model_config.yaml` | Named model profiles (Azure deployment, temperature, retries) and reranker profiles |
| `.env` | Azure endpoints and keys (see `.env.example`) |

To change the evaluation model or prompt, edit `evaluation.model` / `evaluation.prompt_version` in
`pipeline_config.yaml`; prompt versions are folders under `automation/prompts/`. Details:
[config.md](automation/docs/config.md).

## Repository layout

```
automation/
  config/        pipeline_config.yaml, model_config.yaml
  prompts/       translation, translation_qa, quality_eval, discrepancy_diagnosis (one folder per version)
  src/
    run_pipeline.py   entry point and step registry
    markdown/         translation_md, translation_qa_md, md_to_text
    rag/              embedding store, retrieval, reranking
    llm/              Azure client, logprobs and confidence
    legacy/           raw-text translation steps
    common.py         helpers shared by the above
  tests/         pytest suite (tests/golden pins config resolution and pipeline behaviour)
  docs/          design.md, outputs.md, config.md, legacy.md
data/
  raw/, processed/     input corpora and the golden labels
  automation/<run>/    outputs of each run
scripts/clean_mmd.py   turns raw .mmd extractions into the cleaned markdown corpus
legacy/prototype/      early prototype, kept for reference
```

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

## More documentation

- [design.md](automation/docs/design.md): why translation packs sections, how `md_to_text` checks structure, criteria rubric, confidence methods
- [outputs.md](automation/docs/outputs.md): every file a run writes, with schemas
- [config.md](automation/docs/config.md): configuration reference
- [legacy.md](automation/docs/legacy.md): the raw-text path

## Stage 3: human adjustment

Edit anything in `data/<project>/preprocessed/`, keeping the Markdown headers of each
`cleaned_markdown` / `translation_markdown` pair identical in count and level. The pipeline's `validate` step checks
every pair and writes the mismatches to `preprocessed/header_check.log`; you can also run it alone:

```bash
python -m automation.src.markdown.validate --project indonesia
```

Nothing is ever moved for you: **you move the pairs you accept to `processed/`** (both files, same names), because
evaluation reads only `processed/`.

## Pipeline stages

1. **Input processing.**
   - 1.1 (`python -m automation.src.input_process run`): PDF/HTML/TXT in `raw/raw_files/` → `raw/extracted_txt/` →
     quality check, fix and re-check → `preprocessed/cleaned_txt/`, plus the log `raw/input_process.log` (a WARNING
     for every eliminated file and every PDF that needs OCR). See [input_process.md](automation/docs/input_process.md).
   - 1.2 (`python -m automation.src.input_process markdown`): `cleaned_txt` → `preprocessed/cleaned_markdown`.
2. **Translation** (`translation_md`): chunk-and-combine, writes `preprocessed/translation_markdown`.
3. **Human adjustment**: edit `preprocessed/`, then move the pairs you accept to `processed/`.
4. **Evaluation** (`md_to_text`, `storage`, `evaluation`): reads `processed/`, header validation first, RAG evaluation,
   evidence mapping (index pairing, else `alignment.method`), outputs under `data/<project>/automation/` (the project's single, final result folder; a rerun cleans it with `--force`).

`comparison` and `discrepancy_diagnosis` are optional steps; `translation_qa_md` is a legacy step. Install the alignment
dependencies with `pip install -r automation/requirements-align.txt` (plus Vecalign, see that file).
