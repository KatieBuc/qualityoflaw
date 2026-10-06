# qualityoflaw

Pipeline that translates Indonesian local policies to English, has an LLM code each translation against a
56-indicator rubric, and compares the result with a hand-coded golden dataset.

```
translation_md -> md_to_text -> storage -> evaluation -> comparison
```

Two more steps run only when you ask for them: `translation_qa_md` (audits and repairs each translated chunk, after
`translation_md`) and `discrepancy_diagnosis` (LLM root-cause report for every mismatch, after `comparison`).

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

# Resume or extend an existing run; finished items are skipped
python -m automation.src.run_pipeline --run-id 20260928_095010 --steps evaluation,comparison

# Re-run a step even if its output exists
python -m automation.src.run_pipeline --run-id 20260928_095010 --steps evaluation --force
```

| Flag | Meaning |
|------|---------|
| `--steps a,b,c` | Steps to run. Default: the five in the chain above. Steps always execute in pipeline order |
| `--run-id ID` | Use an existing run. Required when `--steps` has no `translation_md` (or legacy `translation`) |
| `--small-scale` | Only the policies listed in `paths.small_scale_stems` |
| `--force` | Redo items that already have output (steps are idempotent by default) |
| `--allow-partial` | Save incomplete evaluation / diagnosis reports instead of failing them |
| `--max-workers N`, `--no-concurrency` | Override or disable parallel API calls |
| `--pipeline-config`, `--model-config` | Use other YAML files than the ones in `automation/config/` |

## Steps

| Step | Reads | Writes (under `data/<project>/automation/<run_id>/` unless noted) |
|------|-------|-----------|
| `translation_md` | `data/<project>/preprocessed/cleaned_markdown/*<suffix>` | `data/<project>/preprocessed/translation_markdown/` (no chunks saved) |
| `translation_qa_md` *(optional)* | source + translated markdown | rewrites the translated markdown in place; `results/translation_qa/` |
| `md_to_text` | translated + source markdown (`preprocessed/`) | `results/translation/*.txt`, `mid_product/chunks/` |
| `storage` | `results/translation/`, `mid_product/chunks/` | `mid_product/rag_store/` (embeddings) |
| `evaluation` | translation, RAG store, `prompts/quality_eval/` | `results/evaluation/`, `mid_product/rag_candidates/` (sliding-window method: `evaluation_sliding_window/`, `sliding_window_candidates/`) |
| `comparison` | evaluation, golden CSV, `data/corrections/manual_overwrites.yaml` | `results/comparison/` (`metrics.csv`, `error_analysis.csv`, `confidence_report.json`, ...) |
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

## Stage 3: human adjustment and promotion

Edit anything in `data/<project>/preprocessed/`, keeping the Markdown headers of each
`cleaned_markdown` / `translation_markdown` pair identical in count and level. Then:

```bash
python -m automation.src.markdown.validate --project indonesia            # check only
python -m automation.src.markdown.validate --project indonesia --promote  # check, then move to processed/
```

Nothing is moved unless every pair passes.
