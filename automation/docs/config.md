# Configuration reference

Two YAML files in `automation/config/` control a run. Credentials are never stored there; they
come from `.env` (see `.env.example`).

## `model_config.yaml`

Named model profiles that `pipeline_config.yaml` refers to by key, plus reranker profiles:

```yaml
models:
  gpt-5.2:
    deployment: gpt-5.2       # Azure deployment name
    temperature: 0.2
    max_tokens: null
    max_retries: 3
    supports_logprobs: true   # optional; false skips the logprobs request parameter
rerankers:
  cohere-rerank-v4.0-fast:
    deployment: ...
```

## `pipeline_config.yaml`

| Block | Controls |
|-------|----------|
| `experiment_name` | Label written to `metadata.json` |
| `concurrency` | `enabled`, `max_workers` (parallel API calls) |
| `translation` | `model`, `prompt_version`, `chunking` (legacy raw-text path; `model` is also used by `translation_md`) |
| `translation_markdown` | `prompt_version`, `qa_prompt_version`, `qa_max_passes`, `chunking.target_chars` / `safe_limit` |
| `translation_qa` | `model`, `prompt_version` (also used by `translation_qa_md`) |
| `evaluation` | `model`, `prompt_version`, `method` (`rag` or `sliding_window`), `confidence`, `sliding_window`, `rag` (storage batch size, retrieval: top_k, hybrid BM25, reranker, evidence verification) |
| `comparison.confidence` | Cutoffs for `confidence_report.json` |
| `discrepancy_diagnosis` | `model`, `prompt_version` |
| `project` | Active project; all folders derive from `data/<project>/` (`automation/src/paths.py`) |
| `paths` | `markdown_input_suffix`, `small_scale_stems`; optional overrides `input_dir`, `markdown_input_dir`, `golden_csv`, `index_schema`, `manual_overwrites` |

Notes:

- Model keys and prompt versions for all four roles are validated on every invocation, whatever `--steps` you pass.
  Prompt files live in `automation/prompts/<kind>/<version>/`.
- The shipped YAML is authoritative. Built-in code defaults only apply to keys you leave out, and some differ from the
  shipped values (for example `sliding_window` is 40/10 in code and 100/15 in the YAML; `qa_max_passes` is 5 in
  code and 2 in the YAML).
- To run on a different corpus, change `project`, `markdown_input_suffix` and `small_scale_stems` together
  (the YAML has commented-out examples for `globallaws_markdown`).
- `paths.index_schema` is parsed but not used: `automation/src/criteria.py` loads the schema from
  the default project's `mapping/index_schema.yaml` at import time.
- Static values (folder names, step lists, suffixes) live in `automation/src/constants.py`;
  frequently changed parameters live in `pipeline_config.yaml`.
- Override the config paths per run with `--pipeline-config` and `--model-config`.

## Confidence (`evaluation.confidence` + `comparison.confidence`)

```yaml
evaluation:
  prompt_version: v4            # v4 adds the verbalized-confidence field
  confidence:
    enabled: true
    methods: [logprobs, verbalized, margin] # any of logprobs / margin / verbalized
    primary: margin             # must be one of the enabled methods
    top_logprobs: 5             # alternatives per answer token; Azure caps at 5

comparison:
  confidence:
    thresholds: [0.5, 0.7, 0.9, 0.95, 0.99]   # absolute cutoffs to sweep
    quantiles: [0.05, 0.10, 0.20, 0.30]       # quantile cutoffs to sweep
    primary_threshold: 0.9                    # drives coverage.primary and low_confidence.csv
    flagged_sample_size: 100                  # least-confident predictions listed in the report
    calibration_bins: 10
```

Both blocks are optional; when omitted the code falls back to built-in defaults. `evaluation.confidence` only affects the `evaluation` step; `comparison.confidence` only affects report cutoffs, so it can be re-tuned and comparison re-run without touching the model. A model profile can also opt out permanently with `supports_logprobs: false`.

`methods: []` disables capture entirely. `primary` must name one of the enabled methods, and `verbalized` requires `prompt_version: v4` — the field is only added to the response schema when the method is enabled, so a v3 run produces byte-identical requests to before the confidence module existed and stays comparable to the existing baseline.

**Switching `primary` after a run is a comparison-only change.** Every enabled method is stored per indicator, so:

```bash
python -m automation.src.run_pipeline --run-id <run> --steps comparison --force
```

recomputes the entire report — coverage, calibration, discrimination, breakdowns, `flagged_sample`, and `low_confidence.csv` — against the newly chosen method. No re-evaluation and no API calls.

The `confidence` field inside an evaluation report holds whichever method was primary *at evaluation time*, so comparison deliberately ignores it when the requested method has its own score, and reads that method's column instead. If the requested method was never captured, the report does **not** silently relabel someone else's numbers: it keeps the method it actually scored, sets `confidence_source.primary_method_warning`, and prints a warning naming what to re-run.

## Golden-label corrections

Before comparing predictions against the golden dataset, the comparison step applies any corrections from `data/corrections/manual_overwrites.yaml` — `filename: {indicator_id: corrected_value}` — directly to the golden `value`s in memory, so `results/comparison/` always reflects the latest corrections even if `data/processed/long_policy_encoding.csv` hasn't been regenerated yet. A correction with no matching (filename, indicator_id) in the golden CSV is skipped with a warning rather than failing the run; a missing/absent corrections file is treated as no corrections. See `evaluate_accuracy.py:apply_manual_overwrites`.

## Concurrency

Translation and evaluation issue Azure OpenAI requests in parallel, capped by a shared `max_workers` semaphore. Evaluation parallelizes both across policies and across the 7 criteria dimensions per policy. The comparison step remains local (no API calls).

| Key | Default | Description |
|-----|---------|-------------|
| `concurrency.enabled` | `true` | When `false`, all API calls run serially |
| `concurrency.max_workers` | `5` | Maximum simultaneous in-flight API requests |

Start with `max_workers: 5` and increase gradually while monitoring for rate-limit (`429`) errors. Real speedup depends on deployment TPM/RPM limits and prompt size.
