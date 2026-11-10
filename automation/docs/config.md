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
    supports_temperature: true  # optional; false never sends `temperature` (reasoning models reject it)
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
| `evaluation` | `model`, `prompt_version`, `method` (`rag` or `sliding_window`), `confidence`, `coverage`, `sliding_window`, `rag` (storage batch size, retrieval: top_k, hybrid BM25, reranker, evidence verification) |
| `comparison.confidence` | Cutoffs for `confidence_report.json` |
| `discrepancy_diagnosis` | `model`, `prompt_version` |
| `alignment` | `method` (`vecalign` / `bertalign` / `heuristic`): step 4.6 aligner for chunks with unequal sentence counts. Needs `automation/requirements-align.txt` |
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

## Prompt size (`evaluation.rag.retrieval.max_prompt_tokens`)

One evaluation call carries a whole dimension: its criteria and rules, and up to `top_k` candidate chunks for each of its
indicators. With whole-section chunks (hundreds of sentences each) that can pass the model's context limit (one Malta
policy sent 413,247 tokens against a limit of 272,000 and the call was refused).

`max_prompt_tokens` (default 200000, below the 272000 of the current deployment) is the budget of one prompt. Characters
are converted at 3 per token, on the cautious side. **A dimension that does not fit is split, not cut:**

1. Its indicators are divided into two runs of consecutive indicators whose prompts weigh about the same, and each run is
   judged by a call of its own. A run that is still too big is split again. Every indicator keeps **all** its retrieved
   candidates; the answers are merged by indicator id, and the tags the judge cites resolve as before.
2. The same split happens when the model refuses a prompt as too long although the estimate said it fits ("context length
   exceeded" is not retried as it was).
3. Only an indicator that is too long **alone** has nothing left to split. Its candidates are then trimmed, lowest-ranked
   first (a candidate that fits goes in whole, one that does not is skipped, and if even the best is larger than the
   budget its leading sentences go in), and if the model still refuses it the budget is cut by the ratio the error reports
   (limit / tokens sent, times 0.9), up to twice, before the dimension fails.

The evaluation report says what happened, and only then: `split_dimensions` (dimension -> number of model calls it took) and
`prompt_trimmed` (dimension -> candidates dropped, chunks truncated, sentences left out, the indicators concerned). A prompt
that fits is one call, byte-identical to before, and the report has neither field. A split costs more calls (so a little
more time and tokens), never evidence.

## Coverage (`evaluation.coverage`)

A prompt whose criteria distinguish *how* something is covered (the global prompt: `Code Yes - Substantive` /
`Code Yes - Referenced only` / `Code No`) has three answers, not two. They are recorded as two fields:

```yaml
evaluation:
  coverage:
    enabled: true                              # off by default; the Indonesian prompts stay Yes / No
    labels: [Substantive, Referenced only]     # the first is the highest level
```

| Field | Values |
|---|---|
| `included` | `Yes` / `No` -- unchanged. Everything that reads it (evidence, accuracy, the Yes / No confidence) works as before; both kinds of Yes are `Yes` |
| `coverage` | one of the labels when `included` is `Yes`; `null` otherwise. Anything else the model writes is turned into `null` and logged |

The level is `No` = 0, then the labels from the last to the first (`Referenced only` = 1, `Substantive` = 2).

**The two answers have separate confidences**, by the same methods (`logprobs`, `margin`, `verbalized`):

- `confidence`, `confidence_scores`, `answer_logprob`, `margin`, ... -- the Yes / No answer, as before.
- `coverage_confidence`, `coverage_confidence_scores`, `coverage_answer_logprob`, `coverage_margin`, ... -- the coverage
  value only; `null` with `coverage_confidence_source: not_applicable` when the indicator is `No`. A judge can be sure the
  policy mentions an indicator and unsure whether the mention is substantive; the two numbers say so separately.
  The `verbalized` method reads one more field from the response, `coverage_self_reported_confidence` (the prompt asks for it).

**Reading the confidence report.** `confidence_report.json` keeps its sections for the Yes / No answer, scored against
`value` collapsed to 0 / 1. With coverage on it gains a `level` section, scored independently:

- It takes only the pairs where the golden label **and** the prediction are above `No` (the level of something that
  was not found cannot be judged), and is `correct` when the predicted level equals the golden one.
- Its confidence is `coverage_confidence`, never the Yes / No one; its errors are `level_over` / `level_under`.
- It needs a golden with levels (values above 1 in the golden `value` column). Without them the section is present
  with `status: no_level_labels`. A run without `evaluation.coverage` has no `level` section at all.

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

Both blocks are optional; when omitted the code falls back to built-in defaults. `evaluation.confidence` only affects the `evaluation` step; `comparison.confidence` only affects report cutoffs, so it can be re-tuned and comparison re-run without touching the model. A model profile can also opt out permanently with `supports_logprobs: false`. Likewise, if a deployment answers 400 "Unsupported parameter: 'temperature'", the wrapper drops `temperature` for the rest of the run and resends immediately (one warning); set `supports_temperature: false` on that profile to skip the probe.

`methods: []` disables capture entirely. `primary` must name one of the enabled methods, and `verbalized` requires `prompt_version: v4` — the field is only added to the response schema when the method is enabled, so a v3 run produces byte-identical requests to before the confidence module existed and stays comparable to the existing baseline.

**Switching `primary` after a run is a comparison-only change.** Every enabled method is stored per indicator, so:

```bash
python -m automation.src.run_pipeline --steps comparison
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
