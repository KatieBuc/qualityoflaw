# Automation Pipeline

Config-driven orchestration for policy translation, LLM quality evaluation, golden-dataset comparison, and discrepancy diagnosis.

Translation reads the curated Markdown corpus at
`data/processed/localpolicies/cleaned_markdown/` (`<POLICY>.cleaned.md`) and
writes translated Markdown, which the `md_to_text` step then renders to the
plain text every downstream stage reads. The older raw-OCR-text path
(`--steps translation,translation_qa,markdown`) is unchanged and still
runnable, so the two can be compared on the same corpus; it is just no longer
part of the default chain. See [Markdown translation path](#markdown-translation-path).

Golden-dataset comparison logic lives in [`src/evaluate_accuracy.py`](src/evaluate_accuracy.py).

## Layout

```
automation/
├── config/
│   ├── model_config.yaml      # Model deployment parameters
│   └── pipeline_config.yaml   # Experiment combination
├── prompts/
│   ├── translation/v1/, v2/   # Raw-text path
│   ├── translation/v3/        # Markdown path (markdown in, markdown out)
│   ├── translation_qa/v1/     # Raw-text path
│   ├── translation_qa/v2/     # Markdown path (adds a Markdown-fidelity check)
│   ├── quality_eval/v1/, v2/, v3/   # v3 is the default RAG judge prompt
│   ├── quality_eval/v4/            # v3 + verbalized-confidence field
│   ├── quality_eval/sliding_window_v1/
│   └── discrepancy_diagnosis/v1/
└── src/
    ├── llm/                   # Shared Azure LLM wrapper
    │   ├── logprobs.py        # Answer-token alignment, probability + margin
    │   └── confidence.py      # Confidence methods and how they attach to results
    ├── markdown/              # Markdown translation path (the default)
    │   ├── chunking.py        # Heading parsing, section packing, structure checks
    │   ├── translate.py       # `translation_md` step
    │   ├── qa.py              # `translation_qa_md` step
    │   ├── to_text.py         # `md_to_text` step + standalone CLI
    │   └── policy_files.py    # `<POLICY>.cleaned.md` -> `<POLICY>` naming
    ├── evaluate_accuracy.py   # Golden-dataset comparison
    ├── confidence_report.py   # Low-confidence coverage of errors, per method
    ├── run_pipeline.py        # Main entry point
    ├── chunking.py            # Raw-text clean/chunk/combine
    ├── translate.py           # Raw-text `translation` + `markdown` steps
    ├── translation_qa.py      # Raw-text Translation QA + re-align step
    ├── translation_qa_prompt_builder.py
    ├── run_eval.py
    ├── diagnose.py            # Discrepancy diagnosis (final pipeline step)
    ├── diagnose_prompt_builder.py
    └── compare.py             # Thin wrapper around evaluate_accuracy
```

## Markdown translation path

The default chain:

```
translation_md -> translation_qa_md -> md_to_text -> storage -> evaluation -> comparison -> discrepancy_diagnosis
```

Input is `data/processed/localpolicies/cleaned_markdown/<POLICY>.cleaned.md`
(`paths.markdown_input_dir`) — 198 curated files whose structure is explicit:
`#` BAB, `##` Bagian/titles, `###` Paragraf, `####` Pasal. The `.cleaned`
suffix is stripped on entry, so every artifact stays keyed on `<POLICY>` and
the golden-dataset join (`<POLICY>.txt`) is unaffected.

### `translation_md`

`automation/src/markdown/translate.py`. Headings are read directly rather than
re-detected by regex — running `chunking.clean_text` over this corpus would
destroy the structure, since `#### Pasal 5` matches neither
`STRUCTURE_MARKER_RE` nor the all-caps-title rule.

Sections are split at headings using the same boundary rule as
`chunking.split_into_sections` (a heading only starts a new section once the
current one has body content, so a title stack stays with its content), then
**packed**: adjacent sections merge up to `target_chars`. This is the one
deliberate difference from the raw-text path. Measured over the corpus:

| target_chars | translation units | per file (mean / max) |
|--------------|-------------------|------------------------|
| 4,000        | 2,194             | 11.1 / 29 |
| **8,000**    | **1,116**         | **5.6 / 16** |
| 16,000       | 578               | 2.9 / 9  |
| unpacked     | 15,291            | 77.2 / 231 |

Unpacked, translation would cost ~15,300 LLM calls per full run (median
section: ~200 chars, half of them under 200) plus as many QA calls. At 8,000
that is 1,116 — a ~93% reduction — with far more context per call. Each unit
carries its heading breadcrumb (e.g. `BAB IV > Bagian Kesatu`) so the model
can resolve cross-references when a unit starts mid-chapter.

A section larger than `safe_limit` still falls through to
`chunking.fallback_split` (sentence packing with 300-char trailing context).
Nothing in the current corpus reaches it — the largest single section is
16,583 chars.

Outputs: `results/translation_markdown/<POLICY>.md`,
`results/source_markdown/<POLICY>.md`, and
`mid_product/translation_chunks/<POLICY>.json`.

### `translation_qa_md`

`automation/src/markdown/qa.py`, prompt `translation_qa/v2`. Same contract as
the raw-text QA step (and it reuses that module's retry-then-degrade logic),
audited per packed unit. The prompt adds Markdown fidelity as its first
check. Corrections are written back to both the Markdown and the chunk
artifact, atomically per policy.

**Each chunk is audited repeatedly, up to `qa_max_passes` (default 5).** A
single pass only guarantees the auditor *reported* what was wrong — a
correction can leave or introduce a further problem, and these sources carry
heavy OCR damage, so one rewrite often does not finish the job. Each pass
re-audits the previous pass's output against the unchanged Indonesian source,
and the loop stops as soon as a pass reports no action required.

What drives the loop is the auditor's own verdict, never a structural count:
heading totals legitimately move when a repair lands, so looping on those
would chase successful repairs forever.

A chunk that still reports problems after the cap keeps its **last**
correction — that is the most-audited version available, and discarding it
would throw away several passes of genuine repair — and is counted in
`chunks_not_converged` for a human to look at. The report also carries
`qa_passes_total` and per-chunk `passes` / `converged`.

Cost: at best one call per chunk (unchanged), at worst `qa_max_passes` × the
number of chunks. Over the full corpus that is 1,116 to 5,580 QA calls,
against the ~15,300 the unpacked design would have needed for a single pass.

### `md_to_text`

`automation/src/markdown/to_text.py`. Writes two artifacts per policy:

1. `results/translation/<POLICY>.txt` — plain text in the same shape the
   raw-text path produced. `policy_files.filter_policy_files` globs `*.txt`,
   so storage, evaluation, comparison and diagnosis need no changes.
2. `mid_product/chunks/<POLICY>.chunks.json` — the **retrieval** units, one
   per heading section, in the schema `rag/store.py:_load_translation_chunks`
   already expects.

Point 2 is what keeps evidence retrieval stable. Translation packs sections to
save API calls, but retrieval must keep its per-Pasal granularity:
`rag/retriever.py` reads only `chunk_id` and `text`, and
`rag/prompt_builder.py` cites individual sentences *within* a chunk, so
coarser chunks would directly coarsen the evidence. Rebuilding chunks.json
here at heading granularity — rather than handing storage the packed units —
means `rag/store.py` sees the same kind of input through the same code path as
before. Measured against the previous run's store:

|                        | total chunks | per file | chars median | p90 | max |
|------------------------|--------------|----------|--------------|-----|-----|
| previous `rag_store`   | 15,405       | 77.4     | 206          | 1,112 | 6,000 |
| via `md_to_text`       | 15,392       | 77.7     | 197          | 1,086 | 5,999 |

Source and translation sections are paired positionally, which is only sound
when the translation preserved the source's heading signature. When it did
not, the records are emitted with `translated_text` only — storage never reads
the source side, so it keeps working, but nothing is silently misaligned.

Because both sides are Markdown, `markdown/chunking.py:check_structure`
compares heading-level sequences exactly, rather than the length-ratio
heuristics (`check_structural_output`) the raw-text path has to rely on. The
message names the levels whose counts changed, which is what separates a
serious finding from a benign one: `#### Pasal` is level 4, so `level 4: 121
-> 120` means an Article was lost, while `level 5: 11 -> 10` means the model
dropped an OCR fragment such as `##### P  P` — arguably the right call.
Individual headings are deliberately not named: the two sides are in
different languages, so the only thing to align on is the level sequence, and
that cannot tell two same-level headings apart.

**Only a lost numbered clause is a defect.** The curated corpus carries OCR
damage that a good translation *repairs*, and every repair moves the heading
counts — so counts alone cannot be the verdict. Observed on the benchmark four:

| corpus artifact | what the translation did |
|---|---|
| `#### Pasal 28 ayat (1) huruf f, meliputi:` (a cross-reference promoted to a heading) | demoted it back into its sentence, which OCR had split |
| `##### 1  AN` (a fragment splitting a run-on list) | dropped it and reconstructed the a.–n. list |
| `#### Pasal 3 1` (OCR split the number) | recovered it as `#### Article 31` |

`clause_ids()` is what separates these from real loss. A numbered clause
heading has a language-independent identity — `HEADING_KEYWORD_LEVELS` maps
`Pasal`/`Article` onto the same canonical level and the number is the same on
both sides, so `Pasal 34` and `Article 34` both yield `(4, "34")`. Comparing
those sequences is exact. Headings with no such identity (OCR fragments,
all-caps titles, spelled-out ordinals like `Bagian Kesatu`/`Part One`, and
cross-reference text) are skipped rather than guessed at.

So `md_to_text` records a `LostClause` failure only when a numbered clause
disappears; everything else increments `repaired` and is logged. On the four
benchmark policies that is 0 failures and 3 repairs, where raw heading counts
had reported 2 failures — and had stayed silent about ACEH_BIREUEN, whose
translation *added* `Article 31`.

**List-item counts are reported but never count as drift either.** The source corpus
is full of OCR-scrambled list markers — a bare `1,` on one line, `2. 3. 10.`
bunched on the next, and the content following with no markers at all — and
repairing exactly that is `translation_qa_md`'s chartered job (problem 3 in
its prompt). Observed on the benchmark four: 68 source list items becoming
277 correctly marked ones. Treating a count change as drift would flag the
repair as a defect. Heading structure is also the only part retrieval depends
on, since `build_retrieval_chunks` splits on headings.

### Standalone conversion

`md_to_text`'s converter is also a CLI, for rendering the source corpus or
checking one file without spending a run:

```bash
python -m automation.src.markdown.to_text data/processed/localpolicies/cleaned_markdown /tmp/plain
```

### Config

```yaml
translation_markdown:
  prompt_version: v3
  qa_prompt_version: v2
  qa_max_passes: 5          # re-audit a chunk until clean, at most this often
  chunking:
    target_chars: 8000
    safe_limit: 32000

paths:
  input_dir: data/raw/localpolicies                              # raw OCR text; still used by diagnose
  markdown_input_dir: data/processed/localpolicies/cleaned_markdown
```

Models are not duplicated — the Markdown path reuses `translation.model` and
`translation_qa.model`.


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
│   ├── source_markdown/       # Copy of the .cleaned.md input, so a run records what it translated
│   │   └── <policy>.md
│   ├── translation_markdown/  # PRIMARY translated artifact — English Markdown, structure preserved
│   │   └── <policy>.md
│   ├── translation/           # English policy .txt files — plain text, read by storage/RAG/evaluation.
│   │   │                      # Derived from translation_markdown by the `md_to_text` step.
│   │   └── <policy>.txt
│   ├── translation_qa/        # QA audit reports (one per policy) — action_required/issues per chunk, not the text itself
│   │   └── <policy>.json
│   ├── evaluation/            # LLM JSON reports (one per policy)
│   ├── comparison/            # Golden-dataset comparison outputs
│   │   ├── metrics.csv
│   │   ├── error_analysis.csv
│   │   ├── confidence_report.json     # Low-confidence coverage of the errors, calibration, breakdowns
│   │   ├── low_confidence.csv         # Only when the primary threshold flags something
│   │   └── unmatched_indicators.csv   # Only when join pairs are missing on one side
│   └── diagnosis/             # Discrepancy diagnosis reports (one per policy with mismatches)
└── mid_product/                # Intermediate artifacts consumed by later stages
    ├── translation_chunks/     # Packed translation units (~6 per policy) + their translations, for QA
    │   └── <policy>.json
    ├── chunks/                # Per-heading retrieval units (~77 per policy) + per-chunk translations.
    │   │                      # Rebuilt by `md_to_text`; read by RAG storage.
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
| `execution_scope.steps_executed` | List of completed steps: `translation_md`, `translation_qa_md`, `md_to_text`, `storage`, `evaluation`, `comparison`, `discrepancy_diagnosis` |
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

### `translation_qa/`

Written by the `translation_qa` step, which runs right after `translation`. For each policy, an LLM checks the translated text against the original (per chunk, when `mid_product/chunks/<stem>.chunks.json` exists — otherwise once for the whole file) for three things: leftover LLM notes/preambles, meaning drift from the original, and numbered/lettered/roman list markers scrambled by OCR layout errors. If a check finds a problem, the LLM's corrected text replaces the translation in place (both `results/translation/<policy>.txt` and, when chunked, `mid_product/chunks/<stem>.chunks.json`'s `translated_text` fields — the latter is what RAG storage actually reads, so corrections must land there too); otherwise the translation is left untouched.

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

### Criteria files and their coding rubric

Each dimension's criteria live in `prompts/quality_eval/<version>/<NN>_<dimension>.txt`. A line of

```
4.1 | Information and education about VAW | Does the policy include education about VAW?
```

starts a criterion; **every following line belongs to it as coding rubric**, until the next criterion:

```
Code Yes if any of the following apply:
campaigns, counseling, or socialization explicitly described to provide education about violence against women;
Code No if the policy refers only to:
education, awareness or information about women's rights, victims' rights, or gender equality;
```

That rubric is what makes borderline calls decidable — the distinction above (education *about VAW* is Yes, education about women's rights generally is No) cannot be inferred from the question alone.

**This was previously discarded.** `parse_criteria_lines` kept only lines that split into exactly three pipe-separated fields, so of the 451 non-empty lines in the v3 criteria, 56 reached the judge and **395 were silently dropped** — every coding rule. v1 and v2 carry no rubric, so they were unaffected, which is why the loss went unnoticed when v3 introduced it. Rules now render under their criterion as an indented `Coding rules:` block, before the retrieved candidate sentences, and the prompt templates instruct the model to treat them as authoritative.

Notes on the current state:

- 38 of the 56 criteria have a rubric; the remaining 18 (listed in `test_criteria_parsing.py`) have none written yet and render exactly as before, question only. Adding rubric for them needs no code change.
- Retrieval is deliberately **not** affected: the embedding query stays the question alone. Folding a rubric's "Code No if…" exclusions into the query would pull retrieval toward the very passages the criterion exists to rule out.
- Prompts grow about 5% (~14k tokens per policy, ~2.8M across a full 198-policy run) — the retrieved candidate sentences still dominate.
- A line that appears before any criterion cannot be attributed and is logged as a warning rather than dropped in silence.

**This changes v3 and v4 outputs**, so results from before the fix are not directly comparable. On a 4-policy spot check (224 pairs) errors went 26 → 23, with false positives falling 19 → 12 and false negatives rising 7 → 11 — the expected signature of adding explicit exclusions, since the judge becomes stricter. That sample is too small to draw conclusions from; a full re-run is needed to measure the real effect.

### `evaluation/`

Per-policy JSON reports named `<timestamp>-<model>-<POLICY_BASENAME>.json`. Each report contains:

- `policy_file` — basename used to join with golden data
- `evaluation_results` — 56 indicators keyed by `indicator_id`, each with `included` (`"Yes"` / `"No"`)
- `model`, `evaluated_at`, `prompt_version`

If multiple reports exist for the same policy, comparison keeps the **latest** by timestamp.

#### Per-indicator confidence

When `evaluation.confidence` is enabled, each indicator also carries how sure the judge was of its Yes/No. Three methods are available, selected in config:

| Method | What it measures | Cost |
|--------|------------------|------|
| `logprobs` | `exp(answer_logprob)` — the probability the model put on the answer it gave | Free; rides along on the evaluation call |
| `margin` | How decisively the winning label beat the losing one, from the same call's `top_logprobs` | Free; same call |
| `verbalized` | The judge's own stated certainty, requested as a response field | Needs `prompt_version: v4` |

Every enabled method is recorded, and `primary` picks which one fills the `confidence` field that the comparison step treats as the headline signal:

| Field | Description |
|-------|-------------|
| `confidence` | The primary method's score |
| `confidence_source` | The primary method's name, or `unaligned` / `unavailable`; suffixed `_merged` on the sliding-window path |
| `confidence_scores` | Every captured method, e.g. `{"logprobs": 0.9998, "margin": 0.9995, "verbalized": 0.85}` |
| `answer_logprob` | Logprob summed over every token spelling the answer, reported unmodified |
| `p_yes` / `p_no` | Normalised over the two labels, from the first answer token's `top_logprobs` |
| `margin` | `abs(p_yes - p_no)` |
| `margin_counterpart_observed` | `false` when the losing label never appeared in `top_logprobs`, so its probability was bounded rather than read — see the caveat below |

One evaluation call covers a whole dimension, so each `included` value is a field inside a large JSON object rather than a standalone token. `llm/logprobs.py` rebuilds the response text from the token stream, pairs each `"included"` value with the `"id"` that precedes it, and sums the logprobs of the tokens covering that value — so a split (`Y` + `es`) or quote-merged (`"Yes`) answer still yields the right probability.

**Margin caveat.** The losing label is usually missing from the returned top-k: when the judge is confident, the runners-up are casing and whitespace variants, not the opposite answer. Its probability is then bounded by the smallest alternative that *was* returned, which makes margin a monotone function of the answer probability rather than an independent signal. Measured on 224 real indicators at `top_logprobs: 5`, the counterpart was observed **10 times out of 224 (4.5%)**, and margin ranked predictions identically to `logprobs` (AUROC 0.62 for both). `confidence_report.json` reports this ratio per run under `methods.margin.notes`. Azure caps `top_logprobs` at 5, so this cannot be improved by asking for more alternatives.

**Verbalized** produces visibly more spread than either token-level method — a probe returned 0.78/0.98/0.99 across three criteria where `logprobs` returned 0.96/1.0/1.0 — which makes it the most promising candidate for `primary`, though it has not yet been measured at scale.

Deployments that reject the `logprobs` parameter are handled automatically: the first 400 triggers one warning and a retry without it, all later calls skip it, and the affected methods yield `null`. A 400 that merely says `top_logprobs` is *too high* is treated differently — the value is clamped to the deployment's own limit and logprobs stay on, so one over-large setting doesn't cost the run its confidence scores. To skip logprobs entirely, set `supports_logprobs: false` on the model profile in `model_config.yaml`.

Note that the token-level methods saturate: on real policies the median is above 0.9999 and roughly 40% of indicators come back at exactly 1.0, which is why the comparison report sweeps quantile cutoffs alongside absolute ones.

Self-consistency (k samples per indicator, scored by vote share) is **not** implemented. It needs repeated sampling rather than a single call, so it belongs in a separate step; `llm/confidence.py` is structured so adding it means adding a method name and a writer for its score, and the comparison report will pick it up automatically.

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

  gpt-5.2-translation-qa:
    deployment: gpt-5.2
    temperature: 0.1
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
translation_qa:
  model: gpt-5.2-translation-qa
  prompt_version: v1
evaluation:
  model: gpt-4o-eval
  prompt_version: v1
discrepancy_diagnosis:
  model: gpt-5.2-diagnosis
  prompt_version: v1
```

`discrepancy_diagnosis.model`/`prompt_version` and `translation_qa.model`/`prompt_version` are required just like `translation`/`evaluation`'s (config is validated eagerly on every invocation, regardless of which `--steps` are selected). Templates live at `automation/prompts/discrepancy_diagnosis/<prompt_version>/prompt_template.txt` and `automation/prompts/translation_qa/<prompt_version>/prompt_template.txt` respectively.

#### Confidence (`evaluation.confidence` + `comparison.confidence`)

```yaml
evaluation:
  prompt_version: v3            # v4 adds the verbalized-confidence field
  confidence:
    enabled: true
    methods: [logprobs, margin] # any of logprobs / margin / verbalized
    primary: logprobs           # must be one of the enabled methods
    top_logprobs: 5             # alternatives per answer token; Azure caps at 5

comparison:
  confidence:
    thresholds: [0.5, 0.7, 0.9, 0.95, 0.99]   # absolute cutoffs to sweep
    quantiles: [0.05, 0.10, 0.20, 0.30]       # quantile cutoffs to sweep
    primary_threshold: 0.9                    # drives coverage.primary and low_confidence.csv
    flagged_sample_size: 100                  # least-confident predictions listed in the report
    calibration_bins: 10
```

Both blocks are optional and fall back to the values above. `evaluation.confidence` only affects the `evaluation` step; `comparison.confidence` only affects report cutoffs, so it can be re-tuned and comparison re-run without touching the model. A model profile can also opt out permanently with `supports_logprobs: false`.

`methods: []` disables capture entirely. `primary` must name one of the enabled methods, and `verbalized` requires `prompt_version: v4` — the field is only added to the response schema when the method is enabled, so a v3 run produces byte-identical requests to before the confidence module existed and stays comparable to the existing baseline.

**Switching `primary` after a run is a comparison-only change.** Every enabled method is stored per indicator, so:

```bash
python -m automation.src.run_pipeline --run-id <run> --steps comparison --force
```

recomputes the entire report — coverage, calibration, discrimination, breakdowns, `flagged_sample`, and `low_confidence.csv` — against the newly chosen method. No re-evaluation and no API calls.

The `confidence` field inside an evaluation report holds whichever method was primary *at evaluation time*, so comparison deliberately ignores it when the requested method has its own score, and reads that method's column instead. If the requested method was never captured, the report does **not** silently relabel someone else's numbers: it keeps the method it actually scored, sets `confidence_source.primary_method_warning`, and prints a warning naming what to re-run.

#### Translation QA + re-align (`translation_qa` step)

Runs right after `translation`, using a different model than translation by default (`gpt-5.2-translation-qa` vs. the translation model) as an independent cross-check. See [`results/translation_qa/`](#translation_qa) above for what it checks and writes. Idempotent like the other LLM steps: skips a policy whose `translation_qa/<stem>.json` report already exists unless `--force` is passed.

#### Markdown rendering (`markdown` step)

`results/cleaned_text/<policy>.cleaned.txt`, `results/cleaned_markdown/<policy>.cleaned.md`,
and `results/translation_markdown/<policy>.md` are written by their own
pipeline step, `markdown` — independent of `translation`, and idempotent: it
only (re)renders a policy whose outputs don't exist yet, so it's safe to run
anytime (`--steps markdown`) to backfill whatever is missing, without
re-running translation.

- `results/cleaned_text/<policy>.cleaned.txt` and
  `results/cleaned_markdown/<policy>.cleaned.md` are both rendered straight
  from the raw original-language input (`data/raw/localpolicies/<policy>.txt`)
  — they have no dependency on translation having run. `cleaned_text` is
  plain text (mirrors `translation/<policy>.txt`); `cleaned_markdown` is the
  same content rendered as Markdown (mirrors `translation_markdown`).
- `results/translation_markdown/<policy>.md` is rendered from
  `results/translation/<policy>.txt` (the translated English output), so a
  given policy is only rendered once its translation exists; policies not
  yet translated are silently skipped, not treated as failures.
- `results/translation/<policy>.txt` itself always stays plain text — it's
  what the storage/RAG and evaluation steps read, and reformatting it as
  Markdown in place would change both.

The Markdown artifacts are rendered by `chunking.render_markdown` on top of
`chunking.clean_text` (OCR-noise removal, paragraph reflow, structural-marker
detection — `automation/src/chunking.py:clean_text`): structural markers
become headings nested by keyword (`BAB`/`Chapter` → `#`,
`Bagian`/`Part`/`Section` → `##`, `Paragraf`/`Paragraph` → `###`,
`Pasal`/`Article` → `####`, with an unrecognized/foreign-language marker
nested one level under the most recent keyword heading), and
numbered/lettered/roman-numeral list starts become proper
ordered/nested-bullet Markdown lists. The plain-text artifacts
(`cleaned_text`, `translation`) skip that rendering step entirely.

`markdown` belongs to the raw-OCR-text path and is **not** part of the
default chain any more — run it explicitly with `--steps markdown` alongside
`--steps translation`. See `run_markdown_step` in
`automation/src/translate.py`. In the Markdown path the source side needs no
rendering at all (the input is already Markdown, and is snapshotted to
`results/source_markdown/`), and the translated side *is* Markdown.

#### Chunked translation (`translation.chunking`)

Every chunked translation run cleans and structurally splits each source file
first — strips OCR noise lines (including stray lone-punctuation residue like
a leftover `.` on its own line), reflows broken lines back into paragraphs,
and keeps structural markers (`BAB` / `Pasal` / `Bagian` / `Paragraf` /
all-caps titles) on their own line (`automation/src/chunking.py:clean_text`) —
this is internal to chunking (it decides the chunk boundaries) and isn't
persisted on its own; the `markdown` step above re-derives the same cleaning
from the raw input when it renders `results/cleaned_text/*.cleaned.txt` and
`results/cleaned_markdown/*.cleaned.md`.

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
| `--steps` | `translation`, `translation_qa`, `markdown`, `storage`, `evaluation`, `comparison`, `discrepancy_diagnosis` (default: all) |
| `--run-id` | Existing run ID (required for eval/comparison/diagnosis without translation) |
| `--force` | Re-run a step even if its output already exists (all steps are idempotent by default — they skip items that already have output) |
| `--allow-partial` | Save incomplete evaluation or discrepancy diagnosis reports |
| `--max-workers` | Override `concurrency.max_workers` from pipeline config |
| `--no-concurrency` | Disable parallel API calls (serial mode) |
| `--pipeline-config` | Path to experiment YAML (default: [`automation\config\pipeline_config.yaml`](.\config\pipeline_config.yaml))|
| `--model-config` | Path to model parameters YAML (default: [`automation\config\model_config.yaml`](.\config\model_config.yaml)) |

### Step rules

- **New run** (no `--run-id`): auto-generates `run_id`, runs all steps by default (`translation`, `translation_qa`, `markdown`, `storage`, `evaluation`, `comparison`, `discrepancy_diagnosis`)
- **Translation QA only**: requires `--run-id` and existing `translation/` outputs
- **Eval only**: requires `--run-id` and existing `translation/` outputs
- **Comparison only**: requires `--run-id` and existing `evaluation/` JSON reports
- **Discrepancy diagnosis**: requires `--run-id` and existing `rag_candidates/`, `evaluation/*.json`, and `comparison/` outputs. Runs whose `evaluation` step predates `rag_candidates/` need `evaluation` re-run with `--force` first. Produces no report files (not an error) when `comparison/error_analysis.csv` doesn't exist, i.e. zero mismatches
- **Idempotent by default**: translation, translation_qa, storage, evaluation, and discrepancy_diagnosis each skip an item (file/policy) that already has output — translation and storage skip an input whose output file exists, translation_qa skips a policy that already has a `translation_qa/<stem>.json` report, evaluation skips a policy that already has an eval report, discrepancy_diagnosis skips a policy that already has a `diagnosis/<stem>.json`. Comparison always re-runs (it recomputes `metrics.csv`/`error_analysis.csv` from whatever `evaluation/` reports currently exist). Pass `--force` to re-run a step's items regardless of existing output
- **Step failures**: if a step fails outright (e.g. a required input directory is missing), the pipeline records the step name and error message under `step_failures` in `failures.json`, prints `Step '<name>' failed: <message>` to stderr, and stops before running any later steps

API retry messages include error type, message, HTTP status, and request ID (SDK-level httpx retry noise is suppressed).
