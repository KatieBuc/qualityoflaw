# Design notes

Why the pipeline works the way it does. For running it see the [README](../../README.md); for
the files a run writes see [outputs.md](outputs.md).

## Markdown translation path

The default chain:

```
translation_md -> md_to_text -> storage -> evaluation -> comparison
```

`translation_qa_md` (after `translation_md`) and `discrepancy_diagnosis` (last) are part of
this path but only run when requested with `--steps`.

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
the raw-text QA step (and it reuses the retry-then-degrade logic in `automation.src.common.qa_unit`),
audited per packed unit. The prompt adds Markdown fidelity as its first
check. Corrections are written back to both the Markdown and the chunk
artifact, atomically per policy.

**Each chunk is audited repeatedly, up to `qa_max_passes` (2 in the shipped `pipeline_config.yaml`).** A
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
number of chunks. Over the full corpus that is 1,116 to 2,232 QA calls (at `qa_max_passes: 2`; 5,580 at 5),
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
  qa_max_passes: 2          # re-audit a chunk until clean, at most this often
  chunking:
    target_chars: 8000
    safe_limit: 32000

paths:
  input_dir: data/raw/localpolicies                              # raw OCR text; still used by diagnose
  markdown_input_dir: data/processed/localpolicies/cleaned_markdown
```

Models are not duplicated — the Markdown path reuses `translation.model` and
`translation_qa.model`.

## Criteria files and their coding rubric

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

## Per-indicator confidence

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

## Embedding chunk size (storage step)

`translation.chunking.safe_limit` sizes chunks in characters against the *translation* LLM's context window — it's much larger than the *embedding* model's hard 8192-token input cap, and legal/citation-heavy text (dense with digits and punctuation) can tokenize less efficiently than prose, so a chunk under `safe_limit` can still exceed 8192 tokens at embedding time. The storage step sub-splits any chunk over `EMBEDDING_SAFE_CHAR_LIMIT` (6000 chars, `automation/src/rag/store.py`) before calling the embedding API, independent of the translation chunk boundaries — this only affects `mid_product/rag_store/`'s retrieval granularity, not `mid_product/chunks/` or the translated output.

