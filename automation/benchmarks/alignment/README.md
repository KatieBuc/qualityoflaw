# Sentence-alignment benchmark

Decides which aligner step 4.6 (evidence mapping) uses when the translated and
source sections do not have the same number of sentences.

Candidates: `index` (pair by position, only valid when counts match),
`heuristic` (the pipeline's current list-marker aligner), `bertalign` and
`vecalign`, both with LaBSE embeddings (the last two are added in
`embedding_aligners.py`).

## Steps

```bash
python -m automation.benchmarks.alignment.build_gold sample --project indonesia --n-gold 120
python -m automation.benchmarks.alignment.build_gold draft  --project indonesia   # LLM calls
# Review data/indonesia/gold_draft.jsonl, fix alignments, set "status": "reviewed", save as gold.jsonl
python -m automation.benchmarks.alignment.run_benchmark --project indonesia --split test --aligners index,heuristic
```

Gold units are sampled half from count-mismatched sections (where aligners are
actually exercised) and tagged dev (~30%, for tuning) or test (reported). The
unannotated remainder is only used to report how often counts already match.

Alignment format: `[[src_indices], [tgt_indices]]` beads over the unit's
0-based `src` / `tgt` sentence lists; an empty side means no counterpart.

Metrics (`metrics.py`): strict bead P/R/F1, lenient (overlap) F1, evidence
recall (per source sentence, did we return exactly the gold target set), and
recall by bead type (1-1, 1-2, ...).
