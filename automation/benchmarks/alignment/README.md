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
# Copy gold_draft.jsonl to gold.jsonl and correct the alignments; set "status": "eliminate" on units to drop
python -m automation.benchmarks.alignment.run_benchmark --project indonesia --aligners index,heuristic
```

Gold units are sampled half from count-mismatched sections (where aligners are
actually exercised) and tagged dev (~30%, for tuning) or test (reported). The
unannotated remainder is only used to report how often counts already match.

Alignment format: `[[src_indices], [tgt_indices]]` beads over the unit's
0-based `src` / `tgt` sentence lists; an empty side means no counterpart.

Metrics (`metrics.py`): strict bead P/R/F1, lenient (overlap) F1, evidence
recall (per source sentence, did we return exactly the gold target set), and
recall by bead type (1-1, 1-2, ...).

## Results (Indonesian gold set, 119 units; 83 test / 36 dev)

Strict bead scores on the test split; evidence recall = per source sentence, exact target set.

| aligner | strict P | strict R | strict F1 | evidence recall |
|---|---|---|---|---|
| index (pair by position) | 0.953 | 0.414 | 0.577 | 0.368 |
| heuristic (list markers) | 0.752 | 0.546 | 0.633 | 0.502 |
| Bertalign + LaBSE | 0.838 | 0.927 | 0.880 | 0.838 |
| Vecalign + LaBSE | 0.850 | 0.937 | 0.891 | 0.848 |

Half of the gold units are count-mismatched by design, so `index` is penalized
here; in the unannotated remainder 97% of units already have equal counts, and
index pairing is exactly right on 57 of the 60 count-matched gold units.
Decision (step 4.6): pair by index when counts match, else Vecalign + LaBSE.
Known limit: both aligners recover almost no 2-2 beads (they emit two 1-1
links instead). Lenient scores are in `results.json` but flatter coarse
aligners, so prefer strict and evidence recall.
