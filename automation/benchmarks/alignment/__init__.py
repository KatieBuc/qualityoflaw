"""Benchmark for sentence-level alignment of translated vs. source text.

Workflow (see README.md in this folder):

1. `build_gold.py sample`  -- sample section pairs from `data/<project>/processed`
   and split them into gold candidates vs. an unannotated remainder.
2. `build_gold.py draft`   -- LLM-draft alignments for the gold candidates.
3. A human corrects `gold_draft.jsonl` and saves it as `gold.jsonl`.
4. `run_benchmark.py`      -- score every aligner on the gold set, and report
   how often the translated/source sentence counts already match on the rest.
"""
