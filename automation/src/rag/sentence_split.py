"""Sentence segmentation for chunk text, so retrieved passages can be cited
at sentence granularity instead of as a whole chunk (see `prompt_builder.py`
and `evidence_check.py`).

Uses `pysbd` rather than a punctuation regex because legal/regulatory text is
full of abbreviations, decimals, and numbered clauses that naive
"split on period" rules mis-segment.
"""

import pysbd

_segmenter = pysbd.Segmenter(language="en", clean=True)


def split_sentences(text: str) -> list[str]:
    if not text or not text.strip():
        return []
    return [s.strip() for s in _segmenter.segment(text) if s.strip()]
