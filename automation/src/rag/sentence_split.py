"""Sentence segmentation for chunk text, so retrieved passages can be cited
at sentence granularity instead of as a whole chunk (see `prompt_builder.py`
and `evidence_check.py`).

Uses `pysbd` rather than a punctuation regex because legal/regulatory text is
full of abbreviations, decimals, and numbered clauses that naive
"split on period" rules mis-segment.

`pysbd` has no Indonesian language model, but the original-language text is
also Latin script with the same terminal punctuation, so `sentence_align.py`
runs it through this same `language="en"` segmenter. An Indonesian miscount
there is harmless: it only fails the strict 1:1 sentence-count gate and drops
that chunk to whole-chunk pairing (see `sentence_align.align_chunk_sentences`).
"""

import pysbd

_segmenter = pysbd.Segmenter(language="en", clean=True)


def split_sentences(text: str) -> list[str]:
    return [sentence for _line_idx, sentence in split_sentences_with_lines(text)]


def split_sentences_with_lines(text: str) -> list[tuple[int, str]]:
    """Segment `text` into sentences, tagged with each sentence's line index.

    `text` is always one block (heading/paragraph/list item) per line, blank
    lines already collapsed (see `markdown.to_text.markdown_to_text`).
    Segmentation runs per line rather than once over the whole joined text:
    `pysbd` does not always agree with itself on where a boundary falls
    inside a short isolated line (e.g. a bare list marker like "a." followed
    by its item text) versus that same line embedded in a longer passage, so
    a single whole-text call cannot be trusted to line up 1:1 with a
    per-line call. `split_sentences` is defined in terms of this function
    specifically so the two can never drift apart -- `sentence_align.py`
    depends on their flat sentence order matching exactly.
    """
    if not text or not text.strip():
        return []
    return [
        (line_idx, sentence)
        for line_idx, line in enumerate(text.split("\n"))
        for sentence in _segment_line(line)
    ]


def _segment_line(line: str) -> list[str]:
    line = line.strip()
    if not line:
        return []
    return [s.strip() for s in _segmenter.segment(line) if s.strip()]
