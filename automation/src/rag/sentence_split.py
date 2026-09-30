"""Sentence segmentation for chunk text, so retrieved passages can be cited
at sentence granularity instead of as a whole chunk (see `prompt_builder.py`
and `evidence_check.py`).

Uses `pysbd` rather than a punctuation regex because legal/regulatory text is
full of abbreviations, decimals, and numbered clauses that naive
"split on period" rules mis-segment.

`pysbd` has no Indonesian language model, but the original-language text is
also Latin script with the same terminal punctuation, so `sentence_align.py`
runs it through this same `language="en"` segmenter -- with
`protect_abbreviations=True`, which shields Indonesian legal abbreviations the
English model would otherwise split on (`dll.`, `Plt.`, `Kab.`, ...).
"""

import re

import pysbd

_segmenter = pysbd.Segmenter(language="en", clean=True)

# Indonesian abbreviations common in regulatory text that `pysbd`'s English
# model does not know, so it treats the trailing "." as a sentence boundary.
# Matched case-insensitively as a whole word; the "." after is neutralised
# before segmentation and restored after.
_SOURCE_ABBREVIATIONS = (
    "dll dst dsb dkk drs tsb ybs sbb "
    "plt plh pj ttd kab kota prov kec kel "
    "setda sekda setwan hlm jo jis no "
    "prof dr ir hj drg "
).split()
_ABBREV_RE = re.compile(
    r"\b(" + "|".join(re.escape(a) for a in _SOURCE_ABBREVIATIONS) + r")\.",
    re.IGNORECASE,
)
# Dotted abbreviations like "a.n." "u.b." "s.d." "d.a." -- each internal dot.
_DOTTED_ABBREV_RE = re.compile(r"\b([a-z](?:\.[a-z])+)\.", re.IGNORECASE)
_DOT = "\x00"


def split_sentences(text: str, *, protect_abbreviations: bool = False) -> list[str]:
    return [s for _line_idx, s in split_sentences_with_lines(text, protect_abbreviations=protect_abbreviations)]


def split_sentences_with_lines(
    text: str, *, protect_abbreviations: bool = False
) -> list[tuple[int, str]]:
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

    `protect_abbreviations` (used for the Indonesian `source_text` side) hides
    the "." in known Indonesian abbreviations from the segmenter and restores
    it afterwards, so `... dll. Ketentuan ...` is not split at `dll.`.
    """
    if not text or not text.strip():
        return []
    return [
        (line_idx, sentence)
        for line_idx, line in enumerate(text.split("\n"))
        for sentence in _segment_line(line, protect_abbreviations)
    ]


def _segment_line(line: str, protect_abbreviations: bool) -> list[str]:
    line = line.strip()
    if not line:
        return []
    if protect_abbreviations:
        line = _DOTTED_ABBREV_RE.sub(lambda m: m.group(0).replace(".", _DOT), line)
        line = _ABBREV_RE.sub(lambda m: m.group(1) + _DOT, line)
    return [s.strip().replace(_DOT, ".") for s in _segmenter.segment(line) if s.strip()]
