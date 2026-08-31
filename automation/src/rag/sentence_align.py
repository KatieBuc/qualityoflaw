"""Best-effort sentence-level pairing between a retrieved chunk's translated
text and its original-language text, for showing evidence side by side (see
`prompt_builder.build_source_text_lookup` and
`evidence_check.resolve_evidence_source_text`).

Both sides are segmented with the same `pysbd` `language="en"` segmenter (see
`sentence_split.py`) -- the original Indonesian text is Latin script with the
same terminal punctuation, so this works well enough in practice. A per-sentence
pairing is only trusted when both sides produce the **same** sentence count for
the whole chunk (strict 1:1); any mismatch -- a merged/split sentence, an
Indonesian abbreviation `pysbd` doesn't know -- falls back to the whole chunk's
original text. Never less informative than always using the whole chunk, which
is what happened before this module existed.
"""

from automation.src.rag.retriever import RetrievedChunk
from automation.src.rag.sentence_split import split_sentences

#: How finely a flat sentence index was resolved. Exposed via
#: `align_chunk_sentences_detailed` so callers can tell an exact per-sentence
#: pairing from a whole-chunk fallback (see run_eval.py's
#: `evidence_original_aligned`).
GRANULARITY_SENTENCE = "sentence"
GRANULARITY_CHUNK = "chunk"


def align_chunk_sentences(candidate: RetrievedChunk) -> dict[int, str]:
    """Map each of `candidate.text`'s flat sentence indices (as produced by
    `split_sentences`) to the best available original-language text for that
    sentence.

    Returns `{}` when `candidate.source_text` is unavailable -- callers treat
    a missing index as "no original text for this sentence", same as today.
    """
    return {idx: text for idx, (text, _granularity) in align_chunk_sentences_detailed(candidate).items()}


def align_chunk_sentences_detailed(candidate: RetrievedChunk) -> dict[int, tuple[str, str]]:
    """Like `align_chunk_sentences`, but each resolved value is paired with
    the granularity it was resolved at (one of the `GRANULARITY_*` constants
    above), so callers can distinguish an exact per-sentence pairing from a
    best-effort whole-chunk fallback.
    """
    if not candidate.source_text:
        return {}

    translated = split_sentences(candidate.text)
    if not translated:
        return {}

    source = split_sentences(candidate.source_text)

    if len(source) == len(translated):
        # Same sentence count on both sides of the whole chunk -- pair
        # positionally.
        return {idx: (source[idx], GRANULARITY_SENTENCE) for idx in range(len(translated))}

    # Counts disagree (translation merged/split a sentence, or the Indonesian
    # segmentation miscounted) -- fall back to the whole chunk's original text
    # for every sentence.
    return {idx: (candidate.source_text, GRANULARITY_CHUNK) for idx in range(len(translated))}
