"""Best-effort pairing between a retrieved chunk's translated text and its
original-language text, for showing evidence side by side (see
`prompt_builder.build_source_text_lookup` and
`evidence_check.resolve_evidence_source_text`).

The corpus is dense with lettered/numbered lists (``a.`` ``b.`` ``c.`` ...).
`translation_qa_md` puts every marker on its own line in the English text, so
`pysbd` segments it cleanly; the Indonesian ``source_text`` often has markers
left glued to neighbouring text by OCR (``berhak: a. untuk...``,
``...Perlindunganb. Pemberdayaan``), so `pysbd` produces fewer segments and a
strict whole-chunk sentence-count match fails.

So when a whole-chunk sentence count match fails, pairing is retried **per list
item**: the label sequence is read off the clean English side, each label is
located in the Indonesian text (line-initial, glued to punctuation, or glued to
a word), and English item *k* is paired with Indonesian item *k*. Within a
paired item, if both sides segment to the same sentence count they pair
positionally (`GRANULARITY_SENTENCE`); otherwise the whole Indonesian item is
used (`GRANULARITY_ITEM`). A chunk with no list markers, and anything the
aligner can't place, falls back to the whole chunk (`GRANULARITY_CHUNK`) --
never less informative than that.
"""

import re

from automation.src.rag.retriever import RetrievedChunk
from automation.src.rag.sentence_split import split_sentences

#: How finely a flat sentence index was resolved, finest first. Exposed via
#: `align_chunk_sentences_detailed` so callers can tell an exact per-sentence
#: pairing from a coarser fallback (see run_eval.py's `evidence_original_aligned`).
GRANULARITY_SENTENCE = "sentence"
GRANULARITY_ITEM = "item"
GRANULARITY_CHUNK = "chunk"

#: A list marker at the head of a sentence segment: ``a.`` ``b)`` ``12.`` ``(3)``.
#: Roman numerals are excluded on purpose -- ``i.``/``v.``/``x.`` collide with
#: real single letters and this corpus's item lists are alphabetic or numeric.
_LABEL_HEAD = re.compile(r"^\(?([a-z]|\d{1,3})[.)](?:\s|$)")
#: The same marker seen mid-segment, right after a chapeau's ``:`` or ``;``
#: (e.g. ``"...include: a. implementing"``, ``"...meliputi:a. penetapan"``).
_LABEL_AFTER_COLON = re.compile(r"[:;]\s*\(?([a-z]|\d{1,3})[.)]\s")


def _next_labels(prev: str) -> set[str]:
    """The label(s) that may legitimately follow `prev` in a list (`prev==""`
    means "the first item"). Keeps a stray ``M.`` / ``No.`` from being read as a
    new item -- only a label that continues the sequence starts one."""
    if prev == "":
        return {"a", "1"}
    if prev.isdigit():
        return {str(int(prev) + 1)}
    if len(prev) == 1 and prev.isalpha():
        return {chr(ord(prev) + 1)}
    return set()


def _group_en_items(sentences: list[str]) -> list[tuple[str, list[int]]]:
    """Group the flat English sentence list into ``(label, [flat_idx, ...])``
    items. ``items[0]`` is the stem (label ``""``) -- everything before the
    first list marker. Only a marker that continues the label sequence opens a
    new item."""
    items: list[tuple[str, list[int]]] = [("", [])]
    for i, sentence in enumerate(sentences):
        allowed = _next_labels(items[-1][0])
        head = _LABEL_HEAD.match(sentence)
        colon = _LABEL_AFTER_COLON.search(sentence)
        label = None
        if head and head.group(1) in allowed:
            label = head.group(1)
        elif colon and colon.group(1) in allowed:
            label = colon.group(1)
        if label is not None:
            items.append((label, [i]))
        else:
            items[-1][1].append(i)
    if not items[0][1]:
        items.pop(0)
    return items


def _locate_labels_in_source(source_text: str, labels: list[str]) -> dict[str, tuple[int, int]]:
    """For each label, in order, find its marker in `source_text` -- line-initial,
    glued after ``:``/``;``/``.``/space, or glued straight onto a word
    (``Perlindungan`` + ``b.``). Returns ``{label: (start, end)}`` for the ones
    found; the search only ever moves forward, so a later label can't match text
    before an earlier one."""
    spans: dict[str, tuple[int, int]] = {}
    cursor = 0
    for label in labels:
        esc = re.escape(label)
        strict = re.compile(
            r"(?:(?<=\n)|(?<=[\s:;.)])|^)\(?(" + esc + r")[.)](?=\s|[A-Z(])"
        )
        m = strict.search(source_text, cursor)
        if m is None:
            glued = re.compile(r"(" + esc + r")[.)](?=\s+[A-Za-z(])")
            m = glued.search(source_text, cursor)
        if m is None:
            continue
        spans[label] = (m.start(), m.end())
        cursor = m.end()
    return spans


def _source_item_texts(source_text: str, labels: list[str]) -> dict[str, str]:
    """Split `source_text` into ``{label: item_text}`` at the located markers.
    ``labels[0]`` must be ``""`` (the stem); its text is whatever precedes the
    first located marker. A label that couldn't be located is absent."""
    spans = _locate_labels_in_source(source_text, [x for x in labels if x])
    ordered = sorted(spans.items(), key=lambda kv: kv[1][0])
    out: dict[str, str] = {}
    first_start = ordered[0][1][0] if ordered else len(source_text)
    out[""] = source_text[:first_start].strip()
    for i, (label, (start, _end)) in enumerate(ordered):
        end = ordered[i + 1][1][0] if i + 1 < len(ordered) else len(source_text)
        out[label] = source_text[start:end].strip()
    return out


def align_chunk_sentences(candidate: RetrievedChunk) -> dict[int, str]:
    """Map each of `candidate.text`'s flat sentence indices (as produced by
    `split_sentences`) to the best available original-language text.

    Returns `{}` when `candidate.source_text` is unavailable -- callers treat a
    missing index as "no original text for this sentence".
    """
    return {idx: text for idx, (text, _g) in align_chunk_sentences_detailed(candidate).items()}


def align_chunk_sentences_detailed(candidate: RetrievedChunk) -> dict[int, tuple[str, str]]:
    """Like `align_chunk_sentences`, but each value is `(text, granularity)` with
    granularity one of the `GRANULARITY_*` constants, so callers can tell an
    exact per-sentence pairing from a coarser item/chunk fallback.
    """
    if not candidate.source_text:
        return {}

    en_sents = split_sentences(candidate.text)
    if not en_sents:
        return {}
    source_text = candidate.source_text
    whole_chunk = (source_text, GRANULARITY_CHUNK)

    # 1. Whole-chunk positional pairing when both sides segment to the same
    #    count -- the simplest sound pairing, and what worked before per-item
    #    alignment existed. Never does worse than this.
    id_sents = split_sentences(source_text, protect_abbreviations=True)
    if len(id_sents) == len(en_sents):
        return {i: (id_sents[i], GRANULARITY_SENTENCE) for i in range(len(en_sents))}

    # 2. Counts disagree -- almost always a list marker glued to its text on the
    #    Indonesian side. Align per list item, anchoring on the label sequence
    #    read off the (clean) English side.
    items = _group_en_items(en_sents)
    labels = [label for label, _ in items]
    if labels == [""]:
        return {i: whole_chunk for i in range(len(en_sents))}

    source_items = _source_item_texts(source_text, labels)

    resolved: dict[int, tuple[str, str]] = {}
    for label, en_idxs in items:
        id_item = source_items.get(label)
        if not id_item:
            for i in en_idxs:
                resolved[i] = whole_chunk
            continue
        id_item_sents = split_sentences(id_item, protect_abbreviations=True)
        if len(id_item_sents) == len(en_idxs):
            for slot, i in enumerate(en_idxs):
                resolved[i] = (id_item_sents[slot], GRANULARITY_SENTENCE)
        else:
            for i in en_idxs:
                resolved[i] = (id_item, GRANULARITY_ITEM)
    return resolved
