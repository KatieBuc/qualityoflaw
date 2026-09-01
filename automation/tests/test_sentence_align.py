from automation.src.rag.retriever import RetrievedChunk
from automation.src.rag.sentence_align import (
    GRANULARITY_ITEM,
    GRANULARITY_SENTENCE,
    align_chunk_sentences,
    align_chunk_sentences_detailed,
)


def test_sentence_level_pairing_when_whole_chunk_counts_match():
    candidate = RetrievedChunk(
        chunk_id=0,
        text="Article 1\nThe Regent shall establish a committee. It shall report annually.",
        source_text="Pasal 1\nBupati harus membentuk komite. Komite harus melapor setiap tahun.",
        score=1.0,
    )
    resolved = align_chunk_sentences(candidate)
    assert resolved == {
        0: "Pasal 1",
        1: "Bupati harus membentuk komite.",
        2: "Komite harus melapor setiap tahun.",
    }


def test_sentence_level_pairing_when_per_line_grouping_differs_but_totals_match():
    candidate = RetrievedChunk(
        chunk_id=0,
        # 2 sentences on line 0, 1 on line 1.
        text="The law is enacted. It takes effect now.\nA committee is formed.",
        # 1 sentence on line 0, 2 on line 1 -- per-line grouping disagrees, but
        # the whole-chunk sentence count still matches (3 == 3), so pairing is
        # positional over the flat sentence lists.
        source_text="Undang-undang ini disahkan.\nUndang-undang ini berlaku sekarang. Sebuah komite dibentuk.",
        score=1.0,
    )
    resolved = align_chunk_sentences(candidate)
    assert resolved == {
        0: "Undang-undang ini disahkan.",
        1: "Undang-undang ini berlaku sekarang.",
        2: "Sebuah komite dibentuk.",
    }


def test_chunk_fallback_when_sentence_counts_disagree():
    candidate = RetrievedChunk(
        chunk_id=0,
        text="Article 1\nThe Regent shall establish a committee. It shall report annually.",
        # The Indonesian merges both sentences into one -- whole-chunk counts
        # disagree (2 vs 3), so every translated sentence falls back to the
        # whole chunk's original text.
        source_text="Pasal 1\nBupati harus membentuk komite yang melapor setiap tahun.",
        score=1.0,
    )
    resolved = align_chunk_sentences(candidate)
    whole_chunk = candidate.source_text
    assert resolved == {0: whole_chunk, 1: whole_chunk, 2: whole_chunk}


def test_chunk_fallback_when_counts_disagree_across_blocks():
    candidate = RetrievedChunk(
        chunk_id=0,
        text="Article 1\nThe Regent shall establish a committee.\nArticle 2\nThis is effective immediately.",
        # Only two lines / two sentences on the source side -- translation split
        # a source block into more blocks, so counts can't be trusted at all.
        source_text="Pasal 1\nBupati harus membentuk komite dan berlaku efektif segera.",
        score=1.0,
    )
    resolved = align_chunk_sentences(candidate)
    whole_chunk = candidate.source_text
    assert all(text == whole_chunk for text in resolved.values())
    assert len(resolved) == 4


def test_list_items_pair_when_indonesian_markers_are_glued_to_a_word():
    # pysbd splits "a." "b." off cleanly on the English side but not on the
    # Indonesian side, where OCR left "b." glued to the previous word. The
    # whole-chunk counts disagree (6 vs 5), so per-item alignment kicks in.
    candidate = RetrievedChunk(
        chunk_id=0,
        text="Article 6\nBased on two pillars:\na. Protection\nb. Empowerment",
        source_text="Pasal 6\nDidasarkan pada dua pilar:\na. Perlindunganb. Pemberdayaan",
        score=1.0,
    )
    detailed = align_chunk_sentences_detailed(candidate)
    assert [g for _, (_t, g) in sorted(detailed.items())] == [GRANULARITY_SENTENCE] * 6
    assert align_chunk_sentences(candidate) == {
        0: "Pasal 6",
        1: "Didasarkan pada dua pilar:",
        2: "a.",
        3: "Perlindungan",
        4: "b.",
        5: "Pemberdayaan",
    }


def test_list_item_falls_back_to_the_whole_item_when_its_sentences_dont_line_up():
    # Item "a." has two English sentences but its Indonesian counterpart is one
    # run-on -- that item resolves at item granularity, the rest stay sentence.
    candidate = RetrievedChunk(
        chunk_id=0,
        text="Article 2\nObjectives:\na. to prevent violence. It must be systemic.\nb. to protect victims.",
        source_text="Pasal 2\nTujuan:\na. mencegah kekerasan yang sistemik.\nb. melindungi korban.",
        score=1.0,
    )
    detailed = align_chunk_sentences_detailed(candidate)
    by_gran = {g for _i, (_t, g) in detailed.items()}
    assert by_gran == {GRANULARITY_SENTENCE, GRANULARITY_ITEM}
    # both sentences of item "a." point at the whole Indonesian item
    assert detailed[3][0] == detailed[4][0] == "a. mencegah kekerasan yang sistemik."
    assert detailed[3][1] == GRANULARITY_ITEM


def test_missing_indonesian_label_falls_back_to_chunk_for_that_item_only():
    candidate = RetrievedChunk(
        chunk_id=0,
        text="Article 3\nRights:\na. to life;\nb. to safety;\nc. to redress.",
        # Indonesian dropped item "b." entirely.
        source_text="Pasal 3\nHak:\na. untuk hidup;\nc. untuk pemulihan.",
        score=1.0,
    )
    detailed = align_chunk_sentences_detailed(candidate)
    grans = [g for _i, (_t, g) in sorted(detailed.items())]
    assert GRANULARITY_SENTENCE in grans and "chunk" in grans
    # the "b." sentence has no Indonesian counterpart
    b_idx = 4
    assert detailed[b_idx] == (candidate.source_text, "chunk")


def test_missing_source_text_returns_empty_mapping():
    candidate = RetrievedChunk(chunk_id=0, text="Article 1\nSome text.", source_text=None, score=1.0)
    assert align_chunk_sentences(candidate) == {}


def test_empty_translated_text_returns_empty_mapping():
    candidate = RetrievedChunk(chunk_id=0, text="", source_text="Pasal 1", score=1.0)
    assert align_chunk_sentences(candidate) == {}
