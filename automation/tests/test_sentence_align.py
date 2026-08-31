from automation.src.rag.retriever import RetrievedChunk
from automation.src.rag.sentence_align import align_chunk_sentences


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


def test_missing_source_text_returns_empty_mapping():
    candidate = RetrievedChunk(chunk_id=0, text="Article 1\nSome text.", source_text=None, score=1.0)
    assert align_chunk_sentences(candidate) == {}


def test_empty_translated_text_returns_empty_mapping():
    candidate = RetrievedChunk(chunk_id=0, text="", source_text="Pasal 1", score=1.0)
    assert align_chunk_sentences(candidate) == {}
