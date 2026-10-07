import pytest

from automation.src.config_loader import parse_alignment_method
from automation.src.rag import evidence_align
from automation.src.rag.embedding_align import AlignerUnavailable
from automation.src.rag.evidence_align import align_chunk_evidence
from automation.src.rag.retriever import RetrievedChunk
from automation.src.rag.sentence_align import GRANULARITY_ITEM, GRANULARITY_SENTENCE


@pytest.fixture(autouse=True)
def _clear_cache():
    evidence_align._embedding_beads.cache_clear()


def chunk(text, source_text):
    return RetrievedChunk(chunk_id=0, text=text, score=0.0, source_text=source_text)


def test_equal_counts_pair_by_index_without_calling_an_aligner(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("aligner must not run when counts match")

    monkeypatch.setattr(evidence_align, "align_sentences", boom)
    result = align_chunk_evidence(
        chunk("First rule. Second rule.", "Aturan pertama. Aturan kedua.")
    )
    assert result == {
        0: ("Aturan pertama.", GRANULARITY_SENTENCE),
        1: ("Aturan kedua.", GRANULARITY_SENTENCE),
    }


def test_unequal_counts_use_the_aligner_and_join_many_to_one(monkeypatch):
    # src: 3 sentences, tgt: 2. Beads: 0->0, (1,2)->1.
    monkeypatch.setattr(
        evidence_align,
        "align_sentences",
        lambda src, tgt, method: [([0], [0]), ([1, 2], [1])],
    )
    result = align_chunk_evidence(
        chunk("One. Two.", "Satu. Dua. Tiga."), method="vecalign"
    )
    assert result[0] == ("Satu.", GRANULARITY_SENTENCE)
    assert result[1] == ("Dua. Tiga.", GRANULARITY_ITEM)


def test_a_translated_sentence_with_no_source_is_left_out(monkeypatch):
    monkeypatch.setattr(
        evidence_align,
        "align_sentences",
        lambda src, tgt, method: [([0], [0]), ([], [1]), ([1], [2])],
    )
    result = align_chunk_evidence(
        chunk("One. Added. Two.", "Satu. Dua."), method="vecalign"
    )
    assert 1 not in result and result[0][0] == "Satu." and result[2][0] == "Dua."


def test_aligner_failure_falls_back_to_the_heuristic(monkeypatch):
    def unavailable(*a, **k):
        raise AlignerUnavailable("not installed")

    monkeypatch.setattr(evidence_align, "align_sentences", unavailable)
    monkeypatch.setattr(evidence_align, "heuristic_align", lambda c: {0: ("fallback", "chunk")})
    result = align_chunk_evidence(chunk("One. Two.", "Satu. Dua. Tiga."))
    assert result == {0: ("fallback", "chunk")}


def test_heuristic_method_skips_the_embedding_aligner(monkeypatch):
    monkeypatch.setattr(
        evidence_align,
        "align_sentences",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not run")),
    )
    monkeypatch.setattr(evidence_align, "heuristic_align", lambda c: {0: ("h", "chunk")})
    assert align_chunk_evidence(chunk("One. Two.", "Satu. Dua. Tiga."), "heuristic") == {
        0: ("h", "chunk")
    }


def test_no_source_text_gives_nothing():
    assert align_chunk_evidence(chunk("One.", None)) == {}


def test_alignment_method_config_is_validated():
    assert parse_alignment_method(None) == "vecalign"
    assert parse_alignment_method({"method": "bertalign"}) == "bertalign"
    with pytest.raises(ValueError):
        parse_alignment_method({"method": "magic"})
