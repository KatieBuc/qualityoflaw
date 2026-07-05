import pytest

from automation.src.chunking import (
    Chunk,
    check_fallback_output,
    chunk_policy_text,
    chunk_policy_text_with_debug,
    clean_text,
    combine_translations,
    fallback_split,
    is_noise_line,
    is_structure_marker,
    split_into_sections,
)


def test_is_noise_line():
    assert is_noise_line("4a")
    assert is_noise_line("dh")
    assert is_noise_line("Vv")
    assert is_noise_line("12")
    assert not is_noise_line("1234")  # too long to be noise
    assert not is_noise_line("Hello world")


def test_is_noise_line_lone_punctuation():
    # A line that's nothing but a single punctuation mark is OCR residue
    # (e.g. a stray "." left over after the paragraph already ended above it).
    assert is_noise_line(".")
    assert is_noise_line(";")
    assert is_noise_line(",")
    # Two punctuation marks, or punctuation attached to a list marker, is not
    # this specific residue pattern.
    assert not is_noise_line("..")
    assert not is_noise_line("1.")
    assert not is_noise_line("a.")


def test_is_structure_marker_regex_forms():
    assert is_structure_marker("BAB I")
    assert is_structure_marker("Pasal 5")
    assert is_structure_marker("Bagian Kedua")
    assert is_structure_marker("Paragraf 1")
    assert not is_structure_marker("Pemerintah Daerah adalah Pemerintah Kota Pekanbaru.")


def test_is_structure_marker_all_caps_title():
    assert is_structure_marker("KETENTUAN UMUM")
    assert is_structure_marker("PERATURAN DAERAH KOTA PEKANBARU")
    # Mixed case, not a recognized keyword -> not a marker.
    assert not is_structure_marker("Pasal")
    # All caps but too long (>= 60 chars) -> not a title.
    long_line = "A" * 60
    assert not is_structure_marker(long_line)
    # Purely numeric/punctuation line has no letters -> not a title.
    assert not is_structure_marker("12.3;")


def test_clean_text_reflows_paragraph_split_across_lines():
    raw = "Daerah adalah Kota\nPekanbaru, sesuai UU."
    cleaned = clean_text(raw)
    assert len(cleaned) == 1
    assert cleaned[0].text == "Daerah adalah Kota Pekanbaru, sesuai UU."
    assert cleaned[0].is_structure is False


def test_clean_text_drops_noise_lines():
    raw = "dh\nDaerah adalah Kota Pekanbaru."
    cleaned = clean_text(raw)
    assert len(cleaned) == 1
    assert cleaned[0].text == "Daerah adalah Kota Pekanbaru."


def test_clean_text_drops_lone_punctuation_residue():
    # A leftover "." on its own line after "...untuk tahun berjalan." must not
    # survive into the cleaned output (or a chunk).
    raw = "Anggaran untuk tahun berjalan.\n.\nPasal 17\nKetentuan berikutnya selesai."
    cleaned = clean_text(raw)
    assert [c.text for c in cleaned] == [
        "Anggaran untuk tahun berjalan.",
        "Pasal 17",
        "Ketentuan berikutnya selesai.",
    ]


def test_clean_text_keeps_structure_markers_standalone():
    raw = "BAB I\nKETENTUAN UMUM\nPasal 1\nDaerah adalah Kota Pekanbaru."
    cleaned = clean_text(raw)
    assert [c.text for c in cleaned] == [
        "BAB I",
        "KETENTUAN UMUM",
        "Pasal 1",
        "Daerah adalah Kota Pekanbaru.",
    ]
    assert [c.is_structure for c in cleaned] == [True, True, True, False]


def test_clean_text_list_start_forces_flush():
    raw = "Pendahuluan diatur sebagai berikut\n1. Daerah adalah Kota Pekanbaru."
    cleaned = clean_text(raw)
    assert [c.text for c in cleaned] == [
        "Pendahuluan diatur sebagai berikut",
        "1. Daerah adalah Kota Pekanbaru.",
    ]


def test_split_into_sections_groups_by_marker():
    raw = (
        "BAB I\nKETENTUAN UMUM\nPasal 1\nIsi pasal satu.\n"
        "BAB II\nMAKSUD DAN TUJUAN\nPasal 2\nIsi pasal dua."
    )
    cleaned = clean_text(raw)
    sections = split_into_sections(cleaned)
    # Consecutive structural markers (BAB / title / Pasal) merge into the same
    # section until body content appears; a new section only starts once a
    # marker follows actual body text.
    assert sections == [
        "BAB I\nKETENTUAN UMUM\nPasal 1\nIsi pasal satu.",
        "BAB II\nMAKSUD DAN TUJUAN\nPasal 2\nIsi pasal dua.",
    ]


def test_split_into_sections_merges_consecutive_title_lines():
    raw = "PERATURAN DAERAH KOTA PEKANBARU\nBAB I\nKETENTUAN UMUM\nPasal 1\nIsi pasal satu."
    cleaned = clean_text(raw)
    sections = split_into_sections(cleaned)
    assert sections == [
        "PERATURAN DAERAH KOTA PEKANBARU\nBAB I\nKETENTUAN UMUM\nPasal 1\nIsi pasal satu."
    ]


def test_split_into_sections_starts_new_section_only_after_body_content():
    raw = "BAB I\nKETENTUAN UMUM\nIsi bab satu.\nBAB II\nKETENTUAN LAIN\nIsi bab dua."
    cleaned = clean_text(raw)
    sections = split_into_sections(cleaned)
    assert sections == [
        "BAB I\nKETENTUAN UMUM\nIsi bab satu.",
        "BAB II\nKETENTUAN LAIN\nIsi bab dua.",
    ]


def test_fallback_split_respects_safe_limit_and_context():
    sentence = "Ketentuan ini mengatur hal penting. "
    section_text = sentence * 20  # well over a small safe_limit
    safe_limit = 100

    sub_chunks = fallback_split(section_text, safe_limit=safe_limit, context_chars=10)

    assert len(sub_chunks) > 1
    # First sub-chunk carries no context.
    assert sub_chunks[0][1] is None
    # Every subsequent sub-chunk's context is the tail of the previous sub-chunk's own text.
    for i in range(1, len(sub_chunks)):
        prev_text = sub_chunks[i - 1][0]
        assert sub_chunks[i][1] == prev_text[-10:]
    # No sub-chunk exceeds the safe limit.
    assert all(len(text) <= safe_limit for text, _ in sub_chunks)


def test_fallback_split_hard_splits_oversized_single_sentence():
    # A single "sentence" with no punctuation at all, longer than safe_limit.
    section_text = "x" * 250
    sub_chunks = fallback_split(section_text, safe_limit=100, context_chars=10)
    assert all(len(text) <= 100 for text, _ in sub_chunks)
    assert "".join(text for text, _ in sub_chunks) == section_text


def test_chunk_policy_text_marks_small_section_structural():
    # BAB I / KETENTUAN UMUM / Pasal 1 merge into a single section since none
    # of them are followed by body content until "Isi pasal satu.".
    raw = "BAB I\nKETENTUAN UMUM\nPasal 1\nIsi pasal satu."
    chunks = chunk_policy_text(raw, safe_limit=32000)
    assert len(chunks) == 1
    assert chunks[0].type == "structural"
    assert chunks[0].context is None
    assert chunks[0].section_id == 0
    assert chunks[0].text == "BAB I\nKETENTUAN UMUM\nPasal 1\nIsi pasal satu."


def test_chunk_policy_text_triggers_fallback_for_oversized_section():
    body = "Ketentuan lampiran ini mengatur hal penting. " * 40
    raw = f"BAB I\nKETENTUAN UMUM\n{body}"
    chunks = chunk_policy_text(raw, safe_limit=500)

    # "BAB I" and "KETENTUAN UMUM" merge into the same (single) section since
    # no body appears between them; that whole section, title stack included,
    # exceeds safe_limit and becomes fallback sub-chunks of section 0.
    assert len(chunks) > 1
    assert all(c.type == "fallback" for c in chunks)
    assert all(c.section_id == 0 for c in chunks)
    assert chunks[0].context is None
    assert chunks[1].context is not None
    assert chunks[0].text.startswith("BAB I\nKETENTUAN UMUM")


def test_chunk_policy_text_multiple_sections_get_distinct_ids():
    raw = (
        "BAB I\nKETENTUAN UMUM\nPasal 1\nIsi pasal satu.\n"
        "BAB II\nMAKSUD DAN TUJUAN\nPasal 2\nIsi pasal dua."
    )
    chunks = chunk_policy_text(raw, safe_limit=32000)
    # Two chapters -> two sections, content stays in the right section.
    section_ids = [c.section_id for c in chunks]
    assert section_ids == sorted(section_ids)
    assert len(set(section_ids)) == 2
    satu_section = next(c.section_id for c in chunks if "Isi pasal satu." in c.text)
    dua_section = next(c.section_id for c in chunks if "Isi pasal dua." in c.text)
    assert satu_section < dua_section


def test_combine_translations_joins_sections_with_blank_line_and_subchunks_directly():
    chunks = [
        Chunk(text="a", type="fallback", context=None, section_id=0, chunk_index=0),
        Chunk(text="b", type="fallback", context="a", section_id=0, chunk_index=1),
        Chunk(text="c", type="structural", context=None, section_id=1),
    ]
    translations = ["A-translated", "B-translated", "C-translated"]

    combined = combine_translations(chunks, translations)

    assert combined == "A-translatedB-translated\n\nC-translated"


def test_combine_translations_length_mismatch_raises():
    chunks = [Chunk(text="a", type="structural", context=None, section_id=0)]
    with pytest.raises(ValueError):
        combine_translations(chunks, [])


def test_check_fallback_output_flags_disproportionately_long_output():
    assert check_fallback_output("short target", "x" * 1000) is True
    assert check_fallback_output("a reasonably sized target text here", "a similar length output") is False
    assert check_fallback_output("", "anything") is False


def test_chunk_policy_text_with_debug_matches_chunk_policy_text():
    raw = (
        "BAB I\nKETENTUAN UMUM\nPasal 1\nIsi pasal satu.\n"
        "BAB II\nMAKSUD DAN TUJUAN\nPasal 2\nIsi pasal dua."
    )
    chunks = chunk_policy_text(raw, safe_limit=32000)
    cleaned_lines, debug_chunks = chunk_policy_text_with_debug(raw, safe_limit=32000)

    assert debug_chunks == chunks
    assert [line.text for line in cleaned_lines] == [
        "BAB I",
        "KETENTUAN UMUM",
        "Pasal 1",
        "Isi pasal satu.",
        "BAB II",
        "MAKSUD DAN TUJUAN",
        "Pasal 2",
        "Isi pasal dua.",
    ]
