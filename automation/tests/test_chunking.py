import pytest

from automation.src.chunking import (
    STRUCTURE_MARKER_EN_RE,
    Chunk,
    check_fallback_output,
    check_structural_output,
    chunk_policy_text,
    chunk_policy_text_with_debug,
    clean_text,
    combine_translations,
    fallback_split,
    is_noise_line,
    is_structure_marker,
    render_markdown,
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


def test_clean_text_splits_heading_concatenated_onto_prior_line():
    # BENGKULU_KEPAHIANG.txt line 182: an all-caps chapter title merged onto
    # the same raw line as the next heading, no punctuation between them.
    raw = "PERLINDUNGAN PEREMPUAN DAN ANAK KORBAN TINDAK KEKERASAN Bagian Kesatu\nPencegahan"
    cleaned = clean_text(raw)
    assert [c.text for c in cleaned] == [
        "PERLINDUNGAN PEREMPUAN DAN ANAK KORBAN TINDAK KEKERASAN",
        "Bagian Kesatu",
        "Pencegahan",
    ]
    assert [c.is_structure for c in cleaned] == [True, True, False]


def test_clean_text_splits_heading_at_end_of_sentence():
    # BENGKULU_KEPAHIANG.txt line 447: body text ending in "." immediately
    # followed by the next Pasal heading on the same raw line.
    raw = (
        "Yang dimaksud dengan pemberdayaan adalah penguatan korban kekerasan "
        "untuk dapat berusaha dan bekerja sendiri setelah mereka dipulihkan "
        "dan diberikan layanan pemulihan kesehatan dan sosial. Pasal 5"
    )
    cleaned = clean_text(raw)
    assert cleaned[-1].text == "Pasal 5"
    assert cleaned[-1].is_structure is True
    assert "Pasal 5" not in cleaned[0].text


def test_clean_text_does_not_split_in_text_cross_reference():
    # A "Pasal N" mention followed by more prose on the same line is a
    # cross-reference, not a heading, and must stay joined.
    raw = "Kekerasan psikis sebagaimana dimaksud dalam Pasal 7 huruf (b) disebabkan karena perbuatan."
    cleaned = clean_text(raw)
    assert len(cleaned) == 1
    assert cleaned[0].is_structure is False
    assert cleaned[0].text == raw


def test_clean_text_does_not_split_wrapped_cross_reference_ending_line():
    # Even when a cross-reference happens to end a wrapped OCR line, the
    # trigger-word guard should keep it from being misread as a heading.
    raw = "Ketentuan ini berlaku sebagaimana dimaksud dalam Pasal 7\nhuruf (b) di atas."
    cleaned = clean_text(raw)
    assert len(cleaned) == 1
    assert cleaned[0].is_structure is False
    assert cleaned[0].text == "Ketentuan ini berlaku sebagaimana dimaksud dalam Pasal 7 huruf (b) di atas."


def test_clean_text_merges_bare_list_marker_with_next_line():
    # JAMBI_BUNGO.txt lines 141-142: "a." alone on one line, its content on
    # the next -- must not orphan "a." as its own paragraph.
    raw = "asas:\na.\nkemanusiaan;\nb. keadilan dan kesetaraan gender;"
    cleaned = clean_text(raw)
    assert [c.text for c in cleaned] == [
        "asas:",
        "a. kemanusiaan;",
        "b. keadilan dan kesetaraan gender;",
    ]


def test_clean_text_holds_consecutive_bare_markers_until_content():
    # DKI_JAKARTA.txt lines 150-152: two bare markers in a row before any
    # content arrives -- both must stay attached to the eventual content
    # instead of being scattered as separate orphan paragraphs.
    raw = "i.\nj.\nhak atas pendampingan."
    cleaned = clean_text(raw)
    assert [c.text for c in cleaned] == ["i. j. hak atas pendampingan."]


def test_clean_text_uppercase_bare_marker_is_not_a_heading():
    # JAWA_BARAT_BEKASI.txt line 274: "C." (OCR noise for "c.") must be
    # treated as a bare list marker, not misclassified as a heading.
    raw = "a.\nb.\nC.\nd.\nmenstruasi;"
    cleaned = clean_text(raw)
    assert len(cleaned) == 1
    assert cleaned[0].is_structure is False
    assert cleaned[0].text == "a. b. C. d. menstruasi;"


def test_is_structure_marker_rejects_single_letter():
    assert not is_structure_marker("C.")
    assert not is_structure_marker("a.")


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


def test_is_structure_marker_english_regex_forms():
    # English equivalents observed in translated output: BAB->Chapter,
    # Pasal->Article, Bagian->Part, Paragraf->Paragraph.
    assert is_structure_marker("CHAPTER I", STRUCTURE_MARKER_EN_RE)
    assert is_structure_marker("Article 5", STRUCTURE_MARKER_EN_RE)
    assert is_structure_marker("Part Two", STRUCTURE_MARKER_EN_RE)
    assert is_structure_marker("Paragraph 1", STRUCTURE_MARKER_EN_RE)
    assert is_structure_marker("Section 3", STRUCTURE_MARKER_EN_RE)
    assert not is_structure_marker(
        "The Regional Government is the Government of Pekanbaru City.", STRUCTURE_MARKER_EN_RE
    )
    # Indonesian markers are not matched by the English regex.
    assert not is_structure_marker("Pasal 5", STRUCTURE_MARKER_EN_RE)


def test_chunk_policy_text_with_english_marker_re():
    raw = (
        "CHAPTER I\nGENERAL PROVISIONS\nArticle 1\nThe content of article one is complete.\n"
        "CHAPTER II\nPURPOSE AND OBJECTIVE\nArticle 2\nThe content of article two is complete."
    )
    chunks = chunk_policy_text(raw, safe_limit=32000, marker_re=STRUCTURE_MARKER_EN_RE)
    assert len(chunks) == 2
    assert chunks[0].section_id == 0
    assert chunks[1].section_id == 1
    assert chunks[0].text.startswith("CHAPTER I\nGENERAL PROVISIONS")
    assert chunks[1].text.startswith("CHAPTER II\nPURPOSE AND OBJECTIVE")


def test_chunk_policy_text_default_marker_re_ignores_english_markers():
    # Without passing marker_re, English structural words aren't recognized as
    # markers (only the all-caps-title fallback heuristic can still catch them),
    # so "Article 1" reflows into the same paragraph as the line after it
    # instead of starting a new structural section.
    raw = "Article 1\nThe content of article one is complete."
    chunks = chunk_policy_text(raw, safe_limit=32000)
    assert len(chunks) == 1
    assert chunks[0].text == "Article 1 The content of article one is complete."


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


def test_check_structural_output_flags_runaway_output():
    # Reproduces the observed hallucination: a 22-char signature-block source
    # producing a 128K-char unrelated document.
    assert check_structural_output("BUPATI BLORA,\nCap Ttd.", "x" * 128_418) is True


def test_check_structural_output_tolerates_normal_legal_expansion():
    source = "Pasal 5\n" + ("Ketentuan ini berlaku efektif sejak diundangkan. " * 20)
    # ~1.3x expansion, typical for Indonesian->English legal translation.
    translated = "Article 5\n" + ("This provision takes effect from the date of promulgation. " * 20)
    assert check_structural_output(source, translated) is False


def test_check_structural_output_ignores_short_titles_despite_high_ratio():
    # "BAB I" -> "CHAPTER I" is a ~1.8x ratio but tiny in absolute size —
    # must not be flagged (below STRUCTURAL_MIN_SUSPICIOUS_CHARS).
    assert check_structural_output("BAB I", "CHAPTER I") is False


def test_check_structural_output_empty_source_never_flagged():
    assert check_structural_output("", "x" * 10_000) is False


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


def test_render_markdown_heading_levels_indonesian():
    raw = "BAB I\nKETENTUAN UMUM\nPasal 1\nIsi pasal satu."
    markdown = render_markdown(clean_text(raw))
    assert "# BAB I" in markdown
    assert "## KETENTUAN UMUM" in markdown
    assert "#### Pasal 1" in markdown
    assert "Isi pasal satu." in markdown


def test_render_markdown_heading_levels_english():
    raw = "CHAPTER I\nGENERAL PROVISIONS\nArticle 1\nThe content is complete."
    markdown = render_markdown(clean_text(raw, marker_re=STRUCTURE_MARKER_EN_RE))
    assert "# CHAPTER I" in markdown
    assert "## GENERAL PROVISIONS" in markdown
    assert "#### Article 1" in markdown


def test_render_markdown_bagian_and_paragraf_levels():
    raw = "BAB I\nBagian Kesatu\nParagraf 1\nPasal 1\nIsi."
    markdown = render_markdown(clean_text(raw))
    assert "# BAB I" in markdown
    assert "## Bagian Kesatu" in markdown
    assert "### Paragraf 1" in markdown
    assert "#### Pasal 1" in markdown


def test_render_markdown_fallback_headings_stay_siblings_under_last_keyword():
    # Two consecutive all-caps title lines with no keyword between them stay
    # at the same nested level (both children of the last keyword heading)
    # instead of nesting deeper into each other.
    raw = "BAB I\nFIRST TITLE\nSECOND TITLE\nPasal 1\nIsi."
    markdown = render_markdown(clean_text(raw))
    assert "## FIRST TITLE" in markdown
    assert "## SECOND TITLE" in markdown


def test_render_markdown_numbered_list_is_ordered():
    raw = "Pasal 1\nDalam Qanun ini:\n1. Daerah adalah Kabupaten.\n2. Kabupaten adalah Kabupaten."
    markdown = render_markdown(clean_text(raw))
    assert "1. Daerah adalah Kabupaten." in markdown
    assert "2. Kabupaten adalah Kabupaten." in markdown


def test_render_markdown_parenthesized_number_strips_parens():
    raw = "Pasal 1\n(1) Setiap perempuan berhak untuk:"
    markdown = render_markdown(clean_text(raw))
    assert "1. Setiap perempuan berhak untuk:" in markdown
    assert "(1)" not in markdown


def test_render_markdown_lettered_list_is_nested_bullet():
    raw = "Pasal 2\na. penghargaan terhadap nilai;\nb. kepastian hukum;"
    markdown = render_markdown(clean_text(raw))
    assert "   - a. penghargaan terhadap nilai;" in markdown
    assert "   - b. kepastian hukum;" in markdown


def test_render_markdown_roman_list_is_double_nested_bullet():
    raw = "Pasal 3\na. berhak atas:\ni. pendampingan;\nii. perlindungan."
    markdown = render_markdown(clean_text(raw))
    assert "   - a. berhak atas:" in markdown
    assert "      - i. pendampingan;" in markdown
    assert "      - ii. perlindungan." in markdown


def test_render_markdown_single_roman_char_needs_preceding_letter_item():
    # A lone "i." with no active lettered list before it is just a
    # letter-tier item, not a nested roman numeral.
    raw = "Pasal 4\ni. item i;\nii. item ii;"
    markdown = render_markdown(clean_text(raw))
    assert "   - i. item i;" in markdown
    assert "      - i. item i;" not in markdown


def test_render_markdown_single_roman_char_after_letter_nests_as_roman():
    # "i." directly after a lettered item is the nested roman numeral under
    # that clause -- the real a./i./ii. pattern seen in the source documents.
    raw = "Pasal 4\na. item a:\ni. item i;\nii. item ii;"
    markdown = render_markdown(clean_text(raw))
    assert "   - a. item a:" in markdown
    assert "      - i. item i;" in markdown
    assert "      - ii. item ii;" in markdown


def test_render_markdown_lettered_list_reaching_roman_letters_stays_letter_tier():
    # "c."/"d." are valid roman-numeral characters, but as the 3rd/4th item
    # of a plain lettered list (a./b./c./d.) they must stay letter-tier
    # siblings, not get misread as a nested roman list starting mid-alphabet
    # (a real a./b./c. "Menimbang" list observed in the source corpus).
    raw = "Pasal 1\na. item a;\nb. item b;\nc. item c;\nd. item d;"
    markdown = render_markdown(clean_text(raw))
    assert "   - a. item a;" in markdown
    assert "   - b. item b;" in markdown
    assert "   - c. item c;" in markdown
    assert "   - d. item d;" in markdown
    assert "      - c. item c;" not in markdown
    assert "      - d. item d;" not in markdown


def test_render_markdown_list_items_stay_together_no_blank_lines_between():
    raw = "Pasal 1\n1. Pertama.\n2. Kedua.\n3. Ketiga."
    markdown = render_markdown(clean_text(raw))
    assert "1. Pertama.\n2. Kedua.\n3. Ketiga." in markdown


def test_render_markdown_heading_ends_list():
    raw = "Pasal 1\na. item a;\nBAB II\nJUDUL\nPasal 2\nIsi."
    markdown = render_markdown(clean_text(raw))
    idx_list = markdown.index("- a. item a;")
    idx_heading = markdown.index("# BAB II")
    assert idx_heading > idx_list
