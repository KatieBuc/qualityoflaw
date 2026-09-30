from automation.src.markdown.chunking import (
    TARGET_CHARS_DEFAULT,
    _clause_identity,
    align_sections,
    chunk_markdown,
    check_structure,
    count_list_items,
    heading_signature,
    pack_sections,
    parse_sections,
    section_clause_key,
)

POLICY_MD = """# BAB I

## KETENTUAN UMUM

#### Pasal 1

Dalam Qanun ini, yang dimaksud dengan:

1. Daerah adalah Kabupaten Bireuen.
2. Kabupaten adalah Kabupaten Bireuen.

#### Pasal 2

Isi pasal dua.

# BAB II

## MAKSUD DAN TUJUAN

### Paragraf 1

#### Pasal 3

Isi pasal tiga.
"""


def test_parse_sections_keeps_title_stack_with_its_content():
    sections = parse_sections(POLICY_MD)

    # "# BAB I" / "## KETENTUAN UMUM" / "#### Pasal 1" are consecutive
    # headings with no body between them, so they belong to one section --
    # the same rule chunking.split_into_sections applies to raw text.
    assert sections[0].text.startswith("# BAB I")
    assert "## KETENTUAN UMUM" in sections[0].text
    assert "#### Pasal 1" in sections[0].text
    assert "Daerah adalah Kabupaten Bireuen." in sections[0].text
    assert "#### Pasal 2" not in sections[0].text


def test_parse_sections_splits_once_body_content_has_appeared():
    headings = [s.heading for s in parse_sections(POLICY_MD)]
    assert headings == ["BAB I", "Pasal 2", "BAB II"]


def test_parse_sections_records_ancestor_breadcrumb():
    sections = parse_sections(POLICY_MD)
    pasal_2 = next(s for s in sections if s.heading == "Pasal 2")

    # Ancestors only -- the section's own heading is not repeated in the path.
    assert pasal_2.heading_path == ["BAB I", "KETENTUAN UMUM"]
    assert pasal_2.level == 4


def test_parse_sections_deeper_heading_drops_stale_ancestors():
    md = (
        "# BAB I\n\n## Bagian Kesatu\n\nBody one.\n\n"
        "# BAB II\n\nBody two.\n\n#### Pasal 9\n\nBody three.\n"
    )
    pasal = next(s for s in parse_sections(md) if s.heading == "Pasal 9")

    # "Bagian Kesatu" belonged to BAB I; a new level-1 heading clears it.
    assert pasal.heading_path == ["BAB II"]


def test_parse_sections_keeps_preamble_before_first_heading():
    sections = parse_sections("PROVINSI ACEH\n\nsome preamble\n\n# BAB I\n\nBody.\n")
    assert sections[0].heading is None
    assert sections[0].level == 0
    assert "PROVINSI ACEH" in sections[0].text


def test_parse_sections_on_empty_input():
    assert parse_sections("") == []


def test_heading_signature_and_list_item_count():
    assert heading_signature(POLICY_MD) == [1, 2, 4, 4, 1, 2, 3, 4]
    assert count_list_items(POLICY_MD) == 2


def test_pack_sections_merges_adjacent_sections_up_to_target():
    sections = parse_sections(POLICY_MD)
    packed = pack_sections(sections, target_chars=TARGET_CHARS_DEFAULT)

    # The whole document is far under 8,000 chars, so it packs into one unit.
    assert len(packed) == 1
    assert packed[0].type == "structural"
    assert packed[0].section_id == 0


def test_pack_sections_flushes_at_the_target_boundary():
    sections = parse_sections(POLICY_MD)
    sizes = [len(s.text) for s in sections]

    # A target that fits the first two sections but not the third.
    target = sizes[0] + sizes[1] + 2
    packed = pack_sections(sections, target_chars=target)

    assert len(packed) == 2
    assert packed[0].text == "\n\n".join(s.text for s in sections[:2])
    assert packed[1].text == sections[2].text
    assert [c.section_id for c in packed] == [0, 1]


def test_pack_sections_gives_each_unit_the_first_sections_breadcrumb():
    sections = parse_sections(POLICY_MD)
    packed = pack_sections(sections, target_chars=len(sections[0].text) + 1)

    assert packed[0].heading_path == sections[0].heading_path
    assert packed[1].heading_path == sections[1].heading_path


def test_pack_sections_passes_an_oversized_section_through_whole():
    # Between target_chars and safe_limit: splitting would gain nothing.
    big = "#### Pasal 1\n\n" + "Kalimat panjang. " * 200
    packed = pack_sections(parse_sections(big), target_chars=500, safe_limit=100000)

    assert len(packed) == 1
    assert packed[0].type == "structural"


def test_pack_sections_sentence_splits_beyond_safe_limit():
    big = "#### Pasal 1\n\n" + "Kalimat yang cukup panjang untuk diuji. " * 100
    packed = pack_sections(parse_sections(big), target_chars=500, safe_limit=800)

    assert len(packed) > 1
    assert all(c.type == "fallback" for c in packed)
    # One section, so one section_id -- combine_translations rejoins them.
    assert {c.section_id for c in packed} == {0}
    assert [c.chunk_index for c in packed] == list(range(len(packed)))
    assert all(len(c.text) <= 800 for c in packed)
    # Every sub-chunk after the first carries the previous one's tail.
    assert packed[0].context is None
    assert all(c.context for c in packed[1:])


def test_chunk_markdown_loses_no_content():
    chunks = chunk_markdown(POLICY_MD, target_chars=200)
    sections = parse_sections(POLICY_MD)
    assert "\n\n".join(c.text for c in chunks) == "\n\n".join(s.text for s in sections)


def test_check_structure_passes_a_faithful_translation():
    translated = (
        POLICY_MD.replace("BAB", "CHAPTER")
        .replace("Pasal", "Article")
        .replace("Paragraf", "Paragraph")
        .replace("Isi pasal dua.", "Content of article two.")
    )
    assert check_structure(POLICY_MD, translated) is None


def test_check_structure_flags_a_dropped_heading():
    translated = POLICY_MD.replace("#### Pasal 2\n\n", "")
    diff = check_structure(POLICY_MD, translated)

    assert diff is not None
    assert not diff.headings_match
    assert "headings 8 -> 7" in diff.describe()


def test_check_structure_reports_which_level_lost_a_heading():
    # "a Pasal (level 4) was lost" and "an OCR fragment (level 5) was
    # dropped" are very different findings; the message has to tell them
    # apart. Observed on the real corpus as "level 4: 121 -> 120" (serious)
    # versus "level 5: 11 -> 10" (the model dropping `##### P  P`).
    translated = POLICY_MD.replace("#### Pasal 2\n\n", "")
    diff = check_structure(POLICY_MD, translated)

    assert diff.level_changes() == {4: (3, 2)}
    assert "level 4: 3 -> 2" in diff.describe()


def test_check_structure_reports_an_invented_heading_at_its_level():
    translated = POLICY_MD.replace("Isi pasal dua.", "#### Pasal 2a\n\nIsi pasal dua.")
    diff = check_structure(POLICY_MD, translated)

    assert diff.level_changes() == {4: (3, 4)}
    assert "level 4: 3 -> 4" in diff.describe()


def test_check_structure_reports_every_changed_level():
    translated = POLICY_MD.replace("#### Pasal 2\n\n", "").replace("## KETENTUAN UMUM\n\n", "")
    assert check_structure(POLICY_MD, translated).level_changes() == {2: (2, 1), 4: (3, 2)}


def test_a_re_levelled_heading_shows_up_on_both_levels():
    translated = POLICY_MD.replace("#### Pasal 2", "### Pasal 2")
    assert check_structure(POLICY_MD, translated).level_changes() == {3: (1, 2), 4: (3, 2)}


def test_check_structure_flags_a_re_levelled_heading():
    translated = POLICY_MD.replace("#### Pasal 2", "### Pasal 2")
    diff = check_structure(POLICY_MD, translated)

    assert diff is not None
    assert diff.translated_headings != diff.source_headings
    assert len(diff.translated_headings) == len(diff.source_headings)


def test_a_changed_list_item_count_is_not_drift():
    # translation_qa_md is chartered to repair list markers that OCR
    # detached from their content, which necessarily changes the count --
    # observed on the real corpus as 68 source items becoming 277 correctly
    # marked ones. Counting that as drift would flag the repair as a defect.
    repaired = POLICY_MD.replace(
        "2. Kabupaten adalah Kabupaten Bireuen.",
        "2. Kabupaten adalah Kabupaten Bireuen.\n3. Bupati adalah Bupati Bireuen.",
    )
    assert check_structure(POLICY_MD, repaired) is None


def test_list_item_counts_are_still_reported_as_context():
    translated = POLICY_MD.replace("#### Pasal 2\n\n", "").replace(
        "2. Kabupaten adalah Kabupaten Bireuen.\n", ""
    )
    assert "list items 2 -> 1" in check_structure(POLICY_MD, translated).describe()


def test_clause_identity_is_language_independent():
    # level is now the heading's real `#` depth, not a keyword-mapped level --
    # same depth + same token is the same clause, regardless of language.
    assert _clause_identity(4, "Pasal 34") == (4, "34")
    assert _clause_identity(4, "Article 34") == (4, "34")
    assert _clause_identity(1, "BAB IV") == (1, "IV")
    assert _clause_identity(1, "CHAPTER IV") == (1, "IV")


def test_clause_identity_resolves_headings_in_other_languages():
    # No keyword dictionary involved -- any keyword works as long as the
    # second word is a number/Roman token. Trailing title text is ignored.
    assert _clause_identity(2, "§ 1") == (2, "1")
    assert _clause_identity(2, "§ 1 Gerichtliche Maßnahmen zum Schutz vor Gewalt") == (2, "1")
    assert _clause_identity(2, "ARTICULO 1º") == (2, "1")  # Spanish ordinal indicator
    assert _clause_identity(2, "Art. 13") == (2, "13")


def test_clause_identity_rejects_non_clause_headings():
    assert _clause_identity(2, "Bagian Kesatu") is None  # spelled-out ordinal
    assert _clause_identity(1, "KETENTUAN UMUM") is None
    assert _clause_identity(5, "1  AN") is None  # OCR fragment


def test_clause_identity_accepts_cross_reference_wrongly_promoted_to_a_heading():
    # Markdown headings are assumed already clean -- unlike the raw-text
    # path, this function no longer defends against a citation the corpus
    # occasionally OCR-promotes to a heading; it now resolves like any other
    # keyword+number heading. Accepted trade-off for a language-agnostic
    # identity check (see markdown/chunking.py module notes).
    assert _clause_identity(4, "Pasal 28 ayat (1) huruf f, meliputi:") == (4, "28")


def test_section_clause_key_returns_the_deepest_clause_in_the_section():
    # parse_sections keeps "# BAB III" and "#### Pasal 4" in one section.
    section = parse_sections("# BAB III\n\n## RUANG LINGKUP\n\n#### Pasal 4\n\nIsi.\n")[0]
    assert section_clause_key(section) == (4, "4")


def test_section_clause_key_is_none_without_a_numbered_clause():
    section = parse_sections("# KETENTUAN UMUM\n\nIsi tanpa pasal.\n")[0]
    assert section_clause_key(section) is None


SRC_SECTIONS = parse_sections(
    "# BAB I\n\n#### Pasal 1\n\nSatu.\n\n#### Pasal 2\n\nDua.\n\n#### Pasal 3\n\nTiga.\n"
)


def _translated(md):
    return parse_sections(md)


def test_align_sections_identity_when_structure_holds():
    tr = _translated(
        "# CHAPTER I\n\n#### Article 1\n\nOne.\n\n#### Article 2\n\nTwo.\n\n#### Article 3\n\nThree.\n"
    )
    assert align_sections(SRC_SECTIONS, tr) == [0, 1, 2]


def test_align_sections_pairs_across_a_relabelled_heading():
    tr = _translated(
        "# CHAPTER I\n\n#### Article 1\n\nOne.\n\n### Article 2\n\nTwo.\n\n#### Article 3\n\nThree.\n"
    )
    assert align_sections(SRC_SECTIONS, tr) == [0, 1, 2]


def test_align_sections_leaves_a_translation_only_section_unpaired():
    tr = _translated(
        "# CHAPTER I\n\n#### Article 1\n\nOne.\n\n#### Article 2\n\nTwo.\n\n"
        "#### Article 3\n\nThree.\n\n#### Article 9\n\nExtra.\n"
    )
    assert align_sections(SRC_SECTIONS, tr) == [0, 1, 2, None]


def test_align_sections_reanchors_after_a_source_only_section():
    # Source has an Article 2 the translation dropped; Article 3 must still
    # pair to Pasal 3, not slip onto Pasal 2.
    tr = _translated(
        "# CHAPTER I\n\n#### Article 1\n\nOne.\n\n#### Article 3\n\nThree.\n"
    )
    assert align_sections(SRC_SECTIONS, tr) == [0, 2]


def test_align_sections_all_none_when_nothing_matches():
    tr = _translated("# PREAMBLE\n\nUnrelated body with no clauses at all.\n")
    assert align_sections(SRC_SECTIONS, tr) == [None]


# An OCR-split heading in the source (`Pasal 3 1`) repaired by the translator
# (`Article 31`): the clause number 31 is then unique in the source (only in
# the elucidation) but repeated in the translation, which used to make the
# anchor matcher shift every later section by one.
_OCR_SPLIT_SRC = (
    "#### Pasal 1\n\nSatu.\n\n"
    "#### Pasal 3 1\n\nTiga satu.\n\n"
    "#### Pasal 32\n\nTiga dua.\n\n"
    "# PENJELASAN\n\n#### Pasal 31\n\nCukup jelas.\n\n"
    "#### Pasal 32\n\nCukup jelas.\n\n"
    "#### Pasal 33\n\nCukup jelas.\n"
)
_OCR_SPLIT_TR = (
    "#### Article 1\n\nOne.\n\n"
    "#### Article 31\n\nThirty-one.\n\n"
    "#### Article 32\n\nThirty-two.\n\n"
    "# ELUCIDATION\n\n#### Article 31\n\nSufficiently clear.\n\n"
    "#### Article 32\n\nSufficiently clear.\n\n"
    "#### Article 33\n\nSufficiently clear.\n"
)


def test_align_sections_equal_counts_pair_by_position_despite_ocr_split_heading():
    source = parse_sections(_OCR_SPLIT_SRC)
    translated = parse_sections(_OCR_SPLIT_TR)
    assert len(source) == len(translated)
    assert align_sections(source, translated) == list(range(len(translated)))


def test_align_sections_equal_counts_ignore_clause_numbers():
    # Same count, but the translation renumbered a clause: position still wins.
    source = parse_sections("#### Pasal 1\n\nSatu.\n\n#### Pasal 2\n\nDua.\n")
    translated = parse_sections("#### Article 7\n\nOne.\n\n#### Article 8\n\nTwo.\n")
    assert align_sections(source, translated) == [0, 1]


def test_align_sections_unequal_counts_still_anchor_on_clause_numbers():
    # Translation dropped Pasal 2: Article 3 must land on Pasal 3, not Pasal 2.
    source = parse_sections(
        "#### Pasal 1\n\nSatu.\n\n#### Pasal 2\n\nDua.\n\n#### Pasal 3\n\nTiga.\n"
    )
    translated = parse_sections("#### Article 1\n\nOne.\n\n#### Article 3\n\nThree.\n")
    assert align_sections(source, translated) == [0, 2]
