from automation.src.markdown.chunking import (
    TARGET_CHARS_DEFAULT,
    chunk_markdown,
    check_structure,
    count_list_items,
    heading_signature,
    pack_sections,
    parse_sections,
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
