from automation.src.markdown.to_text import build_retrieval_chunks, markdown_to_text


def test_headings_keep_their_text_and_lose_their_markers():
    md = "# CHAPTER I\n\n## GENERAL PROVISIONS\n\n#### Article 1\n\nBody text.\n"
    assert markdown_to_text(md) == (
        "CHAPTER I\n\nGENERAL PROVISIONS\n\nArticle 1\n\nBody text.\n"
    )


def test_trailing_hard_break_spaces_are_dropped():
    # The curated corpus terminates most lines with two spaces.
    md = "a. first item;  \nb. second item;  \n"
    assert markdown_to_text(md) == "a. first item;\nb. second item;\n"


def test_nested_bullets_fall_back_to_their_clause_markers():
    md = "1. numbered item\n   - b. lettered item\n      - i. roman item\n"
    assert markdown_to_text(md) == "1. numbered item\nb. lettered item\ni. roman item\n"


def test_heading_text_that_starts_with_a_bullet_loses_it_too():
    # Real corpus line: "## - BUPATI MINAHASA TENGGARA,".
    assert markdown_to_text("## - BUPATI MINAHASA TENGGARA,\n") == (
        "BUPATI MINAHASA TENGGARA,\n"
    )


def test_a_malformed_heading_marker_is_left_as_body_text():
    # "#: PERATURAN" is not a heading (no space after the #), and guessing at
    # malformed source belongs to corpus curation, not to this converter.
    assert markdown_to_text("#: PERATURAN\n") == "#: PERATURAN\n"


def test_numbered_items_keep_their_markers():
    assert markdown_to_text("1. one\n2. two\n") == "1. one\n2. two\n"


def test_emphasis_code_and_links_are_stripped():
    md = "The **Regional** Government shall *coordinate* `services` [here](http://x.id).\n"
    assert markdown_to_text(md) == (
        "The Regional Government shall coordinate services here.\n"
    )


def test_horizontal_rules_are_dropped():
    assert markdown_to_text("Para one.\n\n---\n\nPara two.\n") == "Para one.\n\nPara two.\n"


def test_blank_line_runs_collapse_to_one():
    assert markdown_to_text("One.\n\n\n\nTwo.\n") == "One.\n\nTwo.\n"


def test_empty_input_produces_empty_output():
    assert markdown_to_text("") == ""
    assert markdown_to_text("\n\n   \n") == ""


def test_output_carries_no_markdown_syntax():
    md = "#### Article 5\n\nText.  \n\n   - a. item;  \n"
    text = markdown_to_text(md)
    assert "#" not in text
    assert "- a." not in text
    assert not any(line.endswith(" ") for line in text.splitlines())


SOURCE_MD = """# BAB I

#### Pasal 1

Isi pasal satu.

#### Pasal 2

Isi pasal dua.
"""

TRANSLATED_MD = """# CHAPTER I

#### Article 1

Content of article one.

#### Article 2

Content of article two.
"""


def test_build_retrieval_chunks_pairs_source_to_translation_when_aligned():
    records, aligned = build_retrieval_chunks(SOURCE_MD, TRANSLATED_MD)

    assert aligned is True
    assert len(records) == 2
    assert records[0]["translated_text"].startswith("CHAPTER I")
    assert records[0]["text"].startswith("BAB I")
    assert "Article 2" in records[1]["translated_text"]
    assert "Pasal 2" in records[1]["text"]


def test_build_retrieval_chunks_keeps_one_record_per_heading_section():
    # Retrieval granularity must stay per-Pasal, not per packed translation
    # unit -- rag.retriever cites sentences within a single chunk.
    records, _ = build_retrieval_chunks(SOURCE_MD, TRANSLATED_MD)
    assert [r["section_id"] for r in records] == [0, 1]
    assert all(r["chunk_index"] == 0 for r in records)
    assert all(r["type"] == "structural" for r in records)


def test_build_retrieval_chunks_records_are_plain_text():
    records, _ = build_retrieval_chunks(SOURCE_MD, TRANSLATED_MD)
    assert all("#" not in r["translated_text"] for r in records)


def test_build_retrieval_chunks_drops_the_source_side_when_structure_drifts():
    drifted = TRANSLATED_MD.replace("#### Article 2\n\n", "")
    records, aligned = build_retrieval_chunks(SOURCE_MD, drifted)

    assert aligned is False
    # Pairing by position would misalign, so the source side is left empty --
    # storage never reads it, and the drift is reported separately.
    assert all(r["text"] == "" for r in records)
    assert all(r["translated_text"] for r in records)


def test_build_retrieval_chunks_never_emits_an_empty_translated_text():
    # One empty translated_text makes rag.store discard the whole file, so a
    # section that renders to nothing is dropped instead.
    records, _ = build_retrieval_chunks(SOURCE_MD, TRANSLATED_MD + "\n\n---\n")
    assert all(r["translated_text"].strip() for r in records)
