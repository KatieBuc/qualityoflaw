from automation.src.rag.sentence_split import (
    split_sentences,
    split_sentences_with_lines,
)


def test_splits_multiple_sentences():
    text = "The policy covers domestic violence. It also covers sexual violence."
    assert split_sentences(text) == [
        "The policy covers domestic violence.",
        "It also covers sexual violence.",
    ]


def test_handles_abbreviations_and_decimals_without_over_splitting():
    text = "See Art. 3.1 of the regulation. It defines economic violence."
    result = split_sentences(text)
    assert result == [
        "See Art. 3.1 of the regulation.",
        "It defines economic violence.",
    ]


def test_empty_and_whitespace_only_input_returns_no_sentences():
    assert split_sentences("") == []
    assert split_sentences("   \n  ") == []


def test_single_sentence_returns_one_item():
    assert split_sentences("Just one sentence here.") == ["Just one sentence here."]


def test_split_sentences_with_lines_matches_flattened_split_sentences():
    text = "Article 1\nThe Regent shall establish a committee. It shall report annually.\nArticle 2\nThis regulation is effective immediately."
    with_lines = split_sentences_with_lines(text)
    assert [s for _line_idx, s in with_lines] == split_sentences(text)
    assert [line_idx for line_idx, _s in with_lines] == [0, 1, 1, 2, 3]


def test_split_sentences_with_lines_matches_flattened_split_sentences_for_list_markers():
    # A bare list marker ("a.") followed by its item text on the same line is
    # segmented differently by pysbd depending on whether it sees the line in
    # isolation or embedded in a longer passage -- a single whole-text pysbd
    # call over multi-line text does NOT reliably agree with a per-line call
    # here. split_sentences is defined in terms of split_sentences_with_lines
    # specifically so the two can never diverge on cases like this one.
    text = "Menimbang:\na. that women need to be empowered.\nb. that based on Article 231."
    with_lines = split_sentences_with_lines(text)
    assert [s for _line_idx, s in with_lines] == split_sentences(text)


def test_split_sentences_with_lines_empty_input():
    assert split_sentences_with_lines("") == []
    assert split_sentences_with_lines("   \n  ") == []


# The original Indonesian text is run through this same `language="en"`
# segmenter by sentence_align.py -- it is Latin script with the same terminal
# punctuation, and a miscount only fails the strict 1:1 gate there.
def test_splits_indonesian_text_on_sentence_punctuation():
    text = "Kebijakan ini mencakup kekerasan dalam rumah tangga. Kebijakan ini juga mencakup kekerasan seksual."
    assert split_sentences(text) == [
        "Kebijakan ini mencakup kekerasan dalam rumah tangga.",
        "Kebijakan ini juga mencakup kekerasan seksual.",
    ]


def test_protects_decimal_numbers_in_indonesian_text():
    text = "Lihat Pasal 3.2 tentang kekerasan ekonomi. Ketentuan ini berlaku."
    assert split_sentences(text) == [
        "Lihat Pasal 3.2 tentang kekerasan ekonomi.",
        "Ketentuan ini berlaku.",
    ]


def test_single_indonesian_sentence_returns_one_item():
    assert split_sentences("Satu kalimat saja.") == ["Satu kalimat saja."]


def test_protect_abbreviations_keeps_indonesian_abbreviations_intact():
    text = "Ditetapkan oleh Plt. Sekretaris Daerah dll. Ketentuan lain berlaku."
    # Without protection pysbd's English model splits at "Plt." and "dll.".
    assert len(split_sentences(text)) > 1
    assert split_sentences(text, protect_abbreviations=True) == [
        "Ditetapkan oleh Plt. Sekretaris Daerah dll. Ketentuan lain berlaku.",
    ]


def test_protect_abbreviations_still_splits_real_sentence_boundaries():
    text = "Kewajiban diatur dalam Kab. Bireuen. Pelaksanaannya diawasi Bupati."
    assert split_sentences(text, protect_abbreviations=True) == [
        "Kewajiban diatur dalam Kab. Bireuen.",
        "Pelaksanaannya diawasi Bupati.",
    ]


def test_protect_abbreviations_handles_dotted_forms():
    text = "Surat ditandatangani a.n. Bupati. Salinan sesuai aslinya."
    assert split_sentences(text, protect_abbreviations=True) == [
        "Surat ditandatangani a.n. Bupati.",
        "Salinan sesuai aslinya.",
    ]


# --------------------------------------------------------------------------- backslashes (a US bill's page footer)
import pytest as _pytest  # noqa: E402

from automation.src.rag.sentence_split import split_sentences as _split  # noqa: E402


@_pytest.mark.parametrize("text", [
    r"Filed as G:\COMP\103\103-322.XML here.",
    r"See G:\COMP\103\103-322.XML.Next sentence starts here.",       # the "no space after the period" case pysbd rewrites
    r"A pattern \1 and \g and \n and \d are just characters.",
    "H.R. 3355\nG:\\COMP\\103\\103-322.XML\nSEC. 2. Definitions.",
])
def test_a_backslash_in_the_text_does_not_break_the_segmenter(text):
    sentences = _split(text)
    assert sentences and "".join(sentences).replace(" ", "").replace("\n", "") != ""
    assert text.count("\\") == sum(s.count("\\") for s in sentences)                  # every backslash comes back


def test_the_backslash_path_is_kept_whole_in_the_sentence_text():
    sentences = _split(r"Filed as G:\COMP\103\103-322.XML here. It is done.")
    assert any(r"G:\COMP\103\103-322." in s for s in sentences)


def test_text_without_backslashes_is_segmented_exactly_as_before():
    assert _split("This Act applies. It is law.") == ["This Act applies.", "It is law."]


def test_the_source_side_with_protected_abbreviations_also_survives_a_backslash():
    sentences = _split(r"Lihat G:\COMP\103\103-322.XML dll. Ketentuan lain.", protect_abbreviations=True)
    assert sentences and any("\\" in s for s in sentences)
