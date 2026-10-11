import re

from automation.src.input_process import heading_keywords as KW
from automation.src.input_process import txt_to_md_v7 as conv

INDONESIAN = """QANUN KABUPATEN CONTOH
NOMOR 1 TAHUN 2022

BAB I
KETENTUAN UMUM

Pasal 1
Dalam Qanun ini yang dimaksud dengan:
1. Kabupaten adalah Kabupaten Contoh.
2. Bupati adalah Bupati Contoh.

Pasal 2
Perlindungan perempuan berdasarkan asas keadilan.

BAB II
PENYELENGGARAAN

Bagian Kesatu
Umum

Pasal 3
Pemerintah menyelenggarakan perlindungan sebagaimana dimaksud dalam Pasal 2.

Paragraf 1
Pencegahan

Pasal 4
Pencegahan dilakukan melalui edukasi.

Paragraf 2
Perlindungan

Pasal 4A
Perlindungan dilakukan oleh petugas.

Paragraf 3
Pemulihan

Pasal 4B
Pemulihan dilakukan oleh dinas.

Bagian Kedua
Pelaporan

Pasal 5
Setiap orang wajib melapor.

Pasal 6
Pelaporan dilakukan secara tertulis.

Bagian Ketiga
Pendanaan

Pasal 7
Pendanaan bersumber dari anggaran daerah.
"""


def headings(md):
    return [(len(m.group(1)), m.group(2)) for m in re.finditer(r"(?m)^(#{1,6}) (.+)$", md)]


def convert(text):
    return conv.convert_document(text, "Contoh", depth="relative")


def test_indonesian_hierarchy_is_detected():
    info = conv.analyse(conv.preprocess(INDONESIAN))
    assert info["active"] == ["CHAPTER", "SECTION", "SUBSECTION", "ARTICLE"]


def test_indonesian_headings_are_nested_bab_bagian_paragraf_pasal():
    found = dict((text, depth) for depth, text in headings(convert(INDONESIAN)))
    assert found["BAB I"] < found["Bagian Kesatu"] < found["Paragraf 1"] < found["Pasal 4"]
    assert found["Pasal 1"] == found["Pasal 5"] == found["Pasal 4"]
    for label in ("BAB II", "Bagian Kedua", "Pasal 6"):
        assert label in found


def test_a_pasal_cited_inside_a_sentence_is_not_a_heading():
    labels = [text for _, text in headings(convert(INDONESIAN))]
    assert not any("sebagaimana dimaksud" in text for text in labels)
    assert [t for t in labels if t.startswith("Pasal")] == ["Pasal 1", "Pasal 2", "Pasal 3", "Pasal 4", "Pasal 4A", "Pasal 4B", "Pasal 5", "Pasal 6", "Pasal 7"]


def test_a_pasal_with_ayat_or_a_law_name_after_it_is_a_citation():
    text = INDONESIAN + "\nPasal 6 ayat (1) mengatur sanksi.\nPasal 6 Undang-Undang ini berlaku.\n"
    assert [t for _, t in headings(convert(text)) if t.startswith("Pasal")].count("Pasal 6") == 1


def test_a_word_that_merely_starts_with_part_is_not_a_part_heading():
    text = INDONESIAN.replace("Pasal 2\n", "Pasal 2\nPartisipasi masyarakat dijamin.\n", 1)
    assert "PART" not in conv.analyse(conv.preprocess(text))["active"]
    assert "PART" not in conv.analyse(conv.preprocess("Partisipasi\n" * 5))["active"]


def test_real_roman_numeral_labels_still_work():
    text = "PART I\nGeneral\n\nPART II\nOffences\n\nSection 1\nx\n\nSection 2\ny\n\nSection 3\nz\n"
    assert "PART" in conv.analyse(conv.preprocess(text))["active"]


def test_each_level_lists_its_words_by_language_and_indonesian_is_present():
    assert set(KW.HEADING_WORDS) == set(KW.LEVELS)
    ids = {level: words["id"] for level, words in KW.HEADING_WORDS.items() if "id" in words}
    assert ids == {"BOOK": "BUKU", "CHAPTER": "BAB", "SECTION": "BAGIAN", "SUBSECTION": "PARAGRAF", "ARTICLE": "PASAL"}
    assert "KESATU" in KW.ORDINAL_WORDS["id"]


def test_every_keyword_pattern_compiles():
    for words in KW.HEADING_WORDS.values():
        re.compile(KW.alternation(words), re.I)
    for words in KW.POST_WORDS.values():
        re.compile(KW.alternation(words), re.I)
    re.compile(KW.alternation(KW.ORDINAL_WORDS), re.I)
    re.compile(KW.alternation(KW.XREF_TAILS), re.I)
    re.compile(KW.alternation(KW.PAGE_WORDS), re.I)
    for _, rx in KW.NONLATIN_MARKERS:
        re.compile(rx)


def test_the_converter_has_no_keyword_literals_of_its_own():
    # the vocabulary lives in heading_keywords.py; the composed tables must come from it
    assert conv.TOK["ARTICLE"] == KW.alternation(KW.HEADING_WORDS["ARTICLE"])
    assert "PASAL" in conv.TOK["ARTICLE"] and "BAB" in conv.TOK["CHAPTER"]
    assert "halaman" in conv.PAGE_WORDS
    assert "daftar isi" in conv._TOC_TITLES
