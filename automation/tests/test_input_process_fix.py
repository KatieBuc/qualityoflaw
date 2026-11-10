import pytest

from automation.src.input_process.fix import (
    FIXERS,
    apply_fixes,
    hygiene,
    join_spaced_letters,
    strip_boilerplate,
    strip_html_residue,
)

PROSE = (
    "Bupati melaksanakan sistem informasi dan pelaporan perlindungan perempuan dan anak korban "
    "kekerasan di seluruh wilayah kabupaten."
)


def test_hygiene_normalizes_bom_line_endings_entities_and_blank_lines():
    raw = "﻿Pasal 1\r\nA &amp; B  \r\n\r\n\r\n\r\nisié\r"
    assert hygiene(raw) == "Pasal 1\nA & B\n\nisié\n"


def test_hygiene_composes_unicode_and_keeps_empty_input_empty():
    assert hygiene("é") == "é\n"
    assert hygiene("  \n \n") == ""


def test_strip_boilerplate_drops_chrome_urls_and_page_numbers_but_not_prose():
    text = "\n".join(
        [
            "Pasal 1",
            PROSE,
            "Beranda",
            "Kebijakan Privasi",
            "https://jdih.contoh.go.id/dokumen/1",
            "- 3 -",
            "Halaman 4 dari 20",
            "Page 5 of 20",
            f"{PROSE} Lihat juga https://jdih.contoh.go.id/aturan untuk keterangan lebih lanjut mengenai hal ini.",
        ]
    )
    out = strip_boilerplate(text).split("\n")
    assert out[0] == "Pasal 1" and PROSE in out
    assert not any(x in out for x in ("Beranda", "Kebijakan Privasi", "- 3 -", "Halaman 4 dari 20"))
    assert len(out) == 3  # heading, prose, and the long prose line that merely mentions a URL


def test_strip_boilerplate_removes_repeated_running_headers_but_keeps_repeated_headings():
    header = "PERATURAN DAERAH KABUPATEN CONTOH NOMOR 5 TAHUN 2020"
    text = "\n".join(
        [header, "Pasal 1", PROSE, header, "Pasal 1", "Pasal 1", PROSE + " Satu.", header, "(1) ayat berulang di sini", "(1) ayat berulang di sini", "(1) ayat berulang di sini"]
    )
    out = strip_boilerplate(text).split("\n")
    assert header not in out
    assert out.count("Pasal 1") == 3 and out.count("(1) ayat berulang di sini") == 3


def test_strip_html_residue_removes_tags_only():
    assert strip_html_residue("<div>Pasal 1</div> a < b dan <br/> isi") == "Pasal 1 a < b dan  isi"


def test_join_spaced_letters_keeps_word_breaks_marked_by_double_spaces():
    assert join_spaced_letters("p e r l i n d u n g a n") == "perlindungan"
    assert join_spaced_letters("p e r l i n d u n g a n  a n a k") == "perlindungan anak"
    assert join_spaced_letters("huruf a b c") == "huruf a b c"  # 3 letters: a list, not a word
    assert join_spaced_letters("anak dan ibu") == "anak dan ibu"


@pytest.mark.parametrize("name", sorted(FIXERS))
def test_every_fixer_is_idempotent(name):
    text = "\n".join(["p e r l i n d u n g a n  a n a k", "Beranda", "<p>isi</p>", "- 3 -", PROSE])
    once = FIXERS[name](text)
    assert FIXERS[name](once) == once


def test_apply_fixes_reports_only_the_fixers_that_changed_something():
    text = PROSE + "\n<b>Pasal 1</b>\n"
    fixed, applied = apply_fixes(text, ["strip_html_residue", "strip_boilerplate", "join_spaced_letters"])
    assert fixed == PROSE + "\nPasal 1\n"
    assert applied == ["strip_html_residue"]

    again, applied_again = apply_fixes(fixed, ["strip_html_residue"])
    assert again == fixed and applied_again == []


def test_apply_fixes_names_hygiene_when_it_was_needed():
    _, applied = apply_fixes("A &amp; B\r\n", [])
    assert applied == ["hygiene"]
