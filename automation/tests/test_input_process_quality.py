import fitz
import pytest

from automation.src.input_process.quality.criteria import (
    decide,
    fix_acceptable,
    fix_plan,
    ocr_decision,
    resolve_thresholds,
)
from automation.src.input_process.quality.metrics import (
    corpus_unique_word_ratio,
    numbering_disorder,
    pdf_metrics,
    profile_for,
    text_metrics,
    unit_numbers,
    word_set,
)

ID = profile_for("Indonesian")

SERVICES = [
    "layanan kesehatan", "bantuan hukum", "rumah aman", "pendampingan psikologis",
    "pemulihan sosial", "pemberdayaan ekonomi", "pendidikan keterampilan", "konseling keluarga",
]


def statute(n=20, start=1):
    """A clean Indonesian regulation with unique, prose-length lines."""
    lines = ["PERATURAN DAERAH KABUPATEN CONTOH", "TENTANG PERLINDUNGAN PEREMPUAN DAN ANAK", ""]
    for i in range(start, start + n):
        lines.append(f"Pasal {i}")
        for j in (1, 2, 3):
            lines.append(
                f"({j}) Setiap korban kekerasan berhak memperoleh {SERVICES[(i + j) % len(SERVICES)]} "
                f"dari pemerintah daerah sesuai ketentuan ayat {i + j} peraturan ini pada tahun {2000 + i}."
            )
        lines.append("")
    return "\n".join(lines)


def metrics(text):
    return text_metrics(text, ID)


def rule_ids(verdict):
    return set(verdict.findings)


# --------------------------------------------------------------------------- clean text
def test_a_clean_indonesian_regulation_passes_with_no_findings():
    verdict = decide(metrics(statute()), "txt")
    assert verdict.status == "PASS" and verdict.reason == "" and verdict.findings == {}
    assert verdict.review == ""


def test_pasal_headings_are_counted_but_wrapped_cross_references_are_not():
    nums = unit_numbers(
        ["Pasal 1", "isi", "Pasal 2", "Pasal 5 ayat (1) huruf f, meliputi:", "Pasal 3"]
    )
    assert nums == [1, 2, 3]


def test_scrambled_pasal_order_is_flagged_for_review_but_still_passes():
    text = statute(15, start=1) + "\n" + "\n".join(
        f"Pasal {n}\nBupati menetapkan ketentuan lebih lanjut mengenai pelaksanaan nomor {n} tersebut." for n in
        (30, 22, 31, 23, 32, 24, 33, 25, 34, 26, 35)
    )
    m = metrics(text)
    assert numbering_disorder(unit_numbers(text.split("\n"))) >= 0.10
    verdict = decide(m, "txt")
    assert verdict.status == "PASS" and verdict.review.startswith("R6_numbering_disorder")


# --------------------------------------------------------------------------- elimination
def test_too_little_text_is_eliminated():
    verdict = decide(metrics("Pasal 1\nBupati menetapkan."), "txt")
    assert verdict.status == "ELIMINATE" and "R1_letters" in verdict.reason


def test_replacement_characters_are_eliminated_as_garbage():
    text = statute() + "\n" + "�" * 400
    verdict = decide(metrics(text), "txt")
    assert verdict.status == "ELIMINATE" and "R4_C2_garbage_chars" in verdict.reason


def test_mojibake_is_eliminated():
    text = statute() + "\n" + "\n".join("Pemerintah dan Ã©Ã± rakyat" for _ in range(30))
    verdict = decide(metrics(text), "txt")
    assert verdict.status == "ELIMINATE" and "C7_mojibake" in verdict.reason


def test_paywall_placeholder_is_eliminated():
    text = statute() + "\nBerlangganan untuk membaca dokumen lengkap ini."
    verdict = decide(metrics(text), "html")
    assert verdict.status == "ELIMINATE" and "R3_paywall_markers" in verdict.reason


def test_a_page_of_chrome_has_no_prose_and_is_eliminated():
    text = "\n".join(["Beranda", "Tentang Kami", "Hubungi Kami", "Peta Situs", "Kebijakan Privasi"] * 80)
    verdict = decide(metrics(text), "html")
    assert verdict.status == "ELIMINATE" and "R2_prose_letters" in verdict.reason


# --------------------------------------------------------------------------- fixable findings
def test_web_chrome_urls_repeats_and_html_residue_are_findings_not_removal():
    base = statute()
    chrome = "\n".join(["Kebijakan Privasi", "Hubungi Kami", "Beranda"] * 6)
    verdict = decide(metrics(base + "\n" + chrome), "html")
    assert verdict.status == "PASS" and "P1_web_chrome" in verdict.findings

    urls = "\n".join(f"https://contoh.go.id/halaman/{i}" for i in range(12))
    assert "P2_urls" in decide(metrics(base + "\n" + urls), "html").findings

    repeated = "\n".join(["Halaman ini adalah header yang selalu berulang di setiap halaman"] * 40)
    assert "P3_repeated_lines" in decide(metrics(base + "\n" + repeated), "pdf").findings

    assert "P4_html_residue" in decide(metrics(base + "\n<div>isi</div>"), "html").findings


def test_each_fixable_finding_maps_to_a_fixer():
    verdict = decide(metrics(statute() + "\n<div>x</div>\n" + "\n".join(["Beranda"] * 8)), "html")
    assert fix_plan(verdict.findings) == ["strip_boilerplate", "strip_html_residue"]


def test_shattered_words_are_a_minor_fixable_finding():
    text = statute() + "\n" + "\n".join("p e r l i n d u n g a n dan hukum" for _ in range(8))
    verdict = decide(metrics(text), "txt")
    assert verdict.status == "PASS"
    assert verdict.findings["C5_shattered_words"]["severity"] == "minor"
    assert fix_plan(verdict.findings) == ["join_spaced_letters"]


def test_minor_ocr_noise_does_not_block_a_file():
    text = statute() + "\n" + " ".join(["Pemer1ntah", "daer4h"] * 3)
    verdict = decide(metrics(text), "txt")
    assert verdict.status == "PASS"
    assert verdict.findings["C3_noisy_tokens"]["severity"] == "minor"


# --------------------------------------------------------------------------- OCR routing
def pdf_row(**over):
    base = dict(
        metrics(statute(5)),
        pdf_pages=5,
        pdf_low_text_page_ratio=0.0,
        letters_per_pdf_page=900.0,
    )
    base.update(over)
    return base


def test_a_pdf_with_a_good_text_layer_does_not_need_ocr():
    assert decide(pdf_row(), "pdf").status == "PASS"


def test_a_scanned_pdf_with_no_text_needs_ocr():
    m = dict(metrics(""), pdf_pages=12, pdf_low_text_page_ratio=1.0, letters_per_pdf_page=0.0)
    verdict = decide(m, "pdf")
    assert verdict.status == "NEEDS_OCR" and verdict.reason.startswith("O1")


def test_missing_pages_need_ocr():
    verdict = decide(pdf_row(letters_per_pdf_page=120.0), "pdf")
    assert verdict.status == "NEEDS_OCR" and verdict.reason.startswith("O3")


def test_a_pdf_text_layer_with_garbage_needs_ocr_while_the_same_text_from_txt_is_eliminated():
    text = statute(5) + "\n" + "�" * 300
    assert decide(dict(metrics(text), pdf_pages=5, pdf_low_text_page_ratio=0.0,
                       letters_per_pdf_page=900.0), "pdf").status == "NEEDS_OCR"
    assert decide(metrics(text), "txt").status == "ELIMINATE"


def test_ocr_output_that_is_still_bad_is_eliminated_never_sent_to_ocr_again():
    text = statute(5) + "\n" + "�" * 300
    verdict = decide(metrics(text), "ocr")
    assert verdict.status == "ELIMINATE"
    assert ocr_decision({**metrics(text), "source_format": "ocr"})[0] is False


# --------------------------------------------------------------------------- corpus-relative
def test_unique_word_ratio_applies_to_ocr_text_only():
    texts = [statute(20) for _ in range(12)]
    doc_freq = {}
    for t in texts:
        for w in word_set(t, ID):
            doc_freq[w] = doc_freq.get(w, 0) + 1
    noisy = statute(20) + "\n" + " ".join(f"xqzv{i}abcd" for i in range(300))
    for w in word_set(noisy, ID):
        doc_freq[w] = doc_freq.get(w, 0) + 1

    ratio = corpus_unique_word_ratio(noisy, ID, doc_freq)
    assert ratio > 0.11
    m = dict(metrics(noisy), corpus_unique_word_ratio=ratio)
    assert "C4_unique_words" in decide(m, "ocr").findings
    assert "C4_unique_words" not in decide(m, "txt").findings


# --------------------------------------------------------------------------- fix limits & thresholds
def test_fix_limits_reject_big_letter_loss_and_lost_headings():
    before = metrics(statute(20))
    ok, why = fix_acceptable(before, metrics(statute(20)))
    assert ok and why == ""

    smaller = metrics(statute(10))  # lost half the letters and half the Pasal numbers
    ok, why = fix_acceptable(before, smaller)
    assert not ok and "K2_letter_loss" in why and "K1_headings_lost" in why


def test_threshold_overrides_are_applied_and_unknown_names_rejected():
    t = resolve_thresholds({"min_letters": 5})
    assert decide(metrics("Pasal 1\nBupati menetapkan peraturan daerah ini dengan baik."), "txt", t).findings == {}
    assert t["min_letters"] == 5
    with pytest.raises(ValueError, match="not_a_threshold"):
        resolve_thresholds({"not_a_threshold": 1})


# --------------------------------------------------------------------------- pdf metrics
def make_pdf(path, pages):
    doc = fitz.open()
    for text in pages:
        page = doc.new_page()
        if text:
            page.insert_textbox(fitz.Rect(40, 40, 560, 800), text, fontsize=9)
    doc.save(str(path))
    doc.close()


def test_pdf_metrics_separates_text_pages_from_image_only_pages(tmp_path):
    long_page = " ".join(["Bupati melaksanakan perlindungan perempuan dan anak."] * 12)
    make_pdf(tmp_path / "mixed.pdf", [long_page, "", long_page, ""])
    make_pdf(tmp_path / "scan.pdf", ["", "", ""])

    mixed = pdf_metrics(tmp_path / "mixed.pdf")
    scan = pdf_metrics(tmp_path / "scan.pdf")

    assert mixed["pdf_pages"] == 4 and mixed["pdf_low_text_page_ratio"] == 0.5
    assert scan["pdf_low_text_page_ratio"] == 1.0 and scan["pdf_median_chars_per_page"] == 0
