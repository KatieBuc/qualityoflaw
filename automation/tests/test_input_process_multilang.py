import re

import pytest

from automation.src.input_process.config import parse_input_process_config
from automation.src.input_process.pipeline import analyze, check_all
from automation.src.input_process.quality.criteria import decide
from automation.src.input_process.quality.metrics import (
    canonical_language,
    language_of,
    profile_for,
    text_metrics,
    unit_numbers,
)

RUSSIAN = ["закон", "защита", "семья", "насилие", "орган", "власти", "помощь", "жертвы", "порядок", "статья", "лицо", "право"]
ARABIC = ["القانون", "حماية", "الأسرة", "العنف", "السلطة", "المساعدة", "الضحايا", "النظام", "المادة", "الحق", "الشخص", "الدولة"]
SPANISH = ["protección", "violencia", "administración", "autoridad", "sanción", "mujeres", "víctimas", "población", "también", "ayuda", "familia", "estado"]
SPANISH_NO_ACCENTS = ["proteccion", "violencia", "administracion", "autoridad", "sancion", "mujeres", "victimas", "poblacion", "tambien", "ayuda", "familia", "estado"]
CHINESE = list("法律保护家庭暴力受害人权利国家机关责任条款规定公民组织社会救助措施机构监督执行")


def prose(words, n_lines=120, per_line=10):
    """Distinct, prose-length lines (>= 40 chars) so repeated-line rules stay quiet."""
    return "\n".join(
        " ".join(words[(j * 7 + k) % len(words)] for k in range(per_line)) + f" {j}" for j in range(n_lines)
    )


def chinese(n_lines=60):
    return "\n".join("".join(CHINESE[(j * 3 + k) % len(CHINESE)] for k in range(60)) for j in range(n_lines))


def verdict(text, language, source="txt"):
    return decide(text_metrics(text, profile_for(language)), source)


# --------------------------------------------------------------------------- language tables
def test_profiles_come_from_the_globallaws_tables():
    assert profile_for("Russian").scripts == {"CYRILLIC"}
    assert profile_for("Arabic").scripts == {"ARABIC"}
    assert profile_for("Bosnian").scripts == {"LATIN", "CYRILLIC"}
    assert profile_for("Chinese").spaced is False and profile_for("Thai").spaced is False
    assert profile_for("Spanish").diacritics is True and profile_for("French").diacritics is True


def test_indonesian_malay_english_and_unknown_languages_have_no_diacritic_check():
    for name in ("Indonesian", "Malay", "English", "Klingon", ""):
        profile = profile_for(name)
        assert profile.scripts == {"LATIN"} and profile.spaced and not profile.diacritics


def test_misspelled_language_names_are_corrected():
    assert canonical_language("Potuguese") == "Portuguese"
    assert profile_for("Potuguese").diacritics is True


def test_language_comes_from_the_name_when_the_pattern_matches_else_the_default():
    pattern = re.compile(r"DomViol - ([A-Za-z]+)")
    assert language_of("10. Azerbaijan - DomViol - Azerbaijani - html", pattern, "Indonesian") == "Azerbaijani"
    assert language_of("136. Portugal - DomViol - Potuguese", pattern, "Indonesian") == "Portuguese"
    assert language_of("ACEH_BIREUEN", pattern, "Indonesian") == "Indonesian"
    assert language_of("ACEH_BIREUEN", None, "Indonesian") == "Indonesian"


# --------------------------------------------------------------------------- scripts
@pytest.mark.parametrize("language,words", [("Russian", RUSSIAN), ("Arabic", ARABIC)])
def test_non_latin_text_passes_as_its_own_language_but_not_as_indonesian(language, words):
    text = prose(words)
    assert verdict(text, language).status == "PASS"
    wrong = verdict(text, "Indonesian")
    assert wrong.status == "ELIMINATE" and "C1_wrong_script" in wrong.reason


def test_chinese_is_judged_without_space_based_metrics():
    m = text_metrics(chinese(), profile_for("Chinese"))
    assert m["noisy_token_ratio"] is None and m["glued_tokens_per_1k"] is None
    assert verdict(chinese(), "Chinese").status == "PASS"
    assert verdict(chinese(), "Indonesian").status == "ELIMINATE"


def test_lost_diacritics_are_caught_for_spanish_only():
    accented, stripped = prose(SPANISH), prose(SPANISH_NO_ACCENTS)
    assert verdict(accented, "Spanish").status == "PASS"
    lost = verdict(stripped, "Spanish")
    assert lost.status == "ELIMINATE" and "C8_lost_diacritics" in lost.reason
    assert verdict(stripped, "Indonesian").status == "PASS"  # no diacritic rule for Indonesian


# --------------------------------------------------------------------------- vocabularies
def test_unit_headings_in_several_languages_are_counted():
    lines = ["Artículo 1", "Artigo 2", "Pasal 3", "Article 4", "Artikel 5", "Section 6", "§ 7", "Madde 8"]
    assert unit_numbers(lines) == [1, 2, 3, 4, 5, 6, 7, 8]


def test_spanish_and_french_web_chrome_is_recognised():
    chrome = "\n".join(["Iniciar sesión", "Política de privacidad", "Aviso legal", "Connexion", "Mentions légales"] * 3)
    v = verdict(prose(SPANISH) + "\n" + chrome, "Spanish")
    assert v.status == "PASS" and "P1_web_chrome" in v.findings


def test_original_paywall_phrases_still_eliminate():
    v = verdict(prose(SPANISH) + "\nContenido exclusivo para suscriptores.", "Spanish", "html")
    assert v.status == "ELIMINATE" and "R3_paywall_markers" in v.reason


# --------------------------------------------------------------------------- config
def test_language_pattern_must_be_a_valid_regex_with_a_group():
    with pytest.raises(ValueError, match="valid regex"):
        parse_input_process_config({"input_process": {"language_pattern": "DomViol - ("}})
    with pytest.raises(ValueError, match="one group"):
        parse_input_process_config({"input_process": {"language_pattern": "DomViol - [A-Za-z]+"}})


def test_config_picks_the_language_per_policy():
    cfg = parse_input_process_config(
        {"input_process": {"language": "Indonesian", "language_pattern": r"DomViol - ([A-Za-z]+)"}}
    )
    assert cfg.language_for("3. Algeria - DomViol - French") == "French"
    assert cfg.language_for("ACEH_BIREUEN") == "Indonesian"


# --------------------------------------------------------------------------- pipeline
@pytest.fixture
def mixed(tmp_path):
    def build(pattern):
        cfg = parse_input_process_config(
            {
                "input_process": {
                    "language": "Indonesian",
                    "language_pattern": pattern,
                    "raw_files_dir": str(tmp_path / "raw_files"),
                    "extracted_txt_dir": str(tmp_path / "extracted_txt"),
                    "cleaned_txt_dir": str(tmp_path / "cleaned_txt"),
                    "report_path": str(tmp_path / "report.json"),
                }
            }
        )
        cfg.extracted_txt_dir.mkdir(exist_ok=True)
        return cfg

    return build


def write(cfg, stem, fmt, text):
    (cfg.extracted_txt_dir / f"{stem}__{fmt}.txt").write_text(text, encoding="utf-8")


def test_a_multilingual_folder_is_judged_file_by_file(mixed):
    cfg = mixed(r"DomViol - ([A-Za-z]+)")
    write(cfg, "1. Russia - DomViol - Russian", "txt", prose(RUSSIAN))
    write(cfg, "2. Egypt - DomViol - Arabic", "txt", prose(ARABIC))
    write(cfg, "3. Spain - DomViol - Spanish", "txt", prose(SPANISH))
    write(cfg, "4. China - DomViol - Chinese", "txt", chinese())
    write(cfg, "ACEH_BIREUEN", "txt", prose(["Bupati", "melaksanakan", "perlindungan", "perempuan", "anak", "korban", "kekerasan", "daerah"]))

    outcomes = analyze(cfg).outcomes
    report = check_all(cfg)

    assert report["summary"] == {"files": 5, "passed": 5, "needs_ocr": 0, "eliminated": 0}
    assert outcomes["1. Russia - DomViol - Russian"].language == "Russian"
    assert outcomes["2. Egypt - DomViol - Arabic"].language == "Arabic"
    assert outcomes["ACEH_BIREUEN"].language == "Indonesian"


def test_without_a_language_pattern_every_file_is_judged_as_the_default_language(mixed):
    cfg = mixed("")
    write(cfg, "1. Russia - DomViol - Russian", "txt", prose(RUSSIAN))
    write(cfg, "ACEH_BIREUEN", "txt", prose(["Bupati", "melaksanakan", "perlindungan", "perempuan", "anak", "korban", "kekerasan", "daerah"]))

    report = check_all(cfg)

    assert report["summary"]["passed"] == 1
    [entry] = report["eliminated"]
    assert entry["stem"] == "1. Russia - DomViol - Russian"
    assert "C1_wrong_script" in {f["rule"] for f in entry["failed"]}


def test_corpus_unique_words_are_counted_within_one_language(mixed):
    """12 Spanish OCR files give Spanish a reference corpus; the lone Russian OCR file has none."""
    cfg = mixed(r"DomViol - ([A-Za-z]+)")
    for i in range(12):
        write(cfg, f"{i}. Spain - DomViol - Spanish", "ocr", prose(SPANISH))
    write(cfg, "99. Russia - DomViol - Russian", "ocr", prose(RUSSIAN))

    outcomes = analyze(cfg).outcomes

    assert "corpus_unique_word_ratio" in outcomes["0. Spain - DomViol - Spanish"].metrics
    assert "corpus_unique_word_ratio" not in outcomes["99. Russia - DomViol - Russian"].metrics
