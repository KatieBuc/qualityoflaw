import pytest

from automation.src.constants import AUTOMATION_ROOT
from automation.src.criteria import (
    build_criteria_maps,
    dimension_name,
    extract_schema_from_prompts,
    list_criteria_files,
    load_criteria_file_ids,
    load_index_schema,
    resolve_schema,
    validate_criteria_files,
)
from automation.src.paths import project_dirs

QUALITY_EVAL = AUTOMATION_ROOT / "prompts" / "quality_eval"
INDONESIA_IDS, INDONESIA_DIMENSION, _ = build_criteria_maps(load_index_schema(project_dirs("indonesia").index_schema))


def test_indonesia_schema_has_56_indicators():
    assert len(INDONESIA_IDS) == 56


def test_indonesian_criteria_files_match_the_schema():
    ok, issues = validate_criteria_files(QUALITY_EVAL / "v1", INDONESIA_IDS)
    assert ok, issues


# --------------------------------------------------------------------------- the files say what the prompt has
def test_criteria_files_are_the_numbered_files_of_the_folder_in_order():
    files = list_criteria_files(QUALITY_EVAL / "global_v1")
    assert [f[:2] for f in files] == ["01", "02", "03", "04", "05", "06", "07"]
    assert "prompt_template.txt" not in files
    assert list_criteria_files(QUALITY_EVAL / "missing") == []


def test_dimension_names_come_from_the_file_names():
    assert dimension_name("02_Protection_and_Justice.txt") == "protection_and_justice"
    assert dimension_name("01_scope_of_violence.txt") == "scope_of_violence"


def test_the_indonesian_prompts_give_the_same_schema_as_their_yaml():
    from_prompt = extract_schema_from_prompts(QUALITY_EVAL / "v4")
    ids, dimension, _ = build_criteria_maps(from_prompt)
    assert ids == INDONESIA_IDS
    assert dimension == INDONESIA_DIMENSION                           # same dimension for every indicator


def test_a_project_without_an_index_schema_gets_it_from_its_prompt():
    schema = extract_schema_from_prompts(QUALITY_EVAL / "global_v1")
    ids, dimension, per_dimension = build_criteria_maps(schema)

    assert len(ids) == 47 and len(per_dimension) == 7
    assert [len(v) for v in per_dimension.values()] == [10, 5, 6, 9, 5, 7, 5]
    assert dimension["1.1"] == "scope_of_violence" and dimension["4.9"] == "specialised_support_services"
    assert dimension["6.1"] == "stakeholder_engagement_and_multi-sectoral_cooperation"
    assert schema["scope_of_violence"]["1.1"]["indicator"] == "Domestic violence"


def test_the_schema_of_a_project_comes_from_its_yaml_when_it_has_one_and_from_the_prompt_otherwise(tmp_path, monkeypatch):
    monkeypatch.setattr("automation.src.paths.DATA_ROOT", tmp_path)
    assert resolve_schema("nowhere", QUALITY_EVAL / "global_v1") == extract_schema_from_prompts(QUALITY_EVAL / "global_v1")

    mapping = tmp_path / "withyaml" / "mapping"
    mapping.mkdir(parents=True)
    (mapping / "index_schema.yaml").write_text("dim_a:\n  '1.1':\n    indicator: X\n", encoding="utf-8")
    assert resolve_schema("withyaml", QUALITY_EVAL / "global_v1") == {"dim_a": {"1.1": {"indicator": "X"}}}


def test_the_shipped_global_prompt_is_valid():
    ok, issues = validate_criteria_files(QUALITY_EVAL / "global_v1")
    assert ok, issues


# --------------------------------------------------------------------------- a criterion line the evaluation cannot read
def test_a_criterion_whose_question_is_on_the_next_line_is_reported_not_skipped(tmp_path):
    folder = tmp_path / "v"
    folder.mkdir()
    (folder / "01_dim.txt").write_text(
        "1.1 | Good | A question on the same line?\nCode Yes if x.\n\n"
        "1.2 | Bad indicator\n\nThe question is down here.\n",
        encoding="utf-8",
    )

    ok, issues = validate_criteria_files(folder)

    assert not ok
    assert any("1.2" in issue and "2 field" in issue and "ONE line" in issue for issue in issues)
    assert load_criteria_file_ids(folder / "01_dim.txt") == {"1.1"}               # 1.2 would not reach the judge
    assert extract_schema_from_prompts(folder) == {"dim": {"1.1": {"indicator": "Good"}}}


def test_rubric_lines_are_not_mistaken_for_indicator_ids(tmp_path):
    folder = tmp_path / "v"
    folder.mkdir()
    (folder / "01_dim.txt").write_text(
        "1.1 | A | Q?\n- the framework says | something | with bars\n12. an enumerated line\n", encoding="utf-8"
    )
    assert load_criteria_file_ids(folder / "01_dim.txt") == {"1.1"}


def test_unknown_and_missing_ids_are_reported_against_an_expected_list(tmp_path):
    folder = tmp_path / "v"
    folder.mkdir()
    (folder / "01_dim.txt").write_text("1.1 | A | Q?\n1.3 | C | Q?\n", encoding="utf-8")

    ok, issues = validate_criteria_files(folder, {"1.1", "1.2"})

    assert not ok
    assert any("missing IDs" in i and "1.2" in i for i in issues) and any("unknown IDs" in i and "1.3" in i for i in issues)


def test_a_folder_without_numbered_files_is_invalid(tmp_path):
    ok, issues = validate_criteria_files(tmp_path)
    assert not ok and "No numbered criteria files" in issues[0]


# --------------------------------------------------------------------------- "5. Name | question" is not a criterion
def test_a_numbered_line_that_is_not_in_the_criterion_form_is_reported(tmp_path):
    folder = tmp_path / "v"
    folder.mkdir()
    (folder / "02_dim.txt").write_text(
        "2.1 | Good | A question?\nCode Yes if x.\n\n"
        "5. Marital rape / spousal exemption | Does the framework recognise marital rape?\n\nCode Yes if y.\n",
        encoding="utf-8",
    )

    ok, issues = validate_criteria_files(folder)

    assert not ok
    assert any("line 4" in i and "looks like a criterion" in i and "made-up id" in i for i in issues)


def test_rubric_lines_and_well_formed_criteria_are_not_reported(tmp_path):
    folder = tmp_path / "v"
    folder.mkdir()
    (folder / "01_dim.txt").write_text(
        "1.1 | Good | A question?\n"
        "Code Yes - Substantive if, in the operative provisions of any relevant instrument:\n"
        "- the framework says something;\n"
        "12. an enumerated item without a bar\n"
        "2 | A | Q?\n",
        encoding="utf-8",
    )
    assert validate_criteria_files(folder) == (True, [])


def test_every_shipped_prompt_version_is_free_of_such_lines():
    for version in ("v1", "v2", "v3", "v4", "global_v1"):
        ok, issues = validate_criteria_files(QUALITY_EVAL / version)
        assert ok, (version, issues)
