import pytest

from automation.src.failure_log import (
    clear_failure,
    clear_step_failure,
    load_failures,
    record_failure,
    record_step_failure,
    summarize_failures,
)


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    monkeypatch.setattr("automation.src.config_loader.OUTPUT_DIR_OVERRIDE", tmp_path)
    return tmp_path


def test_record_and_load_failure(data_root):
    entry = {
        "filename": "ACEH_BIREUEN.txt",
        "error_type": "RateLimitError",
        "message": "429 Too Many Requests",
        "details": {"status_code": 429},
        "attempts": 3,
    }
    record_failure("translation", entry)

    log = load_failures()
    assert len(log["translation"]) == 1
    assert log["translation"][0]["filename"] == "ACEH_BIREUEN.txt"
    assert log["translation"][0]["error_type"] == "RateLimitError"
    assert log["updated_at"] is not None


def test_record_failure_replaces_same_item(data_root):
    record_failure(
        "translation",
        {
            "filename": "A.txt",
            "error_type": "Exception",
            "message": "first",
            "details": {},
            "attempts": 1,
        },
    )
    record_failure(
        "translation",
        {
            "filename": "A.txt",
            "error_type": "Exception",
            "message": "second",
            "details": {},
            "attempts": 2,
        },
    )

    log = load_failures()
    assert len(log["translation"]) == 1
    assert log["translation"][0]["message"] == "second"
    assert log["translation"][0]["attempts"] == 2


def test_clear_failure_removes_entry(data_root):
    record_failure(
        "translation",
        {
            "filename": "A.txt",
            "error_type": "Exception",
            "message": "err",
            "details": {},
            "attempts": 1,
        },
    )
    record_failure(
        "evaluation",
        {
            "policy_file": "B.txt",
            "failed_dimensions": [],
            "errors": ["err"],
            "error_type": "EvaluationError",
            "message": "err",
            "details": {},
            "attempts": 1,
        },
    )

    clear_failure("translation", "A.txt")
    log = load_failures()
    assert log["translation"] == []
    assert len(log["evaluation"]) == 1


def test_clear_failure_deletes_file_when_empty(data_root):
    record_failure(
        "translation",
        {
            "filename": "A.txt",
            "error_type": "Exception",
            "message": "err",
            "details": {},
            "attempts": 1,
        },
    )
    clear_failure("translation", "A.txt")
    assert not (data_root / "failures.json").exists()


def test_summarize_failures(data_root):
    record_failure(
        "translation",
        {
            "filename": "A.txt",
            "error_type": "Exception",
            "message": "err",
            "details": {},
            "attempts": 1,
        },
    )
    record_failure(
        "evaluation",
        {
            "policy_file": "B.txt",
            "failed_dimensions": ["dim.txt"],
            "errors": ["err"],
            "error_type": "EvaluationError",
            "message": "err",
            "details": {},
            "attempts": 1,
        },
    )

    summary = summarize_failures()
    assert summary == {
        "translation": 1,
        "translation_qa": 0,
        "storage": 0,
        "evaluation": 1,
        "discrepancy_diagnosis": 0,
        "step_failures": 0,
    }


def test_record_and_clear_step_failure(data_root):
    record_step_failure("evaluation", "No policy files to evaluate")

    log = load_failures()
    assert log["step_failures"] == [
        {"step": "evaluation", "message": "No policy files to evaluate", "at": log["step_failures"][0]["at"]}
    ]
    assert summarize_failures()["step_failures"] == 1

    clear_step_failure("evaluation")
    assert not (data_root / "failures.json").exists()


def test_record_step_failure_replaces_prior_entry_for_same_step(data_root):
    record_step_failure("translation", "first failure")
    record_step_failure("translation", "second failure")

    log = load_failures()
    assert len(log["step_failures"]) == 1
    assert log["step_failures"][0]["message"] == "second failure"


def test_load_failures_empty_run(data_root):
    assert load_failures() == {
        "updated_at": None,
        "translation": [],
        "translation_qa": [],
        "storage": [],
        "evaluation": [],
        "discrepancy_diagnosis": [],
        "step_failures": [],
    }
