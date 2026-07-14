import pytest

from automation.src.failure_log import (
    clear_failure,
    load_failures,
    record_failure,
    summarize_failures,
)


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    monkeypatch.setattr("automation.src.metadata.DEFAULT_DATA_ROOT", tmp_path)
    monkeypatch.setattr("automation.src.config_loader.DEFAULT_DATA_ROOT", tmp_path)
    return tmp_path


def test_record_and_load_failure(data_root):
    run_id = "fail_run"
    entry = {
        "filename": "ACEH_BIREUEN.txt",
        "error_type": "RateLimitError",
        "message": "429 Too Many Requests",
        "details": {"status_code": 429},
        "attempts": 3,
    }
    record_failure(run_id, "translation", entry)

    log = load_failures(run_id)
    assert len(log["translation"]) == 1
    assert log["translation"][0]["filename"] == "ACEH_BIREUEN.txt"
    assert log["translation"][0]["error_type"] == "RateLimitError"
    assert log["updated_at"] is not None


def test_record_failure_replaces_same_item(data_root):
    run_id = "fail_replace"
    record_failure(
        run_id,
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
        run_id,
        "translation",
        {
            "filename": "A.txt",
            "error_type": "Exception",
            "message": "second",
            "details": {},
            "attempts": 2,
        },
    )

    log = load_failures(run_id)
    assert len(log["translation"]) == 1
    assert log["translation"][0]["message"] == "second"
    assert log["translation"][0]["attempts"] == 2


def test_clear_failure_removes_entry(data_root):
    run_id = "fail_clear"
    record_failure(
        run_id,
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
        run_id,
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

    clear_failure(run_id, "translation", "A.txt")
    log = load_failures(run_id)
    assert log["translation"] == []
    assert len(log["evaluation"]) == 1


def test_clear_failure_deletes_file_when_empty(data_root):
    run_id = "fail_empty"
    record_failure(
        run_id,
        "translation",
        {
            "filename": "A.txt",
            "error_type": "Exception",
            "message": "err",
            "details": {},
            "attempts": 1,
        },
    )
    clear_failure(run_id, "translation", "A.txt")
    assert not (data_root / run_id / "failures.json").exists()


def test_summarize_failures(data_root):
    run_id = "fail_summary"
    record_failure(
        run_id,
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
        run_id,
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

    summary = summarize_failures(run_id)
    assert summary == {"translation": 1, "storage": 0, "evaluation": 1}


def test_load_failures_empty_run(data_root):
    assert load_failures("nonexistent") == {
        "updated_at": None,
        "translation": [],
        "storage": [],
        "evaluation": [],
    }
