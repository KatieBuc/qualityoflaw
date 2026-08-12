import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from automation.src.concurrency import ConcurrencyLimiter
from automation.src.translation_qa import run_translation_qa_step

TEMPLATE_TEXT = "ORIGINAL:\n{{ORIGINAL_TEXT}}\n\nTRANSLATED:\n{{TRANSLATED_TEXT}}\n{{CONTEXT}}"


class _Paths:
    def __init__(self, input_dir: Path):
        self.input_dir = input_dir


class _Config:
    """Minimal stand-in for ResolvedPipelineConfig — run_translation_qa_step
    only reads config.paths.input_dir and config.translation_qa_template_path.
    """

    def __init__(self, input_dir: Path, template_path: Path):
        self.paths = _Paths(input_dir)
        self.translation_qa_template_path = template_path


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    monkeypatch.setattr("automation.src.config_loader.DEFAULT_DATA_ROOT", tmp_path)
    return tmp_path


@pytest.fixture
def input_dir(tmp_path):
    d = tmp_path / "raw_input"
    d.mkdir()
    return d


@pytest.fixture
def template_path(tmp_path):
    versioned_dir = tmp_path / "prompts" / "translation_qa" / "v1"
    versioned_dir.mkdir(parents=True)
    path = versioned_dir / "prompt_template.txt"
    path.write_text(TEMPLATE_TEXT, encoding="utf-8")
    return path


@pytest.fixture
def config(input_dir, template_path):
    return _Config(input_dir, template_path)


def _make_wrapper(complete_side_effect):
    wrapper = MagicMock()
    wrapper.profile.deployment = "gpt-5.2-translation-qa"
    wrapper.token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    wrapper.complete_structured.side_effect = complete_side_effect
    return wrapper


def _write_chunks(chunks_dir: Path, stem: str, records: list[dict]) -> None:
    chunks_dir.mkdir(parents=True, exist_ok=True)
    (chunks_dir / f"{stem}.chunks.json").write_text(json.dumps(records), encoding="utf-8")


def test_chunked_path_applies_correction_to_chunks_and_merged_file(config, data_root):
    run_id = "run_chunked"
    run_dir = data_root / run_id
    (config.paths.input_dir / "A.txt").write_text(
        "Pasal 1\nisi asli a.\n\nPasal 2\nisi asli b.", encoding="utf-8"
    )

    translation_dir = run_dir / "results" / "translation"
    translation_dir.mkdir(parents=True)
    (translation_dir / "A.txt").write_text(
        "Article 1\noriginal a.\n\nArticle 2\noriginal b.", encoding="utf-8"
    )

    chunks_dir = run_dir / "mid_product" / "chunks"
    records = [
        {
            "chunk_index": 0,
            "section_id": 0,
            "type": "structural",
            "context": None,
            "text": "Pasal 1\nisi asli a.",
            "translated_text": "Article 1\noriginal a.",
        },
        {
            "chunk_index": 0,
            "section_id": 1,
            "type": "structural",
            "context": None,
            "text": "Pasal 2\nisi asli b.",
            "translated_text": "Article 2\n[Note: unclear] original b.",
        },
    ]
    _write_chunks(chunks_dir, "A", records)

    def complete_fn(prompt, *_):
        if "[Note:" in prompt:
            return {
                "action_required": True,
                "issues": ["leftover note"],
                "corrected_text": "Article 2\noriginal b.",
            }
        return {"action_required": False, "issues": [], "corrected_text": None}

    wrapper = _make_wrapper(complete_fn)
    limiter = ConcurrencyLimiter(max_workers=2, enabled=True)

    result = run_translation_qa_step(run_id=run_id, config=config, wrapper=wrapper, limiter=limiter)

    assert result["counts"]["succeeded"] == 1
    assert result["counts"]["corrected"] == 1
    assert result["counts"]["failed"] == 0

    updated_chunks = json.loads((chunks_dir / "A.chunks.json").read_text(encoding="utf-8"))
    assert updated_chunks[0]["translated_text"] == "Article 1\noriginal a."
    assert updated_chunks[0]["qa_action_required"] is False
    assert updated_chunks[1]["translated_text"] == "Article 2\noriginal b."
    assert updated_chunks[1]["qa_action_required"] is True

    merged = (translation_dir / "A.txt").read_text(encoding="utf-8")
    assert merged == "Article 1\noriginal a.\n\nArticle 2\noriginal b."

    report = json.loads(
        (run_dir / "results" / "translation_qa" / "A.json").read_text(encoding="utf-8")
    )
    assert report["granularity"] == "chunked"
    assert report["chunks_corrected"] == 1


def test_whole_file_path_when_no_chunks_json(config, data_root):
    run_id = "run_whole_file"
    run_dir = data_root / run_id
    (config.paths.input_dir / "B.txt").write_text("Pasal 1\nisi asli.", encoding="utf-8")

    translation_dir = run_dir / "results" / "translation"
    translation_dir.mkdir(parents=True)
    (translation_dir / "B.txt").write_text(
        "Here is the translation:\nArticle 1\noriginal text.", encoding="utf-8"
    )

    wrapper = _make_wrapper(
        lambda prompt, *_: {
            "action_required": True,
            "issues": ["leftover preamble"],
            "corrected_text": "Article 1\noriginal text.",
        }
    )
    limiter = ConcurrencyLimiter(max_workers=2, enabled=True)

    result = run_translation_qa_step(run_id=run_id, config=config, wrapper=wrapper, limiter=limiter)

    assert result["counts"]["succeeded"] == 1
    assert result["counts"]["corrected"] == 1
    assert (translation_dir / "B.txt").read_text(encoding="utf-8") == "Article 1\noriginal text."

    report = json.loads(
        (run_dir / "results" / "translation_qa" / "B.json").read_text(encoding="utf-8")
    )
    assert report["granularity"] == "whole_file"


def test_no_action_required_leaves_translation_untouched(config, data_root):
    run_id = "run_noop"
    run_dir = data_root / run_id
    (config.paths.input_dir / "C.txt").write_text("Pasal 1\nisi asli.", encoding="utf-8")

    translation_dir = run_dir / "results" / "translation"
    translation_dir.mkdir(parents=True)
    original_translation = "Article 1\noriginal text."
    (translation_dir / "C.txt").write_text(original_translation, encoding="utf-8")

    wrapper = _make_wrapper(
        lambda *a, **k: {"action_required": False, "issues": [], "corrected_text": None}
    )
    limiter = ConcurrencyLimiter(max_workers=2, enabled=True)

    result = run_translation_qa_step(run_id=run_id, config=config, wrapper=wrapper, limiter=limiter)

    assert result["counts"]["succeeded"] == 1
    assert result["counts"]["corrected"] == 0
    assert (translation_dir / "C.txt").read_text(encoding="utf-8") == original_translation


def test_chunked_path_aborts_atomically_on_chunk_failure(config, data_root):
    run_id = "run_atomic_fail"
    run_dir = data_root / run_id
    (config.paths.input_dir / "D.txt").write_text("Pasal 1\na.\n\nPasal 2\nb.", encoding="utf-8")

    translation_dir = run_dir / "results" / "translation"
    translation_dir.mkdir(parents=True)
    original_merged = "Article 1\na.\n\nArticle 2\nb."
    (translation_dir / "D.txt").write_text(original_merged, encoding="utf-8")

    chunks_dir = run_dir / "mid_product" / "chunks"
    records = [
        {
            "chunk_index": 0,
            "section_id": 0,
            "type": "structural",
            "context": None,
            "text": "Pasal 1\na.",
            "translated_text": "Article 1\na.",
        },
        {
            "chunk_index": 0,
            "section_id": 1,
            "type": "structural",
            "context": None,
            "text": "Pasal 2\nb.",
            "translated_text": "Article 2\nb.",
        },
    ]
    original_chunks_json = json.dumps(records)
    _write_chunks(chunks_dir, "D", records)

    call_count = {"n": 0}

    def complete_fn(prompt, *_):
        call_count["n"] += 1
        if call_count["n"] == 2:
            raise RuntimeError("simulated LLM failure")
        return {"action_required": False, "issues": [], "corrected_text": None}

    wrapper = _make_wrapper(complete_fn)
    limiter = ConcurrencyLimiter(max_workers=1, enabled=True)

    result = run_translation_qa_step(run_id=run_id, config=config, wrapper=wrapper, limiter=limiter)

    assert result["counts"]["failed"] == 1
    assert result["counts"]["succeeded"] == 0
    assert not (run_dir / "results" / "translation_qa" / "D.json").exists()
    # Nothing written: both artifacts byte-for-byte unchanged.
    assert (translation_dir / "D.txt").read_text(encoding="utf-8") == original_merged
    assert (chunks_dir / "D.chunks.json").read_text(encoding="utf-8") == original_chunks_json

    failures = json.loads((run_dir / "failures.json").read_text(encoding="utf-8"))
    assert failures["translation_qa"][0]["filename"] == "D.txt"


def test_skips_existing_report_by_default(config, data_root):
    run_id = "run_skip"
    run_dir = data_root / run_id
    (config.paths.input_dir / "E.txt").write_text("Pasal 1\na.", encoding="utf-8")

    translation_dir = run_dir / "results" / "translation"
    translation_dir.mkdir(parents=True)
    (translation_dir / "E.txt").write_text("Article 1\na.", encoding="utf-8")

    qa_dir = run_dir / "results" / "translation_qa"
    qa_dir.mkdir(parents=True)
    (qa_dir / "E.json").write_text(json.dumps({"policy_file": "E.txt"}), encoding="utf-8")

    wrapper = _make_wrapper(
        lambda *a, **k: {"action_required": False, "issues": [], "corrected_text": None}
    )
    limiter = ConcurrencyLimiter(max_workers=2, enabled=True)

    result = run_translation_qa_step(run_id=run_id, config=config, wrapper=wrapper, limiter=limiter)

    assert result["counts"]["skipped"] == 1
    assert result["counts"]["succeeded"] == 0
    wrapper.complete_structured.assert_not_called()


def test_force_reruns_existing_report(config, data_root):
    run_id = "run_force"
    run_dir = data_root / run_id
    (config.paths.input_dir / "F.txt").write_text("Pasal 1\na.", encoding="utf-8")

    translation_dir = run_dir / "results" / "translation"
    translation_dir.mkdir(parents=True)
    (translation_dir / "F.txt").write_text("Article 1\na.", encoding="utf-8")

    qa_dir = run_dir / "results" / "translation_qa"
    qa_dir.mkdir(parents=True)
    (qa_dir / "F.json").write_text(json.dumps({"policy_file": "F.txt"}), encoding="utf-8")

    wrapper = _make_wrapper(
        lambda *a, **k: {"action_required": False, "issues": [], "corrected_text": None}
    )
    limiter = ConcurrencyLimiter(max_workers=2, enabled=True)

    result = run_translation_qa_step(
        run_id=run_id, config=config, wrapper=wrapper, limiter=limiter, force=True
    )

    assert result["counts"]["skipped"] == 0
    assert result["counts"]["succeeded"] == 1
    wrapper.complete_structured.assert_called_once()


def test_missing_translation_output_is_skipped_not_failed(config, data_root):
    run_id = "run_missing_translation"
    (config.paths.input_dir / "G.txt").write_text("Pasal 1\na.", encoding="utf-8")
    # No results/translation/G.txt written at all — translation hasn't run yet.

    wrapper = _make_wrapper(
        lambda *a, **k: {"action_required": False, "issues": [], "corrected_text": None}
    )
    limiter = ConcurrencyLimiter(max_workers=2, enabled=True)

    result = run_translation_qa_step(run_id=run_id, config=config, wrapper=wrapper, limiter=limiter)

    assert result["counts"]["skipped"] == 1
    assert result["counts"]["failed"] == 0
    wrapper.complete_structured.assert_not_called()


def test_action_required_without_corrected_text_retries_then_succeeds(config, data_root):
    run_id = "run_retry_succeeds"
    run_dir = data_root / run_id
    (config.paths.input_dir / "H.txt").write_text("Pasal 1\na.", encoding="utf-8")

    translation_dir = run_dir / "results" / "translation"
    translation_dir.mkdir(parents=True)
    (translation_dir / "H.txt").write_text("Article 1\n[Note] a.", encoding="utf-8")

    wrapper = _make_wrapper(
        [
            {"action_required": True, "issues": ["leftover note"], "corrected_text": None},
            {"action_required": True, "issues": ["leftover note"], "corrected_text": "Article 1\na."},
        ]
    )
    limiter = ConcurrencyLimiter(max_workers=2, enabled=True)

    result = run_translation_qa_step(run_id=run_id, config=config, wrapper=wrapper, limiter=limiter)

    assert result["counts"]["succeeded"] == 1
    assert result["counts"]["failed"] == 0
    assert result["counts"]["corrected"] == 1
    assert result["counts"]["incomplete"] == 0
    assert wrapper.complete_structured.call_count == 2
    assert (translation_dir / "H.txt").read_text(encoding="utf-8") == "Article 1\na."

    report = json.loads(
        (run_dir / "results" / "translation_qa" / "H.json").read_text(encoding="utf-8")
    )
    assert report["items"][0]["response_incomplete"] is False


def test_action_required_without_corrected_text_degrades_after_retry_exhausted(config, data_root):
    run_id = "run_retry_exhausted"
    run_dir = data_root / run_id
    (config.paths.input_dir / "I.txt").write_text("Pasal 1\na.", encoding="utf-8")

    translation_dir = run_dir / "results" / "translation"
    translation_dir.mkdir(parents=True)
    original = "Article 1\na."
    (translation_dir / "I.txt").write_text(original, encoding="utf-8")

    wrapper = _make_wrapper(
        lambda *a, **k: {"action_required": True, "issues": ["x"], "corrected_text": None}
    )
    limiter = ConcurrencyLimiter(max_workers=2, enabled=True)

    result = run_translation_qa_step(run_id=run_id, config=config, wrapper=wrapper, limiter=limiter)

    # Never a hard failure: one chunk's malformed response must not discard
    # everything else this step could have checked in the same file.
    assert result["counts"]["failed"] == 0
    assert result["counts"]["succeeded"] == 1
    assert result["counts"]["corrected"] == 0
    assert result["counts"]["incomplete"] == 1
    assert wrapper.complete_structured.call_count == 2
    assert (translation_dir / "I.txt").read_text(encoding="utf-8") == original

    report = json.loads(
        (run_dir / "results" / "translation_qa" / "I.json").read_text(encoding="utf-8")
    )
    assert report["items"][0]["action_required"] is False
    assert report["items"][0]["response_incomplete"] is True
    assert "kept unchanged" in report["items"][0]["issues"][-1]
