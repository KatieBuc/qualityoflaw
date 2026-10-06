"""Golden test for run_pipeline.main(): step order, args passed to each step,
metadata.json / failures.json contents, stdout/stderr and exit code.

All step functions and network clients are faked, so the test pins down the
orchestration behaviour only. It must stay green, byte for byte, across any
refactor of run_pipeline.py. Regenerate deliberately with UPDATE_GOLDEN=1.
"""

import dataclasses
import json
import re
import sys
from pathlib import Path

import pytest

from automation.src import failure_log
from automation.src.config_loader import load_pipeline_config
from automation.src.constants import DEFAULT_MODEL_CONFIG, DEFAULT_PIPELINE_CONFIG

from ._serialize import assert_matches_golden

RUN_ID = "RUN1"

TOKENS = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}


def _translation_result(**over):
    r = {
        "counts": {"total": 3, "succeeded": 3, "skipped": 0, "failed": 0},
        "failed_files": [],
        "elapsed_s": 1.5,
        "token_usage": TOKENS,
    }
    r.update(over)
    return r


def _qa_result(**over):
    r = {
        "counts": {
            "total": 3, "succeeded": 3, "corrected": 1, "incomplete": 0,
            "not_converged": 0, "clauses_lost": 0, "qa_passes": 4, "skipped": 0, "failed": 0,
        },
        "failed_files": [],
        "elapsed_s": 0.5,
        "token_usage": TOKENS,
    }
    r.update(over)
    return r


def _to_text_result(**over):
    r = {
        "counts": {
            "total": 3, "succeeded": 3, "skipped": 0, "failed": 0,
            "clauses_lost": 0, "repaired": 1, "source_unaligned": 2,
        },
        "failed_files": [],
        "elapsed_s": 0.1,
    }
    r.update(over)
    return r


def _markdown_result(**over):
    r = {
        "counts": {"total": 3, "succeeded": 3, "skipped": 0, "failed": 0},
        "failed_files": [],
        "elapsed_s": 0.2,
    }
    r.update(over)
    return r


def _storage_result(**over):
    r = {
        "counts": {"total": 3, "succeeded": 3, "skipped": 0, "failed": 0},
        "failed_files": [],
        "elapsed_s": 0.4,
    }
    r.update(over)
    return r


def _eval_result(**over):
    r = {
        "counts": {"total": 3, "succeeded": 3, "failed": 0, "saved_reports": 3},
        "failed_policies": [],
        "elapsed_s": 2.0,
        "token_usage": TOKENS,
    }
    r.update(over)
    return r


def _comparison_result(**over):
    r = {
        "counts": {"matched_pairs": 280, "evaluated_policies": 3, "accuracy": 0.8213},
        "evaluated_policy_files": ["A.txt", "B.txt"],
        "elapsed_s": 0.1,
        "metrics_path": "metrics.csv",
    }
    r.update(over)
    return r


def _diagnosis_result(**over):
    r = {
        "counts": {
            "total": 2, "succeeded": 2, "failed": 0, "skipped": 0,
            "saved_reports": 2, "discrepancies_total": 7,
        },
        "failed_policies": [],
        "elapsed_s": 0.3,
        "token_usage": TOKENS,
    }
    r.update(over)
    return r


DEFAULT_RESULTS = {
    "run_md_translation_step": _translation_result,
    "run_md_translation_qa_step": _qa_result,
    "run_md_to_text_step": _to_text_result,
    "run_translation_step": _translation_result,
    "run_translation_qa_step": lambda **o: {
        **_qa_result(), "counts": {
            "total": 3, "succeeded": 3, "corrected": 1, "incomplete": 0, "skipped": 0, "failed": 0
        }, **o,
    },
    "run_markdown_step": _markdown_result,
    "run_storage_step": _storage_result,
    "run_evaluation_step": _eval_result,
    "run_sliding_window_evaluation_step": _eval_result,
    "run_comparison_step": _comparison_result,
    "run_diagnosis_step": _diagnosis_result,
}

STEP_FUNCS = tuple(DEFAULT_RESULTS)

FAILED_FILE = {
    "filename": "X.md",
    "error_type": "RateLimit",
    "message": "boom",
    "details": {"status_code": 429},
    "artifact": "chunks",
}


class _FakeWrapper:
    @classmethod
    def from_profile(cls, profile, limiter=None, confidence=None):
        inst = cls()
        inst.tag = f"wrapper:{profile.name}:confidence={confidence is not None}"
        return inst


class _FakeEmbedder:
    @classmethod
    def from_env(cls, batch_size=None):
        inst = cls()
        inst.tag = f"embedder:{batch_size}"
        return inst


def _tag(value):
    if value is None:
        return None
    return getattr(value, "tag", type(value).__name__)


# Each scenario: argv (without program name), optional overrides.
#   config: dict of top-level ResolvedPipelineConfig fields to replace
#   storage_enabled / evaluation_method: convenience overrides
#   results: {step_func_name: result dict | Exception}
#   files: files to create before the run (relative to the data root run dir)
#   prelude: argv of an earlier invocation whose state the scenario builds on
#   side_effects: {step_func_name: callable(run_id)} run inside the fake step
def _record_eval_failure(run_id):
    failure_log.record_failure(
        run_id, "evaluation", {"policy_file": "B.txt", "message": "schema mismatch"}
    )


def _record_translation_failure(run_id):
    failure_log.record_failure(
        run_id, "translation", {"filename": "X.md", "error_type": "E", "message": "bad"}
    )


SCENARIOS = {
    "default_chain": {"argv": []},
    "default_chain_small_scale_force": {"argv": ["--small-scale", "--force"]},
    "all_steps_in_shuffled_order": {
        "argv": [
            "--steps",
            "discrepancy_diagnosis,comparison,evaluation,storage,markdown,translation_qa,"
            "translation,md_to_text,translation_qa_md,translation_md",
            "--allow-partial",
        ]
    },
    "sliding_window_method": {
        "argv": ["--steps", "translation_md,md_to_text,storage,evaluation,comparison"],
        "evaluation_method": "sliding_window",
    },
    "storage_disabled": {
        "argv": ["--steps", "md_to_text,storage,evaluation"],
        "storage_enabled": False,
        "files": ["results/translation_markdown/a.md"],
    },
    "translation_md_raises_stops_pipeline": {
        "argv": [],
        "results": {"run_md_translation_step": RuntimeError("no markdown input")},
    },
    "md_to_text_raises_stops_pipeline": {
        "argv": [],
        "results": {"run_md_to_text_step": FileNotFoundError("missing chunks")},
    },
    "translation_md_failed_files": {
        "argv": ["--steps", "translation_md,md_to_text"],
        "results": {
            "run_md_translation_step": _translation_result(
                counts={"total": 3, "succeeded": 2, "skipped": 0, "failed": 1},
                failed_files=[FAILED_FILE],
            )
        },
    },
    "qa_md_failed_and_lost": {
        "argv": ["--steps", "translation_md,translation_qa_md,md_to_text"],
        "results": {
            "run_md_translation_qa_step": _qa_result(
                counts={
                    "total": 3, "succeeded": 2, "corrected": 1, "incomplete": 1,
                    "not_converged": 1, "clauses_lost": 1, "qa_passes": 5, "skipped": 0,
                    "failed": 1,
                },
                failed_files=[FAILED_FILE],
            )
        },
    },
    "md_to_text_failed_files": {
        "argv": ["--steps", "translation_md,md_to_text"],
        "results": {
            "run_md_to_text_step": _to_text_result(
                counts={
                    "total": 3, "succeeded": 2, "skipped": 0, "failed": 1,
                    "clauses_lost": 1, "repaired": 0,
                },
                failed_files=[FAILED_FILE],
            )
        },
    },
    "storage_failed_files": {
        "argv": ["--steps", "md_to_text,storage"],
        "files": ["results/translation_markdown/a.md"],
        "results": {
            "run_storage_step": _storage_result(
                counts={"total": 3, "succeeded": 2, "skipped": 0, "failed": 1},
                failed_files=[FAILED_FILE],
            )
        },
    },
    "evaluation_failed_policies_strict": {
        "argv": ["--steps", "md_to_text,storage,evaluation,comparison"],
        "results": {
            "run_evaluation_step": _eval_result(
                counts={"total": 3, "succeeded": 2, "failed": 1, "saved_reports": 2, "skipped": 1},
                failed_policies=["B.txt"],
            )
        },
        "side_effects": {"run_evaluation_step": _record_eval_failure},
        "files": ["results/translation_markdown/a.md"],
    },
    "evaluation_failed_policies_allow_partial": {
        "argv": ["--steps", "md_to_text,storage,evaluation,comparison", "--allow-partial"],
        "results": {
            "run_evaluation_step": _eval_result(
                counts={"total": 3, "succeeded": 2, "failed": 1, "saved_reports": 2},
                failed_policies=["B.txt"],
            )
        },
        "side_effects": {"run_evaluation_step": _record_eval_failure},
        "files": ["results/translation_markdown/a.md"],
    },
    "translation_legacy_failed_with_logged_failure": {
        "argv": ["--steps", "translation,translation_qa,markdown"],
        "results": {
            "run_translation_step": _translation_result(
                counts={"total": 3, "succeeded": 2, "skipped": 0, "failed": 1},
                failed_files=[FAILED_FILE],
            ),
            "run_translation_qa_step": {
                "counts": {
                    "total": 3, "succeeded": 2, "corrected": 1, "incomplete": 1,
                    "skipped": 0, "failed": 1,
                },
                "failed_files": [FAILED_FILE],
                "elapsed_s": 0.5,
                "token_usage": TOKENS,
            },
            "run_markdown_step": _markdown_result(
                counts={"total": 3, "succeeded": 2, "skipped": 0, "failed": 1},
                failed_files=[FAILED_FILE],
            ),
        },
        "side_effects": {"run_translation_step": _record_translation_failure},
    },
    "comparison_without_accuracy": {
        "argv": ["--steps", "md_to_text,storage,evaluation,comparison"],
        "results": {
            "run_comparison_step": _comparison_result(
                counts={"matched_pairs": 0, "evaluated_policies": 0}
            )
        },
        "files": ["results/translation_markdown/a.md"],
    },
    "comparison_raises": {
        "argv": ["--steps", "md_to_text,storage,evaluation,comparison"],
        "results": {"run_comparison_step": ValueError("no golden rows")},
        "files": ["results/translation_markdown/a.md"],
    },
    "diagnosis_nothing_to_diagnose": {
        "argv": ["--steps", "md_to_text,storage,evaluation,comparison,discrepancy_diagnosis"],
        "results": {
            "run_diagnosis_step": _diagnosis_result(
                counts={
                    "total": 0, "succeeded": 0, "failed": 0, "skipped": 0,
                    "saved_reports": 0, "discrepancies_total": 0,
                },
                skipped_reason="all policies match golden",
            )
        },
        "files": ["results/translation_markdown/a.md"],
    },
    "diagnosis_nothing_default_reason": {
        "argv": ["--steps", "md_to_text,storage,evaluation,comparison,discrepancy_diagnosis"],
        "results": {
            "run_diagnosis_step": _diagnosis_result(
                counts={
                    "total": 0, "succeeded": 0, "failed": 0, "skipped": 0,
                    "saved_reports": 0, "discrepancies_total": 0,
                },
            )
        },
        "files": ["results/translation_markdown/a.md"],
    },
    "diagnosis_failed_policies_strict": {
        "argv": ["--steps", "md_to_text,storage,evaluation,comparison,discrepancy_diagnosis"],
        "results": {
            "run_diagnosis_step": _diagnosis_result(
                counts={
                    "total": 2, "succeeded": 1, "failed": 1, "skipped": 1,
                    "saved_reports": 1, "discrepancies_total": 5,
                },
                failed_policies=["A.txt"],
            )
        },
        "files": ["results/translation_markdown/a.md"],
    },
    "resume_existing_run_eval_only": {
        "prelude": [],
        "argv": ["--run-id", RUN_ID, "--steps", "evaluation,comparison"],
        "files": [
            "results/translation/a.txt",
            "mid_product/rag_store/a.json",
        ],
    },
    "resume_existing_run_missing_prereq": {
        "prelude": ["--steps", "translation_md"],
        "argv": ["--run-id", RUN_ID, "--steps", "evaluation"],
    },
    "new_run_id_with_legacy_translation_creates_run": {
        "argv": ["--run-id", "BRANDNEW", "--steps", "translation"],
    },
    "new_run_id_with_translation_md": {
        "argv": ["--run-id", "BRANDNEW", "--steps", "translation_md"],
    },
    "new_run_id_eval_only_errors": {
        "argv": ["--run-id", "BRANDNEW", "--steps", "evaluation"],
    },
    "eval_only_without_run_id_errors": {"argv": ["--steps", "evaluation"]},
    "invalid_step_errors": {"argv": ["--steps", "bogus"]},
    "max_workers_override": {"argv": ["--max-workers", "2", "--steps", "translation_md"]},
    "no_concurrency": {"argv": ["--no-concurrency", "--steps", "translation_md"]},
    "max_workers_invalid": {"argv": ["--max-workers", "0", "--steps", "translation_md"]},
}


# Scenarios whose --steps omit a translation step must resume an existing run.
for _name, _scn in SCENARIOS.items():
    argv = _scn["argv"]
    if _name != "eval_only_without_run_id_errors" and "--steps" in argv and "--run-id" not in argv:
        steps = argv[argv.index("--steps") + 1].split(",")
        if "translation_md" not in steps and "translation" not in steps:
            _scn["prelude"] = ["--steps", "translation_md"]
            _scn["argv"] = ["--run-id", RUN_ID, *argv]


def _normalize(text: str, tmp_path: Path) -> str:
    text = text.replace(str(tmp_path), "<tmp>")
    text = text.replace(str(tmp_path).replace("\\", "/"), "<tmp>")
    text = text.replace("\\", "/")
    return text


def _scrub(obj):
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in ("created_at", "updated_at", "at"):
                out[k] = "<ts>" if v else v
            elif k == "total" and isinstance(v, float):
                out[k] = "<elapsed>"
            else:
                out[k] = _scrub(v)
        return out
    if isinstance(obj, list):
        return [_scrub(v) for v in obj]
    return obj


def _read_json(path: Path):
    if not path.exists():
        return None
    return _scrub(json.loads(path.read_text(encoding="utf-8")))


@pytest.fixture
def harness(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("automation.src.config_loader.DEFAULT_DATA_ROOT", tmp_path)
    monkeypatch.setattr("automation.src.run_pipeline.generate_run_id", lambda: RUN_ID)
    monkeypatch.setattr("automation.src.run_pipeline.AzureLLMWrapper", _FakeWrapper)
    monkeypatch.setattr("automation.src.run_pipeline.AzureEmbedder", _FakeEmbedder)
    monkeypatch.setattr("automation.src.run_pipeline.get_cohere_rerank_endpoint", lambda: "x")
    monkeypatch.setattr("automation.src.run_pipeline.get_cohere_rerank_api_key", lambda: "y")

    state = {"calls": [], "results": {}, "side_effects": {}, "config_over": {}}

    def make_fake(name):
        def fake(**kwargs):
            call = {"fn": name}
            for key in ("run_id", "small_scale", "force", "allow_partial"):
                if key in kwargs:
                    call[key] = kwargs[key]
            call["wrapper"] = _tag(kwargs.get("wrapper"))
            call["embedder"] = _tag(kwargs.get("embedder")) if "embedder" in kwargs else "absent"
            call["limiter"] = (
                {
                    "enabled": kwargs["limiter"].enabled,
                    "max_workers": kwargs["limiter"].max_workers,
                }
                if kwargs.get("limiter") is not None
                else None
            )
            state["calls"].append(call)
            effect = state["side_effects"].get(name)
            if effect:
                effect(kwargs["run_id"])
            outcome = state["results"].get(name)
            if outcome is None:
                outcome = DEFAULT_RESULTS[name]()
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        return fake

    for name in STEP_FUNCS:
        monkeypatch.setattr(f"automation.src.run_pipeline.{name}", make_fake(name))

    real_load = load_pipeline_config

    def patched_load(pipeline_config_path=None, model_config_path=None):
        cfg = real_load(DEFAULT_PIPELINE_CONFIG, DEFAULT_MODEL_CONFIG)
        return dataclasses.replace(cfg, **state["config_over"])

    monkeypatch.setattr("automation.src.run_pipeline.load_pipeline_config", patched_load)

    def run(argv):
        from automation.src import run_pipeline

        monkeypatch.setattr(sys, "argv", ["run_pipeline", *argv])
        capsys.readouterr()
        try:
            run_pipeline.main()
            code = 0
        except SystemExit as exc:
            code = exc.code if exc.code is not None else 0
        captured = capsys.readouterr()
        return code, captured.out, captured.err

    return state, run


def _apply_scenario_config(state, scenario):
    over = {}
    from automation.src.config_loader import load_pipeline_config as real

    base = real(DEFAULT_PIPELINE_CONFIG, DEFAULT_MODEL_CONFIG)
    if "storage_enabled" in scenario:
        over["storage"] = dataclasses.replace(base.storage, enabled=scenario["storage_enabled"])
    if "evaluation_method" in scenario:
        over["evaluation_method"] = scenario["evaluation_method"]
    state["config_over"] = over


def _run_scenario(name, scenario, tmp_path, state, run):
    if "prelude" in scenario:
        run(scenario["prelude"])
        state["calls"].clear()
    for rel in scenario.get("files", []):
        run_dir = tmp_path / scenario.get("run_id_dir", RUN_ID)
        path = run_dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x", encoding="utf-8")
    _apply_scenario_config(state, scenario)
    state["results"] = scenario.get("results", {})
    state["side_effects"] = scenario.get("side_effects", {})
    code, out, err = run(scenario["argv"])

    run_ids = [p.name for p in sorted(tmp_path.iterdir()) if p.is_dir()]
    runs = {}
    for run_id in run_ids:
        run_dir = tmp_path / run_id
        runs[run_id] = {
            "files": sorted(
                p.relative_to(run_dir).as_posix() for p in run_dir.rglob("*") if p.is_file()
            ),
            "metadata": _read_json(run_dir / "metadata.json"),
            "failures": _read_json(run_dir / "failures.json"),
        }
    return {
        "exit_code": code,
        "stdout": _normalize(out, tmp_path),
        "stderr": _normalize(err, tmp_path),
        "calls": state["calls"],
        "runs": runs,
    }


@pytest.mark.parametrize("name", list(SCENARIOS))
def test_main_flow_matches_golden(name, harness, tmp_path):
    state, run = harness
    actual = _run_scenario(name, SCENARIOS[name], tmp_path, state, run)
    assert_matches_golden(f"main_flow/{name}", actual)
