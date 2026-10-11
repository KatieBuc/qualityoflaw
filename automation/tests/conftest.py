import pytest

_SKIPPED = {"counts": {"total": 0, "skipped": 1}, "elapsed_s": 0.0, "skipped_reason": "not under test"}


@pytest.fixture(autouse=True)
def _stage_1_and_3_steps_do_nothing_by_default(request, monkeypatch):
    """The default chain now starts with input_process / txt_to_md / human_review / promote, which look at the
    real project folders. Tests of `run_pipeline.main` that are about other steps get inert versions; tests of the
    steps themselves call `automation.src.stages` directly (mark: `real_stages` keeps the real functions)."""
    if request.node.get_closest_marker("real_stages"):
        return
    for name in ("run_input_process_step", "run_txt_to_md_step", "run_human_review_step", "run_promote_step"):
        monkeypatch.setattr(f"automation.src.run_pipeline.{name}", lambda *a, **k: dict(_SKIPPED), raising=False)
