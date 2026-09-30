from automation.src.config_loader import (
    mid_product_dir,
    resolve_mid_product_dir,
    resolve_results_dir,
    results_dir,
)


def test_resolve_results_dir_prefers_new_layout_when_present(tmp_path):
    run_dir = tmp_path / "run"
    new_path = results_dir(run_dir, "translation")
    new_path.mkdir(parents=True)
    (run_dir / "translation").mkdir()  # old flat dir also present — new wins

    assert resolve_results_dir(run_dir, "translation") == new_path


def test_resolve_results_dir_falls_back_to_old_flat_layout(tmp_path):
    run_dir = tmp_path / "run"
    old_path = run_dir / "evaluation"
    old_path.mkdir(parents=True)

    assert resolve_results_dir(run_dir, "evaluation") == old_path


def test_resolve_results_dir_defaults_to_new_layout_when_neither_exists(tmp_path):
    run_dir = tmp_path / "run"

    assert resolve_results_dir(run_dir, "diagnosis") == results_dir(run_dir, "diagnosis")


def test_resolve_mid_product_dir_falls_back_to_old_flat_layout(tmp_path):
    run_dir = tmp_path / "run"
    old_path = run_dir / "rag_candidates"
    old_path.mkdir(parents=True)

    assert resolve_mid_product_dir(run_dir, "rag_candidates") == old_path


def test_resolve_mid_product_dir_defaults_to_new_layout_when_neither_exists(tmp_path):
    run_dir = tmp_path / "run"

    assert resolve_mid_product_dir(run_dir, "chunks") == mid_product_dir(run_dir, "chunks")
