"""Replay `md_to_text` (no LLM) on a committed run and compare with its outputs.

`data/automation/<run>/results/translation/*.txt` and
`mid_product/chunks/*.chunks.json` are what the step produced from that run's
translated + source markdown, so re-running the step on the same inputs must
reproduce them byte for byte. This guards the markdown -> text -> chunks code
path against accidental behaviour changes during refactors.
"""

import shutil

import pytest

from automation.src.config_loader import load_pipeline_config
from automation.src.constants import DEFAULT_MODEL_CONFIG, DEFAULT_PIPELINE_CONFIG, PROJECT_ROOT
from automation.src.markdown.to_text import run_md_to_text_step

REFERENCE_RUN = PROJECT_ROOT / "data" / "automation" / "20260928_095010"
STEMS = [
    "JAWA_BARAT_KARAWANG",
    "JAWA_TIMUR_BANGKALAN",
    "KEPULAUAN_RIAU_KARIMUN",
    "ACEH_BIREUEN",
    "BALI_BADUNG",
]


@pytest.mark.skipif(not REFERENCE_RUN.is_dir(), reason="reference run not available")
def test_md_to_text_reproduces_committed_outputs(tmp_path, monkeypatch):
    monkeypatch.setattr("automation.src.config_loader.OUTPUT_DIR_OVERRIDE", tmp_path)
    run_dir = tmp_path / "REPLAY"
    for stem in STEMS:
        for sub in ("translation_markdown", "source_markdown"):
            src = REFERENCE_RUN / "results" / sub / f"{stem}.md"
            if not src.exists():
                pytest.skip(f"reference file missing: {src}")
            dst = run_dir / "results" / sub / f"{stem}.md"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)

    config = load_pipeline_config(DEFAULT_PIPELINE_CONFIG, DEFAULT_MODEL_CONFIG)
    result = run_md_to_text_step(config)
    assert result["counts"]["failed"] == 0
    assert result["counts"]["total"] == len(STEMS)

    for stem in STEMS:
        new_txt = (run_dir / "results" / "translation" / f"{stem}.txt").read_bytes()
        ref_txt = (REFERENCE_RUN / "results" / "translation" / f"{stem}.txt").read_bytes()
        assert new_txt == ref_txt, f"{stem}.txt differs"
        new_chunks = (run_dir / "mid_product" / "chunks" / f"{stem}.chunks.json").read_bytes()
        ref_chunks = (REFERENCE_RUN / "mid_product" / "chunks" / f"{stem}.chunks.json").read_bytes()
        assert new_chunks == ref_chunks, f"{stem}.chunks.json differs"
