import pytest

from automation.src.ingest import IngestNotImplementedError, run_ingest_step


def test_ingest_stub_fails_loudly(tmp_path):
    with pytest.raises(IngestNotImplementedError, match="not implemented"):
        run_ingest_step(tmp_path / "raw", tmp_path / "cleaned_markdown")
