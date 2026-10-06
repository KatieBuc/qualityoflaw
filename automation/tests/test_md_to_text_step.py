"""The `md_to_text` step, and the guarantee that it keeps evidence retrieval
where it was.

`rag.store` is deliberately unmodified by the Markdown path, so the contract
it already enforces -- `_load_translation_chunks` -- is what these tests
assert against directly, rather than a copy of its rules.
"""

import json

import pytest

from automation.src.markdown.to_text import run_md_to_text_step
from automation.src.rag.store import _load_translation_chunks
from automation.tests.test_markdown_translate import make_config

SOURCE_MD = """# BAB I

## KETENTUAN UMUM

#### Pasal 1

Isi pasal satu selesai.

#### Pasal 2

Isi pasal dua selesai.

# BAB II

#### Pasal 3

Isi pasal tiga selesai.
"""

TRANSLATED_MD = """# CHAPTER I

## GENERAL PROVISIONS

#### Article 1

Content of article one is complete.

#### Article 2

Content of article two is complete.

# CHAPTER II

#### Article 3

Content of article three is complete.
"""


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    monkeypatch.setattr("automation.src.config_loader.DEFAULT_DATA_ROOT", tmp_path)
    return tmp_path


RUN_ID = "md_run"


@pytest.fixture
def run_dir(tmp_path, data_root):
    run_dir = data_root / RUN_ID
    run_dir.mkdir(parents=True)
    (tmp_path / "translation_out").mkdir()
    (tmp_path / "markdown_input").mkdir()
    return run_dir


def seed(run_dir, translated_md=TRANSLATED_MD, source_md=SOURCE_MD, stem="ACEH_BIREUEN"):
    root = run_dir.parent
    (root / "translation_out" / f"{stem}.md").write_text(translated_md, encoding="utf-8")
    if source_md is not None:
        (root / "markdown_input" / f"{stem}.cleaned.md").write_text(
            source_md, encoding="utf-8"
        )


def chunks_of(run_dir, stem="ACEH_BIREUEN"):
    path = run_dir / "mid_product" / "chunks" / f"{stem}.chunks.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_writes_the_plain_text_artifact_downstream_globs_for(tmp_path, data_root, run_dir):
    seed(run_dir)
    result = run_md_to_text_step(RUN_ID, make_config(tmp_path, target_chars=8000))

    assert result["counts"]["succeeded"] == 1
    text_path = run_dir / "results" / "translation" / "ACEH_BIREUEN.txt"
    assert text_path.exists()

    text = text_path.read_text(encoding="utf-8")
    assert "#" not in text
    assert "Article 1" in text
    assert "Content of article one is complete." in text


def test_retrieval_chunks_are_per_heading_section_not_per_packed_unit(
    tmp_path, data_root, run_dir
):
    seed(run_dir)
    run_md_to_text_step(RUN_ID, make_config(tmp_path, target_chars=8000))

    # The whole document is one packed translation unit, but retrieval must
    # still see one chunk per heading section.
    records = chunks_of(run_dir)
    assert len(records) == 3
    assert [r["heading"] for r in records] == ["CHAPTER I", "Article 2", "CHAPTER II"]


def test_chunks_json_validates_against_rag_store_unchanged(tmp_path, data_root, run_dir):
    seed(run_dir)
    run_md_to_text_step(RUN_ID, make_config(tmp_path, target_chars=8000))

    path = run_dir / "mid_product" / "chunks" / "ACEH_BIREUEN.chunks.json"
    loaded = _load_translation_chunks(path)

    # Not None means storage takes its preferred path -- reusing these chunks
    # rather than falling back to regex-chunking the English text.
    assert loaded is not None
    assert len(loaded) == 3
    assert [c["chunk_id"] for c in loaded] == [0, 1, 2]
    assert all(c["text"] for c in loaded)


def test_source_and_translation_are_paired_when_structure_holds(tmp_path, data_root, run_dir):
    seed(run_dir)
    run_md_to_text_step(RUN_ID, make_config(tmp_path, target_chars=8000))

    records = chunks_of(run_dir)
    assert "Pasal 2" in records[1]["text"]
    assert "Article 2" in records[1]["translated_text"]


def failure_entry(run_dir):
    return json.loads((run_dir / "failures.json").read_text(encoding="utf-8"))["translation"][0]


def test_a_dropped_heading_fails_validation_and_writes_nothing(tmp_path, data_root, run_dir):
    # Stage 4 gate: heading count and hierarchy must match between the pair.
    seed(run_dir, translated_md=TRANSLATED_MD.replace("#### Article 2\n\n", ""))
    result = run_md_to_text_step(RUN_ID, make_config(tmp_path, target_chars=8000))

    assert result["counts"]["failed"] == 1
    assert result["counts"]["succeeded"] == 0
    assert not (run_dir / "results" / "translation" / "ACEH_BIREUEN.txt").exists()
    assert not (run_dir / "mid_product" / "chunks" / "ACEH_BIREUEN.chunks.json").exists()

    entry = failure_entry(run_dir)
    assert entry["filename"] == "ACEH_BIREUEN.txt"
    assert entry["error_type"] == "HeaderMismatch"
    assert entry["details"]["source_headings"] == 6
    assert entry["details"]["translated_headings"] == 5


def test_an_added_heading_fails_validation(tmp_path, data_root, run_dir):
    seed(
        run_dir,
        translated_md=TRANSLATED_MD + "\n#### Article 9\n\nAn added provision.\n",
    )
    result = run_md_to_text_step(RUN_ID, make_config(tmp_path, target_chars=8000))

    assert result["counts"]["failed"] == 1
    assert failure_entry(run_dir)["error_type"] == "HeaderMismatch"


def test_a_different_heading_level_fails_validation(tmp_path, data_root, run_dir):
    seed(run_dir, translated_md=TRANSLATED_MD.replace("#### Article 2", "### Article 2"))
    result = run_md_to_text_step(RUN_ID, make_config(tmp_path, target_chars=8000))

    assert result["counts"]["failed"] == 1
    assert "hierarchy" in failure_entry(run_dir)["message"]


def test_a_missing_source_file_fails_validation(tmp_path, data_root, run_dir):
    seed(run_dir, source_md=None)
    result = run_md_to_text_step(RUN_ID, make_config(tmp_path, target_chars=8000))

    assert result["counts"]["failed"] == 1
    assert "no source file" in failure_entry(run_dir)["message"]


def test_a_failed_file_removes_its_stale_outputs(tmp_path, data_root, run_dir):
    config = make_config(tmp_path, target_chars=8000)
    seed(run_dir)
    run_md_to_text_step(RUN_ID, config)
    text_path = run_dir / "results" / "translation" / "ACEH_BIREUEN.txt"
    assert text_path.exists()

    seed(run_dir, translated_md=TRANSLATED_MD.replace("#### Article 2\n\n", ""))
    run_md_to_text_step(RUN_ID, config, force=True)
    assert not text_path.exists()


def test_recovering_a_broken_heading_is_not_a_failure(tmp_path, data_root, run_dir):
    # Observed in ACEH_BIREUEN: the corpus has "#### Pasal 3 1" (OCR split
    # the number), and the translation correctly emits "#### Article 31".
    seed(run_dir, source_md=SOURCE_MD.replace("#### Pasal 2", "#### Pasal 3 1"))
    result = run_md_to_text_step(RUN_ID, make_config(tmp_path, target_chars=8000))

    assert result["counts"]["clauses_lost"] == 0
    assert result["counts"]["repaired"] == 1
    assert not (run_dir / "failures.json").exists()


def test_a_resolved_failure_is_cleared_on_re_run(tmp_path, data_root, run_dir):
    # A header mismatch is a recorded failure; fixing the file clears it.
    config = make_config(tmp_path, target_chars=8000)
    seed(
        run_dir,
        translated_md=TRANSLATED_MD + "\n#### Article 9\n\nAn added provision.\n",
    )
    run_md_to_text_step(RUN_ID, config)
    assert json.loads((run_dir / "failures.json").read_text(encoding="utf-8"))["translation"]

    seed(run_dir)
    run_md_to_text_step(RUN_ID, config, force=True)

    # failure_log deletes the file once nothing is left in it.
    assert not (run_dir / "failures.json").exists()


def test_existing_output_is_skipped_unless_forced(tmp_path, data_root, run_dir):
    seed(run_dir)
    config = make_config(tmp_path, target_chars=8000)
    run_md_to_text_step(RUN_ID, config)

    assert run_md_to_text_step(RUN_ID, config)["counts"]["skipped"] == 1
    assert run_md_to_text_step(RUN_ID, config, force=True)["counts"]["succeeded"] == 1


def test_missing_translation_markdown_raises(tmp_path, data_root):
    with pytest.raises(FileNotFoundError):
        run_md_to_text_step("no_such_run", make_config(tmp_path, target_chars=8000))
