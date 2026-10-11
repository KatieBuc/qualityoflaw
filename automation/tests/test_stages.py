from types import SimpleNamespace

import pytest

from automation.src.breakpoints import PipelineBreakpoint
from automation.src.stages import (
    run_human_review_step,
    run_input_process_step,
    run_promote_step,
    run_txt_to_md_step,
)
from automation.tests.test_txt_to_md_keywords import INDONESIAN

GOOD_SOURCE = "# Doc\n\n## Chapter 1\n\n### Article 1\n\nText.\n\n### Article 2\n\nMore.\n"
SAME_HEADINGS = GOOD_SOURCE.replace("Text.", "Teks.").replace("More.", "Lagi.")
FEWER_HEADINGS = "# Doc\n\n## Chapter 1\n\n### Article 1\n\nOnly one article.\n"


def make_config(tmp_path, *, pause=False, language="Indonesian"):
    root = tmp_path / "data"
    raw = root / "raw"
    pipeline = tmp_path / "pipeline.yaml"
    pipeline.write_text(
        "project: demo\n"
        f"human_review:\n  pause: {str(pause).lower()}\n"
        "input_process:\n"
        f"  language: {language}\n"
        f"  raw_files_dir: {(raw / 'raw_files').as_posix()}\n"
        f"  extracted_txt_dir: {(raw / 'extracted_txt').as_posix()}\n"
        f"  cleaned_txt_dir: {(raw / 'cleaned_txt').as_posix()}\n"
        f"  gold_dir: {(raw / 'gold').as_posix()}\n"
        f"  report_path: {(raw / 'input_report.json').as_posix()}\n",
        encoding="utf-8",
    )
    paths = SimpleNamespace(
        small_scale_stems=("A",),
        markdown_input_dir=root / "preprocessed" / "cleaned_markdown",
        markdown_input_suffix=".cleaned.md",
        translation_markdown_dir=root / "preprocessed" / "translation_markdown",
        processed_cleaned_markdown_dir=root / "processed" / "cleaned_markdown",
        processed_translation_markdown_dir=root / "processed" / "translation_markdown",
    )
    for d in (paths.markdown_input_dir, paths.translation_markdown_dir):
        d.mkdir(parents=True)
    return SimpleNamespace(pipeline_config_path=pipeline, paths=paths, raw=raw)


def put_pair(config, stem, source=GOOD_SOURCE, translation=SAME_HEADINGS):
    (config.paths.markdown_input_dir / f"{stem}.cleaned.md").write_text(source, encoding="utf-8")
    (config.paths.translation_markdown_dir / f"{stem}.md").write_text(translation, encoding="utf-8")


# --- promote -------------------------------------------------------------- #
def test_promote_moves_matching_pairs_and_pauses_for_the_others(tmp_path):
    config = make_config(tmp_path)
    put_pair(config, "A")
    put_pair(config, "B", translation=FEWER_HEADINGS)

    with pytest.raises(PipelineBreakpoint) as stop:
        run_promote_step(config)

    assert (config.paths.processed_cleaned_markdown_dir / "A.cleaned.md").exists()
    assert (config.paths.processed_translation_markdown_dir / "A.md").exists()
    assert not (config.paths.markdown_input_dir / "A.cleaned.md").exists()
    assert (config.paths.markdown_input_dir / "B.cleaned.md").exists()          # the blocked pair stays
    text = stop.value.format()
    assert "B: heading" in text and "--skip-blocked" in text and "automation/docs/pipeline.md" in text


def test_promote_with_skip_blocked_goes_on_and_a_fixed_pair_follows_later(tmp_path):
    config = make_config(tmp_path)
    put_pair(config, "A")
    put_pair(config, "B", translation=FEWER_HEADINGS)

    result = run_promote_step(config, skip_blocked=True)
    assert result["counts"]["succeeded"] == 1 and result["counts"]["failed"] == 1

    (config.paths.translation_markdown_dir / "B.md").write_text(SAME_HEADINGS, encoding="utf-8")
    again = run_promote_step(config)
    assert again["counts"]["succeeded"] == 1 and again["counts"]["skipped"] == 1       # A is already there
    assert not list(config.paths.translation_markdown_dir.glob("*.md"))


def test_promote_never_overwrites_a_pair_already_in_processed(tmp_path):
    config = make_config(tmp_path)
    put_pair(config, "A")
    run_promote_step(config)
    done = config.paths.processed_translation_markdown_dir / "A.md"
    done.write_text(SAME_HEADINGS + "\nedited\n", encoding="utf-8")

    put_pair(config, "A")                      # a stale copy reappears in preprocessed/
    with pytest.raises(PipelineBreakpoint):
        run_promote_step(config)
    assert "edited" in done.read_text(encoding="utf-8")


def test_promote_small_scale_only_touches_the_listed_policies(tmp_path):
    config = make_config(tmp_path)
    put_pair(config, "A")
    put_pair(config, "Z")
    run_promote_step(config, small_scale=True)
    assert (config.paths.markdown_input_dir / "Z.cleaned.md").exists()


def test_promote_with_nothing_anywhere_is_an_error_not_a_pause(tmp_path):
    with pytest.raises(FileNotFoundError, match="Nothing to evaluate"):
        run_promote_step(make_config(tmp_path))


# --- human review --------------------------------------------------------- #
def test_human_review_is_off_by_default(tmp_path):
    config = make_config(tmp_path)
    put_pair(config, "A")
    assert run_human_review_step(config)["skipped_reason"] == "human_review.pause is off"


def test_human_review_pauses_until_reviewed(tmp_path):
    config = make_config(tmp_path, pause=True)
    put_pair(config, "A")

    with pytest.raises(PipelineBreakpoint) as stop:
        run_human_review_step(config)
    assert "--reviewed" in stop.value.format() and "same level" in stop.value.format()

    assert run_human_review_step(config, reviewed=True)["counts"]["succeeded"] == 1


def test_human_review_does_not_pause_when_nothing_is_waiting(tmp_path):
    config = make_config(tmp_path, pause=True)
    assert run_human_review_step(config)["skipped_reason"] == "nothing waiting for review"


# --- input process -------------------------------------------------------- #
def test_input_process_pauses_for_an_unusable_file_and_lists_next_steps(tmp_path):
    config = make_config(tmp_path)
    originals = config.raw / "raw_files"
    originals.mkdir(parents=True)
    (originals / "TOO_SHORT.txt").write_text("abc", encoding="utf-8")

    with pytest.raises(PipelineBreakpoint) as stop:
        run_input_process_step(config)

    text = stop.value.format()
    assert "eliminated: TOO_SHORT" in text and "__txt.txt" in text and "__ocr.txt" in text
    assert "automation/docs/input_process.md" in text and "--skip-blocked" in text
    assert (config.raw / "input_report.json").exists()


def test_input_process_goes_on_past_unresolved_files_with_skip_blocked(tmp_path):
    config = make_config(tmp_path)
    originals = config.raw / "raw_files"
    originals.mkdir(parents=True)
    (originals / "TOO_SHORT.txt").write_text("abc", encoding="utf-8")

    result = run_input_process_step(config, skip_blocked=True)

    assert result["counts"]["eliminated"] == 1


def test_input_process_is_skipped_without_an_originals_folder(tmp_path):
    config = make_config(tmp_path)
    assert "no originals folder" in run_input_process_step(config)["skipped_reason"]


# --- txt_to_md ------------------------------------------------------------ #
def test_txt_to_md_converts_into_the_pipeline_input_folder(tmp_path):
    config = make_config(tmp_path)
    cleaned = config.raw / "cleaned_txt"
    cleaned.mkdir(parents=True)
    (cleaned / "A.txt").write_text(INDONESIAN, encoding="utf-8")

    result = run_txt_to_md_step(config)

    assert result["counts"]["succeeded"] == 1
    assert (config.paths.markdown_input_dir / "A.cleaned.md").exists()


def test_txt_to_md_does_not_redo_a_policy_that_is_already_in_processed(tmp_path):
    config = make_config(tmp_path)
    cleaned = config.raw / "cleaned_txt"
    cleaned.mkdir(parents=True)
    (cleaned / "A.txt").write_text(INDONESIAN, encoding="utf-8")
    config.paths.processed_cleaned_markdown_dir.mkdir(parents=True)
    (config.paths.processed_cleaned_markdown_dir / "A.cleaned.md").write_text(GOOD_SOURCE, encoding="utf-8")

    result = run_txt_to_md_step(config)

    assert result["counts"]["skipped"] == 1 and not (config.paths.markdown_input_dir / "A.cleaned.md").exists()


def test_txt_to_md_is_skipped_without_a_text_folder(tmp_path):
    assert "no text folder" in run_txt_to_md_step(make_config(tmp_path))["skipped_reason"]
