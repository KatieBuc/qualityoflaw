"""The bounded QA convergence loop in `markdown/qa.py`.

The auditor's own verdict drives the loop -- never a structural count, since
heading and list-item totals move whenever a repair lands.
"""

import json

import pytest

from automation.src.markdown.qa import _qa_policy_file, qa_chunk

TEMPLATE = "{{CONTEXT}}{{ORIGINAL_TEXT}}|{{TRANSLATED_TEXT}}"


def responder(*responses):
    """A complete_fn returning each canned QA response in turn."""
    calls = []

    def complete_fn(prompt):
        calls.append(prompt)
        return responses[min(len(calls) - 1, len(responses) - 1)]

    complete_fn.calls = calls
    return complete_fn


def clean():
    return {"action_required": False, "issues": [], "corrected_text": None}


def fix(text, issue="scrambled list markers"):
    return {"action_required": True, "issues": [issue], "corrected_text": text}


def test_a_clean_chunk_costs_one_pass():
    fn = responder(clean())
    outcome = qa_chunk("sumber", "translation", TEMPLATE, fn)

    assert outcome.passes == 1
    assert outcome.converged is True
    assert outcome.corrected is False
    assert outcome.text == "translation"


def test_the_loop_repeats_until_the_auditor_is_satisfied():
    fn = responder(fix("pass 1"), fix("pass 2"), clean())
    outcome = qa_chunk("sumber", "original", TEMPLATE, fn, max_passes=5)

    assert outcome.passes == 3
    assert outcome.converged is True
    assert outcome.corrected is True
    assert outcome.text == "pass 2"


def test_each_pass_re_audits_the_previous_correction():
    fn = responder(fix("first correction"), clean())
    qa_chunk("sumber", "original", TEMPLATE, fn)

    # Pass 2 must check the corrected text, not the original translation,
    # while the Indonesian source stays fixed on both sides.
    assert "|original" in fn.calls[0]
    assert "|first correction" in fn.calls[1]
    assert fn.calls[1].startswith("sumber")


def test_the_pass_cap_is_honoured_and_the_last_correction_kept():
    fn = responder(fix("v1"), fix("v2"), fix("v3"), fix("v4"), fix("v5"), fix("v6"))
    outcome = qa_chunk("sumber", "original", TEMPLATE, fn, max_passes=5)

    assert outcome.passes == 5
    assert outcome.converged is False
    # The most-audited version is kept: discarding it would throw away five
    # passes of genuine repair.
    assert outcome.text == "v5"


def test_the_cap_is_configurable():
    fn = responder(fix("v1"), fix("v2"), fix("v3"))
    assert qa_chunk("s", "o", TEMPLATE, fn, max_passes=2).passes == 2


def test_a_cap_below_one_still_runs_a_single_pass():
    fn = responder(clean())
    assert qa_chunk("s", "o", TEMPLATE, fn, max_passes=0).passes == 1


def test_issues_accumulate_across_passes():
    fn = responder(fix("v1", "issue one"), fix("v2", "issue two"), clean())
    outcome = qa_chunk("sumber", "original", TEMPLATE, fn)

    assert outcome.issues == ["issue one", "issue two"]


def test_an_incomplete_response_stops_the_loop():
    # _qa_unit retries once itself, then degrades the unit to "no action".
    # Looping again would only repeat that.
    incomplete = {"action_required": True, "issues": ["found it"], "corrected_text": None}
    fn = responder(incomplete)
    outcome = qa_chunk("sumber", "original", TEMPLATE, fn, max_passes=5)

    assert outcome.response_incomplete is True
    assert outcome.converged is False
    assert outcome.passes == 1
    assert outcome.text == "original"


SOURCE_CHUNK = "#### Pasal 1\n\nIsi pasal satu."
TRANSLATED_CHUNK = "#### Article 1\n\nContent of article one."


@pytest.fixture
def policy(tmp_path):
    markdown_dir = tmp_path / "translation_markdown"
    source_dir = tmp_path / "cleaned_markdown"
    markdown_dir.mkdir()
    source_dir.mkdir()
    (markdown_dir / "ACEH_BIREUEN.md").write_text(TRANSLATED_CHUNK, encoding="utf-8")
    (source_dir / "ACEH_BIREUEN.cleaned.md").write_text(SOURCE_CHUNK, encoding="utf-8")
    return markdown_dir, source_dir


def test_the_report_records_passes_and_convergence(policy):
    markdown_dir, source_dir = policy
    fn = responder(fix(TRANSLATED_CHUNK + "\n"), clean())
    result = _qa_policy_file("ACEH_BIREUEN", markdown_dir, source_dir, TEMPLATE, fn)

    assert result.status == "succeeded"
    assert result.report["qa_passes_total"] == 2
    assert result.report["chunks_not_converged"] == 0
    assert result.report["max_passes"] == 5
    assert result.report["items"][0]["passes"] == 2
    assert result.report["items"][0]["converged"] is True


def test_an_unconverged_chunk_is_counted_not_discarded(policy):
    markdown_dir, source_dir = policy
    fn = responder(fix(TRANSLATED_CHUNK + "\n\nstill wrong"))
    result = _qa_policy_file(
        "ACEH_BIREUEN", markdown_dir, source_dir, TEMPLATE, fn, max_passes=3
    )

    assert result.report["chunks_not_converged"] == 1
    assert result.report["qa_passes_total"] == 3
    # The correction still lands in the translated Markdown.
    assert "still wrong" in (markdown_dir / "ACEH_BIREUEN.md").read_text(encoding="utf-8")
    assert result.report["items"][0]["passes"] == 3
    assert result.report["items"][0]["converged"] is False


def test_a_changed_heading_count_is_not_reported_as_a_problem(tmp_path):
    # The DOMPU shape: the corpus promoted a cross-reference to a heading and
    # split the sentence around it. Demoting it back is a repair that drops
    # the heading count by one. `_clause_identity` is language-agnostic (no
    # per-language keyword list), so it no longer filters out this kind of
    # OCR artifact -- the demoted "Pasal 28" now surfaces as an informational
    # `lost_clauses` entry, but must still not be reported as a problem
    # (no failure recorded, QA still converges cleanly).
    source = (
        "#### Pasal 34\n\nPelayanan rehabilitasi sosial sebagaimana dimaksud dalam  1)\n\n"
        "#### Pasal 28 ayat (1) huruf f, meliputi:\na. motivasi;"
    )
    translated = (
        "#### Article 34\n\nSocial rehabilitation services as referred to in "
        "Article 28 paragraph (1) letter f, include:\na. motivation;"
    )
    markdown_dir, source_dir = tmp_path / "md", tmp_path / "src"
    markdown_dir.mkdir()
    source_dir.mkdir()
    (markdown_dir / "DOMPU.md").write_text(translated, encoding="utf-8")
    (source_dir / "DOMPU.cleaned.md").write_text(source, encoding="utf-8")

    result = _qa_policy_file("DOMPU", markdown_dir, source_dir, TEMPLATE, responder(clean()))

    # Informational only: the identity collision surfaces in the diagnostic
    # list, but the file still succeeds -- no failure, no forced correction.
    assert result.status == "succeeded"
    assert result.report["lost_clauses"] == ["Article 28"]
    assert "structure_drift" not in result.report


def test_a_lost_numbered_clause_is_reported(policy):
    markdown_dir, source_dir = policy
    fn = responder(fix("Content of article one."), clean())
    result = _qa_policy_file("ACEH_BIREUEN", markdown_dir, source_dir, TEMPLATE, fn)

    assert result.report["lost_clauses"] == ["Article 1"]


def test_pair_chunks_packs_source_and_translation_together():
    from automation.src.markdown.qa import pair_chunks

    source = "# BAB I\n\n#### Pasal 1\n\nIsi satu.\n\n#### Pasal 2\n\nIsi dua."
    translated = "# CHAPTER I\n\n#### Article 1\n\nOne.\n\n#### Article 2\n\nTwo."

    one = pair_chunks(source, translated, target_chars=8000)
    assert len(one) == 1
    assert "Pasal 2" in one[0]["text"] and "Article 2" in one[0]["translated_text"]

    many = pair_chunks(source, translated, target_chars=20)
    assert len(many) > 1
    assert "Article 2" in many[-1]["translated_text"]
    assert "Pasal 2" in many[-1]["text"]
