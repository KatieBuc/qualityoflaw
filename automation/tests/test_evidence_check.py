from automation.src.rag.evidence_check import resolve_evidence_citation


def test_resolve_evidence_citation_returns_real_text():
    lookup = {"1.1-0": "The quick brown fox.", "1.1-1": "Another passage."}
    assert resolve_evidence_citation("1.1-0", lookup) == "The quick brown fox."


def test_resolve_evidence_citation_strips_brackets_and_whitespace():
    lookup = {"1.1-0": "The quick brown fox."}
    assert resolve_evidence_citation("[1.1-0]", lookup) == "The quick brown fox."
    assert resolve_evidence_citation("  1.1-0  ", lookup) == "The quick brown fox."
    assert resolve_evidence_citation(" [1.1-0] ", lookup) == "The quick brown fox."


def test_resolve_evidence_citation_rejects_unknown_tag():
    lookup = {"1.1-0": "The quick brown fox."}
    assert resolve_evidence_citation("9.9-9", lookup) is None


def test_resolve_evidence_citation_rejects_none_or_empty():
    lookup = {"1.1-0": "The quick brown fox."}
    assert resolve_evidence_citation(None, lookup) is None
    assert resolve_evidence_citation("", lookup) is None


def test_resolve_evidence_citation_rejects_transcribed_text_instead_of_tag():
    # If the LLM ignores instructions and returns the passage text itself
    # rather than its tag, that text is not a key in the lookup, so it's
    # correctly rejected rather than silently accepted.
    lookup = {"1.1-0": "The quick brown fox."}
    assert resolve_evidence_citation("The quick brown fox.", lookup) is None
