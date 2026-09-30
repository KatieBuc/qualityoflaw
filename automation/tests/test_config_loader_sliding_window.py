import pytest

from automation.src.config_loader import parse_sliding_window_config


def test_defaults_when_absent():
    config = parse_sliding_window_config(None)
    assert config.window_sentences == 40
    assert config.overlap_sentences == 10
    assert config.prompt_version == "sliding_window_v1"


def test_reads_explicit_values():
    config = parse_sliding_window_config(
        {"window_sentences": 20, "overlap_sentences": 5, "prompt_version": "sliding_window_v2"}
    )
    assert config.window_sentences == 20
    assert config.overlap_sentences == 5
    assert config.prompt_version == "sliding_window_v2"


def test_rejects_window_sentences_below_one():
    with pytest.raises(ValueError, match="window_sentences"):
        parse_sliding_window_config({"window_sentences": 0})


def test_rejects_negative_overlap():
    with pytest.raises(ValueError, match="overlap_sentences"):
        parse_sliding_window_config({"overlap_sentences": -1})


def test_rejects_overlap_greater_than_or_equal_to_window():
    with pytest.raises(ValueError, match="overlap_sentences"):
        parse_sliding_window_config({"window_sentences": 10, "overlap_sentences": 10})
