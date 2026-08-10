import pytest

from automation.src.sliding_window.windowing import split_into_windows


def _sentences(n: int) -> list[str]:
    return [f"s{i}." for i in range(n)]


@pytest.fixture(autouse=True)
def fake_split_sentences(monkeypatch):
    """Windowing tests exercise grouping/overlap logic, not pysbd's actual
    segmentation — monkeypatch to a deterministic split on '|'."""

    def fake(text: str) -> list[str]:
        if not text or not text.strip():
            return []
        return [s.strip() for s in text.split("|") if s.strip()]

    monkeypatch.setattr("automation.src.sliding_window.windowing.split_sentences", fake)


def test_empty_text_returns_no_windows():
    assert split_into_windows("", window_sentences=5, overlap_sentences=1) == []


def test_shorter_than_one_window_returns_single_window():
    text = "|".join(_sentences(2))
    windows = split_into_windows(text, window_sentences=5, overlap_sentences=1)
    assert len(windows) == 1
    assert windows[0].chunk_id == 0
    assert windows[0].text == "s0. s1."


def test_window_count_and_overlap():
    text = "|".join(_sentences(7))
    windows = split_into_windows(text, window_sentences=3, overlap_sentences=1)
    assert [w.text for w in windows] == [
        "s0. s1. s2.",
        "s2. s3. s4.",
        "s4. s5. s6.",
    ]


def test_chunk_ids_are_sequential_from_zero():
    text = "|".join(_sentences(10))
    windows = split_into_windows(text, window_sentences=4, overlap_sentences=2)
    assert [w.chunk_id for w in windows] == list(range(len(windows)))


def test_no_overlap_when_overlap_sentences_is_zero():
    text = "|".join(_sentences(6))
    windows = split_into_windows(text, window_sentences=2, overlap_sentences=0)
    assert [w.text for w in windows] == ["s0. s1.", "s2. s3.", "s4. s5."]


def test_exact_multiple_of_window_size_does_not_add_trailing_empty_window():
    text = "|".join(_sentences(6))
    windows = split_into_windows(text, window_sentences=3, overlap_sentences=0)
    assert len(windows) == 2
