from automation.src.rag.sentence_split import split_sentences


def test_splits_multiple_sentences():
    text = "The policy covers domestic violence. It also covers sexual violence."
    assert split_sentences(text) == [
        "The policy covers domestic violence.",
        "It also covers sexual violence.",
    ]


def test_handles_abbreviations_and_decimals_without_over_splitting():
    text = "See Art. 3.1 of the regulation. It defines economic violence."
    result = split_sentences(text)
    assert result == [
        "See Art. 3.1 of the regulation.",
        "It defines economic violence.",
    ]


def test_empty_and_whitespace_only_input_returns_no_sentences():
    assert split_sentences("") == []
    assert split_sentences("   \n  ") == []


def test_single_sentence_returns_one_item():
    assert split_sentences("Just one sentence here.") == ["Just one sentence here."]
