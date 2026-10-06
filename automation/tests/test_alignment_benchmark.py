from automation.benchmarks.alignment.aligners import align_by_index, align_with_heuristic
from automation.benchmarks.alignment.build_gold import validate_alignment
from automation.benchmarks.alignment.data import Unit, body_text, sample_gold
from automation.benchmarks.alignment.metrics import (
    normalize,
    score_document,
    summarize,
)


def unit(n_src, n_tgt, policy="P", idx=0):
    return Unit(
        id=f"{policy}#{idx}",
        policy=policy,
        section_index=idx,
        heading=None,
        src_text="",
        tgt_text="",
        src=[f"s{i}" for i in range(n_src)],
        tgt=[f"t{i}" for i in range(n_tgt)],
    )


def test_strict_and_lenient_scores():
    gold = normalize([([0], [0]), ([1], [1, 2]), ([2], [3])])
    pred = normalize([([0], [0]), ([1], [1]), ([2], [3])])
    doc = score_document(pred, gold)

    assert doc["strict_correct"] == 2  # the 1-2 bead was predicted as 1-1
    assert doc["lenient_correct"] == 3  # but it overlaps the right place
    summary = summarize([doc])
    assert summary["strict"]["precision"] == 2 / 3
    assert summary["strict"]["recall"] == 2 / 3
    assert summary["recall_by_type"]["1-2"] == {"gold": 1, "found": 0, "recall": 0.0}
    # Source sentence 1 maps to {1,2} in gold but {1} in the prediction.
    assert summary["evidence_recall"] == 2 / 3


def test_perfect_prediction_scores_one():
    gold = normalize([([0], [0]), ([1, 2], [1])])
    summary = summarize([score_document(set(gold), gold)])
    assert summary["strict"]["f1"] == 1.0 and summary["evidence_recall"] == 1.0


def test_index_aligner_only_answers_when_counts_match():
    assert align_by_index(unit(3, 3)) == normalize([([0], [0]), ([1], [1]), ([2], [2])])
    assert align_by_index(unit(3, 2)) == set()


def test_heuristic_aligner_pairs_equal_counts_positionally():
    u = unit(2, 2)
    u.src_text = "Pasal ini mengatur pokok. Ketentuan lain menyusul."
    u.tgt_text = "This article sets the basics. Other rules follow."
    u.src = ["Pasal ini mengatur pokok.", "Ketentuan lain menyusul."]
    u.tgt = ["This article sets the basics.", "Other rules follow."]
    assert align_with_heuristic(u) == normalize([([0], [0]), ([1], [1])])


def test_body_text_drops_heading_lines():
    assert body_text("#### Pasal 1\n\nIsi satu.\nIsi dua.") == "Isi satu.\nIsi dua."


def test_sample_gold_is_stratified_capped_and_reproducible():
    units = [unit(3, 3, f"P{i}", j) for i in range(10) for j in range(4)]
    units += [unit(3, 2, f"M{i}", j) for i in range(10) for j in range(4)]

    gold, remainder = sample_gold(units, 10, seed=1, mismatch_fraction=0.5, max_per_policy=1)
    assert len(gold) == 10 and len(gold) + len(remainder) == len(units)
    assert sum(not u.counts_match for u in gold) == 5
    assert len({u.policy for u in gold}) == 10
    assert {u.split for u in gold} == {"dev", "test"}

    again, _ = sample_gold(units, 10, seed=1, mismatch_fraction=0.5, max_per_policy=1)
    assert [u.id for u in gold] == [u.id for u in again]


def test_validate_alignment_requires_exact_coverage():
    u = unit(2, 2)
    assert validate_alignment(u, [[[0], [0]], [[1], [1]]]) == []
    assert validate_alignment(u, [[[0], [0]]]) != []
    assert validate_alignment(u, [[[0, 0], [0]], [[1], [1]]]) != []
