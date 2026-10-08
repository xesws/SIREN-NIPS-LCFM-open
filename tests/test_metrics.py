import pytest
from siren.evaluation import routing_counts, action_summary, clustered_mean


def test_wrong_rule_is_not_a_true_positive_for_exact_selection():
    rows = [{"id": str(i), "kind": "applicable" if i < 3 else "hard_negative", "rule_id": "r1"} for i in range(5)]
    out = routing_counts(rows, {"0": "r1", "1": "r2", "2": None, "3": "r2", "4": None})
    assert [out[k] for k in ("correct", "wrong", "abstain", "false_fire", "true_reject")] == [1, 1, 1, 1, 1]
    assert out["correct_rule_recall"] == 1 / 3
    assert out["fire_tpr"] == 2 / 3
    assert out["balanced_exact_routing"] == .5 * (1 / 3 + .5)
    assert out["exact_accuracy"] == .4


def test_abstaining_everywhere_is_fifty_percent_balanced():
    rows = [{"id": "p", "kind": "applicable", "rule_id": "r1"}, {"id": "n", "kind": "background"}]
    assert routing_counts(rows, {"p": None, "n": None})["balanced_exact_routing"] == .5


def test_behavior_label_invariants():
    assert action_summary([{"action": False, "positive": True, "repetition": False}])["counts"]["positive"] == 1
    with pytest.raises(ValueError):
        action_summary([{"action": True, "positive": True, "repetition": True}])


def test_rule_balanced_mean_does_not_weight_longer_rules_more():
    out = clustered_mean([("a", 1), ("a", 1), ("b", -1)], draws=100)
    assert out["mean"] == 0
    assert out["n"] == 3 and out["rules"] == 2
