import pytest
from siren.evaluation.judging import merge_votes


def vote(planning=False, action=False, repetition=False):
    return dict(aligned_planning=planning, delivered_action=action, stuck_repetition=repetition)


def test_visible_planning_counts_but_repetition_overrides():
    assert merge_votes([vote(planning=True), vote(planning=True)])["planning_only"]
    labels = merge_votes([vote(True, True, True), vote(True, True, True)])
    assert labels["repetition"] and not labels["positive"] and not labels["action"]


def test_derived_flags_are_aggregated_after_each_votes_override():
    # Majority on raw action and repetition first would yield the wrong action.
    labels = merge_votes([vote(False, True, True), vote(False, True, False), vote(True, False, False)])
    assert not labels["action"] and labels["positive"] and not labels["repetition"]
    assert labels["vote_count"] == 3


def test_disagreement_requires_third_and_diagnostics_do_not_impute_labels():
    with pytest.raises(ValueError):
        merge_votes([vote(True), vote(False)])
    with pytest.raises(ValueError):
        merge_votes([{"aligned_planning": "yes"}, vote(True)])
