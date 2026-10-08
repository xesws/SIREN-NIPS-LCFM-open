import numpy as np
import pytest

from siren.memory import RuleEntry, RuleRegistry
from siren.recognition import BackgroundNull, Recognizer
from siren.routing import Selector, qualify_then_compete


def test_threshold_equality_and_null_margin_abstain():
    assert qualify_then_compete({"r": .5}, {"r": .5}, 0.).reason == "no_eligible"
    assert qualify_then_compete({"r": 1.}, {"r": .5}, 1.).reason == "gap_null"
    assert qualify_then_compete({"r": 1.}, {"r": .5}, 1. - 1e-6).reason == "gap_null"
    assert qualify_then_compete({"r": 1.}, {"r": .5}, .9).selected_rule == "r"


def test_competition_ignores_unqualified_high_score():
    result = qualify_then_compete({"ineligible": 10., "qualified": 5.}, {"ineligible": 11., "qualified": 4.}, 0.)
    assert result.selected_rule == "qualified"
    assert result.qualified_rules == ("qualified",)


def test_tied_qualified_candidates_abstain_deterministically():
    result = qualify_then_compete({"b": 1., "a": 1.}, {"a": 0., "b": 0.}, -1.)
    assert result.reason == "gap_second" and result.selected_rule is None
    assert result.top_rule == "a" and result.second_rule == "b"


def test_empty_library_maturity_and_future_rule_isolation():
    registry = RuleRegistry(BackgroundNull(np.array([-1., 0.]), 0., 1.))
    selector = Selector(registry)
    query = np.array([1., 0.])
    assert not selector.select(query).fire
    state = Recognizer(np.array([1., 0.]), .5, 0., 1., 3)
    entry = RuleEntry("r", "condition", "guidance", state)
    # Preparing an uninstalled entry does not change the available library.
    assert selector.select(query).scores == {}
    registry.install(entry)
    assert selector.select(query).scores == {}
    registry.update_recognizer("r", Recognizer(np.array([1., 0.]), .5, 0., 1., 4))
    assert selector.select(query).selected_rule == "r"
    assert selector.select(-query).selected_rule is None


def test_explicit_maturity_ablation():
    registry = RuleRegistry(BackgroundNull(np.array([-1., 0.]), 0., 1.), require_mature=False)
    registry.install(RuleEntry("r", "condition", "guidance", Recognizer(np.array([1., 0.]), .5, 0., 1., 0)))
    assert Selector(registry).select([1., 0.]).fire


def test_selector_rejects_evaluator_record_and_bad_shapes():
    selector = Selector(RuleRegistry(BackgroundNull(np.array([-1., 0.]), 0., 1.)))
    with pytest.raises(TypeError):
        selector.select({"h": [1., 0.], "target_rule": "r", "answer": "reference"})
    with pytest.raises(ValueError):
        selector.select([1., 2., 3.])
    with pytest.raises(ValueError):
        qualify_then_compete({"r": np.nan}, {"r": 0.}, 0.)
    with pytest.raises(ValueError):
        qualify_then_compete({"r": 1.}, {}, 0.)
