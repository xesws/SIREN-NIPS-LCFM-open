"""Run with `python examples/select_rule.py` after installing the package.

Tiny numeric features demonstrate installation, eligibility and abstention.
These hand-built vectors are a software example, not research data or scores.
"""
import json
import numpy as np

from siren.memory import RuleEntry, RuleRegistry
from siren.recognition import BackgroundNull, Recognizer
from siren.routing import Selector


def main():
    null = BackgroundNull(np.array([-1., 0.]), 0., 1.)
    registry = RuleRegistry(null)
    selector = Selector(registry)
    feature = np.array([1., 0.])
    print(json.dumps({"stage": "empty", **selector.select(feature).to_dict()}))
    immature = Recognizer(np.array([1., 0.]), .5, 0., 1., 2)
    registry.install(RuleEntry("ask_for_evidence", "A conclusion lacks supporting evidence.",
                               "Ask for the missing evidence.", immature))
    print(json.dumps({"stage": "installed_but_immature", **selector.select(feature).to_dict()}))
    mature = Recognizer(np.array([1., 0.]), .5, 0., 1., 4)
    registry.update_recognizer("ask_for_evidence", mature)
    print(json.dumps({"stage": "eligible", **selector.select(feature).to_dict()}))
    print(json.dumps({"stage": "unrelated", **selector.select(np.array([-1., 0.])).to_dict()}))


if __name__ == "__main__":
    main()
