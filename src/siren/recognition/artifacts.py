"""Load released fitted selectors without refitting or accessing private result trees."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from siren.memory import RuleEntry, RuleRegistry
from .core import BackgroundNull, Recognizer


def load_frozen_registry(*, data_root="data", artifact_root="artifacts/frozen", execution: bool = True) -> RuleRegistry:
    """Load the execution-replay selector or the archived recognition-panel selector.

    These share the fitted rule collection but retain distinct feature-generation
    precision/provenance. Execution uses fresh float32 prompt features; recognition
    uses the released historical feature bank. Do not mix their reported panels.
    """
    data_root, artifact_root = Path(data_root), Path(artifact_root)
    rules = {r["rule_id"]: r for r in json.loads((data_root / "rules.json").read_text(encoding="utf-8"))}
    directory = artifact_root / "routing"
    if execution:
        payload = json.loads((directory / "execution_selector.json").read_text(encoding="utf-8"))
        order = payload["order"]
        states = {s["rule_id"]: s for s in payload["states"]}
        keys = {rid: states[rid]["key"] for rid in order}
        null = BackgroundNull(payload["null"], payload["null_mu"], payload["null_sigma"])
    else:
        payload = json.loads((directory / "states_siren.json").read_text(encoding="utf-8"))
        order, states = payload["rule_ids"], payload["final"]
        with np.load(directory / "siren_keys.npz", allow_pickle=False) as arrays:
            if arrays["final_keys"].shape[0] != len(order):
                raise ValueError("published key/order alignment differs")
            keys = dict(zip(order, arrays["final_keys"]))
        null = BackgroundNull.from_dict(json.loads((directory / "siren_null.json").read_text(encoding="utf-8")))
    if len(order) != len(set(order)) or set(order) != set(states) or not set(order).issubset(rules):
        raise ValueError("frozen selector roster differs")
    registry = RuleRegistry(null)
    for rid in order:
        state, rule = states[rid], rules[rid]
        if state.get("scale") is not None or state["score"] != "hopfield":
            raise ValueError("unexpected frozen selector scoring")
        model = Recognizer(keys[rid], state["theta"], state["fit_mu"], state["fit_sigma"], int(state["n"]))
        registry.install(RuleEntry(rid, rule["condition"], rule["guidance"], model))
    return registry
