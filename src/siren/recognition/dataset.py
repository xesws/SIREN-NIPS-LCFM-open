"""Explicit fit-bank joins for the released benchmark, independent of evaluator targets."""
from __future__ import annotations

import numpy as np

from siren.baselines.text_selectors import fit_maxsim, fit_minilm_discriminant
from siren.memory import RuleEntry, RuleRegistry
from .core import BackgroundNull, fit_discriminant


def fit_registry(rules, annotations, fit_banks, features, *, method: str = "siren",
                 null: BackgroundNull | None = None) -> RuleRegistry:
    """Fit a terminal collection from ID-addressed feature arrays and explicit fit banks.

    Supported methods: siren, minilm_discriminant, minilm_maxsim,
    minilm_maxsim_self. Text feature extraction is a separate, input-only operation.
    ``null`` can provide the paper's frozen prototype when regenerating its banks.
    """
    if method not in ("siren", "minilm_discriminant", "minilm_maxsim", "minilm_maxsim_self"):
        raise ValueError("unknown selector method")
    rules, annotations = list(rules), list(annotations)
    if len({r["rule_id"] for r in rules}) != len(rules):
        raise ValueError("duplicate rule IDs")
    # Ignore evaluation annotation content completely. It cannot choose a fit bank.
    fit = {r["id"]: r for r in annotations if r["split"] == "fit"}
    if len(fit) != sum(r["split"] == "fit" for r in annotations):
        raise ValueError("duplicate fit sample IDs")
    if set(fit_banks) != {r["rule_id"] for r in rules}:
        raise ValueError("fit banks must cover the known collection")

    def stack(ids):
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("empty or duplicate fit bank")
        if any(uid not in fit for uid in ids):
            raise ValueError("fit bank contains a missing or non-fit ID")
        return np.stack([features[uid] for uid in ids])

    background_ids = [r["id"] for r in annotations if r["split"] == "fit" and r["kind"] == "background"]
    registry = RuleRegistry(BackgroundNull.fit(stack(background_ids)) if null is None else null)
    for rule in rules:
        rid = rule["rule_id"]
        banks = fit_banks[rid]
        pos_ids, neg_ids = banks["positive"], banks["negative"]
        if set(pos_ids) & set(neg_ids):
            raise ValueError("positive and negative fit banks overlap")
        positive, negative = stack(pos_ids), stack(neg_ids)
        if any(fit[uid]["rule_id"] != rid or fit[uid]["kind"] != "applicable" for uid in pos_ids):
            raise ValueError("positive bank must contain own applicable training contexts")
        if method == "siren":
            state = fit_discriminant(positive, negative)
        elif method == "minilm_discriminant":
            state = fit_minilm_discriminant(positive, negative)
        else:
            state = fit_maxsim(positive, negative, leave_one_out=method != "minilm_maxsim_self")
        registry.install(RuleEntry(rid, rule["condition"], rule["guidance"], state))
    return registry
