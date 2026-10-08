"""Frozen-paper accounting, separated from all runtime selection interfaces."""
from collections import defaultdict
import numpy as np


def routing_counts(annotations, decisions):
    """Count five exclusive exact-routing outcomes on a protocol panel.

    A hard-negative annotation is target-relative. The paper's library protocol
    prescribes abstention on it; this function does not infer universal labels.
    """
    counts = dict(correct=0, wrong=0, abstain=0, false_fire=0, true_reject=0)
    for row in annotations:
        selected = decisions[row["id"]]
        if row["kind"] in ("applicable", "positive"):
            if row.get("rule_id") is None:
                raise ValueError("applicable context needs a target rule in evaluator annotations")
            key = "correct" if selected == row["rule_id"] else "abstain" if selected is None else "wrong"
        elif row["kind"] in ("hard_negative", "background"):
            key = "true_reject" if selected is None else "false_fire"
        else:
            raise ValueError("unknown annotation kind")
        counts[key] += 1
    c, w, u, f, t = (counts[k] for k in ("correct", "wrong", "abstain", "false_fire", "true_reject"))
    p, n = c + w + u, f + t
    return {**counts, "positive_n": p, "negative_n": n,
            "correct_rule_recall": c / p if p else None,
            "fire_tpr": (c + w) / p if p else None,
            "correct_rejection": t / n if n else None,
            "false_positive_rate": f / n if n else None,
            "balanced_exact_routing": .5 * (c / p + t / n) if p and n else None,
            "exact_accuracy": (c + t) / (p + n) if p + n else None}


def binary_labels(labels):
    """Preserve the historical repetition override, reject malformed labels."""
    values = {k: labels[k] for k in ("action", "positive", "repetition")}
    if any(type(v) is not bool for v in values.values()):
        raise ValueError("behavior labels must be booleans")
    if values["repetition"] and (values["action"] or values["positive"]):
        raise ValueError("frozen label violates repetition override")
    if values["action"] and not values["positive"]:
        raise ValueError("action completion must also be rule-consistent")
    return values


def action_summary(labels):
    rows = [binary_labels(x) for x in labels]
    count = {k: sum(r[k] for r in rows) for k in ("action", "positive", "repetition")}
    return {"n": len(rows), "counts": count,
            "rates": {k: v / len(rows) if rows else None for k, v in count.items()}}


def clustered_mean(values, *, seed=42, draws=10000):
    """Equally weight rule means and percentile-bootstrap whole rule clusters."""
    grouped = defaultdict(list)
    for rule, value in values:
        if not np.isfinite(value):
            raise ValueError("nonfinite observation")
        grouped[rule].append(float(value))
    means = np.array([np.mean(v) for _, v in sorted(grouped.items())])
    if not len(means):
        return {"rules": 0, "n": 0, "mean": None, "low": None, "high": None}
    rng = np.random.default_rng(seed)
    boot = means[rng.integers(0, len(means), size=(draws, len(means)))].mean(axis=1)
    return {"rules": len(means), "n": sum(map(len, grouped.values())), "mean": float(means.mean()),
            "low": float(np.quantile(boot, .025)), "high": float(np.quantile(boot, .975))}
