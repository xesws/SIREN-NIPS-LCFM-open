"""Label-free qualify-then-compete selection and explicit abstention."""
from __future__ import annotations

from dataclasses import dataclass, asdict
from collections.abc import Mapping

import numpy as np

from siren.memory import RuleRegistry


@dataclass(frozen=True)
class RouteDecision:
    selected_rule: str | None
    reason: str
    qualified_rules: tuple[str, ...]
    top_rule: str | None
    second_rule: str | None
    null_score: float
    scores: dict[str, float]

    @property
    def fire(self) -> bool:
        return self.selected_rule is not None

    def to_dict(self) -> dict:
        return asdict(self)


def qualify_then_compete(scores: Mapping[str, float], thresholds: Mapping[str, float],
                         null_score: float, *, gap_second: float = 1e-6,
                         gap_null: float = 1e-6) -> RouteDecision:
    """Scores contain only installed, maturity-eligible rules, not labels or targets."""
    if set(scores) != set(thresholds):
        raise ValueError("thresholds must cover exactly the competing rules")
    if any(not isinstance(rid, str) or not rid for rid in scores):
        raise ValueError("rule IDs must be nonempty strings")
    values = [*scores.values(), *thresholds.values(), null_score, gap_second, gap_null]
    if not np.isfinite(values).all() or gap_second < 0 or gap_null < 0:
        raise ValueError("scores must be finite and comparison gaps nonnegative")
    sc = {r: float(x) for r, x in scores.items()}
    qualified = tuple(sorted((r for r in sc if sc[r] > thresholds[r]), key=lambda r: (-sc[r], r)))
    all_rank = sorted(sc, key=lambda r: (-sc[r], r))
    top = qualified[0] if qualified else (all_rank[0] if all_rank else None)
    second = qualified[1] if len(qualified) > 1 else None
    if not qualified:
        reason = "no_eligible"
    elif second is not None and sc[top] <= sc[second] + gap_second:
        reason = "gap_second"
    elif sc[top] <= null_score + gap_null:
        reason = "gap_null"
    else:
        reason = "ok"
    return RouteDecision(top if reason == "ok" else None, reason, qualified, top, second, float(null_score), sc)


class Selector:
    def __init__(self, registry: RuleRegistry):
        if not isinstance(registry, RuleRegistry):
            raise TypeError("expected RuleRegistry")
        self.registry = registry

    def select(self, feature) -> RouteDecision:
        """Accept a single feature vector only; never an annotated sample dictionary."""
        if isinstance(feature, Mapping):
            raise TypeError("selection accepts a feature vector, not an annotated record")
        h = np.asarray(feature, dtype=np.float64)
        if h.shape != self.registry.null.key.shape or not np.isfinite(h).all():
            raise ValueError("invalid selection feature")
        eligible = [r for r in self.registry.order if self.registry.eligible(r)]
        states = {r: self.registry[r].recognizer for r in eligible}
        scores = {r: float(st.scores(h[None])[0]) for r, st in states.items()}
        thresholds = {r: st.standardized_threshold for r, st in states.items()}
        null = float(self.registry.null.scores(h[None])[0])
        return qualify_then_compete(scores, thresholds, null)

    def select_text(self, text: str, encoder) -> RouteDecision:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("selection accepts nonempty visible interaction text")
        return self.select(encoder.encode_one(text))
