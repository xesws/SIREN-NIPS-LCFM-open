"""Fit-only numerical recognition used by SIREN and the matched text controls.

Ported from the authors' recognition lifecycle and matched-selector routines.
No experiment labels, model APIs, or filesystem paths enter numerical scoring.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from sklearn.covariance import ledoit_wolf


def matrix(value, *, dim: int | None = None, allow_empty: bool = False) -> np.ndarray:
    x = np.asarray(value, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] == 0 or (not allow_empty and x.shape[0] == 0):
        raise ValueError("expected a nonempty (examples, features) matrix")
    if dim is not None and x.shape[1] != dim:
        raise ValueError("feature dimensions differ")
    if not np.isfinite(x).all():
        raise ValueError("features must be finite")
    return x


def normalize(x, axis: int = -1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-8)


def hopfield_scores(features, key) -> np.ndarray:
    """Published single-key, one-step scoring; alpha=.1, with epsilon normalization."""
    h = matrix(features, allow_empty=True)
    k = np.asarray(key, dtype=np.float64)
    if k.shape != (h.shape[1],) or not np.isfinite(k).all():
        raise ValueError("invalid recognition key")
    q, k = normalize(h), normalize(k.reshape(1, -1))
    # Keep the original single-key softmax and normalization arithmetic.
    retrieved = normalize((np.ones((len(q), 1)) / (1 + 1e-8)) @ k)
    return (normalize(.9 * q + .1 * retrieved) @ k.T)[:, 0]


def maxsim_scores(features, examples) -> np.ndarray:
    e = matrix(examples)
    h = matrix(features, dim=e.shape[1], allow_empty=True)
    return (normalize(h) @ normalize(e).T).max(axis=1)


def discriminant_key(positive, negative) -> np.ndarray:
    p = matrix(positive)
    n = matrix(negative, dim=p.shape[1])
    covariance, _ = ledoit_wolf(np.vstack([p, n]))
    direction = np.linalg.solve(covariance + 1e-6 * np.eye(p.shape[1]), p.mean(0) - n.mean(0))
    return normalize(direction, axis=0)


def score_moments(scores) -> tuple[float, float]:
    x = np.asarray(scores, dtype=np.float64)
    if x.ndim != 1 or not x.size or not np.isfinite(x).all():
        raise ValueError("expected finite, nonempty score vector")
    return float(x.mean()), max(float(x.std()), 1e-8)


def conservative_threshold(negative_scores) -> float:
    x = np.asarray(negative_scores, dtype=np.float64)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError("invalid negative scores")
    if not x.size:
        return 1.0
    high, low = np.quantile(x, .99), np.quantile(x, .90)
    return float(high + abs(high - low))


def calibrate(positive_scores, negative_scores) -> tuple[float, float, float]:
    p, n = np.asarray(positive_scores, dtype=np.float64), np.asarray(negative_scores, dtype=np.float64)
    if p.ndim != 1 or n.ndim != 1 or not np.isfinite(p).all() or not np.isfinite(n).all():
        raise ValueError("invalid fit-side calibration scores")
    mu, sigma = score_moments(np.r_[p, n])
    theta = (float(.5 * (np.quantile(p, .10) + np.quantile(n, .90))) if p.size and n.size
             else float(np.quantile(p, .10)) if p.size else conservative_threshold(n))
    return theta, mu, sigma


@dataclass(frozen=True)
class Recognizer:
    """One fitted rule. ``evidence_count`` counts own applicable examples only."""

    key: np.ndarray
    threshold: float
    mean: float
    std: float
    evidence_count: int
    mode: Literal["hopfield", "maxsim"] = "hopfield"
    initial_key: np.ndarray | None = None
    scale: np.ndarray | None = None

    def __post_init__(self):
        if self.mode not in ("hopfield", "maxsim"):
            raise ValueError("unsupported recognizer mode")
        key = np.array(self.key, dtype=np.float64, copy=True)
        wanted_ndim = 1 if self.mode == "hopfield" else 2
        if key.ndim != wanted_ndim or not key.size or not np.isfinite(key).all():
            raise ValueError("invalid key or exemplar matrix")
        if type(self.evidence_count) is not int or self.evidence_count < 0:
            raise ValueError("evidence_count must be a nonnegative integer")
        if not np.isfinite([self.threshold, self.mean, self.std]).all() or self.std < 1e-8:
            raise ValueError("invalid fit calibration")
        key.setflags(write=False)
        object.__setattr__(self, "key", key)
        for name in ("initial_key", "scale"):
            value = getattr(self, name)
            if value is not None:
                a = np.array(value, dtype=np.float64, copy=True)
                if a.shape != (key.shape[-1],) or not np.isfinite(a).all():
                    raise ValueError(f"invalid {name}")
                if name == "scale" and (a <= 0).any():
                    raise ValueError("diagonal scale must be positive")
                a.setflags(write=False)
                object.__setattr__(self, name, a)

    @property
    def dimension(self) -> int:
        return self.key.shape[-1]

    @property
    def standardized_threshold(self) -> float:
        return float((self.threshold - self.mean) / self.std)

    def raw_scores(self, features) -> np.ndarray:
        h = matrix(features, dim=self.dimension, allow_empty=True)
        if self.scale is not None:
            h = h * self.scale
        return hopfield_scores(h, self.key) if self.mode == "hopfield" else maxsim_scores(h, self.key)

    def scores(self, features) -> np.ndarray:
        return (self.raw_scores(features) - self.mean) / self.std

    def to_dict(self) -> dict:
        return {"key": self.key.tolist(), "threshold": self.threshold, "mean": self.mean,
                "std": self.std, "evidence_count": self.evidence_count, "mode": self.mode,
                "initial_key": None if self.initial_key is None else self.initial_key.tolist(),
                "scale": None if self.scale is None else self.scale.tolist()}

    @classmethod
    def from_dict(cls, value: dict) -> "Recognizer":
        return cls(**value)


def fit_discriminant(positive, negative, *, split: str = "fit") -> Recognizer:
    """Fit the common SIREN/MiniLM discriminant on explicit training banks."""
    require_fit(split)
    p = matrix(positive)
    n = matrix(negative, dim=p.shape[1])
    key = discriminant_key(p, n)
    theta, mu, sigma = calibrate(hopfield_scores(p, key), hopfield_scores(n, key))
    return Recognizer(key, theta, mu, sigma, len(p), initial_key=key)


def require_fit(split: str):
    if split != "fit":
        raise ValueError("fitting and calibration accept split='fit' only")


def initialize(condition_feature, negative, *, split: str = "fit") -> Recognizer:
    """Condition-based initialization; historical inherited keys can be loaded separately."""
    require_fit(split)
    key = np.asarray(condition_feature, dtype=np.float64)
    if key.ndim != 1 or not key.size or not np.isfinite(key).all():
        raise ValueError("invalid condition feature")
    n = matrix(negative, dim=key.size)
    key = normalize(key, axis=0)
    scores = hopfield_scores(n, key)
    mu, sigma = score_moments(scores)
    return Recognizer(key, conservative_threshold(scores), mu, sigma, 0, initial_key=key)


def refine(state: Recognizer, positive, negative, *, split: str = "fit") -> Recognizer:
    """Refit from the observed evidence prefix, preserving the original low-evidence key."""
    require_fit(split)
    if state.mode != "hopfield":
        raise ValueError("SIREN refinement requires a Hopfield recognizer")
    p = matrix(positive, dim=state.dimension, allow_empty=True)
    n = matrix(negative, dim=state.dimension)
    if len(p) < state.evidence_count:
        raise ValueError("evidence cannot move backwards")
    if not len(p):
        return state
    initial = state.initial_key if state.initial_key is not None else state.key
    if len(p) < 4:
        key, scale = normalize(4.0 * initial + p.mean(0), axis=0), state.scale
    else:
        key, scale = discriminant_key(p, n), None
    hpos, hneg = (p, n) if scale is None else (p * scale, n * scale)
    theta, mu, sigma = calibrate(hopfield_scores(hpos, key), hopfield_scores(hneg, key))
    return Recognizer(key, theta, mu, sigma, len(p), initial_key=initial, scale=scale)


def negative_bank(background, own_hard_negative, other_applicable) -> np.ndarray:
    """Known-collection protocol: all other rules' fit positives are contrastive negatives."""
    bg = matrix(background)
    own = matrix(own_hard_negative, dim=bg.shape[1], allow_empty=True)
    others = [matrix(x, dim=bg.shape[1], allow_empty=True) for x in other_applicable]
    return np.vstack([bg, own, *others])


@dataclass(frozen=True)
class BackgroundNull:
    key: np.ndarray
    mean: float
    std: float

    def __post_init__(self):
        key = np.array(self.key, dtype=np.float64, copy=True)
        if key.ndim != 1 or not key.size or not np.isfinite(key).all():
            raise ValueError("invalid background-null prototype")
        if not np.isfinite([self.mean, self.std]).all() or self.std < 1e-8:
            raise ValueError("invalid background-null calibration")
        key.setflags(write=False)
        object.__setattr__(self, "key", key)

    def scores(self, features) -> np.ndarray:
        return (hopfield_scores(features, self.key) - self.mean) / self.std

    @classmethod
    def fit(cls, background, *, calibration_background=None, split: str = "fit") -> "BackgroundNull":
        require_fit(split)
        bg = matrix(background)
        key = normalize(bg.mean(0), axis=0)
        calibration = bg if calibration_background is None else matrix(calibration_background, dim=bg.shape[1])
        mu, sigma = score_moments(hopfield_scores(calibration, key))
        return cls(key, mu, sigma)

    def to_dict(self) -> dict:
        return {"key": self.key.tolist(), "mean": self.mean, "std": self.std}

    @classmethod
    def from_dict(cls, value: dict) -> "BackgroundNull":
        return cls(**value)
