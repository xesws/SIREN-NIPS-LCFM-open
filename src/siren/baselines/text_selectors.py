"""Matched MiniLM discriminant and maximum-similarity controls."""
from __future__ import annotations

import numpy as np

from siren.recognition.core import (Recognizer, calibrate, fit_discriminant, matrix,
                                    maxsim_scores, normalize, require_fit)


def fit_minilm_discriminant(positive_embeddings, negative_embeddings, *, split: str = "fit") -> Recognizer:
    """Same learner and calibration as SIREN; only the input representation changes."""
    return fit_discriminant(positive_embeddings, negative_embeddings, split=split)


def fit_maxsim(positive_embeddings, negative_embeddings, *, leave_one_out: bool = True,
               split: str = "fit") -> Recognizer:
    require_fit(split)
    p = matrix(positive_embeddings)
    n = matrix(negative_embeddings, dim=p.shape[1])
    if type(leave_one_out) is not bool:
        raise TypeError("leave_one_out must be boolean")
    if len(p) < 2:
        raise ValueError("the paper's MaxSim calibration requires at least two fit positives")
    similarities = normalize(p) @ normalize(p).T
    if leave_one_out:
        np.fill_diagonal(similarities, -np.inf)
    theta, mu, sigma = calibrate(similarities.max(axis=1), maxsim_scores(n, p))
    return Recognizer(p, theta, mu, sigma, len(p), mode="maxsim")


class MiniLMEncoder:
    """Optional text encoder; no implicit model download by default."""

    def __init__(self, model):
        self.model = model
        if model.max_seq_length != 256:
            raise ValueError("paper MiniLM encoder requires max_seq_length=256")

    @classmethod
    def from_pretrained(cls, model_path: str = "sentence-transformers/all-MiniLM-L6-v2", *,
                        revision: str | None = None, device: str = "cpu", local_files_only: bool = True):
        from sentence_transformers import SentenceTransformer
        return cls(SentenceTransformer(model_path, revision=revision, device=device, local_files_only=local_files_only))

    def encode(self, texts, *, batch_size: int = 32) -> np.ndarray:
        if isinstance(texts, str):
            raise TypeError("encode expects a sequence of texts")
        texts = list(texts)
        if not texts or any(not isinstance(t, str) or not t.strip() for t in texts):
            raise ValueError("encoder accepts nonempty visible texts only")
        # The published model includes its own Normalize module. Preserve its output.
        h = self.model.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)
        h = matrix(h)
        if not np.allclose(np.linalg.norm(h, axis=1), 1, atol=1e-5):
            raise RuntimeError("unexpected unnormalized MiniLM embeddings")
        return h

    def encode_one(self, text: str) -> np.ndarray:
        return self.encode([text])[0]
