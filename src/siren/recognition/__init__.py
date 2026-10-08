"""Rule applicability fitting and optional language-model features."""
from .core import (BackgroundNull, Recognizer, calibrate, discriminant_key,
                   fit_discriminant, hopfield_scores, initialize, maxsim_scores,
                   negative_bank, refine)

__all__ = ["BackgroundNull", "Recognizer", "calibrate", "discriminant_key",
           "fit_discriminant", "hopfield_scores", "initialize", "maxsim_scores",
           "negative_bank", "refine"]
