"""Text-based selector controls; prompt execution lives in siren.execution."""
from .text_selectors import MiniLMEncoder, fit_maxsim, fit_minilm_discriminant

__all__ = ["MiniLMEncoder", "fit_maxsim", "fit_minilm_discriminant"]
