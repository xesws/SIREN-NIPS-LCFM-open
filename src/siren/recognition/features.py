"""Optional prompt-only Llama features with the paper's tokenization convention."""
from __future__ import annotations

import math
import numpy as np


def suffix_span(start: int, end: int, fraction: float = .60) -> tuple[int, int]:
    if type(start) is not int or type(end) is not int or start < 0 or end < start:
        raise ValueError("expected an inclusive nonempty token span")
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    return end - max(1, math.ceil((end - start + 1) * fraction)) + 1, end


def interaction_span(tokenizer, rendered: str, text: str) -> tuple[int, int]:
    """Reproduce published offsets: map the visible interaction in the chat template.

    The historical protocol tokenizes the model input with special tokens, while
    offset mapping uses add_special_tokens=False. This convention is preserved
    for reproduction; it must not be silently changed for an existing feature bank.
    """
    if not isinstance(text, str) or not text.strip():
        raise ValueError("interaction must be nonempty text")
    needle = text if text in rendered else text.rstrip()
    if not needle or needle not in rendered:
        raise ValueError("interaction not found in rendered chat template")
    start = rendered.rindex(needle)
    end = start + len(needle)
    offsets = tokenizer(rendered, return_offsets_mapping=True, add_special_tokens=False)["offset_mapping"]
    indices = [i for i, (a, b) in enumerate(offsets) if b > a and a < end and b > start]
    if not indices:
        raise ValueError("interaction has no mapped tokens")
    return indices[0], indices[-1]


class LlamaFeatures:
    """Extract block output before any rule execution. No target labels are accepted."""

    def __init__(self, model, tokenizer, *, layer: int = 12):
        if type(layer) is not int or layer < 0 or layer >= len(model.model.layers):
            raise ValueError("invalid zero-indexed decoder layer")
        self.model, self.tokenizer, self.layer = model, tokenizer, layer

    @classmethod
    def from_pretrained(cls, model_path: str, *, revision: str | None = None,
                        device: str = "cuda", precision: str = "bfloat16",
                        local_files_only: bool = True, layer: int = 12):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if precision not in ("bfloat16", "float32"):
            raise ValueError("precision must be bfloat16 or float32")
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        common = dict(revision=revision, local_files_only=local_files_only, trust_remote_code=False)
        tok = AutoTokenizer.from_pretrained(model_path, **common)
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=getattr(torch, precision), **common).to(device)
        model.requires_grad_(False)
        model.eval()
        return cls(model, tok, layer=layer)

    def encode_one(self, text: str) -> np.ndarray:
        import torch

        if not isinstance(text, str) or not text.strip():
            raise ValueError("encode_one accepts nonempty interaction text only")
        tok, model = self.tokenizer, self.model
        rendered = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
        start, end = suffix_span(*interaction_span(tok, rendered, text))
        enc = tok(rendered, add_special_tokens=True, truncation=False, return_tensors="pt")
        device = next(model.parameters()).device
        enc = {k: v.to(device) for k, v in enc.items()}
        values = []

        def capture(_module, _args, output):
            h = output[0] if isinstance(output, tuple) else output
            if h.shape[0] != 1 or h.shape[1] <= end:
                raise RuntimeError("unexpected feature batch or token span")
            values.append(h[0, start:end + 1].mean(0).detach().float().cpu().numpy())

        handle = model.model.layers[self.layer].register_forward_hook(capture)
        training = model.training
        try:
            model.eval()
            with torch.no_grad():
                model(**enc, use_cache=False)
        finally:
            handle.remove()
            model.train(training)
        if len(values) != 1 or not np.isfinite(values[0]).all():
            raise RuntimeError("invalid or repeated feature capture")
        return values[0]

    def encode(self, texts) -> np.ndarray:
        if isinstance(texts, str):
            raise TypeError("encode expects a sequence of texts; use encode_one for one text")
        texts = list(texts)
        if not texts:
            raise ValueError("empty feature request")
        return np.stack([self.encode_one(text) for text in texts])
