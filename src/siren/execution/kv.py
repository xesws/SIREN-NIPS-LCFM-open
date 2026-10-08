"""SIREN-KV: rule-local, bounded edits to a native attention prefix.

Adapted from original e23_native_schedule_r13/core.py. Backbone weights are
frozen; cached keys are already post-RoPE and MUST NOT be rotated again.
"""
from contextlib import nullcontext
import torch
from torch import nn
from torch.nn import functional as F


def native_cache(pairs):
    from transformers.cache_utils import DynamicCache
    cache = DynamicCache()
    for layer, (key, value) in enumerate(pairs):
        cache.update(key, value, layer)
    return cache


def cache_pairs(cache):
    return tuple((layer.keys, layer.values) for layer in cache.layers)


class Bank(nn.Module):
    def __init__(self, record):
        super().__init__()
        for name in ("K0", "V0", "sK", "sV"):
            self.register_buffer(name, record[name].detach().clone())
        self.UK = nn.Parameter(torch.zeros_like(self.K0))
        self.UV = nn.Parameter(torch.zeros_like(self.V0))

    def memory(self):
        return self.K0 + .5 * self.sK * self.UK.tanh(), self.V0 + .5 * self.sV * self.UV.tanh()


class RuleMemory(nn.Module):
    def __init__(self, capsule):
        super().__init__()
        validate_capsule(capsule)
        self.length = capsule["prefix_length"]
        self.banks = nn.ModuleList([Bank(layer) for layer in capsule["layers"]])

    def cache(self):
        # Fresh cache per request: decode must never modify persistent memory.
        return native_cache([(k[None], v[None]) for b in self.banks for k, v in [b.memory()]])

    def regularizer(self):
        return torch.stack([.5 * (b.UK.tanh().square().mean() + b.UV.tanh().square().mean())
                            for b in self.banks]).mean()

    def optimizer_groups(self, learning_rate):
        return [{"name": name, "params": [getattr(b, name) for b in self.banks], "lr": learning_rate}
                for name in ("UK", "UV")]


def validate_capsule(capsule):
    if capsule.get("schema") != "siren-kv-prefix-v1" or capsule.get("post_rope_keys") is not True:
        raise ValueError("expected native post-RoPE key coordinates")
    length = capsule["prefix_length"]
    if length <= 0 or length != len(capsule["prefix_ids"]) or not capsule["layers"]:
        raise ValueError("invalid prefix geometry")
    shape = None
    for layer in capsule["layers"]:
        if set(layer) != {"K0", "V0", "sK", "sV"}:
            raise ValueError("invalid prefix fields")
        for name in ("K0", "V0"):
            value, scale = layer[name], layer["s" + name[0]]
            if value.ndim != 3 or value.shape[1] != length or value.dtype != torch.float32:
                raise ValueError("prefix must have shape [KV heads, prefix length, head dimension] in fp32")
            shape = tuple(value.shape) if shape is None else shape
            if tuple(value.shape) != shape or not torch.isfinite(value).all():
                raise ValueError("inconsistent/nonfinite prefix")
            expected = value.double().square().mean(-1, keepdim=True).sqrt().float()
            if not torch.equal(scale, expected) or not (scale > 1e-8).all():
                raise ValueError("invalid row RMS scale")


@torch.no_grad()
def compile_prefix(model, frame):
    ids = torch.tensor([frame["prefix_ids"]], device=next(model.parameters()).device)
    out = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=True)
    layers = []
    for key, value in cache_pairs(out.past_key_values):
        key, value = key[0].detach().cpu().float().contiguous(), value[0].detach().cpu().float().contiguous()
        layers.append({"K0": key, "V0": value,
                       "sK": key.double().square().mean(-1, keepdim=True).sqrt().float(),
                       "sV": value.double().square().mean(-1, keepdim=True).sqrt().float()})
    capsule = {"schema": "siren-kv-prefix-v1", "post_rope_keys": True,
               "prefix_length": len(frame["prefix_ids"]), "prefix_ids": frame["prefix_ids"], "layers": layers}
    validate_capsule(capsule)
    return capsule


def forward_with_memory(model, memory, ids, *, math_attention=True):
    if ids.ndim != 2 or ids.shape[0] != 1:
        raise ValueError("rule-local forward expects batch size 1")
    prefix, length = memory.length, ids.shape[1]
    positions = torch.arange(prefix, prefix + length, device=ids.device)
    backend = (torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH)
               if math_attention else nullcontext())
    with backend:
        out = model(input_ids=ids,
                    attention_mask=torch.ones((1, prefix + length), dtype=torch.long, device=ids.device),
                    position_ids=positions[None], cache_position=positions,
                    past_key_values=memory.cache(), use_cache=True)
    if out.past_key_values.get_seq_length() != prefix + length:
        raise RuntimeError("cache length does not match prefix plus suffix")
    return out.logits


def example_weights(lengths):
    if not lengths or any(n <= 0 for n in lengths):
        raise ValueError("positive action-plus-EOS lengths required")
    return [.5 / len(lengths) + .5 * n / sum(lengths) for n in lengths]


def early_mask(length, early_tokens=8, device=None):
    if length < 1 or early_tokens < 0:
        raise ValueError("invalid target length/window")
    # Final EOS is always assigned the late objective, including short actions.
    return torch.arange(length, device=device) < min(early_tokens, length - 1)


def scheduled_objective(logits, packing, teacher_logp, early_tokens=8):
    selected = logits[0, packing["prediction_positions"]].float()
    if selected.shape != teacher_logp.shape or selected.shape != (len(packing["target_ids"]), packing["vocab_size"]):
        raise ValueError("full-vocabulary teacher/target shapes differ")
    logp = selected.log_softmax(-1)
    logq = teacher_logp.detach().to(logp.device)
    if not torch.isfinite(logq).all() or not torch.allclose(logq.logsumexp(-1), torch.zeros(len(logq), device=logq.device), atol=1e-5, rtol=0):
        raise ValueError("teacher distribution must be finite and normalized")
    gold = torch.tensor(packing["target_ids"], device=logits.device)
    ce = F.nll_loss(logp, gold, reduction="none")
    kl = (logq.exp() * (logq - logp)).sum(-1)
    early = early_mask(len(gold), early_tokens, logits.device)
    # Both contributions use the ORIGINAL full sequence denominator.
    return {"action_ce": ce.mean(), "teacher_kl": kl.mean(),
            "ce_contribution": (torch.where(early, 1., .5) * ce).mean(),
            "kl_contribution": (torch.where(early, 0., .5) * kl).mean()}
