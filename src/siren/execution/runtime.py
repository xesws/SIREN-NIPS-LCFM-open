"""Local-only model loading and independent-request delivery.

Generation mirrors the original e23_native_prefix_r11/worker.py and
PI controls. This module never decides which rule is applicable. On abstention,
call generate_reply(method='base'); on selection pass the selected rule/state.
"""
import torch
from .kv import RuleMemory
from .prompts import render_prompt
from .tokens import pack_prompt
from .value import SinglePositionValue
from .identity import validate_binding


def load_local_model(model_path, device="cuda", *, precision="float32", revision=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .training import freeze_model
    if precision not in ("float32", "bfloat16"):
        raise ValueError("precision must be float32 or bfloat16")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True,
                                                torch_dtype=getattr(torch, precision), attn_implementation="sdpa", revision=revision).to(device)
    model._siren_declared_revision = revision
    freeze_model(model, required_precision=precision)
    return model, tokenizer


def stop_ids(model, tokenizer):
    values = {tokenizer.eos_token_id}
    for config in (model.config, model.generation_config):
        eos = getattr(config, "eos_token_id", None)
        values.update(eos if isinstance(eos, (list, tuple)) else [eos])
    return sorted(int(x) for x in values if x is not None)


def memory_from_checkpoint(checkpoint, device):
    unit = RuleMemory(checkpoint["capsule"]).to(device)
    offsets = checkpoint["offsets"]
    if len(offsets) != len(unit.banks):
        raise ValueError("checkpoint layer count differs")
    with torch.no_grad():
        for bank, values in zip(unit.banks, offsets):
            if set(values) != {"UK", "UV"}:
                raise ValueError("unexpected trainable memory scope")
            for name, value in values.items():
                parameter = getattr(bank, name)
                if value.shape != parameter.shape or value.dtype != torch.float32 or not torch.isfinite(value).all():
                    raise ValueError("invalid memory offset")
                parameter.copy_(value)
    return unit


@torch.no_grad()
def generate_reply(model, tokenizer, stream, *, method="base", rule=None, checkpoint=None,
                   max_new_tokens=1024):
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be positive")
    if method not in ("base", "pi-v1", "pi-v2", "siren-kv", "siren-value"):
        raise ValueError("unknown delivery method")
    model.eval()
    device = next(model.parameters()).device
    unit = None
    if method.startswith("siren-"):
        if checkpoint is None or checkpoint.get("schema") != "siren-execution-v1" or checkpoint["method"] != method:
            raise ValueError("matching trained rule checkpoint required")
        validate_binding(model, tokenizer, checkpoint.get("model_identity"))
        saved_rule = checkpoint["rule"]
        if rule is not None and rule != saved_rule:
            raise ValueError("selected rule differs from checkpoint")
        rule = saved_rule
    if method == "siren-kv":
        record = pack_prompt(tokenizer, rule, checkpoint["frame"], stream)
        ids = torch.tensor([record["suffix_ids"]], device=device)
        unit = memory_from_checkpoint(checkpoint, device)
        prefix = unit.length
        kwargs = {"past_key_values": unit.cache(),
                  "attention_mask": torch.ones((1, prefix + ids.shape[1]), device=device, dtype=torch.long),
                  "cache_position": torch.arange(prefix, prefix + ids.shape[1], device=device)}
    else:
        if method.startswith("pi-") and rule is None:
            raise ValueError("PI delivery requires a rule selected by the caller")
        text = render_prompt(tokenizer, stream, rule if method.startswith("pi-") else None, method)
        ids = torch.tensor([tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"]], device=device)
        kwargs = {"attention_mask": torch.ones_like(ids)}
        if method == "siren-value":
            unit = SinglePositionValue(model.config.hidden_size, checkpoint["layer"]).to(device)
            value = checkpoint["value"]
            if value.shape != unit.value.shape or not torch.isfinite(value).all():
                raise ValueError("invalid single-position value")
            unit.value.copy_(value)
    options = dict(input_ids=ids, use_cache=True, do_sample=False, max_new_tokens=max_new_tokens,
                   pad_token_id=tokenizer.eos_token_id, eos_token_id=stop_ids(model, tokenizer), **kwargs)
    if method == "siren-value":
        with unit.inject(model, ids.shape[1]):
            output = model.generate(**options)
    else:
        output = model.generate(**options)
    generated = output[0, ids.shape[1]:].cpu().tolist()
    return {"method": method, "selected_rule": rule["rule_id"] if method != "base" else None,
            "text": tokenizer.decode(generated, skip_special_tokens=True),
            "generated_token_ids": generated, "generated_tokens": len(generated),
            "prompt_tokens": ids.shape[1], "memory_prefix_tokens": unit.length if method == "siren-kv" else 0,
            "hit_cap": len(generated) == max_new_tokens, "eos_seen": bool(generated and generated[-1] in stop_ids(model, tokenizer))}
