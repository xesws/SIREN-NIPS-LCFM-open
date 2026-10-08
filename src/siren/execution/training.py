"""Single-rule training with final-only checkpoints and no remote services.

Algorithm sources: e23_native_schedule_r13/worker.py (KV),
e5_react/train_value_react.py (Value). Validation data never enters gradients.
Teacher logits are teacher-forced on original fit actions, never new examples.
"""
import random
import numpy as np
import torch
from .kv import RuleMemory, compile_prefix, forward_with_memory, example_weights, scheduled_objective
from .tokens import prefix_frame, pack_action
from .value import SinglePositionValue, pack_value_example, value_objective
from .identity import model_binding


def freeze_model(model, seed=42, *, required_precision="float32"):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    model.eval()
    model.requires_grad_(False)
    if required_precision is not None and any(p.dtype != getattr(torch, required_precision) for p in model.parameters()):
        raise ValueError("backbone precision differs from required " + required_precision)


@torch.no_grad()
def teacher_distribution(model, packing):
    ids = torch.tensor([packing["full_ids"]], device=next(model.parameters()).device)
    out = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
    return out.logits[0, packing["full_prediction_positions"]].float().log_softmax(-1).cpu()


def _validate_training(rule, examples):
    if not examples or not rule.get("guidance", "").strip():
        raise ValueError("a rule and nonempty training examples are required")
    ids = [x["id"] for x in examples]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate training sample ID")
    for example in examples:
        if example.get("rule_id") != rule["rule_id"] or not example["action"].strip():
            raise ValueError("training example does not belong to this rule")
        if example.get("split", "train") != "train":
            raise ValueError("only the explicit train split may enter fitting")


def train_kv(model, tokenizer, rule, examples, *, steps=25, learning_rate=.01, early_tokens=8,
             progress=None):
    _validate_training(rule, examples)
    if steps < 1 or learning_rate <= 0:
        raise ValueError("steps and learning rate must be positive")
    freeze_model(model)
    device = next(model.parameters()).device
    frame = prefix_frame(tokenizer, rule)
    capsule = compile_prefix(model, frame)
    memory = RuleMemory(capsule).to(device)
    packs = [pack_action(tokenizer, rule, frame, item) for item in examples]
    teachers = [teacher_distribution(model, record) for record in packs]
    weights = example_weights([len(record["target_ids"]) for record in packs])
    optimizer = torch.optim.Adam(memory.optimizer_groups(learning_rate), betas=(.9, .999), eps=1e-8, weight_decay=0)
    history = []
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        sums = {k: 0. for k in ("action_ce", "teacher_kl", "ce_contribution", "kl_contribution", "regularizer", "loss")}
        for weight, record, teacher in zip(weights, packs, teachers):
            ids = torch.tensor([record["suffix_ids"]], device=device)
            logits = forward_with_memory(model, memory, ids)
            terms = scheduled_objective(logits, record, teacher, early_tokens)
            terms["regularizer"] = memory.regularizer()
            terms["loss"] = terms["ce_contribution"] + terms["kl_contribution"] + .01 * terms["regularizer"]
            (weight * terms["loss"]).backward()
            for key in sums:
                sums[key] += weight * float(terms[key].detach())
            del logits, terms
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in memory.parameters()):
            raise FloatingPointError("invalid memory gradients")
        optimizer.step()
        history.append({"step": step, **sums})
        if progress:
            progress(history[-1])
    return {"schema": "siren-execution-v1", "method": "siren-kv", "rule": dict(rule),
            "model_identity": model_binding(model, tokenizer),
            "frame": frame, "capsule": capsule,
            "offsets": [{"UK": b.UK.detach().cpu(), "UV": b.UV.detach().cpu()} for b in memory.banks],
            "training_ids": [x["id"] for x in examples], "history": history,
            "recipe": {"steps": steps, "learning_rate": learning_rate, "early_tokens": early_tokens,
                       "seed": 42, "relative_radius": .5, "regularizer": .01},
            "example_weights": weights}


def train_value(model, tokenizer, rule, examples, *, steps=100, learning_rate=.1,
                layer=29, action_weight=.5, progress=None):
    """Original full-trace Value recipe, summed example gradients and norm cap."""
    _validate_training(rule, examples)
    if steps < 1 or learning_rate <= 0:
        raise ValueError("steps and learning rate must be positive")
    freeze_model(model, required_precision=None)
    device = next(model.parameters()).device
    unit = SinglePositionValue(model.config.hidden_size, layer).to(device)
    packs = [pack_value_example(tokenizer, rule, x) for x in examples]
    optimizer = torch.optim.Adam(unit.parameters(), lr=learning_rate)
    history = []
    for iteration in range(steps):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for packing in packs:
            ids = torch.tensor([packing["full_ids"]], device=device)
            with unit.inject(model, packing["prompt_length"]):
                logits = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False).logits
            terms = value_objective(logits, packing, unit.value, action_weight=action_weight)
            terms["loss"].backward()  # Original recipe sums, rather than averages, fit gradients.
            losses.append(float(terms["loss"].detach()))
        if unit.value.grad is None or not torch.isfinite(unit.value.grad).all():
            raise FloatingPointError("invalid value gradient")
        optimizer.step()
        with torch.no_grad():
            norm = unit.value.norm()
            if norm > 160:
                unit.value.mul_(160 / (norm + 1e-8))
        history.append({"step": iteration + 1, "loss": sum(losses) / len(losses),
                        "value_norm": float(unit.value.detach().norm())})
        if progress:
            progress(history[-1])
        if history[-1]["loss"] < .12 and iteration > 10:
            break
    return {"schema": "siren-execution-v1", "method": "siren-value", "rule": dict(rule),
            "model_identity": model_binding(model, tokenizer),
            "layer": layer, "value": unit.value.detach().cpu(), "history": history,
            "training_ids": [x["id"] for x in examples],
            "recipe": {"steps": steps, "learning_rate": learning_rate, "action_weight": action_weight,
                       "seed": 42, "norm_cap": 160., "norm_coefficient": .01, "early_stop_loss": .12}}
