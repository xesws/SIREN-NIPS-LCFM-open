"""Single-position SIREN-Value, independent implementation of the paper write.

Source algorithms: e2_edit/split_hook.py and e5_react/train_value_react.py.
No external HoReN implementation is vendored. A selected rule's vector is added
at the last prompt token of the layer-29 MLP down-projection, once per request.
"""
from contextlib import contextmanager
import torch
from torch import nn
from torch.nn import functional as F
from .prompts import render_prompt
from .tokens import supervised_target


class SinglePositionValue(nn.Module):
    def __init__(self, hidden_size, layer=29):
        super().__init__()
        self.layer = layer
        self.value = nn.Parameter(torch.zeros(hidden_size, dtype=torch.float32))

    @contextmanager
    def inject(self, model, prompt_length):
        if prompt_length <= 0:
            raise ValueError("prompt cannot be empty")
        target = model.model.layers[self.layer].mlp.down_proj
        count = 0

        def hook(_module, _args, output):
            nonlocal count
            if count:
                return output
            if output.ndim != 3 or output.shape[0] != 1 or output.shape[1] < prompt_length:
                raise ValueError("single-position write expects an unpadded batch of one")
            count += 1
            edited = output.clone()
            edited[:, prompt_length - 1, :] += self.value.to(output.dtype)
            return edited

        handle = target.register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()


def pack_value_example(tokenizer, rule, item):
    thought = rule["guidance"].strip()
    text = render_prompt(tokenizer, item["stream"])
    ids, boundary = supervised_target(tokenizer, text, thought + " " + item["action"].strip())
    content = ids[boundary:-1]
    target = " ".join(thought.split())
    end = next((k for k in range(1, len(content) + 1)
                if " ".join(tokenizer.decode(content[:k], skip_special_tokens=True).split()) == target), None)
    if end is None or end >= len(content):
        raise ValueError("cannot separate guidance from action in supervised continuation")
    return {"id": item["id"], "full_ids": ids, "prompt_length": boundary,
            "thought_positions": list(range(boundary, boundary + end)),
            "action_positions": list(range(boundary + end, len(ids)))}


def value_objective(logits, packing, value, action_weight=.5, norm_coefficient=.01):
    ids = torch.tensor(packing["full_ids"], device=logits.device)

    def loss(positions):
        if not positions or min(positions) <= 0:
            raise ValueError("invalid supervised token span")
        return F.cross_entropy(logits[0, [p - 1 for p in positions]], ids[positions])

    thought, action = loss(packing["thought_positions"]), loss(packing["action_positions"])
    norm = value.float().square().mean()
    return {"thought_ce": thought, "action_ce": action, "regularizer": norm,
            "loss": thought + action_weight * action + norm_coefficient * norm}
