"""End-to-end two-pass SIREN, with a label-free visible-input interface."""
from pathlib import Path
import torch
from siren.recognition.features import LlamaFeatures
from .runtime import generate_reply


def respond(model, tokenizer, selector, sample, *, delivery="siren-kv", checkpoint_dir=None,
            max_new_tokens=1024, encoder=None):
    """Recognize the original request, select/abstain, then generate independently.

    ``sample`` accepts only {id, stream}. Evaluation labels cannot enter routing.
    The optional encoder enables CPU tests and must encode visible text only.
    Missing selected-rule state raises: it must not silently change the decision.
    """
    if not isinstance(sample, dict) or set(sample) - {"id", "stream"} or "stream" not in sample:
        raise ValueError("system inputs accept only {id, stream}; no target or rule labels")
    if delivery not in ("base", "pi-v1", "pi-v2", "siren-kv"):
        raise ValueError("system delivery must be base, pi-v1, pi-v2 or siren-kv")
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("the execution-replay selector requires original-request float32 features")
    encoder = encoder or LlamaFeatures(model, tokenizer, layer=12)
    decision = selector.select_text(sample["stream"], encoder)
    selected = decision.selected_rule
    method = "base" if selected is None or delivery == "base" else delivery
    rule, checkpoint = None, None
    if method != "base":
        entry = selector.registry[selected]
        rule = {"rule_id": entry.rule_id, "condition": entry.condition, "guidance": entry.guidance}
        if method == "siren-kv":
            if checkpoint_dir is None:
                raise ValueError("selected numeric rule requires --checkpoint-dir")
            directory = Path(checkpoint_dir).resolve()
            path = (directory / (selected + ".pt")).resolve()
            if path.parent != directory:
                raise ValueError("rule ID must identify a file inside checkpoint directory")
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            if checkpoint.get("rule") != rule:
                raise ValueError("selected rule does not match numeric checkpoint")
    result = generate_reply(model, tokenizer, sample["stream"], method=method, rule=rule,
                            checkpoint=checkpoint, max_new_tokens=max_new_tokens)
    return {"id": sample.get("id"), "route": decision.to_dict(), "generation": result}
