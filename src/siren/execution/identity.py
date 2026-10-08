"""Portable model/tokenizer compatibility bindings for local rule checkpoints.

These bind configuration, tokenizer vocabulary/template and an optional declared
upstream revision. They are not a hash of every backbone weight shard.
"""
import hashlib
import json


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()


def model_binding(model, tokenizer):
    config = model.config.to_dict()
    fields = ("model_type", "architectures", "hidden_size", "intermediate_size", "num_hidden_layers",
              "num_attention_heads", "num_key_value_heads", "vocab_size", "max_position_embeddings",
              "rope_theta", "rope_scaling", "bos_token_id", "eos_token_id", "rms_norm_eps", "hidden_act")
    configuration = {key: config.get(key) for key in fields}
    vocabulary = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {"length": len(tokenizer)}
    tokenization = {"vocabulary": vocabulary, "chat_template": getattr(tokenizer, "chat_template", None),
                    "special_tokens": getattr(tokenizer, "special_tokens_map", {}),
                    "eos_token_id": tokenizer.eos_token_id}
    return {"configuration_sha256": _digest(configuration), "configuration": configuration,
            "tokenizer_sha256": _digest(tokenization),
            "revision": getattr(model, "_siren_declared_revision", None) or getattr(model.config, "_commit_hash", None),
            "precision": str(next(model.parameters()).dtype).removeprefix("torch.")}


def validate_binding(model, tokenizer, expected):
    if not isinstance(expected, dict):
        raise ValueError("checkpoint is missing model/tokenizer compatibility metadata")
    actual = model_binding(model, tokenizer)
    for key in ("configuration_sha256", "tokenizer_sha256", "precision"):
        if actual[key] != expected.get(key):
            raise ValueError("checkpoint model/tokenizer compatibility mismatch: " + key)
    if expected.get("revision") is not None and actual["revision"] != expected["revision"]:
        raise ValueError("checkpoint backbone revision differs; provide the same --revision declaration")
