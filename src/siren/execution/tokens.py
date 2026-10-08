"""Exact fixed-prefix decomposition and causal action/EOS supervision.

Adapted from original e23_native_schedule_r13/tokens.py. The original tokenizer
patch's public behavior is independently implemented here without vendoring
HoReN: count real prompt tokens, append EOS, supervise EOS even if pad == EOS.
The historical fit format uses TWO spaces between rendered prompt and target.
"""
from .prompts import render_prompt


def prefix_frame(tokenizer, rule):
    slot = "SIREN_STREAM_SLOT"
    marker = "Current dialogue:\n"
    text = render_prompt(tokenizer, slot, rule)
    if text.count(slot) != 1 or marker + slot not in text:
        raise ValueError("chat template does not preserve the stream marker")
    cut = text.index(marker + slot) + len("Current")
    encoded = tokenizer(text, add_special_tokens=True, return_offsets_mapping=True, truncation=False)
    boundaries = [i for i, (a, b) in enumerate(encoded["offset_mapping"]) if a < b and b == cut]
    if len(boundaries) != 1:
        raise ValueError("tokenizer must expose the frozen boundary after 'Current'")
    prefix_ids = list(encoded["input_ids"][:boundaries[0] + 1])
    suffix = tokenizer(text[cut:], add_special_tokens=False, truncation=False)["input_ids"]
    if prefix_ids + suffix != encoded["input_ids"]:
        raise ValueError("fixed-prefix split changes tokenization")
    return {"prefix_ids": prefix_ids, "prefix_length": len(prefix_ids), "split_character": cut}


def pack_prompt(tokenizer, rule, frame, stream):
    text = render_prompt(tokenizer, stream, rule)
    full = list(tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])
    suffix = list(tokenizer(text[frame["split_character"]:], add_special_tokens=False, truncation=False)["input_ids"])
    if frame["prefix_ids"] + suffix != full:
        raise ValueError("current stream changes the frozen token boundary")
    return {"full_ids": full, "suffix_ids": suffix, "prefix_length": frame["prefix_length"]}


def supervised_target(tokenizer, rendered_prompt, target):
    """Return one unpadded causal sequence and its first supervised token index."""
    if tokenizer.eos_token_id is None or not target.strip():
        raise ValueError("a nonempty target and EOS token are required")
    prompt_ids = list(tokenizer(rendered_prompt, add_special_tokens=True, truncation=False)["input_ids"])
    full_ids = list(tokenizer(rendered_prompt + "  " + target.strip(), add_special_tokens=True,
                              truncation=False)["input_ids"]) + [int(tokenizer.eos_token_id)]
    if not prompt_ids or full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError("target tokenization changes the prompt prefix")
    if len(full_ids) <= len(prompt_ids) + 1:
        raise ValueError("target has no content tokens")
    return full_ids, len(prompt_ids)


def pack_action(tokenizer, rule, frame, item):
    prompt = pack_prompt(tokenizer, rule, frame, item["stream"])
    full_ids, length = supervised_target(tokenizer, render_prompt(tokenizer, item["stream"], rule), item["action"])
    prefix = frame["prefix_length"]
    if full_ids[:length] != prompt["full_ids"] or full_ids[:prefix] != frame["prefix_ids"]:
        raise ValueError("teacher and student token prefixes differ")
    return {"id": item["id"], "full_ids": full_ids, "suffix_ids": full_ids[prefix:],
            "prompt_length": length - prefix, "full_prompt_length": length,
            "target_ids": full_ids[length:],
            "prediction_positions": list(range(length - prefix - 1, len(full_ids) - prefix - 1)),
            "full_prediction_positions": list(range(length - 1, len(full_ids) - 1)),
            "vocab_size": len(tokenizer)}
