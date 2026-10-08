"""Run one explicitly selected rule (delivery-only), or an unmodified base reply.

--rule-id represents a caller's selection, NOT evaluation ground truth. This
example does not perform routing. Do not use oracle rule IDs for system scores.
"""
import argparse
import json
from pathlib import Path
import torch
from siren.execution.runtime import load_local_model, generate_reply


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--revision", help="Must match checkpoint revision metadata, when recorded")
    parser.add_argument("--precision", choices=["float32", "bfloat16"], help="Defaults: Value bfloat16; other methods float32")
    parser.add_argument("--method", choices=["base", "pi-v1", "pi-v2", "siren-kv", "siren-value"], default="base")
    parser.add_argument("--input", type=Path, required=True, help="JSON containing stream, or public inputs JSONL")
    parser.add_argument("--sample-id", help="Select one ID from inputs JSONL")
    parser.add_argument("--rules", type=Path, default=Path("data/rules.json"))
    parser.add_argument("--rule-id", help="Rule selected independently by caller; required for PI")
    parser.add_argument("--checkpoint", type=Path, help="Required for numeric methods")
    parser.add_argument("--max-new-tokens", type=int, help="Defaults: Value 96; other methods 1024")
    args = parser.parse_args()
    text = args.input.read_text()
    if args.sample_id:
        selected = [json.loads(line) for line in text.splitlines() if line.strip()]
        selected = [row for row in selected if row["id"] == args.sample_id]
        if len(selected) != 1:
            parser.error("sample ID must identify exactly one input")
        sample = selected[0]
    else:
        sample = json.loads(text)
    if set(sample) - {"id", "stream"}:
        parser.error("input accepts only id/stream; evaluator annotations are not model inputs")
    rule = None
    if args.rule_id:
        rules = json.loads(args.rules.read_text())
        matches = [row for row in rules if row["rule_id"] == args.rule_id]
        if len(matches) != 1:
            parser.error("rule ID must identify exactly one rule")
        rule = matches[0]
    if args.method.startswith("pi-") and rule is None:
        parser.error("PI requires --rule-id selected by caller")
    if args.method.startswith("siren-") and not args.checkpoint:
        parser.error("numeric delivery requires --checkpoint")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True) if args.checkpoint else None
    precision = args.precision or ("bfloat16" if args.method == "siren-value" else "float32")
    limit = args.max_new_tokens or (96 if args.method == "siren-value" else 1024)
    model, tokenizer = load_local_model(args.model_path, args.device, precision=precision, revision=args.revision)
    result = generate_reply(model, tokenizer, sample["stream"], method=args.method, rule=rule,
                            checkpoint=checkpoint, max_new_tokens=limit)
    print(json.dumps({"id": sample.get("id"), **result}, ensure_ascii=False))


if __name__ == "__main__":
    main()
