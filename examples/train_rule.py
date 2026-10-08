"""Train one rule from its explicit fit split (never evaluation labels)."""
import argparse
import json
from pathlib import Path
import torch
from siren.execution.runtime import load_local_model
from siren.execution.training import train_kv, train_value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="Existing local model snapshot; never downloaded")
    parser.add_argument("--rule-id", required=True)
    parser.add_argument("--rules", type=Path, default=Path("data/rules.json"))
    parser.add_argument("--train", type=Path, help="Defaults: train.jsonl for KV, value_train.jsonl for historical Value")
    parser.add_argument("--method", choices=["siren-kv", "siren-value"], default="siren-kv")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--revision", help="Backbone revision recorded in checkpoint compatibility metadata")
    parser.add_argument("--precision", choices=["float32", "bfloat16"], help="Defaults: KV float32; historical Value bfloat16")
    parser.add_argument("--steps", type=int, help="Recipe defaults: KV 25, Value 100")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output exists; choose a new final checkpoint path")
    rules = json.loads(args.rules.read_text())
    selected = [rule for rule in rules if rule["rule_id"] == args.rule_id]
    if len(selected) != 1:
        parser.error("rule ID must identify exactly one rule")
    training_path = args.train or Path("data/execution/train.jsonl" if args.method == "siren-kv"
                                       else "data/execution/value_train.jsonl")
    rows = [json.loads(line) for line in training_path.read_text().splitlines() if line.strip()]
    examples = [row for row in rows if row["rule_id"] == args.rule_id]
    if not examples:
        parser.error("no training examples for the requested rule")
    if args.method == "siren-value" and any("thought" in row for row in examples):
        thoughts = {row.get("thought") for row in examples}
        if len(thoughts) != 1 or None in thoughts:
            parser.error("historical Value thought must be identical across a rule's fit examples")
        selected[0] = {**selected[0], "guidance": thoughts.pop()}
    if args.steps is not None and args.steps < 1:
        parser.error("steps must be positive")
    precision = args.precision or ("bfloat16" if args.method == "siren-value" else "float32")
    model, tokenizer = load_local_model(args.model_path, args.device, precision=precision, revision=args.revision)
    kwargs = {"progress": lambda row: print(json.dumps(row), flush=True)}
    if args.steps is not None:
        kwargs["steps"] = args.steps
    train = train_kv if args.method == "siren-kv" else train_value
    checkpoint = train(model, tokenizer, selected[0], examples, **kwargs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output)
    print(json.dumps({"event": "final_checkpoint_saved", "path": str(args.output),
                      "method": args.method, "rule_id": args.rule_id}), flush=True)


if __name__ == "__main__":
    main()
