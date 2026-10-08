"""Run SIREN's real selector and then numeric/text delivery; no oracle rule option.

Default selector: artifacts/frozen/routing/execution_selector.json, the released
terminal installed-rule collection. A custom --registry must use RuleRegistry's
JSON schema and contain only the rules installed for that deployment.
"""
import argparse
import json
from pathlib import Path
from siren.memory import RuleRegistry
from siren.recognition.artifacts import load_frozen_registry
from siren.routing import Selector
from siren.execution.runtime import load_local_model
from siren.execution.system import respond


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, help="Existing local model snapshot; no downloads")
    parser.add_argument("--revision", help="Declared backbone revision, matching trained checkpoints")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--artifact-root", default="artifacts/frozen")
    parser.add_argument("--registry", type=Path, help="Optional custom installed RuleRegistry JSON")
    parser.add_argument("--delivery", choices=["siren-kv", "pi-v1", "pi-v2", "base"], default="siren-kv")
    parser.add_argument("--checkpoint-dir", type=Path, help="Final numeric checkpoints named <rule_id>.pt")
    parser.add_argument("--input", type=Path, required=True, help="Input-only JSON or JSONL, {id,stream}")
    parser.add_argument("--sample-id", help="Select an input ID from JSONL; never a target rule")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("max-new-tokens must be positive")
    if args.delivery == "siren-kv" and args.checkpoint_dir is None:
        parser.error("numeric system execution requires --checkpoint-dir")
    if args.sample_id is not None:
        rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
        rows = [row for row in rows if row.get("id") == args.sample_id]
        if len(rows) != 1:
            parser.error("sample ID must identify exactly one visible input")
        sample = rows[0]
    else:
        sample = json.loads(args.input.read_text())
    if not isinstance(sample, dict) or set(sample) - {"id", "stream"} or "stream" not in sample:
        parser.error("input accepts only id and stream; use inputs.jsonl, not annotations")
    registry = (RuleRegistry.load(args.registry) if args.registry else
                load_frozen_registry(data_root=args.data_root, artifact_root=args.artifact_root, execution=True))
    model, tokenizer = load_local_model(args.model_path, args.device, precision="float32", revision=args.revision)
    result = respond(model, tokenizer, Selector(registry), sample, delivery=args.delivery,
                     checkpoint_dir=args.checkpoint_dir, max_new_tokens=args.max_new_tokens)
    print(json.dumps(result, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
