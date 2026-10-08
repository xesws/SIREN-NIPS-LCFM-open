#!/usr/bin/env python3
"""Explicit paid judging of one new response; never updates frozen evidence."""
import argparse
import json
from pathlib import Path
from siren.evaluation.judging import judge_one

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--item", required=True, type=Path, help="JSON with rule_condition, required_behavior, dialogue, candidate_reply")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--allow-paid-api", action="store_true")
    args = parser.parse_args()
    if not args.allow_paid_api:
        parser.error("--allow-paid-api is required; frozen result reproduction needs no API")
    if args.output.exists():
        parser.error("output already exists; choose a new path")
    result = judge_one(json.loads(args.item.read_text()), (ROOT / "prompts/judge_behavior.txt").read_text(),
                       json.loads((ROOT / "prompts/judge_behavior_config.json").read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(args.output), "votes": len(result["votes"]), "usage": result["usage"]}))


if __name__ == "__main__":
    main()
