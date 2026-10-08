"""Portable extraction, fitting and evaluation of the released selector panels."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from siren.memory import RuleRegistry
from siren.routing import Selector
from .core import BackgroundNull
from .dataset import fit_registry


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def load_features(path) -> tuple[dict[str, np.ndarray], dict]:
    """NPZ interface: ids (Unicode N), features (N,D), optional metadata_json scalar."""
    with np.load(path, allow_pickle=False) as data:
        ids, features = data["ids"], data["features"]
        if ids.ndim != 1 or ids.dtype.kind not in ("U", "S"):
            raise ValueError("feature IDs must be a one-dimensional string array")
        ids = [x.decode("utf-8") if isinstance(x, bytes) else str(x) for x in ids]
        if len(ids) != len(set(ids)) or any(not uid for uid in ids):
            raise ValueError("feature IDs are empty or duplicated")
        if features.ndim != 2 or features.shape[0] != len(ids) or features.shape[1] == 0 or not np.isfinite(features).all():
            raise ValueError("invalid feature matrix")
        metadata = json.loads(str(data["metadata_json"].item())) if "metadata_json" in data else {}
        if not isinstance(metadata, dict):
            raise ValueError("feature metadata must be a JSON object")
        return dict(zip(ids, features)), metadata


def fit(args):
    root = Path(args.data_root)
    features, metadata = load_features(args.features)
    null = BackgroundNull.from_dict(read_json(args.null_state)) if args.null_state else None
    registry = fit_registry(read_json(root / "rules.json"),
                            read_jsonl(root / "recognition/annotations.jsonl"),
                            read_json(root / "recognition/fit_banks.json"), features,
                            method=args.method, null=null)
    registry.save(args.output)
    return {"event": "selector_fitted", "method": args.method, "rules": len(registry),
            "output": str(args.output), "features": metadata,
            "null_source": "provided_frozen_state" if null is not None else "fit_background_mean"}


def summarize_routing(rows) -> dict:
    counts = {name: 0 for name in ("correct", "wrong", "abstain", "false_fire", "true_reject")}
    for row in rows:
        if row["kind"] == "applicable":
            result = "abstain" if row["selected_rule"] is None else "correct" if row["selected_rule"] == row["rule_id"] else "wrong"
        elif row["kind"] in ("hard_negative", "background"):
            result = "true_reject" if row["selected_rule"] is None else "false_fire"
        else:
            raise ValueError("unknown routing annotation kind")
        counts[result] += 1
    c, w, u, f, t = [counts[x] for x in ("correct", "wrong", "abstain", "false_fire", "true_reject")]
    p, n = c + w + u, f + t
    return {**counts, "applicable_count": p, "negative_count": n,
            "correct_rule_recall": c / p if p else None,
            "activation_tpr": (c + w) / p if p else None,
            "false_activation_rate": f / n if n else None,
            "correct_rejection_rate": t / n if n else None,
            "exact_routing_accuracy": (c + t) / (p + n) if p + n else None,
            "balanced_exact_routing_score": .5 * (c / p + t / n) if p and n else None}


def evaluate(args):
    root = Path(args.data_root)
    registry = RuleRegistry.load(args.registry)
    features, metadata = load_features(args.features)
    annotations = [r for r in read_jsonl(root / "recognition/annotations.jsonl") if r["split"] == "eval"]
    if len({r["id"] for r in annotations}) != len(annotations):
        raise ValueError("duplicate evaluation IDs")
    selector = Selector(registry)
    predictions, auc_groups = [], defaultdict(list)
    for ann in annotations:
        # Only numerical input enters selection. Annotation joins happen afterward.
        h = features[ann["id"]]
        decision = selector.select(h)
        row = {"id": ann["id"], "view": ann["view"], "kind": ann["kind"],
               "rule_id": ann["rule_id"], **decision.to_dict()}
        rid = ann["rule_id"]
        if ann["kind"] != "background" and rid in registry.entries:
            raw = float(registry[rid].recognizer.raw_scores(np.asarray(h)[None])[0])
            row["own_rule_score"] = raw
            auc_groups[(ann["view"], rid)].append((int(ann["kind"] == "applicable"), raw))
        predictions.append(row)
    aucs = defaultdict(dict)
    for (view, rid), pairs in auc_groups.items():
        target, score = zip(*pairs)
        if len(set(target)) == 2:
            aucs[view][rid] = float(roc_auc_score(target, score))
    views = sorted({row["view"] for row in predictions})
    report = {"schema": "siren-selector-evaluation-v1", "features": metadata,
              "rows": len(predictions), "installed_rules": list(registry.order),
              "all": summarize_routing(predictions),
              "views": {v: {**summarize_routing([r for r in predictions if r["view"] == v]),
                             "own_rule_auc_macro": float(np.mean(list(aucs[v].values()))) if aucs[v] else None,
                             "own_rule_auc": aucs[v]} for v in views},
              "negative_protocol": "Every activation on a hard-negative or background row is false under the published protocol; hard-negative labels are target-relative.",
              "predictions": predictions}
    write_json(args.output, report)
    return {"event": "selector_evaluated", "rows": len(predictions), "output": str(args.output), "views": report["views"]}


def extract(args):
    records = read_jsonl(Path(args.data_root) / "recognition/inputs.jsonl")
    if any(set(row) != {"id", "stream"} for row in records):
        raise ValueError("feature extraction accepts input-only {id, stream} records")
    ids = [row["id"] for row in records]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate input IDs")
    if args.encoder == "minilm":
        from siren.baselines.text_selectors import MiniLMEncoder
        encoder = MiniLMEncoder.from_pretrained(args.model_path, revision=args.revision, device=args.device,
                                                local_files_only=not args.allow_download)
    else:
        from .features import LlamaFeatures
        encoder = LlamaFeatures.from_pretrained(args.model_path, revision=args.revision, device=args.device,
                                                precision=args.precision, local_files_only=not args.allow_download)
    embeddings = encoder.encode([row["stream"] for row in records])
    metadata = {"encoder": args.encoder, "revision": args.revision,
                "precision": args.precision if args.encoder == "llama" else "float32",
                "pooling": "last_60pct_mean_published_offsets" if args.encoder == "llama" else "model_default",
                "model_source": Path(args.model_path).name}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        np.savez_compressed(f, ids=np.asarray(ids, dtype=str), features=embeddings,
                            metadata_json=np.asarray(json.dumps(metadata)))
    return {"event": "features_extracted", "rows": len(ids), "output": str(out), "metadata": metadata}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name, function in (("fit", fit), ("evaluate", evaluate), ("extract", extract)):
        p = sub.add_parser(name)
        p.add_argument("--data-root", default="data")
        p.add_argument("--output", required=True)
        p.set_defaults(function=function)
        if name in ("fit", "evaluate"):
            p.add_argument("--features", required=True, help="NPZ with ids and features arrays")
        if name == "fit":
            p.add_argument("--method", required=True, choices=("siren", "minilm_discriminant", "minilm_maxsim", "minilm_maxsim_self"))
            p.add_argument("--null-state", help="optional frozen BackgroundNull JSON for exact reproduction")
        elif name == "evaluate":
            p.add_argument("--registry", required=True)
        else:
            p.add_argument("--encoder", required=True, choices=("llama", "minilm"))
            p.add_argument("--model-path", required=True)
            p.add_argument("--revision")
            p.add_argument("--device", default="cpu")
            p.add_argument("--precision", choices=("bfloat16", "float32"), default="bfloat16")
            p.add_argument("--allow-download", action="store_true", help="explicitly allow upstream model download")
    args = parser.parse_args(argv)
    print(json.dumps(args.function(args), ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
