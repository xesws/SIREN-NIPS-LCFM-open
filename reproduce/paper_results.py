#!/usr/bin/env python3
"""Recompute paper measurements from frozen records; never call a model/API."""
from pathlib import Path
from collections import Counter, defaultdict
import argparse
import hashlib
import json
import numpy as np
from sklearn.metrics import roc_auc_score
from siren.evaluation import routing_counts, action_summary, clustered_mean

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("siren", "minilm_discriminant", "minilm_maxsim", "minilm_maxsim_self")
EXECUTION = ("siren_kv", "pi_v1", "pi_v2")
VIEWS = ("fresh", "original", "transformed", "seed", "background")


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_bundle(artifacts, data=None):
    manifest = read(artifacts / "manifest.json")
    for name, item in manifest["files"].items():
        rel = Path(name)
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError("unsafe evidence path")
        if rel.parts[:2] == ("artifacts", "frozen"):
            path = artifacts / rel.relative_to("artifacts/frozen")
        elif rel.parts[0] == "data" and data is not None:
            path = data / rel.relative_to("data")
        else:
            path = ROOT / rel
        if not path.is_file() or digest(path) != item["sha256"]:
            raise ValueError(f"Missing or changed evidence: {name}")
    return len(manifest["files"])


def routing(artifacts, data):
    annotations = [r for r in rows(data / "recognition/annotations.jsonl") if r["split"] == "eval"]
    prediction = {m: {r["id"]: r for r in rows(artifacts / f"routing/predictions_{m}.jsonl")} for m in METHODS}
    expected = read(artifacts / "routing/published_metrics.json")
    source_names = read(artifacts / "routing/index.json")["methods"]
    out = {}
    rng = np.random.default_rng(42)
    for view in VIEWS:
        subset = [r for r in annotations if r["view"] == view]
        ids = sorted({r["rule_id"] for r in subset if r.get("rule_id")})
        sample_draws = rng.integers(0, len(subset) if view == "background" else len(ids),
                                    size=(10000, len(subset) if view == "background" else len(ids)))
        result, boot = {}, {}
        for method in METHODS:
            pred = prediction[method]
            assert set(pred) == {r["id"] for r in annotations}
            decisions = {i: r["pred_rule_id"] for i, r in pred.items()}
            aggregate = routing_counts(subset, decisions)
            aggregate["reasons"] = dict(Counter(pred[r["id"]]["reason"] for r in subset))
            if view != "background":
                per = {rid: routing_counts([r for r in subset if r["rule_id"] == rid], decisions) for rid in ids}
                auc = []
                for rid in ids:
                    panel = [r for r in subset if r["rule_id"] == rid]
                    auc.append(roc_auc_score([r["kind"] == "applicable" for r in panel], [pred[r["id"]]["host_raw_score"] for r in panel]))
                aggregate["macro_auc"] = float(np.mean(auc))
                positive = [r for r in subset if r["kind"] == "applicable"]
                aggregate["target_top_ranked"] = sum(pred[r["id"]]["rank_top_rule"] == r["rule_id"] for r in positive) / len(positive)
                count_matrix = np.array([[per[r][k] for k in ("correct", "wrong", "abstain", "false_fire", "true_reject")] for r in ids])
                c, w, u, f, t = count_matrix[sample_draws].sum(axis=1).T
                boot[method] = .5 * (c / (c + w + u) + t / (f + t))
                aggregate["balanced_interval"] = np.quantile(boot[method], [.025, .975]).tolist()
            historical = expected["views"][view]["arms"][source_names[method]]
            for key in ("correct", "wrong", "abstain", "false_fire", "true_reject"):
                if key in historical:
                    assert aggregate[key] == historical[key], (view, method, key)
            if "macro_auc" in historical:
                assert abs(aggregate["macro_auc"] - historical["macro_auc"]) < 1e-12
            result[method] = aggregate
        if view != "background":
            delta = boot["siren"] - boot["minilm_discriminant"]
            result["siren_minus_minilm_discriminant"] = {
                "delta": result["siren"]["balanced_exact_routing"] - result["minilm_discriminant"]["balanced_exact_routing"],
                "low": float(np.quantile(delta, .025)), "high": float(np.quantile(delta, .975))}
        out[view] = result
    return out


def execution(artifacts, data):
    annotations = {r["id"]: r for r in rows(data / "execution/annotations.jsonl")}
    outputs = {r["output_id"]: r for r in rows(artifacts / "execution/outputs.jsonl")}
    cases = {(r["method"], r["mode"], r["sample_id"]): r["output_id"] for r in rows(artifacts / "execution/cases.jsonl")}
    stages = {(r["method"], r["installed"]): r for r in rows(artifacts / "execution/stages.jsonl")}
    order = read(data / "protocols/execution_installation.json")["rule_order"]
    assert len(annotations) == 666 and len(order) == 82
    by_split, curves, retention = {}, {m: [] for m in EXECUTION}, {}
    for split in ("main", "development", "seed"):
        ids = [sid for sid, a in annotations.items() if a["split"] == split]
        correct = [s for s in ids if stages["siren_kv", 82]["routes"][s]["selected_rule"] == annotations[s]["rule_id"]]
        by_split[split] = {"base": action_summary([outputs[cases["base", "base", s]]["labels"] for s in ids])}
        for method in EXECUTION:
            by_split[split][method] = {
                "correct_rule": action_summary([outputs[cases[method, "correct_rule", s]]["labels"] for s in ids]),
                "system": action_summary([outputs[cases[method, "system", s]]["labels"] for s in ids]),
                "after_correct_selection": action_summary([outputs[cases[method, "system", s]]["labels"] for s in correct])}
        for n in range(83):
            selections = [{s: r["selected_rule"] for s, r in stages[m, n]["routes"].items()} for m in EXECUTION]
            assert selections[0] == selections[1] == selections[2], ("selection mismatch", n)
        paired = []
        for s in ids:
            a = outputs[cases["siren_kv", "correct_rule", s]]["labels"]["action"]
            b = outputs[cases["pi_v2", "correct_rule", s]]["labels"]["action"]
            paired.append((annotations[s]["rule_id"], int(a) - int(b)))
        by_split[split]["kv_minus_pi_v2_action"] = clustered_mean(paired)
    for method in EXECUTION:
        for n in range(83):
            stage = stages[method, n]
            assert len(stage["system"]) == len(annotations)
            summary = {split: action_summary([outputs[stage["system"][sid]]["labels"] for sid, a in annotations.items() if a["split"] == split])
                       for split in ("main", "development", "seed")}
            summary["installed"] = n
            curves[method].append(summary)
        ret = {}
        for arrival, rid in enumerate(order, 1):
            ids = [s for s, a in annotations.items() if a["rule_id"] == rid]
            series = [sum(outputs[stages[method, n]["system"][s]]["labels"]["action"] for s in ids) / len(ids) for n in range(83)]
            ret[rid] = {"arrival": arrival, "post_installation": series[arrival],
                        "final_minus_post": series[-1] - series[arrival],
                        "worst_decline_from_post": min(series[arrival:]) - series[arrival]}
        retention[method] = {"rules": ret, "unchanged": sum(abs(r["final_minus_post"]) < 1e-12 for r in ret.values()),
                             "improved": sum(r["final_minus_post"] > 1e-12 for r in ret.values()),
                             "ever_declined": sum(r["worst_decline_from_post"] < -1e-12 for r in ret.values())}
    expected = {"siren_kv": (430, 127, 221), "pi_v1": (430, 128, 221), "pi_v2": (409, 118, 212)}
    for method, counts in expected.items():
        got = tuple(by_split["main"][method][k]["counts"]["action"] for k in ("correct_rule", "after_correct_selection", "system"))
        assert got == counts, (method, got, counts)
    assert by_split["main"]["base"]["counts"]["action"] == 136
    assert retention["siren_kv"]["unchanged"] == 81 and retention["siren_kv"]["improved"] == 1
    assert retention["siren_kv"]["ever_declined"] == 0
    return {"panels": by_split, "curves": curves, "retention": retention}


def wrong_decisions(step):
    # n_offdiag is a matrix-size denominator, not a count of wrong calls.
    return sum(round(float(rate) * step["independent"][rid]["n_pos"])
               for rid, rate in step["steal"].items())


def recognition(artifacts):
    histories = {name: read(artifacts / f"recognition/replay_{name}.json")["steps"]
                 for name in ("with_eligibility", "without_eligibility")}
    assert all(len(steps) == 492 for steps in histories.values())
    own = defaultdict(list)
    for step in histories["with_eligibility"]:
        rid = step["rule_id"]
        own[step["k"]].append({"rule_id": rid, "auc": step["independent"][rid]["auc"],
                                "correct": step["qtc_tpr"][rid], "wrong": step["steal"][rid], "abstain": step["silent"][rid]})
    curve = []
    for k, rr in sorted(own.items()):
        assert len(rr) == len({r["rule_id"] for r in rr}) == 82
        auc = np.array([r["auc"] for r in rr])
        curve.append({"pairs": k, "auc": float(auc.mean()), "auc_q25": float(np.quantile(auc, .25)), "auc_q75": float(np.quantile(auc, .75)),
                      **{key: float(np.mean([r[key] for r in rr])) for key in ("correct", "wrong", "abstain")}})
    eligibility = {}
    for name, steps in histories.items():
        wrong = [wrong_decisions(s) for s in steps]
        macro = [float(np.mean(list(s["qtc_tpr"].values()))) for s in steps]
        eligibility[name] = {"wrong_peak": max(wrong), "wrong_burden": sum(wrong), "terminal_wrong": wrong[-1],
                             "trajectory_correct": float(np.mean(macro)), "wrong_series": wrong, "correct_series": macro}
    a, b = eligibility["without_eligibility"], eligibility["with_eligibility"]
    assert (a["wrong_peak"], b["wrong_peak"], a["wrong_burden"], b["wrong_burden"]) == (18, 3, 1313, 485)
    assert a["terminal_wrong"] == b["terminal_wrong"] == 3
    mature_cells = 0
    for x, y in zip(histories["without_eligibility"], histories["with_eligibility"], strict=True):
        assert x["tag"] == y["tag"] and x["slim"] == y["slim"] and x["independent"] == y["independent"]
        for rid, state in x["slim"].items():
            if state["n_real"] >= 4:
                mature_cells += 1
                assert x["qtc_tpr"][rid] == y["qtc_tpr"][rid]
    assert mature_cells == 20595
    eligibility["mature_rule_snapshot_cells_with_identical_correct_selection"] = mature_cells
    eligibility["wrong_burden_reduction"] = 1 - b["wrong_burden"] / a["wrong_burden"]
    assert round(curve[0]["auc"], 3) == .743 and round(curve[-1]["auc"], 3) == .922
    return {"own_evidence_curve": curve, "eligibility": eligibility}


def expression(artifacts):
    out = {}
    for method in ("pi_v1", "pi_v2"):
        pairs = rows(artifacts / f"expression/{method}_pairs.jsonl")
        out[method] = {}
        for name, selected in {"main_all": [r for r in pairs if r["split"] == "main"],
                               "main_both_action": [r for r in pairs if r["split"] == "main" and r["outcome"] == "both"]}.items():
            counts = Counter(r["preference"] for r in selected)
            net = clustered_mean([(r["rule_id"], 1 if r["preference"] == "siren_kv" else -1 if r["preference"] == method else 0) for r in selected])
            out[method][name] = {"n": len(selected), "counts": dict(counts), "rule_balanced_net_preference": net}
    assert out["pi_v1"]["main_both_action"]["n"] == 402
    assert out["pi_v1"]["main_both_action"]["counts"]["siren_kv"] == 227
    assert out["pi_v1"]["main_both_action"]["counts"]["pi_v1"] == 123
    return out


def figures(result, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    curve = result["recognition"]["own_evidence_curve"]
    fig, ax = plt.subplots(figsize=(6, 4))
    x = [c["pairs"] for c in curve]
    ax.plot(x, [c["auc"] for c in curve], marker="o", label="Mean per-rule AUC")
    ax.fill_between(x, [c["auc_q25"] for c in curve], [c["auc_q75"] for c in curve], alpha=.2, label="Rule interquartile range")
    ax.set(xlabel="Observed own-rule evidence pairs", ylabel="ROC AUC", ylim=(0, 1.02)); ax.legend(); fig.tight_layout()
    fig.savefig(output / "recognition_curve.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for name in ("with_eligibility", "without_eligibility"):
        ax.plot(result["recognition"]["eligibility"][name]["wrong_series"], label=name.replace("_", " "))
    ax.set(xlabel="Recognition replay snapshot", ylabel="Wrong-rule decisions"); ax.legend(); fig.tight_layout()
    fig.savefig(output / "eligibility.png", dpi=180); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4))
    for method, cc in result["execution"]["curves"].items():
        ax.plot([r["installed"] for r in cc], [100 * r["main"]["rates"]["action"] for r in cc], label=method)
    base = 100 * result["execution"]["panels"]["main"]["base"]["rates"]["action"]
    ax.axhline(base, color="gray", linestyle="--", label="base")
    ax.set(xlabel="Installed execution memories", ylabel="Main-panel action completion (%)"); ax.legend(); fig.tight_layout()
    fig.savefig(output / "sequential_execution.png", dpi=180); plt.close(fig)


def markdown(result):
    lines = ["# Recomputed SIREN paper results", "", "Source: frozen per-example predictions, outputs and judgments. No model or API invoked.", "",
             "## Complete routing: new contexts", "", "| Method | Correct | Wrong | Abstain | False activation | Balanced routing |", "|---|---:|---:|---:|---:|---:|"]
    for method in METHODS:
        r = result["routing"]["fresh"][method]
        lines.append(f"| {method} | {r['correct']}/576 | {r['wrong']}/576 | {r['abstain']}/576 | {r['false_fire']}/288 | {100*r['balanced_exact_routing']:.2f}% |")
    lines += ["", "## Main execution panel", "", "| Method | Correct rule supplied | After correct selection | End to end |", "|---|---:|---:|---:|"]
    for method in EXECUTION:
        r = result["execution"]["panels"]["main"][method]
        cells = [f"{r[k]['counts']['action']}/{r[k]['n']}" for k in ("correct_rule", "after_correct_selection", "system")]
        lines.append("| " + " | ".join([method] + cells) + " |")
    lines += ["", "Base action completion: 136/464.", "", "## Evidence and retention", "",
              f"Initial / final own-rule AUC: {result['recognition']['own_evidence_curve'][0]['auc']:.6f} / {result['recognition']['own_evidence_curve'][-1]['auc']:.6f}.",
              f"Eligibility wrong-decision reduction: {100*result['recognition']['eligibility']['wrong_burden_reduction']:.2f}%.",
              "SIREN-KV: 81 rules unchanged, one improved, no post-installation action-completion decline.", "",
              "Complete metrics and rule-bootstrap intervals are in metrics.json. Figures are regenerated presentations of the frozen measurements, not new model experiments."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts/frozen")
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/paper")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    verified = verify_bundle(args.artifacts, args.data)
    result = {"verified_artifact_files": verified, "seed": 42, "bootstrap_draws": 10000,
              "routing": routing(args.artifacts, args.data), "execution": execution(args.artifacts, args.data),
              "recognition": recognition(args.artifacts), "expression": expression(args.artifacts)}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.output / "results.md").write_text(markdown(result))
    if not args.no_plots:
        figures(result, args.output)
    print(json.dumps({"verified_artifact_files": verified, "paper_counts_match": True, "output": str(args.output)}))


if __name__ == "__main__":
    main()
