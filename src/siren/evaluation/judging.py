"""Behavior vote semantics, plus an explicitly invoked optional API client.

The paper's practical rubric treats evidence quote checks as diagnostics,
not as grounds to impute or change the three required boolean judgments.
Derived labels are formed per vote before majority aggregation.
"""
import json
import os

FIELDS = ("aligned_planning", "delivered_action", "stuck_repetition")


def normalize_vote(value):
    if not isinstance(value, dict) or any(type(value.get(k)) is not bool for k in FIELDS):
        raise ValueError("required judge boolean missing or wrong type")
    nested = value.get("evidence")
    nested = nested if isinstance(nested, dict) else {}
    return {**{k: value[k] for k in FIELDS},
            "evidence": {k: value.get(k, nested.get(k, default)) for k, default in
                         (("planning_quote", ""), ("action_quote", ""), ("repetition_quotes", []))},
            "rationale": value.get("rationale", "")}


def flags(vote):
    p, a, r = (vote[k] for k in FIELDS)
    return {"positive": (p or a) and not r, "action": a and not r, "repetition": r}


def merge_votes(votes):
    if len(votes) not in (2, 3):
        raise ValueError("two agreeing votes or three votes are required")
    votes = [normalize_vote(v) for v in votes]
    agree = all(votes[0][k] == votes[1][k] for k in FIELDS)
    if len(votes) == 2 and not agree:
        raise ValueError("third vote required")
    if len(votes) == 3 and agree:
        raise ValueError("unnecessary third vote")
    derived = [flags(v) for v in votes]
    out = {key: sum(v[key] for v in derived) > len(votes) / 2 for key in derived[0]}
    return {**out, "planning_only": out["positive"] and not out["action"],
            "other_nonpositive": not out["positive"] and not out["repetition"],
            "initial_disagreement": not agree, "vote_count": len(votes)}


def judge_one(item, system, config):
    """Paid API call only when invoked by the caller; no implicit judging.

    Uses exactly the frozen system prompt/config supplied by the caller. It
    stops on failed/truncated/malformed responses rather than silently changing
    the model, rubric or labels. Raw votes and token usage are returned.
    """
    import httpx
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        raise RuntimeError("DEEPSEEK_API_KEY is required for optional paid judging")
    payload = {k: item[k] for k in ("rule_condition", "required_behavior", "dialogue", "candidate_reply")}
    body = {k: config[k] for k in ("model", "temperature", "thinking", "response_format", "max_tokens")}
    body.update(stream=False, messages=[{"role": "system", "content": system},
                                       {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}])
    votes, usage, returned = [], [], []
    with httpx.Client(timeout=120) as client:
        for pass_id in range(3):
            try:
                response = client.post(config["base_url"].rstrip("/") + "/chat/completions",
                                       headers={"Authorization": "Bearer " + key}, json=body)
            except httpx.HTTPError:
                raise RuntimeError("judge request failed; usage may be unknown") from None
            if response.status_code != 200:
                raise RuntimeError(f"judge HTTP status {response.status_code}; request stopped")
            raw = response.json()
            choice = raw["choices"][0]
            if choice.get("finish_reason") != "stop":
                raise RuntimeError("judge output incomplete; request stopped")
            votes.append(normalize_vote(json.loads(choice["message"]["content"])))
            usage.append(raw.get("usage"))
            returned.append(raw.get("model"))
            if pass_id == 1 and all(votes[0][k] == votes[1][k] for k in FIELDS):
                break
    return {"labels": merge_votes(votes), "votes": votes, "usage": usage,
            "returned_models": returned, "configured_model": config["model"]}
