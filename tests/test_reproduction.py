import importlib.util
import json
from pathlib import Path
import pytest
from siren.evaluation.judging import merge_votes

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("paper_results", ROOT / "reproduce/paper_results.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_wrong_calls_use_context_counts_not_offdiagonal_matrix_size():
    step = {"n_offdiag": 6642, "steal": {"seed": 1/9, "expanded": 2/6},
            "independent": {"seed": {"n_pos": 9}, "expanded": {"n_pos": 6}}}
    assert module.wrong_decisions(step) == 3


def test_frozen_behavior_votes_reconstruct_every_saved_label():
    path = ROOT / "artifacts/frozen/execution/judge_packets.jsonl"
    if not path.exists():
        pytest.skip("download optional frozen evidence first")
    count = 0
    for line in path.read_text().splitlines():
        row = json.loads(line)
        got = merge_votes(row["votes"])
        assert got == row["labels"]
        count += 1
    assert count == 2676


def test_frozen_eligibility_pair_matches_paper():
    path = ROOT / "artifacts/frozen/recognition/replay_with_eligibility.json"
    if not path.exists():
        pytest.skip("download optional frozen evidence first")
    result = module.recognition(ROOT / "artifacts/frozen")
    assert result["eligibility"]["with_eligibility"]["wrong_burden"] == 485
    assert result["eligibility"]["without_eligibility"]["wrong_burden"] == 1313
