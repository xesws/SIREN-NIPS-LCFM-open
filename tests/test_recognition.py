import numpy as np
import pytest
from sklearn.covariance import ledoit_wolf

from siren.recognition.core import (BackgroundNull, Recognizer, calibrate, fit_discriminant,
                                    hopfield_scores, initialize, negative_bank, normalize, refine)
from siren.recognition.features import interaction_span, suffix_span
from siren.baselines.text_selectors import fit_maxsim, fit_minilm_discriminant
from siren.recognition.dataset import fit_registry


def test_discriminant_matches_published_equations():
    rng = np.random.default_rng(42)
    p, n = rng.normal(size=(6, 5)) + .5, rng.normal(size=(15, 5)) - .5
    model = fit_discriminant(p, n)
    cov, _ = ledoit_wolf(np.vstack([p, n]))
    expected = np.linalg.solve(cov + 1e-6 * np.eye(5), p.mean(0) - n.mean(0))
    expected /= np.linalg.norm(expected) + 1e-8
    np.testing.assert_allclose(model.key, expected, atol=1e-15)
    ps, ns = hopfield_scores(p, expected), hopfield_scores(n, expected)
    assert model.threshold == .5 * (np.quantile(ps, .1) + np.quantile(ns, .9))
    assert model.mean == np.r_[ps, ns].mean()
    assert model.std == np.r_[ps, ns].std()
    baseline = fit_minilm_discriminant(p, n)
    np.testing.assert_array_equal(model.key, baseline.key)
    assert baseline.threshold == model.threshold


def test_single_key_score_matches_full_softmax_arithmetic():
    h = np.array([[1., 2., -1.], [0., 0., 0.], [-3., 1., 2.]])
    key = np.array([3., -1., 2.])
    q, k = normalize(h), normalize(key[None])
    logits = 20 * q @ k.T
    weights = np.exp(logits - logits.max(axis=1, keepdims=True))
    weights /= weights.sum(axis=1, keepdims=True) + 1e-8
    expected = normalize(.9 * q + .1 * normalize(weights @ k)) @ k.T
    np.testing.assert_array_equal(hopfield_scores(h, key), expected[:, 0])


def test_low_evidence_refits_from_initial_key_not_previous_key():
    p = np.array([[1., .4], [.8, .1], [.9, .3], [1.1, -.1]])
    n = np.array([[-1., .2], [-.8, -.2], [-.9, .4], [-1.1, -.4]])
    initial = initialize([0., 1.], n)
    one = refine(initial, p[:1], n)
    two = refine(one, p[:2], n)
    np.testing.assert_array_equal(two.key, normalize(4 * initial.key + p[:2].mean(0)))
    four = refine(two, p, n)
    np.testing.assert_array_equal(four.key, fit_discriminant(p, n).key)
    assert initial.evidence_count == 0 and two.evidence_count == 2
    with pytest.raises(ValueError, match="backwards"):
        refine(four, p[:2], n)


def test_maxsim_leave_one_out_changes_calibration_not_prediction():
    p = np.array([[1., 0.], [0., 1.], [.6, .8]])
    n = np.array([[-1., 0.], [0., -1.]])
    loo, own = fit_maxsim(p, n), fit_maxsim(p, n, leave_one_out=False)
    assert loo.threshold < own.threshold
    np.testing.assert_array_equal(loo.raw_scores(p), own.raw_scores(p))
    with pytest.raises(ValueError):
        fit_maxsim(p[:1], n)


def test_fit_rejects_evaluation_and_nonfinite_arrays():
    p, n = np.eye(3), -np.eye(3)
    for fn in (fit_discriminant, fit_maxsim, fit_minilm_discriminant):
        with pytest.raises(ValueError, match="fit"):
            fn(p, n, split="eval")
    with pytest.raises(ValueError):
        fit_discriminant([[np.nan, 0]], [[1, 0]])
    with pytest.raises(ValueError):
        Recognizer(np.ones(2), .2, 0., 0., 4)
    assert calibrate([1., 1.], [1., 1.])[2] == 1e-8


def test_negative_bank_preserves_other_rule_positives():
    result = negative_bank([[1., 0.]], [[2., 0.]], [np.array([[3., 0.], [4., 0.]])])
    np.testing.assert_array_equal(result[:, 0], [1., 2., 3., 4.])


def test_null_can_use_separate_frozen_calibration_bank():
    background = np.array([[1., 0.], [1., 1.]])
    calibration = np.array([[0., 1.], [-1., 0.]])
    null = BackgroundNull.fit(background, calibration_background=calibration)
    expected = hopfield_scores(calibration, normalize(background.mean(0)))
    assert null.mean == expected.mean()


def test_pooling_inclusive_suffix_and_template_whitespace():
    assert suffix_span(2, 6) == (4, 6)
    assert suffix_span(0, 5) == (2, 5)
    assert suffix_span(3, 3) == (3, 3)
    class CharacterTokenizer:
        def __call__(self, text, **kw):
            return {"offset_mapping": [(i, i + 1) for i in range(len(text))]}
    assert interaction_span(CharacterTokenizer(), "<user>hello</user>", "hello  ") == (6, 10)
    with pytest.raises(ValueError):
        suffix_span(3, 2)


def test_dataset_fit_is_independent_of_eval_annotations_and_rejects_eval_bank():
    rules = [{"rule_id": "r", "condition": "condition", "guidance": "guidance"}]
    ann = [{"id": f"p{i}", "split": "fit", "rule_id": "r", "kind": "applicable"} for i in range(4)]
    ann += [{"id": "b", "split": "fit", "rule_id": None, "kind": "background"},
            {"id": "e", "split": "eval", "rule_id": "r", "kind": "applicable"}]
    h = {f"p{i}": np.array([1., i / 10]) for i in range(4)}
    h["b"] = np.array([-1., .1])
    banks = {"r": {"positive": [f"p{i}" for i in range(4)], "negative": ["b"]}}
    first = fit_registry(rules, ann, banks, h)
    ann[-1] = {"id": "e", "split": "eval", "rule_id": "invented", "kind": "hard_negative", "answer": "changed"}
    second = fit_registry(rules, ann, banks, h)
    assert first.to_dict() == second.to_dict()
    banks["r"]["negative"].append("e")
    with pytest.raises(ValueError, match="non-fit"):
        fit_registry(rules, ann, banks, h)


def test_fit_and_evaluate_cli_from_portable_feature_cache(tmp_path, capsys):
    import json
    from siren.recognition.cli import main, load_features

    data = tmp_path / "data"
    rec = data / "recognition"
    rec.mkdir(parents=True)
    rules = [{"rule_id": "r", "condition": "c", "guidance": "g"}]
    (data / "rules.json").write_text(json.dumps(rules))
    annotations = [{"id": f"p{i}", "split": "fit", "view": "fit", "rule_id": "r", "kind": "applicable"} for i in range(4)]
    annotations += [{"id": f"b{i}", "split": "fit", "view": "fit", "rule_id": None, "kind": "background"} for i in range(3)]
    annotations += [{"id": "test_p", "split": "eval", "view": "test", "rule_id": "r", "kind": "applicable"},
                    {"id": "test_n", "split": "eval", "view": "test", "rule_id": "r", "kind": "hard_negative"}]
    (rec / "annotations.jsonl").write_text("\n".join(json.dumps(x) for x in annotations))
    (rec / "fit_banks.json").write_text(json.dumps({"r": {"positive": [f"p{i}" for i in range(4)], "negative": [f"b{i}" for i in range(3)]}}))
    feature_path = tmp_path / "features.npz"
    np.savez(feature_path, ids=np.array([r["id"] for r in annotations]),
             features=np.array([[1., .1], [.9, .2], [1., -.1], [.9, -.2], [-1., 0.], [-.9, .1], [-.9, -.1], [1., 0.], [-1., 0.]]))
    registry, output = tmp_path / "registry.json", tmp_path / "evaluation.json"
    main(["fit", "--data-root", str(data), "--features", str(feature_path), "--method", "minilm_discriminant", "--output", str(registry)])
    main(["evaluate", "--data-root", str(data), "--features", str(feature_path), "--registry", str(registry), "--output", str(output)])
    result = json.loads(output.read_text())
    assert result["rows"] == 2 and result["views"]["test"]["own_rule_auc_macro"] == 1.
    assert result["all"]["correct"] == 1 and result["all"]["true_reject"] == 1
    assert result["all"]["balanced_exact_routing_score"] == 1.
    bad = tmp_path / "bad.npz"
    np.savez(bad, ids=np.array(["x", "x"]), features=np.ones((2, 2)))
    with pytest.raises(ValueError, match="duplicated"):
        load_features(bad)


def test_feature_extractor_captures_only_prompt_and_cleans_hook():
    torch = pytest.importorskip("torch")
    from types import SimpleNamespace
    from siren.recognition.features import LlamaFeatures

    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert messages == [{"role": "user", "content": "abcd"}]
            return "<abcd>"

        def __call__(self, text, **kwargs):
            if kwargs.get("return_offsets_mapping"):
                return {"offset_mapping": [(i, i + 1) for i in range(len(text))]}
            return {"input_ids": torch.arange(len(text))[None], "attention_mask": torch.ones((1, len(text)), dtype=torch.long)}

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))
            self.block = torch.nn.Identity()
            self.model = SimpleNamespace(layers=[self.block])
            self.fail = False

        def forward(self, input_ids, attention_mask, use_cache):
            assert not torch.is_grad_enabled() and not use_cache
            if self.fail:
                raise RuntimeError("test failure")
            return self.block(input_ids.float()[..., None].expand(-1, -1, 2))

    model = Model()
    encoder = LlamaFeatures(model, Tokenizer(), layer=0)
    # Interaction [1,4], ceil(.6*4)=3, mean positions [2,3,4].
    np.testing.assert_array_equal(encoder.encode_one("abcd"), [3., 3.])
    assert model.training and not model.block._forward_hooks
    model.fail = True
    with pytest.raises(RuntimeError, match="test failure"):
        encoder.encode_one("abcd")
    assert model.training and not model.block._forward_hooks
