"""CPU-only algorithm tests; no downloads, pretrained weights or paid calls."""
import unittest
import tempfile
from pathlib import Path
import numpy as np
import pytest
torch = pytest.importorskip("torch", reason="install .[model] to run local model mechanics tests")
pytest.importorskip("transformers")
from transformers import LlamaConfig, LlamaForCausalLM
from siren.execution.kv import RuleMemory, compile_prefix, forward_with_memory, early_mask, example_weights, scheduled_objective
from siren.execution.tokens import prefix_frame, pack_prompt, pack_action, supervised_target
from siren.execution.prompts import render_prompt, PI_V1, PI_V2
from siren.execution.training import freeze_model, train_kv, train_value
from siren.execution.runtime import memory_from_checkpoint, generate_reply
from siren.execution.identity import model_binding
from siren.execution.value import SinglePositionValue, pack_value_example


class CharacterTokenizer:
    """A tokenizer with a pad/EOS alias and exact offsets, independent of HF data."""
    eos_token_id = pad_token_id = 1

    def __len__(self):
        return 256

    def __call__(self, text, add_special_tokens=True, return_offsets_mapping=False, **_):
        ids = ([2] if add_special_tokens else []) + [ord(c) + 3 for c in text]
        result = {"input_ids": ids}
        if return_offsets_mapping:
            result["offset_mapping"] = ([(0, 0)] if add_special_tokens else []) + [(i, i + 1) for i in range(len(text))]
        if _.get("return_tensors") == "pt":
            result = {"input_ids": torch.tensor([ids]), "attention_mask": torch.ones((1, len(ids)), dtype=torch.long)}
        return result

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "U:" + messages[0]["content"].strip() + "\nA:"

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(i - 3) for i in ids if i > 2)


def tiny_model(layers=2):
    torch.manual_seed(42)
    model = LlamaForCausalLM(LlamaConfig(vocab_size=256, hidden_size=16, intermediate_size=24,
                                       num_hidden_layers=layers, num_attention_heads=2,
                                       num_key_value_heads=1, max_position_embeddings=1024,
                                       eos_token_id=1, pad_token_id=1, attention_dropout=0.0,
                                       attn_implementation="sdpa"))
    freeze_model(model)
    return model


class ExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.tok = CharacterTokenizer()
        self.rule = {"rule_id": "r1", "condition": "Needs a reason.", "guidance": "Ask why."}
        self.item = {"id": "fit1", "rule_id": "r1", "stream": "It is so.", "action": "Why?"}

    def test_frozen_templates_and_field_projection(self):
        self.assertIn("Write only the next assistant reply", PI_V2)
        self.assertTrue(PI_V2.startswith(PI_V1))
        enriched = {**self.rule, "target_rule": "DO_NOT_READ", "reference_answer": "DO_NOT_READ"}
        self.assertEqual(render_prompt(self.tok, "x", self.rule), render_prompt(self.tok, "x", enriched))
        self.assertEqual(render_prompt(self.tok, "x"), "U:x\nA:")

    def test_exact_decomposition_action_alignment_and_eos(self):
        frame = prefix_frame(self.tok, self.rule)
        prompt = pack_prompt(self.tok, self.rule, frame, self.item["stream"])
        packing = pack_action(self.tok, self.rule, frame, self.item)
        self.assertEqual(prompt["full_ids"], frame["prefix_ids"] + prompt["suffix_ids"])
        self.assertEqual(packing["target_ids"][-1], self.tok.eos_token_id)
        self.assertEqual(self.tok.decode(packing["target_ids"]), "  Why?")
        self.assertEqual(packing["target_ids"], [packing["suffix_ids"][p + 1] for p in packing["prediction_positions"]])
        self.assertEqual(packing["full_ids"][:packing["full_prompt_length"]], prompt["full_ids"])
        pack_prompt(self.tok, self.rule, frame, "")  # Empty dialogue still uses frozen boundary.

    def test_native_cache_equivalence_causality_and_request_isolation(self):
        model = tiny_model()
        prefix, suffix = [5, 9, 11], [22, 25, 17, 31]
        capsule = compile_prefix(model, {"prefix_ids": prefix})
        memory = RuleMemory(capsule)
        full = torch.tensor([prefix + suffix])
        with torch.no_grad():
            baseline = model(input_ids=full, attention_mask=torch.ones_like(full), use_cache=False).logits[:, len(prefix):]
            numeric = forward_with_memory(model, memory, torch.tensor([suffix]))
            repeated = forward_with_memory(model, memory, torch.tensor([suffix]))
            changed = forward_with_memory(model, memory, torch.tensor([suffix[:-1] + [49]]))
        torch.testing.assert_close(numeric, baseline, atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(numeric, repeated, atol=0, rtol=0)
        torch.testing.assert_close(numeric[:, :-1], changed[:, :-1], atol=2e-6, rtol=2e-5)
        self.assertEqual(memory.cache().get_seq_length(), len(prefix))
        self.assertTrue(all(not p.requires_grad for p in model.parameters()))

    def test_memory_bounds_regularizer_and_gradients(self):
        model = tiny_model()
        memory = RuleMemory(compile_prefix(model, {"prefix_ids": [5, 6, 7]}))
        self.assertEqual(float(memory.regularizer().detach()), 0.)
        with torch.no_grad():
            for bank in memory.banks:
                bank.UK.fill_(20.)
                bank.UV.fill_(-20.)
        for bank in memory.banks:
            for edited, initial in zip(bank.memory(), (bank.K0, bank.V0)):
                self.assertTrue(torch.all((edited - initial).norm(dim=-1) <= .5 * initial.norm(dim=-1) + 1e-6))
        memory = RuleMemory(compile_prefix(model, {"prefix_ids": [5, 6, 7]}))
        out = forward_with_memory(model, memory, torch.tensor([[10, 11]]))
        out.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in memory.parameters()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))

    def test_objective_full_denominator_eos_and_weighting(self):
        logits = torch.tensor([[[0., 1., 2.], [2., 1., 0.], [1., 0., 2.]]], requires_grad=True)
        packing = {"prediction_positions": [0, 1, 2], "target_ids": [2, 0, 1], "vocab_size": 3}
        teacher = torch.log_softmax(torch.zeros(3, 3), -1)
        terms = scheduled_objective(logits, packing, teacher, early_tokens=8)
        ce = torch.nn.functional.cross_entropy(logits[0], torch.tensor([2, 0, 1]), reduction="none")
        logp = logits[0].log_softmax(-1)
        kl = (teacher.exp() * (teacher - logp)).sum(-1)
        self.assertEqual(early_mask(3, 8).tolist(), [True, True, False])
        torch.testing.assert_close(terms["ce_contribution"], (ce[0] + ce[1] + .5 * ce[2]) / 3)
        torch.testing.assert_close(terms["kl_contribution"], .5 * kl[2] / 3)
        self.assertEqual(early_mask(1, 8).tolist(), [False])
        self.assertEqual(example_weights([2, 6]), [.375, .625])
        with self.assertRaises(ValueError):
            scheduled_objective(logits, packing, torch.zeros(3, 3))

    def test_single_position_hook_only_writes_once(self):
        model = tiny_model()
        unit = SinglePositionValue(16, layer=0)
        with torch.no_grad():
            unit.value.fill_(1.)
        target = model.model.layers[0].mlp.down_proj
        inputs = torch.randn(1, 4, 24)
        baseline = target(inputs)
        with unit.inject(model, 3):
            first = target(inputs)
            second = target(inputs[:, :1])
        delta = first - baseline
        torch.testing.assert_close(delta[:, 2], torch.ones(1, 16))
        torch.testing.assert_close(delta[:, [0, 1, 3]], torch.zeros(1, 3, 16))
        torch.testing.assert_close(second, target(inputs[:, :1]))
        self.assertEqual(len(target._forward_hooks), 0)

    def test_single_rule_training_reload_and_generation(self):
        model = tiny_model()
        before = {k: value.clone() for k, value in model.state_dict().items()}
        checkpoint = train_kv(model, self.tok, self.rule, [self.item], steps=1)
        self.assertEqual(len(checkpoint["history"]), 1)
        self.assertTrue(all(torch.equal(value, model.state_dict()[key]) for key, value in before.items()))
        memory = memory_from_checkpoint(checkpoint, "cpu")
        self.assertGreater(sum(float(p.detach().abs().sum()) for p in memory.parameters()), 0.)
        first = generate_reply(model, self.tok, "Hello.", method="siren-kv", checkpoint=checkpoint, max_new_tokens=3)
        second = generate_reply(model, self.tok, "Hello.", method="siren-kv", checkpoint=checkpoint, max_new_tokens=3)
        self.assertEqual(first["generated_token_ids"], second["generated_token_ids"])
        for method in ("base", "pi-v1", "pi-v2"):
            result = generate_reply(model, self.tok, "Hello.", method=method, rule=self.rule, max_new_tokens=2)
            self.assertTrue(1 <= result["generated_tokens"] <= 2)
        frame = prefix_frame(self.tok, self.rule)
        capsule = compile_prefix(model, frame)
        pristine = RuleMemory(capsule)
        zero = {"schema": "siren-execution-v1", "method": "siren-kv", "rule": self.rule,
                "frame": frame, "capsule": capsule, "model_identity": model_binding(model, self.tok),
                "offsets": [{"UK": b.UK.detach(), "UV": b.UV.detach()} for b in pristine.banks]}
        zero_output = generate_reply(model, self.tok, "Hello.", method="siren-kv", checkpoint=zero, max_new_tokens=8)
        pi_output = generate_reply(model, self.tok, "Hello.", method="pi-v1", rule=self.rule, max_new_tokens=8)
        self.assertEqual(zero_output["generated_token_ids"], pi_output["generated_token_ids"])
        value = train_value(model, self.tok, self.rule, [self.item], steps=1, layer=0)
        packing = pack_value_example(self.tok, self.rule, self.item)
        self.assertEqual(packing["full_ids"][packing["action_positions"][-1]], self.tok.eos_token_id)
        result = generate_reply(model, self.tok, "Hello.", method="siren-value", checkpoint=value, max_new_tokens=2)
        self.assertTrue(result["generated_token_ids"])
        with self.assertRaises(ValueError):
            train_kv(model, self.tok, self.rule, [{**self.item, "split": "validation"}], steps=1)

    def test_system_routes_original_input_and_abstains(self):
        from siren.memory import RuleEntry, RuleRegistry
        from siren.recognition import BackgroundNull, Recognizer
        from siren.recognition.features import LlamaFeatures
        from siren.routing import Selector
        from siren.execution.system import respond
        model = tiny_model(layers=13)
        sample = {"id": "visible", "stream": "Give a reason."}
        feature = LlamaFeatures(model, self.tok, layer=12).encode_one(sample["stream"])
        key = feature / np.linalg.norm(feature)
        registry = RuleRegistry(BackgroundNull(-key, 0., 1.))
        registry.install(RuleEntry("r1", self.rule["condition"], self.rule["guidance"],
                                   Recognizer(key, .5, 0., 1., 4)))
        selector = Selector(registry)
        frame = prefix_frame(self.tok, self.rule)
        capsule = compile_prefix(model, frame)
        unit = RuleMemory(capsule)
        checkpoint = {"schema": "siren-execution-v1", "method": "siren-kv", "rule": self.rule,
                      "frame": frame, "capsule": capsule, "model_identity": model_binding(model, self.tok),
                      "offsets": [{"UK": bank.UK.detach(), "UV": bank.UV.detach()} for bank in unit.banks]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "r1.pt"
            torch.save(checkpoint, path)
            numeric = respond(model, self.tok, selector, sample, checkpoint_dir=directory, max_new_tokens=2)
            textual = respond(model, self.tok, selector, sample, delivery="pi-v1", max_new_tokens=2)
            self.assertEqual(numeric["route"]["selected_rule"], "r1")
            self.assertEqual(numeric["route"], textual["route"])
            self.assertEqual(numeric["generation"]["generated_token_ids"], textual["generation"]["generated_token_ids"])
            checkpoint["rule"] = {**self.rule, "rule_id": "wrong"}
            torch.save(checkpoint, path)
            with self.assertRaisesRegex(ValueError, "selected rule"):
                respond(model, self.tok, selector, sample, checkpoint_dir=directory, max_new_tokens=2)
        registry.update_recognizer("r1", Recognizer(key, 2., 0., 1., 4))
        abstained = respond(model, self.tok, selector, sample, max_new_tokens=2)
        self.assertIsNone(abstained["route"]["selected_rule"])
        self.assertEqual(abstained["generation"]["method"], "base")
        with self.assertRaisesRegex(ValueError, "no target"):
            respond(model, self.tok, selector, {**sample, "target_rule": "r1"}, max_new_tokens=2)

    def test_historical_value_bfloat16_train_and_decode(self):
        model = tiny_model().to(torch.bfloat16)
        checkpoint = train_value(model, self.tok, self.rule, [self.item], steps=1, layer=0)
        self.assertEqual(checkpoint["model_identity"]["precision"], "bfloat16")
        result = generate_reply(model, self.tok, "Hello.", method="siren-value", checkpoint=checkpoint, max_new_tokens=2)
        self.assertTrue(result["generated_token_ids"])

    def test_checkpoint_backbone_and_tokenizer_compatibility(self):
        from siren.execution.identity import validate_binding
        model = tiny_model()
        binding = model_binding(model, self.tok)
        validate_binding(model, self.tok, binding)
        model.config.rope_theta += 1.
        with self.assertRaisesRegex(ValueError, "compatibility"):
            validate_binding(model, self.tok, binding)
        model = tiny_model()
        model._siren_declared_revision = "revision-a"
        binding = model_binding(model, self.tok)
        model._siren_declared_revision = "revision-b"
        with self.assertRaisesRegex(ValueError, "revision"):
            validate_binding(model, self.tok, binding)


if __name__ == "__main__":
    unittest.main()
