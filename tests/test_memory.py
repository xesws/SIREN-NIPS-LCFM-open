import numpy as np
import pytest

from siren.memory import RuleEntry, RuleRegistry
from siren.recognition import BackgroundNull, Recognizer
from siren.routing import Selector


def example():
    registry = RuleRegistry(BackgroundNull(np.array([-1., 0.]), 0., 1.))
    registry.install(RuleEntry("r", "condition", "guidance", Recognizer(np.array([1., 0.]), .5, 0., 1., 4), "weights/r.safetensors"))
    return registry


def test_json_roundtrip_preserves_order_state_and_decision(tmp_path):
    registry = example()
    registry.install(RuleEntry("s", "second condition", "second guidance", Recognizer(np.array([0., 1.]), .5, 0., 1., 2)))
    path = tmp_path / "state" / "registry.json"
    registry.save(path)
    restored = RuleRegistry.load(path)
    assert restored.order == ("r", "s")
    assert restored.to_dict() == registry.to_dict()
    assert Selector(restored).select([1., 0.]).to_dict() == Selector(registry).select([1., 0.]).to_dict()
    assert not restored.eligible("s")


def test_no_duplicate_install_or_silent_refinement_rollback():
    registry = example()
    with pytest.raises(ValueError, match="already installed"):
        registry.install(registry["r"])
    with pytest.raises(ValueError, match="decrease"):
        registry.update_recognizer("r", Recognizer(np.array([1., 0.]), .5, 0., 1., 3))
    with pytest.raises(KeyError):
        registry.update_recognizer("future", registry["r"].recognizer)
    with pytest.raises(TypeError):
        registry.entries["new"] = registry["r"]


def test_registry_rejects_bad_schema_dimensions_and_nonfinite_state():
    registry = example()
    with pytest.raises(ValueError, match="dimensions"):
        registry.install(RuleEntry("different", "c", "g", Recognizer(np.ones(3), .5, 0., 1., 4)))
    payload = registry.to_dict()
    payload["schema"] = "unknown"
    with pytest.raises(ValueError, match="schema"):
        RuleRegistry.from_dict(payload)
    payload = registry.to_dict()
    payload["entries"][0]["recognizer"]["key"][0] = float("nan")
    with pytest.raises(ValueError):
        RuleRegistry.from_dict(payload)


def test_fitted_arrays_are_copied_and_read_only():
    array = np.array([1., 0.])
    model = Recognizer(array, .5, 0., 1., 4)
    array[:] = 0
    assert model.key[0] == 1
    with pytest.raises(ValueError):
        model.key[0] = 0
