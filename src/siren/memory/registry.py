"""Persistent installed-rule state; storing a rule does not authorize execution."""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile
from types import MappingProxyType

from siren.recognition import BackgroundNull, Recognizer


@dataclass(frozen=True)
class RuleEntry:
    rule_id: str
    condition: str
    guidance: str
    recognizer: Recognizer
    execution_ref: str | None = None

    def __post_init__(self):
        if any(not isinstance(x, str) or not x.strip() for x in (self.rule_id, self.condition, self.guidance)):
            raise ValueError("rule ID, condition and guidance must be nonempty strings")
        if not isinstance(self.recognizer, Recognizer):
            raise TypeError("recognizer must be a Recognizer")
        if self.execution_ref is not None and (not isinstance(self.execution_ref, str) or not self.execution_ref):
            raise ValueError("execution_ref must be a nonempty artifact identifier or None")

    def to_dict(self) -> dict:
        return {"rule_id": self.rule_id, "condition": self.condition, "guidance": self.guidance,
                "recognizer": self.recognizer.to_dict(), "execution_ref": self.execution_ref}

    @classmethod
    def from_dict(cls, value: dict) -> "RuleEntry":
        return cls(**{**value, "recognizer": Recognizer.from_dict(value["recognizer"])})


class RuleRegistry:
    """Only installed entries exist here; each entry remains separate from its eligibility.

    ``require_mature=False`` is the paper's explicit eligibility ablation, never
    the default. The null prototype and calibration are shared fit-only state.
    """

    schema = "siren-rule-registry-v1"

    def __init__(self, null: BackgroundNull, *, minimum_evidence: int = 4, require_mature: bool = True):
        if not isinstance(null, BackgroundNull):
            raise TypeError("null must be a BackgroundNull")
        if type(minimum_evidence) is not int or minimum_evidence < 0:
            raise ValueError("minimum_evidence must be a nonnegative integer")
        if type(require_mature) is not bool:
            raise TypeError("require_mature must be boolean")
        self.null = null
        self.minimum_evidence = minimum_evidence
        self.require_mature = require_mature
        self._entries: dict[str, RuleEntry] = {}

    @property
    def entries(self):
        return MappingProxyType(self._entries)

    @property
    def order(self) -> tuple[str, ...]:
        return tuple(self._entries)

    def __len__(self):
        return len(self._entries)

    def __getitem__(self, rule_id: str) -> RuleEntry:
        return self._entries[rule_id]

    def _check(self, entry: RuleEntry):
        if not isinstance(entry, RuleEntry):
            raise TypeError("expected RuleEntry")
        if entry.recognizer.dimension != len(self.null.key):
            raise ValueError("rule/null feature dimensions differ")

    def install(self, entry: RuleEntry):
        self._check(entry)
        if entry.rule_id in self._entries:
            raise ValueError("rule already installed; use update_recognizer for refinement")
        self._entries[entry.rule_id] = entry

    def eligible(self, rule_id: str) -> bool:
        entry = self._entries[rule_id]
        return not self.require_mature or entry.recognizer.evidence_count >= self.minimum_evidence

    def update_recognizer(self, rule_id: str, recognizer: Recognizer):
        old = self._entries[rule_id]
        if recognizer.evidence_count < old.recognizer.evidence_count:
            raise ValueError("evidence count cannot decrease")
        entry = replace(old, recognizer=recognizer)
        self._check(entry)
        self._entries[rule_id] = entry

    def to_dict(self) -> dict:
        return {"schema": self.schema, "null": self.null.to_dict(),
                "minimum_evidence": self.minimum_evidence, "require_mature": self.require_mature,
                "entries": [entry.to_dict() for entry in self._entries.values()]}

    @classmethod
    def from_dict(cls, value: dict) -> "RuleRegistry":
        if value.get("schema") != cls.schema:
            raise ValueError("unsupported registry schema")
        result = cls(BackgroundNull.from_dict(value["null"]), minimum_evidence=value["minimum_evidence"],
                     require_mature=value["require_mature"])
        for item in value["entries"]:
            result.install(RuleEntry.from_dict(item))
        return result

    def save(self, path):
        """Atomic JSON snapshot; contains no executable pickle payload."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        temporary = None
        try:
            with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as f:
                temporary = Path(f.name)
                f.write(payload)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temporary, path)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()

    @classmethod
    def load(cls, path) -> "RuleRegistry":
        with Path(path).open(encoding="utf-8") as f:
            return cls.from_dict(json.load(f))
