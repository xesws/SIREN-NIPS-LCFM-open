"""Frozen text delivery templates, adapted from SIREN's original PI controls.

Sources: e21_rag_comparison/core.py; e26_pi_template_control/runtime.py and
configs/e26/pi_template_control.json. PI is a delivery channel after selection,
not an independent retrieval implementation. Only condition/guidance are read.
"""
PI_V1 = (
    "Use the selected behavioral guidance to continue the current dialogue. "
    "Adapt it to this situation. Examples, when provided, illustrate the guidance."
)
PI_V2 = (
    PI_V1 + "\nWrite only the next assistant reply in the current dialogue."
    "\nDo not invent subsequent dialogue turns unless the user explicitly asks for a dialogue example."
)


def render_prompt(tokenizer, stream: str, rule: dict | None = None, version="pi-v1") -> str:
    if not isinstance(stream, str):
        raise TypeError("stream must be text")
    content = stream
    if rule is not None:
        if version not in ("pi-v1", "pi-v2"):
            raise ValueError("version must be pi-v1 or pi-v2")
        instruction = PI_V1 if version == "pi-v1" else PI_V2
        content = (instruction + "\n\nCondition: " + rule["condition"]
                   + "\nGuidance: " + rule["guidance"] + "\n\nCurrent dialogue:\n" + stream)
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
    )
