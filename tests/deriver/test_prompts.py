import importlib.util
from pathlib import Path

_PROMPTS_PATH = Path(__file__).resolve().parents[2] / "src" / "deriver" / "prompts.py"
_PROMPTS_SPEC = importlib.util.spec_from_file_location(
    "deriver_prompts", _PROMPTS_PATH
)
assert _PROMPTS_SPEC is not None and _PROMPTS_SPEC.loader is not None
_PROMPTS_MODULE = importlib.util.module_from_spec(_PROMPTS_SPEC)
_PROMPTS_SPEC.loader.exec_module(_PROMPTS_MODULE)

minimal_deriver_prompt = _PROMPTS_MODULE.minimal_deriver_prompt


def test_minimal_deriver_prompt_prefers_durable_observations() -> None:
    """The deriver prompt should bias toward durable memory, not session noise."""
    prompt = minimal_deriver_prompt(
        peer_id="alice",
        messages="alice: Please always give me the short version first.",
    )

    assert "likely to remain useful beyond the current session" in prompt
    assert "Do NOT extract one-off tasks" in prompt
    assert "temporary workflow steps" in prompt
    assert "ephemeral process updates" in prompt
    assert 'SKIP: "I\'m going to copy this into a markdown file now"' in prompt
    assert 'SKIP: "The sync job is at step 1000 of 3000"' in prompt
    assert 'EXPLICIT: "Please always give me the short version first"' in prompt
