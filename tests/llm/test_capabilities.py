import pytest

from src.llm.capabilities import anthropic_capabilities, capabilities_for


@pytest.mark.parametrize(
    ("model", "thinking", "prefill", "forced", "preserved"),
    [
        ("claude-3-5-haiku-20241022", "budget", True, "conflicts_with_thinking", False),
        ("claude-3-7-sonnet-latest", "budget", True, "conflicts_with_thinking", False),
        ("claude-opus-4-20250514", "budget", False, "conflicts_with_thinking", False),
        ("claude-haiku-4-5", "budget", False, "conflicts_with_thinking", False),
        (
            "claude-haiku-4-5-20251001",
            "budget",
            False,
            "conflicts_with_thinking",
            False,
        ),
        ("claude-sonnet-4-6", "adaptive", False, "conflicts_with_thinking", False),
        ("claude-opus-4-8", "adaptive", False, "conflicts_with_thinking", False),
        ("claude-opus-5", "adaptive", False, "conflicts_with_thinking", False),
        ("claude-opus-5-5", "adaptive", False, "rejected", True),
        ("claude-sonnet-5-5", "adaptive", False, "rejected", True),
        ("claude-fable-5-1", "adaptive", False, "rejected", True),
        ("claude-haiku-5-5", "adaptive", False, "skips_thinking", True),
        # Vendor prefixes and suffixes still resolve to the model.
        ("anthropic/claude-sonnet-5-5", "adaptive", False, "rejected", True),
        ("anthropic.claude-haiku-5-5", "adaptive", False, "skips_thinking", True),
        ("claude-opus-4-5@20251101", "budget", False, "conflicts_with_thinking", False),
        (
            "us.anthropic.claude-opus-4-1-20250805-v1:0",
            "budget",
            False,
            "conflicts_with_thinking",
            False,
        ),
        # Proxies spell the version with a dot.
        (
            "anthropic/claude-opus-4.6",
            "adaptive",
            False,
            "conflicts_with_thinking",
            False,
        ),
        ("claude-opus-5.5", "adaptive", False, "rejected", True),
        # An unlisted newer Claude gets the strictest known behavior.
        ("claude-sonnet-6", "adaptive", False, "rejected", True),
        ("claude-haiku-5-7", "adaptive", False, "rejected", True),
        # Not a Claude model id at all.
        ("my-local-model", "budget", True, "conflicts_with_thinking", False),
    ],
)
def test_anthropic_capabilities(
    model: str, thinking: str, prefill: bool, forced: str, preserved: bool
) -> None:
    caps = anthropic_capabilities(model)

    assert caps.thinking == thinking
    assert caps.supports_prefill is prefill
    assert caps.forced_tool_choice == forced
    assert caps.preserved_thinking is preserved


def test_capabilities_for_other_transports_do_not_read_claude_rows() -> None:
    assert capabilities_for("openai", "anthropic/claude-sonnet-5-5") == (
        anthropic_capabilities("my-local-model")
    )
    assert capabilities_for("anthropic", "claude-sonnet-5-5").preserved_thinking
