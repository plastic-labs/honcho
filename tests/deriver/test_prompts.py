from unittest.mock import patch

import pytest

from src.deriver.prompts import (
    estimate_deriver_prompt_tokens,
    estimate_minimal_deriver_prompt_tokens,
    minimal_deriver_prompt,
)


def test_minimal_deriver_prompt_includes_custom_instructions_when_present() -> None:
    prompt = minimal_deriver_prompt(
        peer_id="alice",
        messages="alice: hello",
        custom_instructions="Prefer concrete timeline facts.",
    )

    assert "CUSTOM INSTRUCTIONS:" in prompt
    assert "Prefer concrete timeline facts." in prompt


def test_minimal_deriver_prompt_omits_custom_instructions_when_absent() -> None:
    prompt = minimal_deriver_prompt(
        peer_id="alice",
        messages="alice: hello",
        custom_instructions=None,
    )

    assert "CUSTOM INSTRUCTIONS:" not in prompt


def test_minimal_deriver_prompt_limits_evidence_to_target_speaker() -> None:
    """Keep another peer's assertions out of the target representation."""

    prompt = minimal_deriver_prompt(
        peer_id="user",
        messages="assistant: user plays tennis on Tuesdays",
    )

    assert "the target peer's own messages" in prompt
    assert "Only messages whose speaker label exactly matches `Target peer:`" in prompt
    assert "Messages from other speakers are context only" in prompt
    assert "even when it explicitly names the target peer" in prompt
    assert "including with `you`/`your`" in prompt
    assert (
        "TARGET `user`, MESSAGE `assistant: user plays tennis on Tuesdays` "
        "→ no observation about `user`"
    ) in prompt
    assert (
        "TARGET `assistant`, MESSAGE `assistant: You play tennis on Tuesdays` "
        "→ no observation about `assistant`"
    ) in prompt


def test_target_speaker_rules_preserve_static_cache_prefix() -> None:
    """Keep target-specific values outside the reusable static prompt prefix."""

    alice_prompt = minimal_deriver_prompt(
        peer_id="alice",
        messages="alice: I like tea",
    )
    bob_prompt = minimal_deriver_prompt(
        peer_id="bob",
        messages="bob: I like coffee",
    )

    alice_prefix = alice_prompt.split("Target peer:", maxsplit=1)[0]
    bob_prefix = bob_prompt.split("Target peer:", maxsplit=1)[0]
    assert alice_prefix == bob_prefix


def test_estimate_deriver_prompt_tokens_increases_with_custom_instructions() -> None:
    base_tokens = estimate_minimal_deriver_prompt_tokens()
    custom_tokens = estimate_deriver_prompt_tokens(
        "Prefer explicit facts with absolute dates and keep the subject precise."
    )

    assert custom_tokens > base_tokens


def test_estimate_deriver_prompt_tokens_propagates_token_estimation_errors() -> None:
    estimate_minimal_deriver_prompt_tokens.cache_clear()

    with patch(
        "src.deriver.prompts.estimate_tokens",
        side_effect=RuntimeError("tokenizer unavailable"),
    ):
        with pytest.raises(RuntimeError, match="tokenizer unavailable"):
            estimate_deriver_prompt_tokens(None)

        with pytest.raises(RuntimeError, match="tokenizer unavailable"):
            estimate_deriver_prompt_tokens("Prefer concrete facts.")
