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


def test_minimal_deriver_prompt_separates_speaker_from_subject() -> None:
    prompt = minimal_deriver_prompt(
        peer_id="assistant",
        messages="assistant: You play tennis on Tuesdays",
    )

    assert (
        "it identifies the speaker, not necessarily the subject" in prompt
    )
    assert "`I`/`me`/`my` refer to the speaker" in prompt
    assert "`you`/`your` refer to the addressee" in prompt
    assert (
        "Another speaker's statement about the target peer is valid evidence" in prompt
    )
    assert (
        "Facts about the target peer that are directly supported by any speaker's "
        "message"
    ) in prompt
    assert (
        "Speaking, hearing, receiving, acknowledging, repeating, or knowing another "
        "subject's fact is transient conversation state"
    ) in prompt
    assert "return no observations" in prompt
    assert "Omit any observation whose subject is ambiguous" in prompt
    assert (
        "TARGET `assistant`, MESSAGE `assistant: You play tennis on Tuesdays` "
        "→ no observation about `assistant`"
    ) in prompt
    assert (
        "TARGET `assistant`, MESSAGE `assistant: I prefer concise responses` "
        '→ "assistant prefers concise responses"'
    ) in prompt
    assert (
        "TARGET `user`, MESSAGE `assistant: You play tennis on Tuesdays` "
        '→ "user plays tennis on Tuesdays"'
    ) in prompt
    assert "alice works remotely on Fridays" in prompt
    assert "general knowledge" not in prompt


def test_subject_attribution_rule_matches_timestamped_message_format() -> None:
    prompt = minimal_deriver_prompt(
        peer_id="user",
        messages="2026-08-13 03:20:00 assistant: The user likes tennis",
    )

    assert "<timestamp> <speaker>: <content>" in prompt
    assert "text after the timestamp" in prompt


def test_subject_attribution_rules_preserve_static_cache_prefix() -> None:
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
