from src.dialectic.prompts import agent_system_prompt, workspace_agent_system_prompt
from src.dreamer.specialists import custom_instructions_prompt
from src.schemas import (
    ResolvedConfiguration,
    ResolvedDreamConfiguration,
    ResolvedPeerCardConfiguration,
    ResolvedReasoningConfiguration,
    ResolvedSummaryConfiguration,
)
from src.utils.summarizer import long_summary_prompt, short_summary_prompt


def _configuration(
    dream: str | None = None, peer_card: str | None = None
) -> ResolvedConfiguration:
    return ResolvedConfiguration(
        reasoning=ResolvedReasoningConfiguration(enabled=True),
        peer_card=ResolvedPeerCardConfiguration(
            use=True, create=True, custom_instructions=peer_card
        ),
        summary=ResolvedSummaryConfiguration(
            enabled=True, messages_per_short_summary=20, messages_per_long_summary=60
        ),
        dream=ResolvedDreamConfiguration(enabled=True, custom_instructions=dream),
    )


def test_summary_prompts_include_custom_instructions() -> None:
    for build in (short_summary_prompt, long_summary_prompt):
        prompt = build("<messages>", 50, "previous", "Write in German.")
        assert "CUSTOM INSTRUCTIONS:\nWrite in German." in prompt


def test_summary_prompts_omit_blank_custom_instructions() -> None:
    for build in (short_summary_prompt, long_summary_prompt):
        assert "CUSTOM INSTRUCTIONS:" not in build("<messages>", 50, "prev", "  ")


def test_dialectic_prompts_include_custom_instructions() -> None:
    pair = agent_system_prompt("a", "b", None, None, custom_instructions="German.")
    workspace = workspace_agent_system_prompt(custom_instructions="German.")

    assert "CUSTOM INSTRUCTIONS:\nGerman." in pair
    assert "CUSTOM INSTRUCTIONS:\nGerman." in workspace
    assert "CUSTOM INSTRUCTIONS:" not in agent_system_prompt("a", "b", None, None)


def test_dreamer_peer_card_instructions_are_additive() -> None:
    prompt = custom_instructions_prompt(
        _configuration(dream="Write in German.", peer_card="Track plan tier."),
        peer_card_enabled=True,
    )

    assert "CUSTOM INSTRUCTIONS:\nWrite in German." in prompt
    assert "PEER CARD CUSTOM INSTRUCTIONS:" in prompt
    assert prompt.endswith("Track plan tier.")


def test_dreamer_skips_peer_card_instructions_when_card_disabled() -> None:
    prompt = custom_instructions_prompt(
        _configuration(peer_card="Track plan tier."), peer_card_enabled=False
    )

    assert prompt == ""
