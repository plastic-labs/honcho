"""The pair dialectic searches the messages before saying it has no record.

Prefetched conclusions are summaries: exact wording, tables, and anything said
only by a peer whose messages aren't derived (often the assistant) aren't in
them. The prompt sends the agent to the messages before it abstains.
"""

from unittest.mock import MagicMock

from src.dialectic.core import DialecticAgent
from src.dialectic.prompts import agent_system_prompt


def test_the_pair_prompt_searches_messages_before_abstaining() -> None:
    prompt = agent_system_prompt("assistant", "user", None, None)

    assert "Search the messages for it first" in prompt
    assert "Before you abstain, search the messages" in prompt
    # A prefetch counts as the memory search, so a level with one tool call
    # spends it on the messages.
    assert "search the messages next rather than memory again" in prompt


def test_prefetched_conclusions_point_to_the_messages() -> None:
    intro = DialecticAgent._prefetch_intro(  # pyright: ignore[reportPrivateUsage]
        MagicMock(spec=DialecticAgent)
    )

    assert "not the messages themselves" in intro
    assert "search the messages before answering" in intro
