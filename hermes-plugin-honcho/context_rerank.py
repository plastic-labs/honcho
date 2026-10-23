"""Context-rerank injection: one large peer.context() fetch, narrowed by one LLM call to what this turn needs."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

_PLACEHOLDER_RE = re.compile(r"\{(peer_representation|history_of_trial|user_message)\}")
_HEADING_RE = re.compile(r"^#+\s*(.+?)\s*$")
# Unindented, so a deductive conclusion's indented premises are not counted.
_DATED_RE = re.compile(r"^(?:\[id:[^\]]*\]\s*)?\[\d{4}-")
# Representation headings, in the order counts are reported; a heading matches by its first word.
_HEADINGS = ("explicit", "deductive", "inductive", "contradictions")

DEFAULT_RERANK_BASE_URL = "https://api.anthropic.com"
DEFAULT_RERANK_PROMPT_TEMPLATE = (
    "Here is a list of potentially useful conclusions about the user, derived by a memory system with reasoning "
    "capabilities. Many of these will not be relevant to the current task.\n\n"
    "<conclusions>\n{peer_representation}\n</conclusions>\n\n"
    "An AI coding agent is working with this user. Here is the recent history of their session (user requests, "
    "the agent's actions summarised, and its replies):\n\n"
    "<history>\n{history_of_trial}\n</history>\n\n"
    "Here is the user's latest message:\n\n"
    "<latest_message>\n{user_message}\n</latest_message>\n\n"
    "Identify and summarize the 2-5 conclusions the agent most needs in order to carry out the latest message as "
    "faithfully as possible to the user's intent, especially relevant latent intent: preferences, conventions or "
    "expectations the user holds but has not restated in this message. Base every point on the conclusions above. "
    "Do not invent conclusions or guess at user intent that the conclusions do not support. If none are relevant, "
    "say so in one line.\n\n"
    "Your output is shown verbatim to the agent. Write it as concise, actionable points addressed to the agent. "
    "Do not restate the task."
)


def render_rerank_prompt(template: str, *, peer_representation: str, history_of_trial: str, user_message: str) -> str:
    """Substitute the three placeholders in one pass, so braces inside the values are left alone."""
    values = {"peer_representation": peer_representation, "history_of_trial": history_of_trial,
              "user_message": user_message}
    return _PLACEHOLDER_RE.sub(lambda m: values[m.group(1)], template)


def peer_representation_block(representation: str, card: list[str]) -> str:
    """The peer card (if any) under its own heading, then the representation markdown."""
    parts = []
    if card:
        parts.append("## Peer Card\n" + "\n".join(card))
    if representation.strip():
        parts.append(representation.strip())
    return "\n\n".join(parts)


def count_conclusions(markdown: str) -> dict[str, int]:
    """Conclusions per representation heading. Explicit and deductive conclusions are dated lines,
    inductive ones are ``**Pattern**`` entries, contradictions are ``**CONTRADICTION**`` entries."""
    counts = dict.fromkeys(_HEADINGS, 0)
    section: str | None = None
    for line in markdown.splitlines():
        if heading := _HEADING_RE.match(line):
            words = heading.group(1).split()
            first = words[0].lower() if words else ""
            section = first if first in counts else None
            continue
        if section in ("explicit", "deductive"):
            counts[section] += bool(_DATED_RE.match(line))
        elif section == "inductive":
            counts[section] += line.count("**Pattern**")
        elif section == "contradictions":
            counts[section] += line.count("**CONTRADICTION**")
    return counts


def call_rerank(prompt: str, *, api_key: str, base_url: str, model: str, max_tokens: int, timeout: float | None,
                client_factory: Callable[..., Any] | None = None) -> tuple[str, dict[str, Any]]:
    """One streamed Messages call. Returns the text blocks joined (thinking is skipped) and the token usage.
    Streaming keeps a proxy that bounds time-to-headers from cutting a long call."""
    if client_factory is None:
        import anthropic  # optional dependency: only this injection mode needs it
        client_factory = anthropic.Anthropic
    client = client_factory(api_key=api_key, base_url=base_url, timeout=timeout)
    with client.messages.stream(model=model, max_tokens=max_tokens,
                                messages=[{"role": "user", "content": prompt}]) as stream:
        message = stream.get_final_message()
    text = "".join(block.text for block in message.content if getattr(block, "type", None) == "text")
    usage = {key: getattr(message.usage, key, None) for key in (
        "input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")}
    usage["stop_reason"] = getattr(message, "stop_reason", None)
    return text, usage
