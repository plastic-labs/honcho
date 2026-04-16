"""Retrieve Honcho conversation context formatted for LLM injection."""

from __future__ import annotations

from .client import HonchoContext, get_client


def get_context(
    ctx: HonchoContext,
    tokens: int = 2000,
) -> list[dict[str, str]]:
    """Retrieve conversation context ready for injection into an LLM prompt.

    Fetches recent messages from a Honcho session within the given token
    budget and converts them to OpenAI-compatible message format. The
    returned list is suitable for use as prior messages in a
    ``Runner.run()`` input or as context in dynamic agent instructions.

    Args:
        ctx: ``HonchoContext`` holding the user, session, and assistant IDs.
        tokens: Maximum number of tokens to include. Defaults to ``2000``.

    Returns:
        A list of message dicts in OpenAI format:
        ``[{"role": "user" | "assistant", "content": "..."}]``.
        Returns an empty list if the session has no messages yet.
    """
    honcho = get_client()
    session = honcho.session(ctx.session_id)
    context = session.context(tokens=tokens)
    # Strip 'name' field — the OpenAI Responses API does not accept it
    return [
        {"role": m["role"], "content": m["content"]}
        for m in context.to_openai(assistant=ctx.assistant_id)
    ]
