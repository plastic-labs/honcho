"""Retrieve Honcho conversation context formatted for LLM injection."""

from __future__ import annotations

from .client import HonchoContext, get_client


def get_context(
    ctx: HonchoContext,
    tokens: int = 2000,
) -> list[dict[str, str]]:
    """Retrieve conversation context ready for injection into an LLM prompt.

    Args:
        ctx: ``HonchoContext`` holding the user, session, and assistant IDs.
        tokens: Maximum number of tokens to include. Defaults to ``2000``.

    Returns:
        A list of message dicts: ``[{"role": "user" | "assistant", "content": "..."}]``.
        Returns an empty list if the session has no messages yet.
    """
    honcho = get_client()
    user_peer = honcho.peer(ctx.user_id)
    assistant_peer = honcho.peer(ctx.assistant_id)
    session = honcho.session(ctx.session_id)

    session.add_peers([user_peer, assistant_peer])

    context = session.context(tokens=tokens)
    return context.to_openai(assistant=ctx.assistant_id)
