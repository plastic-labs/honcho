"""Ingestion-time filtering of low-value messages.

Some clients (notably Claude Code session ingestion) emit tool-run breadcrumb
lines such as ``[Tool] Ran: ...`` as messages. These dominate the message store
by volume while carrying no durable signal, diluting search and distillation.

This module provides a narrow, conservative predicate that identifies those
breadcrumbs so the API can skip persisting them. It deliberately matches only
well-known breadcrumb prefixes to avoid dropping legitimate user/assistant
content that merely mentions a tool.
"""

from collections.abc import Callable, Sequence
from logging import getLogger
from typing import TypeVar

logger = getLogger(__name__)

# Prefixes that identify a message whose *entire* content is a tool-run
# breadcrumb rather than conversational content. Matched against the message
# content after stripping leading whitespace. Extend this tuple if new
# breadcrumb shapes appear; keep entries specific so legitimate prose that
# happens to start with one of these words is not caught.
TOOL_BREADCRUMB_PREFIXES: tuple[str, ...] = ("[Tool] Ran:",)

_T = TypeVar("_T")


def is_tool_run_breadcrumb(content: str) -> bool:
    """Return True if ``content`` is a tool-run breadcrumb, not real content.

    Args:
        content: The raw message content.

    Returns:
        True when the message is a known tool-run breadcrumb that should not be
        persisted to the message store.
    """
    if not content:
        return False
    return content.lstrip().startswith(TOOL_BREADCRUMB_PREFIXES)


def filter_tool_run_breadcrumbs(
    messages: Sequence[_T],
    *,
    content_of: Callable[[_T], str],
) -> list[_T]:
    """Drop messages whose content is a tool-run breadcrumb.

    Args:
        messages: Messages (or message-like objects) to filter.
        content_of: Callable returning the content string for each message.

    Returns:
        A new list preserving input order with breadcrumb messages removed.
    """
    kept = [m for m in messages if not is_tool_run_breadcrumb(content_of(m))]
    dropped = len(messages) - len(kept)
    if dropped:
        logger.debug("Filtered %d tool-run breadcrumb message(s) at ingestion", dropped)
    return kept
