"""Scope and session-allowlist option handling for the Honcho Python SDK.

The ``scope`` and ``sessions`` options appear on several read surfaces (chat,
representation, session context, search). Their validation and their wire
translation live here so those surfaces cannot drift apart — the server enforces
the same exclusions with a 422, and this raises before the round trip.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from .resolve import resolve_id

if TYPE_CHECKING:
    from ..base import ScopeBase, SessionBase

__all__ = [
    "MAX_SCOPES_PER_OPTION",
    "MAX_SESSIONS_PER_ADD",
    "MAX_SESSION_ALLOWLIST_ENTRIES",
    "resolve_scope_membership",
    "resolve_scope_option",
    "resolve_scope_session",
    "resolve_session_allowlist",
    "scope_context_fields",
    "scope_recall_fields",
    "validate_scope_id",
]

# Scope IDs are stored server-side as peer names with this prefix prepended, so
# they must leave room for it within the 512-character peer name limit.
_SCOPE_PEER_PREFIX = "scope."
MAX_SCOPE_ID_LENGTH = 512 - len(_SCOPE_PEER_PREFIX)

MAX_SCOPES_PER_OPTION = 100
MAX_SESSION_ALLOWLIST_ENTRIES = 1000
# The server accepts at most this many sessions per membership call.
MAX_SESSIONS_PER_ADD = 100

_RESOURCE_NAME_PATTERN = r"^[a-zA-Z0-9_-]+$"


def validate_scope_id(value: str) -> str:
    """Validate an unprefixed scope ID.

    Args:
        value: The scope ID as the caller supplied it.

    Returns:
        The validated scope ID, unchanged.

    Raises:
        ValueError: If the ID is empty, too long, carries the reserved prefix,
            or contains characters outside the resource-name charset.
    """
    if not 1 <= len(value) <= MAX_SCOPE_ID_LENGTH:
        raise ValueError(
            f"Scope ID must be between 1 and {MAX_SCOPE_ID_LENGTH} characters"
        )
    # Checked before the charset: the reserved prefix contains '.', which is
    # itself outside the charset, so a charset-first check would report the
    # charset instead of the real mistake for a double-prefixed ID.
    if value.startswith(_SCOPE_PEER_PREFIX):
        raise ValueError(
            f"Scope ID must not start with the reserved prefix '{_SCOPE_PEER_PREFIX}' (scope IDs are unprefixed)"
        )
    if not re.fullmatch(_RESOURCE_NAME_PATTERN, value):
        raise ValueError(f"Scope ID must match pattern {_RESOURCE_NAME_PATTERN}")
    return value


def resolve_scope_option(
    scope: "str | ScopeBase | Sequence[str | ScopeBase]",
) -> str | list[str]:
    """Resolve the ``scope`` read option to its wire value.

    A single scope stays a string; a sequence becomes a list of IDs. The two
    shapes mean different things to the server — one scope reads that scope's own
    view, a list restricts recall to the union of their member sessions — so the
    distinction is preserved rather than normalized away.

    Args:
        scope: One scope (ID or ``Scope``) or a sequence of them.

    Returns:
        A single validated scope ID, or a list of them.

    Raises:
        ValueError: On an empty sequence, an over-cap sequence, or an invalid ID.
    """
    # ``str`` is itself a Sequence, so both single-scope forms — an ID and a
    # ``Scope`` — are taken first; whatever remains is the list form.
    if isinstance(scope, str) or not isinstance(scope, Sequence):
        return validate_scope_id(resolve_id(scope))

    ids = [validate_scope_id(resolve_id(entry)) for entry in scope]
    if not ids:
        # An empty list would resolve to an empty allowlist server-side and
        # silently recall nothing, which is never the intent.
        raise ValueError("scope must name at least one scope")
    if len(ids) > MAX_SCOPES_PER_OPTION:
        raise ValueError(f"scope can name at most {MAX_SCOPES_PER_OPTION} scopes")
    return ids


def resolve_session_allowlist(
    sessions: "Sequence[str | SessionBase]",
) -> list[str]:
    """Resolve the ``sessions`` allowlist option to a list of session IDs.

    Args:
        sessions: Sessions to allow, as IDs or ``Session`` objects.

    Returns:
        The session IDs, in the order given.

    Raises:
        ValueError: On an empty list or one over the server's cap.
    """
    ids = [resolve_id(entry) for entry in sessions]
    if not ids:
        # The server treats an empty allowlist as fail-closed (recalls nothing),
        # so an empty list here is a caller mistake rather than a query.
        raise ValueError("sessions must name at least one session")
    if len(ids) > MAX_SESSION_ALLOWLIST_ENTRIES:
        raise ValueError(
            f"sessions can name at most {MAX_SESSION_ALLOWLIST_ENTRIES} sessions"
        )
    return ids


def _validate_session_id(value: str) -> str:
    """Validate a session ID against the charset the server accepts.

    Args:
        value: The session ID as the caller supplied it.

    Returns:
        The validated session ID, unchanged.

    Raises:
        ValueError: If the ID is empty or contains characters outside the
            resource-name charset.
    """
    if not value:
        raise ValueError("Session ID must be a non-empty string")
    if not re.fullmatch(_RESOURCE_NAME_PATTERN, value):
        raise ValueError(f"Session ID must match pattern {_RESOURCE_NAME_PATTERN}")
    return value


def resolve_scope_session(session: "str | SessionBase") -> str:
    """Resolve and validate a single session ID for a scope membership change.

    Validated rather than passed through because this ID is interpolated into a
    request *path*: an unvalidated value silently changes which resource the
    request addresses. ``valid-session?typo`` would target ``valid-session``
    with a stray query string, removing the wrong session from the scope and
    triggering reconciliation against it.

    Args:
        session: The session, as an ID or a ``Session`` object.

    Returns:
        The validated session ID.

    Raises:
        ValueError: If the ID is empty or malformed.
    """
    return _validate_session_id(resolve_id(session))


def resolve_scope_membership(
    sessions: "Sequence[str | SessionBase]",
) -> list[str]:
    """Resolve a scope membership change to a list of session IDs.

    Capped at the server's per-call limit rather than silently chunking, so a
    rejected batch is the batch the caller passed.

    Args:
        sessions: Sessions to add, as IDs or ``Session`` objects.

    Returns:
        The session IDs, in the order given.

    Raises:
        ValueError: On an empty list, one over the server's per-call cap, or a
            malformed session ID.
    """
    ids = [_validate_session_id(resolve_id(session)) for session in sessions]
    if not ids:
        raise ValueError("At least one session must be given")
    if len(ids) > MAX_SESSIONS_PER_ADD:
        raise ValueError(
            f"At most {MAX_SESSIONS_PER_ADD} sessions can be added per call"
        )
    return ids


def scope_context_fields(
    *,
    scope: "str | ScopeBase | None",
    sessions: "Sequence[str | SessionBase] | None",
    peer_target: str | None,
    peer_perspective: str | None,
    limit_to_session: bool,
) -> dict[str, Any]:
    """Build the query fields for ``scope``/``sessions`` on the context route.

    Unlike the recall endpoints, session context takes these as query parameters
    — ``sessions`` is sent as a repeated parameter, not as a ``filters`` body.

    Only a single scope is accepted: a scope is the *perspective source* for the
    target's representation and card, which is one observer, so a list has no
    meaning here.

    Args:
        scope: The ``scope`` option, if given.
        sessions: The ``sessions`` allowlist option, if given.
        peer_target: The observed peer. Required by either option, since both only
            reach the representation and there is none without a target.
        peer_perspective: The observing peer, if given — a scope replaces it.
        limit_to_session: Whether recall is already pinned to this session alone.

    Returns:
        The fields to merge into the query. Empty when neither option is set.

    Raises:
        ValueError: If either option is combined with something it contradicts, or
            used without ``peer_target``.
    """
    # A scope already determines what the context can see, and limit_to_session
    # already pins recall to this session alone, so combining them with a
    # perspective or an allowlist is a contradiction rather than a narrowing.
    # Raised here so the caller does not pay a round trip for a 422.
    if sessions is not None:
        if scope is not None:
            raise ValueError("`sessions` and `scope` are mutually exclusive")
        if limit_to_session:
            raise ValueError("`sessions` and `limit_to_session` are mutually exclusive")
        if peer_target is None:
            raise ValueError(
                "You must provide a `peer_target` when `sessions` is provided"
            )
        return {"sessions": resolve_session_allowlist(sessions)}

    if scope is None:
        return {}

    if peer_perspective is not None:
        raise ValueError("`scope` and `peer_perspective` are mutually exclusive")
    if peer_target is None:
        raise ValueError("You must provide a `peer_target` when `scope` is provided")
    return {"scope": validate_scope_id(resolve_id(scope))}


def scope_recall_fields(
    *,
    scope: "str | ScopeBase | Sequence[str | ScopeBase] | None",
    sessions: "Sequence[str | SessionBase] | None",
    session_id: str | None = None,
) -> dict[str, Any]:
    """Build the request-body fields for the ``scope``/``sessions`` options.

    ``sessions`` is sugar: it goes on the wire as the constrained
    ``filters: {"session_id": [...]}`` body the recall endpoints accept, never as
    a field of its own, which the server would reject as an unknown key.

    Args:
        scope: The ``scope`` option, if given.
        sessions: The ``sessions`` allowlist option, if given.
        session_id: A single session already set on the request, if any — a scope
            already determines what can be seen, so the two conflict.

    Returns:
        The fields to merge into the request body. Empty when neither option is
        set.

    Raises:
        ValueError: If ``scope`` is combined with ``sessions`` or ``session_id``,
            or if either option is itself invalid.
    """
    if scope is None:
        if sessions is None:
            return {}
        return {"filters": {"session_id": resolve_session_allowlist(sessions)}}

    if sessions is not None:
        raise ValueError("`scope` and `sessions` are mutually exclusive")
    if session_id is not None:
        raise ValueError("`scope` and `session` are mutually exclusive")
    return {"scope": resolve_scope_option(scope)}
