"""Scope namespace helpers.

A *scope* is a named grouping of sessions that provides a visibility boundary
within a peer. Under the hood a scope named ``therapy`` is a peer named
``scope__therapy`` that observes its member sessions and never speaks.
Developers manage scopes exclusively through the ``/scopes`` routes (and the
``scopes`` field on session creation) and never see the observer/observed
mechanics.

This module is the single source of truth for the reserved name prefix and
the ``kind`` configuration flag. The name prefix is the namespace; the
``{"kind": "scope"}`` flag inside the peer's configuration JSONB is the
authoritative marker that guardrails key off.
"""

from collections.abc import Iterable
from typing import Any

from src.exceptions import ValidationException

# Reserved peer-name prefix for scope peers. User-created peers may not use it.
SCOPE_PEER_PREFIX = "scope__"

# Value of the `kind` configuration flag carried by scope peers.
SCOPE_KIND = "scope"


def scope_peer_name(scope_name: str) -> str:
    """Return the peer name backing the given (unprefixed) scope name."""
    return f"{SCOPE_PEER_PREFIX}{scope_name}"


def is_scope_peer_name(name: str) -> bool:
    """Return whether a peer name lives in the reserved scope namespace."""
    return name.startswith(SCOPE_PEER_PREFIX)


def scope_name_from_peer(peer_name: str) -> str:
    """Return the unprefixed scope name for a scope peer name.

    Raises:
        ValueError: If the peer name is not in the scope namespace.
    """
    if not is_scope_peer_name(peer_name):
        raise ValueError(f"{peer_name} is not a scope peer name")
    return peer_name[len(SCOPE_PEER_PREFIX) :]


def is_scope_peer_configuration(configuration: dict[str, Any] | None) -> bool:
    """Return whether a peer configuration carries the authoritative scope flag."""
    return bool(configuration) and configuration.get("kind") == SCOPE_KIND


def validate_no_scope_peer_names(names: Iterable[str], *, action: str) -> None:
    """Reject any peer name that uses the reserved scope namespace.

    Args:
        names: Peer names to check.
        action: Human-readable guidance appended to the error, directing the
            caller to the supported path (e.g. the ``/scopes`` routes).

    Raises:
        ValidationException: If any name starts with the reserved prefix.
    """
    offenders = sorted({name for name in names if is_scope_peer_name(name)})
    if offenders:
        raise ValidationException(
            f"Peer name(s) {offenders} use the reserved scope prefix "
            + f"'{SCOPE_PEER_PREFIX}'. {action}"
        )
