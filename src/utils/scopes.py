"""Scope namespace helpers.

A *scope* is a named grouping of sessions that provides a visibility boundary
within a peer. Under the hood a scope named ``therapy`` is a peer named
``scope.therapy`` that observes its member sessions and never speaks.
Developers manage scopes exclusively through the ``/scopes`` routes (and the
``scopes`` field on session creation) and never see the observer/observed
mechanics.

This module is the single source of truth for the reserved name prefix and the
``kind`` flag. Being a scope requires **both**: the reserved name prefix (the
namespace) and ``{"kind": "scope"}`` in the peer's ``internal_metadata`` JSONB
(the authoritative marker). Neither half is forgeable — the prefix sits outside
``RESOURCE_NAME_PATTERN``, and ``internal_metadata`` appears in no API schema —
so requiring both means a peer that merely occupies the namespace, or merely
carries a look-alike ``configuration``, is not a scope.
"""

from collections.abc import Iterable
from typing import Any

from src.exceptions import AuthenticationException, ValidationException
from src.security import JWTParams

# Reserved peer-name prefix for scope peers. User-created peers may not use it.
#
# The '.' is load-bearing: it is outside RESOURCE_NAME_PATTERN
# (^[a-zA-Z0-9_-]+$), the charset every peer name created through the API must
# match. No peer created through the validated API can therefore occupy this
# namespace. (Peers carried over by the users->peers rename in
# d429de0e5338 predate that pattern and were never charset-validated, so the
# legacy-collision path in crud/scope.py stays as a backstop.)
SCOPE_PEER_PREFIX = "scope."

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


def is_scope_peer(name: str, internal_metadata: dict[str, Any] | None) -> bool:
    """Authoritative scope test: reserved name AND the internal kind flag.

    Takes ``(name, internal_metadata)`` rather than a ``Peer`` so it is callable
    from an ORM instance, the cached plain dict built by ``crud.peer._fetch_peer``,
    or a raw row.
    """
    return (
        is_scope_peer_name(name)
        and bool(internal_metadata)
        and internal_metadata.get("kind") == SCOPE_KIND
    )


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


def validate_scope_read_option(
    *,
    filters: dict[str, Any] | None,
    session_id: str | None,
    jwt_params: JWTParams,
) -> None:
    """Refuse `scope` combined with `filters`/`session_id`, or a peer-scoped key."""
    if filters is not None:
        raise ValidationException("`scope` and `filters` are mutually exclusive")
    if session_id:
        raise ValidationException("`scope` and `session_id` are mutually exclusive")
    if jwt_params.p is not None:
        raise AuthenticationException(
            "`scope` requires a workspace- or admin-level key"
        )
