"""Named-scope credentials and the SDK calls they cannot make.

Honcho 3.4.0 named-scope keys (audience ``honcho:named-scope``, claim ``sc``)
may recall and ingest inside one scope. ``POST /v3/workspaces`` and
``POST /peers`` are not on that list. The high-level SDK calls the first
before nearly every method, and ``Honcho.peer()`` is the second, so a
restricted key never reaches chat or representation unless the CLI skips both.
"""

from __future__ import annotations

import base64
import json
from typing import Any

from honcho import Peer

NAMED_SCOPE_AUDIENCE = "honcho:named-scope"
_CLIENT_FLAG = "_honcho_cli_named_scope"
_SCOPE_NAME = "_honcho_cli_scope_name"


def jwt_claims(token: str | None) -> dict[str, Any] | None:
    """Decode a JWT payload without verifying it.

    Returns None when ``token`` is not a three-segment JWT or the payload is
    not a JSON object. Signature checks stay with the server.
    """
    if not token or token.count(".") != 2:
        return None
    payload = token.split(".", 2)[1]
    pad = "=" * (-len(payload) % 4)
    try:
        data = json.loads(base64.urlsafe_b64decode(payload + pad))
    except (ValueError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def is_named_scope_key(token: str | None) -> bool:
    """True for a 3.4.0 named-scope JWT, or any token that carries ``sc``.

    A token with ``sc`` but the dedicated audience is rejected by a 3.4.0
    server. It still must not be sent through workspace provisioning.
    """
    claims = jwt_claims(token)
    if claims is None:
        return False
    aud = claims.get("aud")
    if aud == NAMED_SCOPE_AUDIENCE or (
        isinstance(aud, list) and NAMED_SCOPE_AUDIENCE in aud
    ):
        return True
    scope_name = claims.get("sc")
    return isinstance(scope_name, str) and bool(scope_name)


def arm_named_scope_client(client: Any, token: str | None) -> None:
    """Skip SDK workspace provisioning and remember the key's scope name.

    ``_workspace_ensured`` is the SDK's own latch: once set, ``_ensure_workspace``
    does not post ``POST /v3/workspaces``.
    """
    claims = jwt_claims(token) or {}
    scope_name = claims.get("sc")
    client._workspace_ensured = True
    setattr(client, _CLIENT_FLAG, True)
    setattr(
        client,
        _SCOPE_NAME,
        scope_name if isinstance(scope_name, str) and scope_name else None,
    )


def is_named_scope_client(client: Any) -> bool:
    """True only when this process armed the client.

    A bare ``MagicMock`` answers ``getattr`` with another mock, which is
    truthy. Compare with ``True`` so test doubles stay on the normal path.
    """
    return getattr(client, _CLIENT_FLAG, False) is True


def named_scope_name(client: Any) -> str | None:
    name = getattr(client, _SCOPE_NAME, None)
    return name if isinstance(name, str) else None


def peer_for_recall(client: Any, peer_id: str) -> Any:
    """A peer handle for chat or representation.

    ``Honcho.peer()`` get-or-creates the peer. A named-scope key cannot call
    that route; chat and representation address the peer on their own routes,
    so the handle is built locally and makes no request of its own.
    """
    if is_named_scope_client(client):
        return Peer(peer_id, client)
    return client.peer(peer_id)
