"""Route-policy enumeration for the scopes facade.

Three review passes over the scopes work each found the same *class* of defect —
a route nobody had checked, not logic that was subtly wrong. One of them
(`PUT /sessions/{id}/peers/{id}/config`, which let any caller set a scope to
``observe_others=false`` and silently stop all fan-out into it) predated the
work entirely: the guardrail set was assembled guardrail-by-guardrail rather
than derived from the route list. Sampling review cannot close that kind of gap.

So this module enumerates instead. It derives every route through which a peer
name can reach the system, and requires each to be classified as either
GUARDED (a real scope is refused) or EXEMPT (with a stated reason). A newly
added peer-touching route fails `test_every_peer_touching_route_is_classified`
until someone consciously classifies it.

Two invariants are then asserted *behaviorally* — by calling the routes, not by
inspecting annotations, because the guards deliberately live in crud (which is
what makes `messages/upload` guarded for free, via `crud.create_messages`):

1. A real scope peer is refused on every GUARDED route.
2. An *unflagged* peer merely occupying the reserved namespace is NOT a scope and
   is unaffected. This is the half most easily broken by a well-meaning guard —
   it regressed once already when `update_peer` used a name-based check.

Known limitation: this covers the HTTP surface only. Peer names also reach the
system through the deriver, dreamer, and queue, which have no route table to
enumerate; a guard gap there would not be caught here.
"""

from collections.abc import Callable, Iterator

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from nanoid import generate as generate_nanoid
from pydantic import BaseModel
from pydantic.fields import FieldInfo
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.main import app
from src.models import Peer, Workspace
from src.utils.scopes import scope_peer_name

# Parameter and model-field names that carry a peer name.
_PEER_PARAM_NAMES = {"peer_id", "peer_name", "peer_names"}

# The scopes router *is* the facade; scope peers are its whole subject.
_SCOPES_PREFIX = "/v3/workspaces/{workspace_id}/scopes"

Builder = Callable[[TestClient, str, str, str], object]


def _request_update_peer(client: TestClient, ws: str, _session: str, peer: str):
    return client.put(
        f"/v3/workspaces/{ws}/peers/{peer}", json={"metadata": {"k": "v"}}
    )


def _request_create_peer(client: TestClient, ws: str, _session: str, peer: str):
    return client.post(f"/v3/workspaces/{ws}/peers", json={"id": peer})


def _request_chat(client: TestClient, ws: str, _session: str, peer: str):
    return client.post(
        f"/v3/workspaces/{ws}/peers/{peer}/chat", json={"query": "what do you know?"}
    )


def _request_representation(client: TestClient, ws: str, _session: str, peer: str):
    return client.post(f"/v3/workspaces/{ws}/peers/{peer}/representation", json={})


def _request_create_session_with_peer(
    client: TestClient, ws: str, _session: str, peer: str
):
    return client.post(
        f"/v3/workspaces/{ws}/sessions",
        json={"id": str(generate_nanoid()), "peers": {peer: {}}},
    )


def _request_create_message(client: TestClient, ws: str, session: str, peer: str):
    return client.post(
        f"/v3/workspaces/{ws}/sessions/{session}/messages",
        json={"messages": [{"peer_id": peer, "content": "hello"}]},
    )


def _request_upload_message(client: TestClient, ws: str, session: str, peer: str):
    return client.post(
        f"/v3/workspaces/{ws}/sessions/{session}/messages/upload",
        data={"peer_id": peer},
        files={"file": ("note.txt", b"hello there", "text/plain")},
    )


def _request_add_session_peers(client: TestClient, ws: str, session: str, peer: str):
    return client.post(f"/v3/workspaces/{ws}/sessions/{session}/peers", json={peer: {}})


def _request_set_session_peers(client: TestClient, ws: str, session: str, peer: str):
    return client.put(f"/v3/workspaces/{ws}/sessions/{session}/peers", json={peer: {}})


def _request_remove_session_peers(client: TestClient, ws: str, session: str, peer: str):
    return client.request(
        "DELETE", f"/v3/workspaces/{ws}/sessions/{session}/peers", json=[peer]
    )


def _request_set_peer_config(client: TestClient, ws: str, session: str, peer: str):
    return client.put(
        f"/v3/workspaces/{ws}/sessions/{session}/peers/{peer}/config",
        json={"observe_others": False, "observe_me": True},
    )


# Routes that must refuse a real scope peer. Each maps to a request builder so
# the guard is proven by calling it, not by trusting an annotation.
SCOPE_GUARDED_ROUTES: dict[tuple[str, str], Builder] = {
    ("POST", "/v3/workspaces/{workspace_id}/peers"): _request_create_peer,
    ("PUT", "/v3/workspaces/{workspace_id}/peers/{peer_id}"): _request_update_peer,
    ("POST", "/v3/workspaces/{workspace_id}/peers/{peer_id}/chat"): _request_chat,
    (
        "POST",
        "/v3/workspaces/{workspace_id}/peers/{peer_id}/representation",
    ): _request_representation,
    (
        "POST",
        "/v3/workspaces/{workspace_id}/sessions",
    ): _request_create_session_with_peer,
    (
        "POST",
        "/v3/workspaces/{workspace_id}/sessions/{session_id}/messages",
    ): _request_create_message,
    (
        "POST",
        "/v3/workspaces/{workspace_id}/sessions/{session_id}/messages/upload",
    ): _request_upload_message,
    (
        "POST",
        "/v3/workspaces/{workspace_id}/sessions/{session_id}/peers",
    ): _request_add_session_peers,
    (
        "PUT",
        "/v3/workspaces/{workspace_id}/sessions/{session_id}/peers",
    ): _request_set_session_peers,
    (
        "DELETE",
        "/v3/workspaces/{workspace_id}/sessions/{session_id}/peers",
    ): _request_remove_session_peers,
    (
        "PUT",
        "/v3/workspaces/{workspace_id}/sessions/{session_id}/peers/{peer_id}/config",
    ): _request_set_peer_config,
}

# Routes a scope peer may legitimately reach. Every entry needs a reason: adding
# one is a deliberate decision that a scope in this position is harmless.
SCOPE_EXEMPT_ROUTES: dict[tuple[str, str], str] = {
    ("GET", "/v3/workspaces/{workspace_id}/peers/{peer_id}/card"): (
        "peer_id is the *observer*. The Dreamer writes (scope, observed) cards, so "
        "reading a scope's card of another peer is normal operation."
    ),
    ("PUT", "/v3/workspaces/{workspace_id}/peers/{peer_id}/card"): (
        "Same: peer_id is the observer. See DEV-1998 follow-up for the narrower "
        "question of a scope's *self* card (target omitted), which is junk data "
        "rather than a leak since nothing forms a representation of a scope."
    ),
    ("GET", "/v3/workspaces/{workspace_id}/peers/{peer_id}/context"): (
        "Read-only. Scope-as-path-peer on the context routes is closed in Phase 2b "
        "(DEV-1998), which owns the read-side scope surface."
    ),
    ("POST", "/v3/workspaces/{workspace_id}/peers/{peer_id}/search"): (
        "Read-only, and empty by construction: a scope can never author messages, "
        "so there is nothing to search."
    ),
    ("POST", "/v3/workspaces/{workspace_id}/peers/{peer_id}/sessions"): (
        "Read-only. A scope legitimately has member sessions; this is the "
        "observer-mechanics view of what GET /scopes/{id}/sessions exposes."
    ),
    ("GET", "/v3/workspaces/{workspace_id}/sessions/{session_id}/peers"): (
        "Read-only membership listing, which includes the scope observer."
    ),
    (
        "GET",
        "/v3/workspaces/{workspace_id}/sessions/{session_id}/peers/{peer_id}/config",
    ): "Read-only. The write side of this route IS guarded.",
    ("POST", "/v3/keys"): (
        "Mints a scoped JWT rather than touching a peer. Scope-bound keys are "
        "Phase 3 (DEV-2002)."
    ),
}

# Routes excluded from the squatter check, with why. The guarded direction is
# still asserted for all of these.
_SQUATTER_CHECK_SKIPS: dict[tuple[str, str], str] = {
    ("POST", "/v3/workspaces/{workspace_id}/peers"): (
        "Creating a reserved-prefix peer is refused for ANY name in the namespace, "
        "flagged or not — that is the point of reserving it. Covered by "
        "test_scopes.py::test_peer_create_rejects_reserved_prefix."
    ),
}

# Routes whose 422 legitimately comes from request-schema validation rather than a
# scope guard, so the response detail is pydantic's rather than ours. Everything
# else must name the offending peer AND say why, or the test would happily pass on
# an unrelated 422 (a malformed body, say) and prove nothing.
_SCHEMA_LEVEL_REJECTION: dict[tuple[str, str], str] = {
    ("POST", "/v3/workspaces/{workspace_id}/peers"): (
        "PeerCreate.name carries RESOURCE_NAME_PATTERN, which the reserved prefix "
        "violates, so pydantic refuses it before the route body runs."
    ),
}


def _nested_models(annotation: object, seen: set[object]) -> Iterator[type[BaseModel]]:
    """Yield `annotation` and every pydantic model nested inside it."""
    if (
        not isinstance(annotation, type)
        or not issubclass(annotation, BaseModel)
        or annotation in seen
    ):
        return
    model: type[BaseModel] = annotation
    seen.add(model)
    yield model
    fields: dict[str, FieldInfo] = model.model_fields
    for field in fields.values():
        stack = [field.annotation]
        while stack:
            current = stack.pop()
            yield from _nested_models(current, seen)
            stack.extend(getattr(current, "__args__", ()) or ())


def _peer_param_names(route: APIRoute) -> set[str]:
    """Peer-name-carrying params anywhere in a route's dependant tree.

    Walks sub-dependencies so `Form(...)` params behind a parser dependency are
    seen (this is how `messages/upload` takes its `peer_id`), and descends into
    request-body models so `MessageCreate.peer_name` is seen too.
    """
    found: set[str] = set()
    seen: set[object] = set()
    stack = [route.dependant]
    while stack:
        dependant = stack.pop()
        params = (
            dependant.path_params
            + dependant.query_params
            + dependant.header_params
            + dependant.body_params
        )
        for param in params:
            if param.name in _PEER_PARAM_NAMES:
                found.add(param.name)
            annotations = [param.field_info.annotation]
            while annotations:
                annotation = annotations.pop()
                for model in _nested_models(annotation, seen):
                    found |= set(model.model_fields) & _PEER_PARAM_NAMES
                annotations.extend(getattr(annotation, "__args__", ()) or ())
        stack.extend(dependant.dependencies)
    return found


def _peer_touching_routes() -> set[tuple[str, str]]:
    """Every (method, path) through which a peer name can reach the system.

    Union of two signals, because neither alone is sufficient: parameter names
    miss routes where peer names are dict *keys* (`POST /sessions/{id}/peers`
    takes `dict[str, SessionPeerConfig]`), and path shape misses names carried in
    a body or form field.
    """
    found: set[tuple[str, str]] = set()
    for route in app.routes:
        if not isinstance(route, APIRoute):
            continue
        path = route.path.rstrip("/") or route.path
        if path.startswith(_SCOPES_PREFIX):
            continue
        by_shape = "{peer_id}" in path or path.endswith("/peers")
        if not (by_shape or _peer_param_names(route)):
            continue
        for method in route.methods or set():
            if method not in ("HEAD", "OPTIONS"):
                found.add((method, path))
    return found


def test_every_peer_touching_route_is_classified():
    """Each peer-touching route is either guarded or explicitly exempt.

    A new route fails here until classified. If this fails for a route you added,
    decide whether a scope peer reaching it is harmful: add it to
    SCOPE_GUARDED_ROUTES with a request builder, or to SCOPE_EXEMPT_ROUTES with a
    reason. Do not add a mutating route to the exempt set.
    """
    classified = set(SCOPE_GUARDED_ROUTES) | set(SCOPE_EXEMPT_ROUTES)
    actual = _peer_touching_routes()

    unclassified = actual - classified
    assert not unclassified, (
        "peer-touching routes with no scope policy: "
        + f"{sorted(unclassified)} — classify each as guarded or exempt"
    )

    stale = classified - actual
    assert not stale, f"classified routes that no longer exist: {sorted(stale)}"


def test_guarded_and_exempt_sets_are_disjoint():
    overlap = set(SCOPE_GUARDED_ROUTES) & set(SCOPE_EXEMPT_ROUTES)
    assert not overlap, f"routes classified both ways: {sorted(overlap)}"


def test_exempt_routes_all_state_a_reason():
    for route, reason in SCOPE_EXEMPT_ROUTES.items():
        assert len(reason.strip()) > 30, f"{route} needs a real reason, got {reason!r}"


def test_skip_list_only_covers_guarded_routes():
    """The exception lists must not drift out of the guarded set."""
    for name, entries in (
        ("_SQUATTER_CHECK_SKIPS", _SQUATTER_CHECK_SKIPS),
        ("_SCHEMA_LEVEL_REJECTION", _SCHEMA_LEVEL_REJECTION),
    ):
        unknown = set(entries) - set(SCOPE_GUARDED_ROUTES)
        assert not unknown, f"{name} references non-guarded routes: {sorted(unknown)}"


@pytest.mark.parametrize(
    ("method", "path"),
    sorted(SCOPE_GUARDED_ROUTES),
    ids=lambda v: v if isinstance(v, str) else str(v),
)
def test_guarded_route_refuses_a_real_scope(
    client: TestClient,
    sample_data: tuple[Workspace, Peer],
    method: str,
    path: str,
):
    """Every guarded route refuses a peer that is a real scope."""
    test_workspace, _ = sample_data
    scope_name = str(generate_nanoid())
    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/scopes", json={"id": scope_name}
    )
    assert response.status_code == 201, response.text
    backing = scope_peer_name(scope_name)

    session_name = str(generate_nanoid())
    assert client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions", json={"id": session_name}
    ).status_code in (200, 201)
    assert (
        client.post(
            f"/v3/workspaces/{test_workspace.name}/scopes/{scope_name}/sessions",
            json={"session_ids": [session_name]},
        ).status_code
        == 200
    )

    build = SCOPE_GUARDED_ROUTES[(method, path)]
    result = build(client, test_workspace.name, session_name, backing)
    status = getattr(result, "status_code", None)
    assert status == 422, f"{method} {path} accepted a scope peer (got {status})"

    # A 422 alone proves nothing — a malformed request body would also produce
    # one. Require the rejection to actually be about this scope.
    detail = str(getattr(result, "text", ""))
    if (method, path) in _SCHEMA_LEVEL_REJECTION:
        assert "pattern" in detail, (
            f"{method} {path} was expected to be refused by schema validation, "
            f"but the detail does not mention the pattern: {detail[:200]}"
        )
    else:
        assert "scope" in detail.lower() and backing in detail, (
            f"{method} {path} returned 422 but not because of the scope — "
            f"detail: {detail[:200]}"
        )


@pytest.mark.parametrize(
    ("method", "path"),
    sorted(set(SCOPE_GUARDED_ROUTES) - set(_SQUATTER_CHECK_SKIPS)),
    ids=lambda v: v if isinstance(v, str) else str(v),
)
async def test_guarded_route_allows_unflagged_squatter(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
    method: str,
    path: str,
):
    """A peer merely occupying the reserved namespace is not a scope.

    Peer names were length-validated only before migration d429de0e5338, so
    `scope.production` is a possible real user name. Such a peer has only the name
    half of the invariant and must keep working — guards that key off the prefix
    alone lock a tenant out of its own data.
    """
    test_workspace, _ = sample_data
    squatter = scope_peer_name(str(generate_nanoid()))
    db_session.add(models.Peer(workspace_name=test_workspace.name, name=squatter))
    await db_session.commit()

    session_name = str(generate_nanoid())
    assert client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions", json={"id": session_name}
    ).status_code in (200, 201)
    # Give the squatter a membership so config/removal routes have a row to act on.
    assert (
        client.post(
            f"/v3/workspaces/{test_workspace.name}/sessions/{session_name}/peers",
            json={squatter: {}},
        ).status_code
        == 200
    )

    build = SCOPE_GUARDED_ROUTES[(method, path)]
    result = build(client, test_workspace.name, session_name, squatter)
    status = getattr(result, "status_code", None)
    assert status != 422, (
        f"{method} {path} refused an unflagged squatter (got {status}) — "
        "the guard is keying off the name prefix rather than the scope flag"
    )
