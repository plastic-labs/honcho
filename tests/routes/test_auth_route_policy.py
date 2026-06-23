"""Route-policy regression tests for auth scoping.

Two invariants this guards:

1. `allow_member_read=True` grants peer-scoped keys read access to sessions
   their peer belongs to. It must appear ONLY on intended read routes — never on
   a mutating route, where it would hand session members write access. HTTP
   method is not a reliable read/write signal in this codebase (some read
   endpoints use POST for a richer request body), so we assert against an
   explicit allowlist instead of deriving from the method.

2. The messages router dropped its router-level auth dependency in favor of
   per-route dependencies. Every route on it must still carry auth, or a future
   route added without an explicit dependency would serve unauthenticated.
"""

from fastapi.routing import APIRoute

from src.main import app

# (method, path) pairs intentionally granting member peers read access. Adding a
# route here is a deliberate security decision: it must be read-only. Never add
# a mutating route. See CLAUDE.md "Auth scoping" for the rule.
EXPECTED_MEMBER_READ_ROUTES = {
    ("POST", "/v3/workspaces/{workspace_id}/sessions/{session_id}/messages/list"),
    (
        "GET",
        "/v3/workspaces/{workspace_id}/sessions/{session_id}/messages/{message_id}",
    ),
    ("GET", "/v3/workspaces/{workspace_id}/sessions/{session_id}/context"),
    ("GET", "/v3/workspaces/{workspace_id}/sessions/{session_id}/summaries"),
    ("GET", "/v3/workspaces/{workspace_id}/sessions/{session_id}/peers"),
    (
        "GET",
        "/v3/workspaces/{workspace_id}/sessions/{session_id}/peers/{peer_id}/config",
    ),
    ("POST", "/v3/workspaces/{workspace_id}/sessions/{session_id}/search"),
}

# Unambiguously mutating methods. POST is intentionally excluded: this codebase
# uses POST for some read endpoints (`/messages/list`, `/search`) to take a
# richer request body, so POST is not a write signal. The allowlist test above
# is the real guard against a write route opting into member read; this test
# additionally catches the clear-cut PUT/PATCH/DELETE mistakes.
MUTATING_METHODS = {"PUT", "PATCH", "DELETE"}


def _auth_dependency_calls(route: APIRoute):
    """Yield the callables of every honcho auth dependency attached to a route.

    `require_auth(...)` closures are tagged with `honcho_allow_member_read`, so a
    dependency is a honcho auth dependency iff its callable has that attribute.
    Walks the dependant tree to cover both `dependencies=[Depends(...)]` and
    parameter-level `Depends(...)`.
    """
    stack = list(route.dependant.dependencies)
    while stack:
        dep = stack.pop()
        if hasattr(dep.call, "honcho_allow_member_read"):
            yield dep.call
        stack.extend(dep.dependencies)


def _method_path_pairs(route: APIRoute):
    for method in route.methods or set():
        if method in ("HEAD", "OPTIONS"):
            continue
        yield (method, route.path)


def test_member_read_allowlist_matches_routes():
    """Exactly the allowlisted routes opt into member read — no more, no less."""
    actual: set[tuple[str, str]] = set()
    for route in app.routes:
        if not isinstance(route, APIRoute):
            continue
        if any(
            getattr(call, "honcho_allow_member_read", False)
            for call in _auth_dependency_calls(route)
        ):
            actual.update(_method_path_pairs(route))

    assert actual == EXPECTED_MEMBER_READ_ROUTES


def test_member_read_never_on_mutating_route():
    """A member-read route must never use a mutating HTTP method."""
    for method, path in EXPECTED_MEMBER_READ_ROUTES:
        assert method not in MUTATING_METHODS, (
            f"{method} {path} grants member-read on a mutating method — "
            "member peers would gain write access"
        )


def test_every_message_route_requires_auth():
    """The messages router has no router-level auth dependency; assert each route
    carries its own so a newly added route cannot be silently unauthenticated."""
    prefix = "/v3/workspaces/{workspace_id}/sessions/{session_id}/messages"
    message_routes = [
        route
        for route in app.routes
        if isinstance(route, APIRoute) and route.path.startswith(prefix)
    ]
    assert message_routes, "expected to find message routes mounted under the prefix"
    for route in message_routes:
        assert any(
            _auth_dependency_calls(route)
        ), f"{route.methods} {route.path} has no auth dependency"
