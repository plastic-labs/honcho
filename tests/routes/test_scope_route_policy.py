"""Route-policy enumeration for the scopes facade, per peer *position*.

Four review passes over the scopes work each found the same class of defect: a
place nobody had checked, rather than logic that was subtly wrong. The first
version of this module enumerated routes and classified each one guarded or
exempt — and that model was itself the fifth defect. A binary per-route verdict
cannot express the actual invariant, which is positional:

    A scope may be an OBSERVER. A scope may never be OBSERVED.

`POST /conclusions` is the case that proves it: a scope as `observer_id` is how
scoped conclusions are stored and must work, while a scope as `observed_id`
persisted a conclusion about something that carries ``observe_me=false``. One
route, two positions, opposite verdicts. The same split applies to
`schedule_dream`, the peer-card routes, and session context.

One refinement, added with the `scope` read option: on the *read* routes an
observer position is refused too, even though a scope there is mechanically
legitimate. Asking for a scope's perspective is what `scope` is for, and routing
through it is what keeps the observer mechanics hidden — so `peer_perspective`,
`GET /peers/{peer_id}/context`, chat and representation all refuse a raw scope
peer name and point at `scope` instead. The invariant above still governs the
storage side, where `observer_id` / `observer` remain ALLOW: a scope observing is
the entire mechanism. Read "OBSERVER" as "may observe", not "may be named as one
on any route".

So classification here is keyed by ``(method, path, position)``, where position is
the request parameter carrying the peer name. Every derived triple must appear in
`POLICY` as either REFUSE or ALLOW-with-a-reason; a new one fails
`test_every_peer_position_is_classified` until someone classifies it.

Each REFUSE case is then asserted behaviorally — by calling the route, because the
guards deliberately live in crud (which is what makes `messages/upload` guarded
for free via `crud.create_messages`) — in both directions:

1. a real scope is refused, and the rejection must actually name it, so an
   unrelated 422 cannot pass the assertion;
2. an *unflagged* peer merely occupying the reserved namespace is NOT a scope and
   is unaffected. That half regressed once already when `update_peer` keyed off
   the name prefix.

Known limitation: this covers the HTTP surface only. Peer names also reach the
system through the deriver, dreamer, and queue, which have no route table to
enumerate; a gap there would not be caught here.
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from httpx import Response
from nanoid import generate as generate_nanoid
from pydantic import BaseModel
from pydantic.fields import FieldInfo
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.main import app
from src.models import Peer, Workspace
from src.utils.scopes import scope_peer_name

# Request parameters and model fields that carry a peer name, in any position.
_PEER_PARAM_NAMES = {
    "peer_id",
    "peer_name",
    "peer_names",
    "observer",
    "observer_id",
    "observed",
    "observed_id",
    "sender_id",
    "target",
    "peer_target",
    "peer_perspective",
}

# Peer names arriving as dict keys or an aliased body field are invisible to
# parameter-name detection, so these paths are matched by shape instead. The
# position recorded for them is the body field or key role.
_KEY_POSITION = "body_peer_keys"

# The scopes router *is* the facade; scope peers are its whole subject.
_SCOPES_PREFIX = "/v3/workspaces/{workspace_id}/scopes"

# A builder places `peer` into one position of one route and returns the response.
Builder = Callable[[TestClient, str, str, str], Response]


@dataclass(frozen=True)
class Case:
    """Policy for one peer position on one route."""

    method: str
    path: str
    position: str
    refuse: bool
    reason: str = ""
    build: Builder | None = None
    # REFUSE cases only. Whether a reserved name that does NOT YET EXIST is also
    # refused — the third axis, and the one that is not derivable from `refuse`.
    # It follows from the guard the call site picked:
    #
    #   validate_no_scope_peer_names  name-only, no DB          refuses missing
    #   reject_scope_observed         strict on the observed     refuses missing
    #   reject_scope_peers            flag-based, permissive     allows missing
    #
    # Permissive is correct where something downstream still stops it (the create
    # path validates new names) or where the name simply resolves to nothing (404
    # before any guard runs). Each False therefore needs `missing_reason`.
    refuse_missing: bool | None = None
    missing_reason: str = ""
    # Required when refuse_missing is False: the exact status(es) a missing
    # reserved name may receive. Deliberately not a bare `!= 422` — that passes on
    # a 5xx too, which is the same hole the squatter assertion below had.
    missing_status: tuple[int, ...] = ()
    # Set when the 422 legitimately comes from request-schema validation rather
    # than a scope guard, so the detail is pydantic's rather than ours.
    schema_level: bool = False
    # Set when the squatter direction cannot be asserted here, with why.
    skip_squatter: str = ""
    # ALLOW cases only: builder plus the status a real scope must receive, so the
    # suite proves legitimate observer positions keep working.
    allow_status: tuple[int, ...] = ()

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.method, self.path, self.position)


_W = "/v3/workspaces/{workspace_id}"


def _b_create_peer(c: TestClient, ws: str, _s: str, p: str):
    return c.post(f"/v3/workspaces/{ws}/peers", json={"id": p})


def _b_update_peer(c: TestClient, ws: str, _s: str, p: str):
    return c.put(f"/v3/workspaces/{ws}/peers/{p}", json={"metadata": {"k": "v"}})


def _b_chat_observer(c: TestClient, ws: str, _s: str, p: str):
    return c.post(f"/v3/workspaces/{ws}/peers/{p}/chat", json={"query": "hi"})


def _b_chat_target(c: TestClient, ws: str, _s: str, p: str):
    return c.post(
        f"/v3/workspaces/{ws}/peers/{_OTHER}/chat", json={"query": "hi", "target": p}
    )


def _b_repr_observer(c: TestClient, ws: str, _s: str, p: str):
    return c.post(f"/v3/workspaces/{ws}/peers/{p}/representation", json={})


def _b_repr_target(c: TestClient, ws: str, _s: str, p: str):
    return c.post(
        f"/v3/workspaces/{ws}/peers/{_OTHER}/representation", json={"target": p}
    )


def _b_card_target(c: TestClient, ws: str, _s: str, p: str):
    return c.put(
        f"/v3/workspaces/{ws}/peers/{_OTHER}/card?target={p}",
        json={"peer_card": ["note"]},
    )


def _b_conclusion_observed(c: TestClient, ws: str, _s: str, p: str):
    return c.post(
        f"/v3/workspaces/{ws}/conclusions",
        json={
            "conclusions": [
                {
                    "observer_id": _OTHER,
                    "observed_id": p,
                    "content": "something",
                    "level": "explicit",
                }
            ]
        },
    )


def _b_dream_observed(c: TestClient, ws: str, _s: str, p: str):
    return c.post(
        f"/v3/workspaces/{ws}/schedule_dream",
        json={"observer": _OTHER, "observed": p, "dream_type": "omni"},
    )


def _b_session_create(c: TestClient, ws: str, _s: str, p: str):
    return c.post(
        f"/v3/workspaces/{ws}/sessions",
        json={"id": str(generate_nanoid()), "peers": {p: {}}},
    )


def _b_session_context_target(c: TestClient, ws: str, s: str, p: str):
    return c.get(f"/v3/workspaces/{ws}/sessions/{s}/context?peer_target={p}")


def _b_message(c: TestClient, ws: str, s: str, p: str):
    return c.post(
        f"/v3/workspaces/{ws}/sessions/{s}/messages",
        json={"messages": [{"peer_id": p, "content": "hello"}]},
    )


def _b_upload(c: TestClient, ws: str, s: str, p: str):
    return c.post(
        f"/v3/workspaces/{ws}/sessions/{s}/messages/upload",
        data={"peer_id": p},
        files={"file": ("note.txt", b"hello there", "text/plain")},
    )


def _b_add_peers(c: TestClient, ws: str, s: str, p: str):
    return c.post(f"/v3/workspaces/{ws}/sessions/{s}/peers", json={p: {}})


def _b_set_peers(c: TestClient, ws: str, s: str, p: str):
    return c.put(f"/v3/workspaces/{ws}/sessions/{s}/peers", json={p: {}})


def _b_remove_peers(c: TestClient, ws: str, s: str, p: str):
    return c.request("DELETE", f"/v3/workspaces/{ws}/sessions/{s}/peers", json=[p])


def _b_conclusion_observer(c: TestClient, ws: str, _s: str, p: str):
    return c.post(
        f"/v3/workspaces/{ws}/conclusions",
        json={
            "conclusions": [
                {
                    "observer_id": p,
                    "observed_id": _OTHER,
                    "content": "something",
                    "level": "explicit",
                }
            ]
        },
    )


def _b_dream_observer(c: TestClient, ws: str, _s: str, p: str):
    return c.post(
        f"/v3/workspaces/{ws}/schedule_dream",
        json={"observer": p, "observed": _OTHER, "dream_type": "omni"},
    )


def _b_card_observer_put(c: TestClient, ws: str, _s: str, p: str):
    return c.put(
        f"/v3/workspaces/{ws}/peers/{p}/card?target={_OTHER}",
        json={"peer_card": ["note"]},
    )


def _b_card_observer_get(c: TestClient, ws: str, _s: str, p: str):
    return c.get(f"/v3/workspaces/{ws}/peers/{p}/card?target={_OTHER}")


def _b_peer_context_observer(c: TestClient, ws: str, _s: str, p: str):
    return c.get(f"/v3/workspaces/{ws}/peers/{p}/context")


def _b_peer_context_target(c: TestClient, ws: str, _s: str, p: str):
    return c.get(f"/v3/workspaces/{ws}/peers/{_OTHER}/context?target={p}")


def _b_context_perspective(c: TestClient, ws: str, s: str, p: str):
    query = f"?peer_perspective={p}&peer_target={_OTHER}"
    return c.get(f"/v3/workspaces/{ws}/sessions/{s}/context{query}")


def _b_queue_status_observer(c: TestClient, ws: str, _s: str, p: str):
    return c.get(f"/v3/workspaces/{ws}/queue/status?observer_id={p}")


def _b_queue_status_sender(c: TestClient, ws: str, _s: str, p: str):
    return c.get(f"/v3/workspaces/{ws}/queue/status?sender_id={p}")


def _b_peer_config(c: TestClient, ws: str, s: str, p: str):
    return c.put(
        f"/v3/workspaces/{ws}/sessions/{s}/peers/{p}/config",
        json={"observe_others": False, "observe_me": True},
    )


def _b_peer_config_get(c: TestClient, ws: str, s: str, p: str):
    return c.get(f"/v3/workspaces/{ws}/sessions/{s}/peers/{p}/config")


# A plain peer used for the *other* side of two-position routes, so the position
# under test is the only scope in the request. Created by the fixtures below.
_OTHER = "policy-counterparty"

_OBSERVER_OK = (
    "Observer position. A scope observing others is the entire mechanism scopes "
    "are built on, so this must keep working."
)
_READ_ONLY_OK = (
    "Read-only. Returns nothing meaningful for a scope rather than creating or "
    "mutating knowledge about one."
)

POLICY: tuple[Case, ...] = (
    # ---- observed position: a scope must never be the subject ----
    Case(
        "POST",
        f"{_W}/conclusions",
        "observed_id",
        True,
        refuse_missing=False,
        missing_reason=(
            "Every observer and observed peer is resolved before the scope check, "
            "so a name that does not exist is a 404 and no conclusion is written. "
            "The guard is still the strict variant, for if that ever changes."
        ),
        missing_status=(404,),
        build=_b_conclusion_observed,
    ),
    Case(
        "POST",
        f"{_W}/schedule_dream",
        "observed",
        True,
        refuse_missing=True,
        build=_b_dream_observed,
    ),
    Case(
        "PUT",
        f"{_W}/peers/{{peer_id}}/card",
        "target",
        True,
        refuse_missing=True,
        build=_b_card_target,
    ),
    Case(
        "POST",
        f"{_W}/peers/{{peer_id}}/chat",
        "target",
        True,
        refuse_missing=True,
        build=_b_chat_target,
    ),
    Case(
        "POST",
        f"{_W}/peers/{{peer_id}}/representation",
        "target",
        True,
        refuse_missing=True,
        build=_b_repr_target,
    ),
    Case(
        "GET",
        f"{_W}/sessions/{{session_id}}/context",
        "peer_target",
        True,
        refuse_missing=True,
        build=_b_session_context_target,
    ),
    # ---- observer position: legitimately a scope ----
    Case(
        "POST",
        f"{_W}/conclusions",
        "observer_id",
        False,
        reason=_OBSERVER_OK,
        build=_b_conclusion_observer,
        allow_status=(200, 201),
    ),
    Case(
        "POST",
        f"{_W}/schedule_dream",
        "observer",
        False,
        reason=_OBSERVER_OK,
        build=_b_dream_observer,
        allow_status=(204,),
    ),
    Case(
        "PUT",
        f"{_W}/peers/{{peer_id}}/card",
        "peer_id",
        False,
        reason=_OBSERVER_OK,
        build=_b_card_observer_put,
        allow_status=(200,),
    ),
    Case(
        "GET",
        f"{_W}/peers/{{peer_id}}/card",
        "peer_id",
        False,
        reason=_OBSERVER_OK,
        build=_b_card_observer_get,
        allow_status=(200,),
    ),
    Case(
        "GET",
        f"{_W}/sessions/{{session_id}}/context",
        "peer_perspective",
        True,
        refuse_missing=False,
        missing_reason=(
            "The perspective peer is resolved before the flag-based guard runs, so a "
            "reserved name that does not exist yet is a 404 — the same answer any "
            "absent peer gets here — and nothing on this path creates it."
        ),
        missing_status=(404,),
        build=_b_context_perspective,
    ),
    Case(
        "GET",
        f"{_W}/queue/status",
        "observer_id",
        False,
        reason=_OBSERVER_OK,
        build=_b_queue_status_observer,
        allow_status=(200,),
    ),
    Case(
        "GET",
        f"{_W}/queue/status",
        "sender_id",
        False,
        reason=(
            "Filter only. `sender_id` reaches CRUD as `observed`, but it selects "
            "existing queue rows rather than creating knowledge about a peer."
        ),
        build=_b_queue_status_sender,
        allow_status=(200,),
    ),
    # ---- peer identity / membership mutation: never a scope ----
    Case(
        "POST",
        f"{_W}/peers",
        _KEY_POSITION,
        True,
        refuse_missing=True,
        build=_b_create_peer,
        schema_level=True,
        skip_squatter=(
            "Creating any name in the reserved namespace is refused whether flagged "
            "or not — that is what reserving it means. Covered by "
            "test_scopes.py::test_peer_create_rejects_reserved_prefix."
        ),
    ),
    Case(
        "PUT",
        f"{_W}/peers/{{peer_id}}",
        "peer_id",
        True,
        refuse_missing=True,
        build=_b_update_peer,
    ),
    Case(
        "POST",
        f"{_W}/sessions",
        "peer_names",
        True,
        refuse_missing=True,
        build=_b_session_create,
    ),
    Case(
        "POST",
        f"{_W}/sessions/{{session_id}}/messages",
        "peer_name",
        True,
        refuse_missing=True,
        build=_b_message,
    ),
    Case(
        "POST",
        f"{_W}/sessions/{{session_id}}/messages/upload",
        "peer_id",
        True,
        refuse_missing=True,
        build=_b_upload,
    ),
    Case(
        "POST",
        f"{_W}/sessions/{{session_id}}/peers",
        _KEY_POSITION,
        True,
        refuse_missing=True,
        build=_b_add_peers,
    ),
    Case(
        "PUT",
        f"{_W}/sessions/{{session_id}}/peers",
        _KEY_POSITION,
        True,
        refuse_missing=True,
        build=_b_set_peers,
    ),
    Case(
        "DELETE",
        f"{_W}/sessions/{{session_id}}/peers",
        _KEY_POSITION,
        True,
        refuse_missing=False,
        missing_reason=(
            "Removal creates nothing and a name that does not exist has no "
            "membership row, so the request is a no-op. Refusing here would give a "
            "reserved name a different removal result than any other absent peer."
        ),
        missing_status=(200,),
        build=_b_remove_peers,
    ),
    Case(
        "PUT",
        f"{_W}/sessions/{{session_id}}/peers/{{peer_id}}/config",
        "peer_id",
        True,
        refuse_missing=False,
        missing_reason=(
            "The peer is resolved before the scope check, so a name that does not "
            "exist is a 404 and never reaches the guard. Nothing is created, so "
            "there is no window for the name to be claimed here."
        ),
        missing_status=(404,),
        build=_b_peer_config,
    ),
    Case(
        "GET",
        f"{_W}/sessions/{{session_id}}/peers/{{peer_id}}/config",
        "peer_id",
        True,
        refuse_missing=False,
        missing_reason=(
            "Same resolution order as the write side of this route: an absent peer "
            "is a 404 before the scope check, and a read creates nothing."
        ),
        missing_status=(404,),
        build=_b_peer_config_get,
    ),
    # ---- path peer on the dialectic surface ----
    Case(
        "POST",
        f"{_W}/peers/{{peer_id}}/chat",
        "peer_id",
        True,
        refuse_missing=True,
        build=_b_chat_observer,
    ),
    Case(
        "POST",
        f"{_W}/peers/{{peer_id}}/representation",
        "peer_id",
        True,
        refuse_missing=True,
        build=_b_repr_observer,
    ),
    # ---- reads that neither create nor mutate knowledge about a scope ----
    Case(
        "POST", f"{_W}/peers/{{peer_id}}/search", "peer_id", False, reason=_READ_ONLY_OK
    ),
    Case(
        "POST",
        f"{_W}/peers/{{peer_id}}/sessions",
        "peer_id",
        False,
        reason=(
            "Read-only. A scope legitimately has member sessions; this is the "
            "observer-mechanics view of POST /scopes/{scope_id}/sessions/list."
        ),
    ),
    Case(
        "GET",
        f"{_W}/peers/{{peer_id}}/context",
        "peer_id",
        True,
        refuse_missing=True,
        build=_b_peer_context_observer,
    ),
    Case(
        "GET",
        f"{_W}/peers/{{peer_id}}/context",
        "target",
        True,
        refuse_missing=True,
        build=_b_peer_context_target,
    ),
    Case(
        "GET",
        f"{_W}/peers/{{peer_id}}/card",
        "target",
        False,
        reason=(
            "Read-only. The write side (PUT with target) IS refused, so this can only "
            "return pre-existing rows, never create them."
        ),
    ),
    Case(
        "POST",
        "/v3/keys",
        "peer_id",
        False,
        reason=(
            "Mints a scoped JWT rather than touching a peer, so no peer row is read "
            "or written. Keys cannot be bound to a scope yet."
        ),
    ),
)

_BY_KEY = {case.key: case for case in POLICY}

# Routes whose peer names arrive as dict keys or an aliased body field, invisible
# to parameter-name detection and therefore matched by path shape.
_KEY_POSITION_PATHS = {
    ("POST", f"{_W}/peers"),
    ("POST", f"{_W}/sessions/{{session_id}}/peers"),
    ("PUT", f"{_W}/sessions/{{session_id}}/peers"),
    ("DELETE", f"{_W}/sessions/{{session_id}}/peers"),
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
    for f in fields.values():
        stack = [f.annotation]
        while stack:
            current = stack.pop()
            yield from _nested_models(current, seen)
            stack.extend(getattr(current, "__args__", ()) or ())


def _peer_positions(route: APIRoute) -> set[str]:
    """Peer-name-carrying parameter names anywhere in a route's dependant tree.

    Walks sub-dependencies so `Form(...)` params behind a parser dependency are
    seen — this is how `messages/upload` takes its `peer_id` — and descends into
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


def _derived_positions() -> set[tuple[str, str, str]]:
    """Every (method, path, position) through which a peer name can be supplied."""
    found: set[tuple[str, str, str]] = set()
    for route in app.routes:
        if not isinstance(route, APIRoute):
            continue
        path = route.path.rstrip("/") or route.path
        if path.startswith(_SCOPES_PREFIX):
            continue
        positions = _peer_positions(route)
        for method in route.methods or set():
            if method in ("HEAD", "OPTIONS"):
                continue
            if (method, path) in _KEY_POSITION_PATHS:
                found.add((method, path, _KEY_POSITION))
            for position in positions:
                found.add((method, path, position))
    return found


def test_every_peer_position_is_classified():
    """Each (route, peer position) pair has an explicit scope policy.

    A new one fails here until classified. Decide whether a scope in that
    *position* is harmful — the rule is that a scope may be an observer but never
    observed — then add a Case with `refuse=True` and a builder, or `refuse=False`
    and a reason.
    """
    derived = _derived_positions()
    classified = set(_BY_KEY)

    # _peer_positions walks FastAPI/Pydantic internals (route.dependant, its
    # *_params lists, field_info.annotation). An upgrade that reshapes any of them
    # would make derivation silently return nothing, and every assertion below
    # would then pass vacuously. Anchor on a position that must always be found.
    assert (
        "POST",
        f"{_W}/sessions/{{session_id}}/messages",
        "peer_name",
    ) in derived, (
        "derived no peer positions for a route that certainly has one — the "
        "FastAPI internals _peer_positions() traverses have probably changed shape"
    )

    unclassified = derived - classified
    assert not unclassified, (
        "peer positions with no scope policy: "
        + f"{sorted(unclassified)} — classify each as refuse or allow"
    )

    stale = classified - derived
    assert not stale, f"classified positions that no longer exist: {sorted(stale)}"


def test_policy_entries_are_well_formed():
    assert len(_BY_KEY) == len(POLICY), "duplicate (method, path, position) in POLICY"
    for case in POLICY:
        if case.refuse:
            assert case.build is not None, f"{case.key} refuses but has no builder"
            assert not case.reason, f"{case.key} refuses; reason is for allow cases"
            # The missing-name axis is not derivable from `refuse`, so it must be
            # stated rather than defaulted — that gap is what this field closes.
            assert case.refuse_missing is not None, (
                f"{case.key} refuses a real scope but does not say whether a "
                "reserved name that does not exist yet is also refused"
            )
            if case.refuse_missing:
                assert not case.missing_status, (
                    f"{case.key} refuses a missing reserved name, so the expected "
                    "status is 422 — missing_status is for the permissive cases"
                )
            else:
                assert (
                    len(case.missing_reason.strip()) > 30
                ), f"{case.key} tolerates a missing reserved name; say why"
                assert case.missing_status, (
                    f"{case.key} tolerates a missing reserved name; name the exact "
                    "status(es) it should get, so a 5xx cannot satisfy the case"
                )
        else:
            assert len(case.reason.strip()) > 30, f"{case.key} needs a real reason"
            assert bool(case.build) == bool(case.allow_status), (
                f"{case.key}: an allow case needs a builder and an expected "
                "allow_status together, or neither"
            )
            assert (
                case.refuse_missing is None
            ), f"{case.key}: refuse_missing applies to REFUSE cases only"


_REFUSING = tuple(case for case in POLICY if case.refuse)
# Allow cases that additionally prove, behaviorally, that a real scope works here.
_ALLOWING_EXERCISED = tuple(
    case for case in POLICY if not case.refuse and case.build is not None
)


def _setup(client: TestClient, workspace: str) -> tuple[str, str]:
    """Create the counterparty peer and a session, returning (session, scope name)."""
    assert client.post(
        f"/v3/workspaces/{workspace}/peers", json={"id": _OTHER}
    ).status_code in (200, 201)
    session_name = str(generate_nanoid())
    assert client.post(
        f"/v3/workspaces/{workspace}/sessions", json={"id": session_name}
    ).status_code in (200, 201)
    return session_name, str(generate_nanoid())


def _real_scope(
    client: TestClient, workspace: str, session_name: str, scope_name: str
) -> str:
    """Create a scope, attach the session to it, and return its backing peer name."""
    assert (
        client.post(
            f"/v3/workspaces/{workspace}/scopes", json={"id": scope_name}
        ).status_code
        == 201
    )
    assert (
        client.post(
            f"/v3/workspaces/{workspace}/scopes/{scope_name}/sessions",
            json={"session_ids": [session_name]},
        ).status_code
        == 204
    )
    return scope_peer_name(scope_name)


@pytest.mark.parametrize("case", _REFUSING, ids=lambda c: f"{c.method}:{c.position}")
def test_refusing_position_rejects_a_real_scope(
    client: TestClient,
    sample_data: tuple[Workspace, Peer],
    case: Case,
):
    """A real scope is refused in every position marked REFUSE."""
    test_workspace, _ = sample_data
    session_name, scope_name = _setup(client, test_workspace.name)
    backing = _real_scope(client, test_workspace.name, session_name, scope_name)

    assert case.build is not None
    result = case.build(client, test_workspace.name, session_name, backing)
    status = result.status_code
    assert status == 422, (
        f"{case.method} {case.path} accepted a scope in position "
        f"{case.position!r} (got {status})"
    )

    # A 422 alone proves nothing — a malformed body would also produce one.
    detail = result.text
    if case.schema_level:
        assert (
            "pattern" in detail
        ), f"{case.key} expected a schema-level refusal; detail: {detail[:200]}"
    else:
        assert "scope" in detail.lower() and backing in detail, (
            f"{case.key} returned 422 but not because of the scope; "
            f"detail: {detail[:200]}"
        )


@pytest.mark.parametrize(
    "case", _ALLOWING_EXERCISED, ids=lambda c: f"{c.method}:{c.position}"
)
def test_allowing_position_accepts_a_real_scope(
    client: TestClient,
    sample_data: tuple[Workspace, Peer],
    case: Case,
):
    """A real scope works in every position marked ALLOW.

    The other half of the contract. Refusal tests alone would be satisfied by a
    guard that rejected scopes everywhere, which would break the feature: scoped
    conclusions, scoped dreams and scoped peer cards all require a scope in the
    observer position.
    """
    test_workspace, _ = sample_data
    session_name, scope_name = _setup(client, test_workspace.name)
    backing = _real_scope(client, test_workspace.name, session_name, scope_name)

    assert case.build is not None
    result = case.build(client, test_workspace.name, session_name, backing)
    assert result.status_code in case.allow_status, (
        f"{case.method} {case.path} refused a scope in the legitimate position "
        f"{case.position!r}: expected {case.allow_status}, got "
        f"{result.status_code} — {result.text[:200]}"
    )


@pytest.mark.parametrize(
    "case",
    tuple(c for c in _REFUSING if not c.skip_squatter),
    ids=lambda c: f"{c.method}:{c.position}",
)
async def test_refusing_position_allows_unflagged_squatter(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
    case: Case,
):
    """A peer merely occupying the reserved namespace is not a scope.

    Peer names were length-validated only before migration d429de0e5338, so
    `scope.production` is a possible real user name. Such a peer has only the name
    half of the invariant and must keep working — a guard keying off the prefix
    alone locks a tenant out of its own data.
    """
    test_workspace, _ = sample_data
    session_name, _ = _setup(client, test_workspace.name)
    squatter = scope_peer_name(str(generate_nanoid()))
    db_session.add(models.Peer(workspace_name=test_workspace.name, name=squatter))
    await db_session.commit()

    # Give it a membership so config and removal have a row to act on.
    assert (
        client.post(
            f"/v3/workspaces/{test_workspace.name}/sessions/{session_name}/peers",
            json={squatter: {}},
        ).status_code
        == 200
    )

    assert case.build is not None
    result = case.build(client, test_workspace.name, session_name, squatter)
    # Deliberately not `!= 422`: that also passes on a 5xx, so a guard regressing
    # into an unhandled error (the psycopg DataError path this feature defends
    # against) would keep this green.
    assert result.status_code < 400, (
        f"{case.method} {case.path} did not accept an unflagged squatter in "
        f"position {case.position!r} (got {result.status_code}) — a 422 means the "
        "guard is keying off the name prefix rather than the scope flag; anything "
        f"else means the request blew up. Body: {result.text[:200]}"
    )


@pytest.mark.parametrize("case", _REFUSING, ids=lambda c: f"{c.method}:{c.position}")
async def test_refusing_position_and_a_missing_reserved_name(
    client: TestClient,
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
    case: Case,
):
    """The third axis: a reserved name that does not exist yet.

    Neither of the other two tests reaches it — both resolve an existing subject.
    A permissive guard here is sometimes correct (the create path refuses the name
    itself, or it simply resolves to nothing), which is why the expected verdict is
    declared per case rather than assumed.

    What is NOT negotiable in either direction is that the request must not MINT
    the reserved name. Minting it would let any caller squat a scope name before
    the workspace owner can create it, and would leave a peer that the facade can
    never adopt.
    """
    test_workspace, _ = sample_data
    session_name, _ = _setup(client, test_workspace.name)
    missing = scope_peer_name(str(generate_nanoid()))

    assert case.build is not None
    result = case.build(client, test_workspace.name, session_name, missing)

    if case.refuse_missing:
        assert result.status_code == 422, (
            f"{case.method} {case.path} accepted a not-yet-existing reserved name "
            f"in position {case.position!r} (got {result.status_code}) — it could "
            f"become a scope later. Body: {result.text[:200]}"
        )
    else:
        # Exact, not `!= 422`: a permissive position still has one correct answer,
        # and a 5xx must not read as tolerance.
        assert result.status_code in case.missing_status, (
            f"{case.method} {case.path} gave {result.status_code} for a missing "
            f"reserved name in position {case.position!r}; the policy expects "
            f"{case.missing_status} because {case.missing_reason!r} — update the "
            f"policy or the guard. Body: {result.text[:200]}"
        )

    minted = await db_session.scalar(
        select(models.Peer)
        .where(models.Peer.workspace_name == test_workspace.name)
        .where(models.Peer.name == missing)
    )
    assert minted is None, (
        f"{case.method} {case.path} minted the reserved name {missing!r} from "
        f"position {case.position!r} — the scope namespace is now squatted"
    )
