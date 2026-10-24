# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownLambdaType=false, reportAny=false
"""Every /v3 operation, fuzzed with auth enabled and a spread of JWTs.

Same schemathesis operations as `test_api_no_server_errors`, but each request
carries a token drawn from: nothing, garbage, wrong-secret, expired, and valid
tokens at every scope (workspace, peer, session, named scope, admin) in the
seeded workspace and in a second, structurally identical workspace.

Two properties hold for every request:

1. No 5xx and no unhandled exception.
2. A 2xx is only returned when the token is entitled to the resource. The
   entitlement oracle (`_permitted`) is derived from the route policy in
   `src/security.py` and the handlers (and `tests/routes/test_auth_route_policy`),
   not from observed behaviour. It only ever answers "may this 2xx?"; it never
   requires a 2xx, because the fuzzer's bodies are usually invalid.

The second workspace reuses the seeded workspace's peer/session/scope names, so
a workspace-A key that wrongly ignored the workspace would find real data.
"""

import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import unquote, urlsplit

import jwt as pyjwt
import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from schemathesis import GenerationMode
from schemathesis.core.result import Ok
from starlette.routing import Match

from src.config import settings as app_settings
from src.main import app
from src.security import JWTParams, create_admin_jwt, create_jwt
from tests.property.conftest import API_SCHEMA
from tests.routes.test_auth_route_policy import (
    EXPECTED_MEMBER_READ_ROUTES,
    EXPECTED_NAMED_SCOPE_ROUTES,
)

MAX_EXAMPLES = int(os.environ.get("HONCHO_FUZZ_EXAMPLES", "25"))
# 32+ bytes, so PyJWT does not warn on every encode and decode
SECRET = "property-test-secret-0123456789ab"
WRONG_SECRET = "property-test-wrong-secret-01234567"

WS_A, WS_B = "auth-ws-a", "auth-ws-b"
ALICE, BOB = "auth-alice", "auth-bob"
SESS_1, SESS_2 = "auth-session-1", "auth-session-2"
SCOPE_1, SCOPE_2 = "auth-scope-1", "auth-scope-2"

# alice and bob are members of SESS_1; only bob is in SESS_2, so alice has a
# session she can name but must not be able to read.
SESSION_MEMBERS = {SESS_1: {ALICE, BOB}, SESS_2: {BOB}}

ADMIN_ONLY = {("POST", "/v3/keys"), ("POST", "/v3/workspaces/list")}

OPERATIONS = [
    result.ok()
    for result in API_SCHEMA.get_all_operations()
    if isinstance(result, Ok) and result.ok().path.startswith("/v3")
]

# Body fields that name an existing resource, mapped to the id kind to use.
BODY_ID_FIELDS = {
    "peer_id": "peer_id",
    "observer_id": "peer_id",
    "observed_id": "peer_id",
    "target": "peer_id",
    "session_id": "session_id",
}

P_ROOT = "/v3/workspaces/{workspace_id}/peers"
S_ROOT = "/v3/workspaces/{workspace_id}/sessions"


@dataclass(frozen=True)
class Token:
    """A bearer token plus the claims it was minted with."""

    kind: str  # invalid | admin | ws | peer | session | scope
    value: str | None
    w: str | None = None
    p: str | None = None
    s: str | None = None
    sc: str | None = None

    @property
    def label(self) -> str:
        return f"{self.kind}(w={self.w},p={self.p},s={self.s},sc={self.sc})"


def _build_tokens() -> list[Token]:
    def mint(kind: str, **claims: str) -> Token:
        return Token(kind, create_jwt(JWTParams(**claims)), **claims)  # pyright: ignore[reportArgumentType]

    tokens = [
        Token("invalid", None, sc="no-token"),
        Token("invalid", "", sc="empty-token"),
        Token("invalid", "garbage", sc="garbage"),
        Token("invalid", "a.b.c", sc="dotted-garbage"),
        Token("invalid", create_admin_jwt()[:-3] + "AAA", sc="bad-signature"),
        Token(
            "invalid",
            pyjwt.encode({"t": "", "ad": True}, WRONG_SECRET, algorithm="HS256"),
            sc="wrong-secret-admin",
        ),
        Token(
            "invalid",
            pyjwt.encode(
                {"t": "", "w": WS_A, "p": ALICE}, WRONG_SECRET, algorithm="HS256"
            ),
            sc="wrong-secret-peer",
        ),
        Token(
            "invalid",
            pyjwt.encode({"t": "", "ad": True}, None, algorithm="none"),  # pyright: ignore[reportArgumentType]
            sc="alg-none-admin",
        ),
        Token(
            "invalid",
            pyjwt.encode({"t": "", "ad": True, "exp": 1}, SECRET, algorithm="HS256"),
            sc="expired-admin",
        ),
        Token(
            "invalid",
            pyjwt.encode({"t": "", "w": WS_A, "exp": 1}, SECRET, algorithm="HS256"),
            sc="expired-ws",
        ),
        Token("admin", create_admin_jwt()),
    ]
    for ws in (WS_A, WS_B):
        tokens.append(mint("ws", w=ws))
        tokens.extend(mint("peer", w=ws, p=p) for p in (ALICE, BOB))
        tokens.extend(mint("session", w=ws, s=s) for s in (SESS_1, SESS_2))
        tokens.extend(mint("scope", w=ws, sc=sc) for sc in (SCOPE_1, SCOPE_2))
    return tokens


def _hdr(token: Token | str | None) -> dict[str, str]:
    value = token.value if isinstance(token, Token) else token
    return {"Authorization": f"Bearer {value}"} if value else {}


def _seed_workspace(
    client: TestClient, ws: str, admin: dict[str, str]
) -> dict[str, dict[str, str]]:
    """Idempotently create one workspace's resources; return its per-kind IDs."""

    def ok(response: Any) -> Any:
        assert response.status_code < 300, response.text
        return response.json()

    base = f"/v3/workspaces/{ws}"
    ok(client.post("/v3/workspaces", json={"id": ws}, headers=admin))
    for peer in (ALICE, BOB):
        ok(client.post(f"{base}/peers", json={"id": peer}, headers=admin))
    for session, members in SESSION_MEMBERS.items():
        ok(
            client.post(
                f"{base}/sessions",
                json={"id": session, "peers": {m: {} for m in members}},
                headers=admin,
            )
        )
    messages = ok(
        client.post(
            f"{base}/sessions/{SESS_1}/messages",
            json={
                "messages": [
                    {"content": "I like hiking", "peer_id": ALICE},
                    {"content": "Me too", "peer_id": BOB},
                ]
            },
            headers=admin,
        )
    )
    conclusions = ok(
        client.post(
            f"{base}/conclusions",
            json={
                "conclusions": [
                    {
                        "content": "alice likes hiking",
                        "observer_id": BOB,
                        "observed_id": ALICE,
                        "session_id": SESS_1,
                    }
                ]
            },
            headers=admin,
        )
    )
    for scope in (SCOPE_1, SCOPE_2):
        ok(client.post(f"{base}/scopes", json={"id": scope}, headers=admin))
    webhook = ok(
        client.post(
            f"{base}/webhooks", json={"url": "https://example.com/hook"}, headers=admin
        )
    )
    return {
        "fixed": {
            "workspace_id": ws,
            "message_id": messages[0]["id"],
            "conclusion_id": conclusions[0]["id"],
            "endpoint_id": webhook["id"],
        }
    }


@pytest.fixture
def auth_world(
    fuzz_client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> dict[str, Any]:
    """Turn auth on, seed both workspaces as admin, mint every token."""
    monkeypatch.setattr(app_settings.AUTH, "USE_AUTH", True)
    monkeypatch.setattr(app_settings.AUTH, "JWT_SECRET", SECRET)
    fuzz_client.headers.pop("Authorization", None)
    admin = _hdr(create_admin_jwt())
    ids = {ws: _seed_workspace(fuzz_client, ws, admin)["fixed"] for ws in (WS_A, WS_B)}
    return {"ids": ids, "tokens": _build_tokens(), "admin": admin}


def _resolve(method: str, path: str) -> tuple[str | None, dict[str, str]]:
    """Which route template and path params the server will pick for this URL."""
    scope = {
        "type": "http",
        "method": method,
        "path": unquote(path),
        "root_path": "",
        "query_string": b"",
        "headers": [],
    }
    for route in app.routes:
        if isinstance(route, APIRoute):
            match, child = route.matches(scope)
            if match is Match.FULL:
                return route.path, dict(child["path_params"])
    return None, {}


def _as_list(value: Any) -> list[str] | None:
    """Normalize a recall `filters.session_id` shape to a list of ids."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict) and isinstance(value.get("in"), list):
        value = value["in"]
    if isinstance(value, list) and all(isinstance(v, str) for v in value):
        return value
    return None


class Oracle:
    """Answers "may this token get a 2xx here?" from the documented policy."""

    client: TestClient
    admin: dict[str, str]

    def __init__(self, client: TestClient, admin: dict[str, str]):
        self.client = client
        self.admin = admin

    def session_state(self, ws: str, sid: str, peer: str) -> str:
        """`member`, `non-member`, or `absent`, read live as admin.

        Live because other fuzzed requests (ws/admin keys) change membership.
        """
        exists = self.client.post(
            f"/v3/workspaces/{ws}/sessions/list",
            json={"filters": {"id": sid}},
            headers=self.admin,
        )
        if exists.status_code != 200 or not exists.json()["items"]:
            return "absent"
        response = self.client.get(
            f"/v3/workspaces/{ws}/sessions/{sid}/peers", headers=self.admin
        )
        assert response.status_code == 200, response.text
        names = {item["id"] for item in response.json()["items"]}
        return "member" if peer in names else "non-member"

    def permitted(
        self,
        tok: Token,
        method: str,
        tmpl: str,
        pp: dict[str, str],
        query: dict[str, Any],
        body: Any,
    ) -> bool:
        if tok.kind == "invalid":
            return False
        if tok.kind == "admin":
            return True
        if (method, tmpl) in ADMIN_ONLY:
            return False
        body = body if isinstance(body, dict) else {}
        body_id = body.get("id") if isinstance(body.get("id"), str) else None

        if tmpl == "/v3/workspaces":
            # Self-authorizing: body id must be the token's workspace, or absent
            # (then the token's own workspace is used). Scope keys never opt in.
            if tok.kind == "scope":
                return False
            return (body_id or tok.w) == tok.w
        if pp.get("workspace_id") != tok.w:
            return False
        ws = tok.w or ""

        if tok.kind == "ws":
            return True

        if tok.kind == "scope":
            if (method, tmpl) not in EXPECTED_NAMED_SCOPE_ROUTES:
                return False
            return pp.get("scope_id") in (None, tok.sc)

        if tok.kind == "session":
            if (method, tmpl) == ("POST", S_ROOT):
                return (body_id or tok.s) == tok.s
            return pp.get("session_id") == tok.s

        # Peer-scoped.
        assert tok.kind == "peer"
        me = tok.p or ""
        if (method, tmpl) == ("POST", P_ROOT):
            return (body_id or me) == me
        if (method, tmpl) == ("POST", S_ROOT):
            peers = body.get("peers")
            if isinstance(peers, dict) and set(peers) - {me}:
                return False
            if body.get("scopes"):
                return False
            return not (body_id and self.session_state(ws, body_id, me) == "non-member")
        if tmpl.startswith(P_ROOT + "/{peer_id}"):
            if pp.get("peer_id") != me:
                return False
            if tmpl.endswith(("/chat", "/representation")):
                if body.get("scope") is not None:
                    return False
                sid = body.get("session_id")
                sessions = [sid] if isinstance(sid, str) and sid else []
                filters = body.get("filters")
                if isinstance(filters, dict) and "session_id" in filters:
                    listed = _as_list(filters["session_id"])
                    if listed is None:
                        return True  # rejected as 422 by the handler
                    sessions += listed
                return all(self.session_state(ws, s, me) == "member" for s in sessions)
            return True
        if (method, tmpl) in EXPECTED_MEMBER_READ_ROUTES:
            sid = pp.get("session_id", "")
            if self.session_state(ws, sid, me) != "member":
                return False
            if "peer_id" in pp and pp["peer_id"] != me:
                return False
            if tmpl.endswith("/context"):
                observer = query.get("peer_perspective") or query.get("peer_target")
                if observer and observer != me:
                    return False
                sessions = query.get("sessions")
                if sessions:
                    listed = _as_list(sessions)
                    if listed is None or any(
                        self.session_state(ws, s, me) != "member" for s in listed
                    ):
                        return False
            return True
        return False


def _point_at_seeded(value: Any, ids: dict[str, str], data: st.DataObject) -> None:
    """Swap ID fields anywhere in a JSON body for seeded IDs, most of the time."""
    if isinstance(value, list):
        for item in value:
            _point_at_seeded(item, ids, data)
    elif isinstance(value, dict):
        for key, item in value.items():
            if key in BODY_ID_FIELDS and isinstance(item, str):
                if data.draw(st.integers(0, 3)):
                    value[key] = ids[BODY_ID_FIELDS[key]]
            else:
                _point_at_seeded(item, ids, data)


@pytest.mark.parametrize("mode", [GenerationMode.POSITIVE, GenerationMode.NEGATIVE])
@pytest.mark.parametrize("operation", OPERATIONS, ids=lambda op: op.label)
def test_operation_respects_auth(
    operation,
    mode: GenerationMode,
    fuzz_client: TestClient,
    auth_world: dict[str, Any],
):
    tokens: list[Token] = auth_world["tokens"]
    oracle = Oracle(fuzz_client, auth_world["admin"])

    @settings(
        max_examples=MAX_EXAMPLES,
        deadline=None,
        database=None,
        suppress_health_check=[
            HealthCheck.function_scoped_fixture,
            HealthCheck.too_slow,
            HealthCheck.filter_too_much,
            HealthCheck.data_too_large,
        ],
    )
    @given(data=st.data())
    def run(data: st.DataObject) -> None:
        case = data.draw(operation.as_strategy(generation_mode=mode))
        token = data.draw(st.sampled_from(tokens))
        if mode is GenerationMode.POSITIVE:
            # Mostly aim at the token's own resources, so entitled requests
            # reach the handlers; otherwise anywhere, which probes the walls.
            own = token.kind != "invalid" and data.draw(st.integers(0, 2)) > 0
            ws = (token.w if own and token.w else None) or data.draw(
                st.sampled_from([WS_A, WS_B])
            )
            ids = {
                **auth_world["ids"][ws],
                "peer_id": (token.p if own and token.p else None)
                or data.draw(st.sampled_from([ALICE, BOB])),
                "session_id": (token.s if own and token.s else None)
                or data.draw(st.sampled_from([SESS_1, SESS_2])),
                "scope_id": (token.sc if own and token.sc else None)
                or data.draw(st.sampled_from([SCOPE_1, SCOPE_2])),
            }
            for name in case.path_parameters:
                if name in ids and data.draw(st.integers(0, 3)):
                    case.path_parameters[name] = ids[name]
            _point_at_seeded(case.body, ids, data)
        kwargs = case.as_transport_kwargs(base_url="http://testserver")
        headers = {
            k: v
            for k, v in (kwargs.get("headers") or {}).items()
            if k.lower() != "authorization"
        }
        headers.update(_hdr(token))
        # Decide entitlement from the state *before* the request: a successful
        # create changes what the oracle reads (the session now exists).
        method = kwargs["method"].upper()
        tmpl, pp = _resolve(method, urlsplit(kwargs["url"]).path)
        allowed = tmpl is not None and oracle.permitted(
            token,
            method,
            tmpl,
            pp,
            dict(kwargs.get("params") or {}),
            kwargs.get("json"),
        )
        try:
            response = fuzz_client.request(
                method=kwargs["method"],
                url=kwargs["url"],
                params=kwargs.get("params"),
                headers=headers,
                json=kwargs.get("json"),
                follow_redirects=False,
            )
        except Exception as exc:
            first_line = str(exc).splitlines()[0] if str(exc) else ""
            pytest.fail(
                f"{type(exc).__name__}: {first_line}\n{case.as_curl_command()}\n"
                + f"token={token.label}",
                pytrace=False,
            )
        status = response.status_code
        context = (
            f"token={token.label}\n{case.as_curl_command()}\n{response.text[:300]}"
        )
        assert status < 500, f"{status} {context}"

        if 200 <= status < 300:
            assert tmpl is not None, f"2xx for an unrouted URL\n{context}"
            if token.kind == "invalid":
                pytest.fail(f"unauthenticated 2xx ({status})\n{context}", pytrace=False)
            assert allowed, f"unauthorized 2xx ({status}) on {tmpl} {pp}\n{context}"
        elif token.kind == "invalid":
            # Authentication runs before body validation and before any lookup,
            # so a bad token is always 401 (or a routing miss), never 403/404/422.
            assert status in (401, 404, 405, 307, 308), (
                f"invalid token got {status}\n{context}"
            )

    run()


@pytest.mark.parametrize("path_ids", [(WS_A, WS_B), (WS_B, WS_A)])
def test_cross_workspace_matrix_is_wired(
    fuzz_client: TestClient, auth_world: dict[str, Any], path_ids: tuple[str, str]
):
    """Guard against a vacuous fuzz run: keys really work in their own workspace.

    Each token kind gets a 2xx on a resource it owns and a non-2xx on the same
    resource name in the other workspace.
    """
    own, other = path_ids
    by_label = {t.label: t for t in auth_world["tokens"]}

    def get(token: Token, ws: str, suffix: str) -> int:
        return fuzz_client.get(
            f"/v3/workspaces/{ws}{suffix}", headers=_hdr(token)
        ).status_code

    cases = [
        (by_label[f"ws(w={own},p=None,s=None,sc=None)"], "/queue/status"),
        (by_label[f"peer(w={own},p={ALICE},s=None,sc=None)"], f"/peers/{ALICE}/card"),
        (
            by_label[f"session(w={own},p=None,s={SESS_1},sc=None)"],
            f"/sessions/{SESS_1}/summaries",
        ),
        (
            by_label[f"scope(w={own},p=None,s=None,sc={SCOPE_1})"],
            f"/scopes/{SCOPE_1}",
        ),
    ]
    for token, suffix in cases:
        assert get(token, own, suffix) == 200, (token.label, suffix)
        assert get(token, other, suffix) in (401, 403, 404), (token.label, suffix)


def _directed_requests(ws: str, peer: str, session: str) -> list[tuple[str, str, Any]]:
    """Well-formed requests that read or touch `peer`'s data inside `ws`."""
    root = f"/v3/workspaces/{ws}"
    return [
        ("GET", f"{root}/peers/{peer}/card", None),
        ("PUT", f"{root}/peers/{peer}/card", {"peer_card": ["x"]}),
        ("GET", f"{root}/peers/{peer}/context", None),
        ("POST", f"{root}/peers/{peer}/representation", {}),
        ("POST", f"{root}/peers/{peer}/search", {"query": "hiking"}),
        ("POST", f"{root}/peers/{peer}/sessions", {}),
        ("PUT", f"{root}/peers/{peer}", {"metadata": {"a": 1}}),
        ("POST", f"{root}/peers/{peer}/chat", {"query": "what do they like?"}),
        (
            "POST",
            f"{root}/peers/{peer}/chat",
            {"query": "q", "session_id": session},
        ),
        (
            "POST",
            f"{root}/peers/{peer}/chat",
            {"query": "q", "filters": {"session_id": [session]}},
        ),
        ("GET", f"{root}/sessions/{session}/peers/{peer}/config", None),
        ("GET", f"{root}/sessions/{session}/context?peer_target={peer}", None),
        (
            "GET",
            f"{root}/sessions/{session}/context?peer_target={BOB}&peer_perspective={peer}",
            None,
        ),
        ("POST", f"{root}/conclusions/list", {"filters": {"observed_id": peer}}),
        ("POST", f"{root}/conclusions/query", {"query": "hiking"}),
        ("POST", f"{root}/search", {"query": "hiking"}),
        ("POST", f"{root}/sessions/list", {}),
        ("POST", f"{root}/peers/list", {}),
        ("GET", f"{root}/queue/status", None),
        ("GET", f"{root}/sessions/{session}/messages/list", None),
        ("POST", f"{root}/sessions/{session}/messages/list", {}),
        ("POST", f"{root}/sessions/{session}/search", {"query": "hiking"}),
        ("GET", f"{root}/sessions/{session}/summaries", None),
        ("GET", f"{root}/sessions/{session}/peers", None),
        ("PUT", f"{root}/sessions/{session}/peers", {peer: {}}),
        ("POST", f"{root}/sessions/{session}/peers", {peer: {}}),
        ("POST", f"{root}/sessions/{session}/clone", None),
        ("DELETE", f"{root}/sessions/{session}", None),
        ("POST", f"{root}/sessions", {"id": session, "peers": {peer: {}}}),
        ("POST", f"{root}/sessions", {"id": session, "peers": {ALICE: {}, BOB: {}}}),
        ("POST", f"{root}/sessions", {"id": "new-s", "scopes": [SCOPE_1]}),
        ("POST", f"{root}/peers", {"id": peer}),
        ("POST", f"{root}/scopes", {"id": SCOPE_1}),
        ("POST", f"{root}/scopes/list", {}),
        ("GET", f"{root}/webhooks", None),
        ("POST", "/v3/keys?workspace_id=" + ws, None),
        ("POST", "/v3/workspaces", {"id": ws}),
        ("POST", "/v3/workspaces/list", {}),
        ("DELETE", root, None),
    ]


def _status(client: TestClient, token: Token, method: str, url: str, body: Any) -> int:
    return client.request(
        method,
        url,
        json=body,
        headers=_hdr(token),
        follow_redirects=False,
    ).status_code


@pytest.mark.parametrize(
    "kind,claim",
    [
        ("peer", ALICE),
        ("peer", BOB),
        ("session", SESS_1),
        ("session", SESS_2),
        ("scope", SCOPE_1),
        ("scope", SCOPE_2),
    ],
)
def test_scoped_key_cannot_touch_unentitled_resources(
    fuzz_client: TestClient, auth_world: dict[str, Any], kind: str, claim: str
):
    """Directed isolation matrix for peer, session and named-scope keys.

    Every request that the policy does not entitle the key to must not 2xx:
    the other peer's routes, the session the key's peer is not in, workspace-
    level routes, and everything in the other workspace.
    """
    token = next(
        t
        for t in auth_world["tokens"]
        if t.kind == kind and t.w == WS_A and claim in (t.p, t.s, t.sc)
    )
    oracle = Oracle(fuzz_client, auth_world["admin"])
    bad: list[str] = []
    checks = oks = 0
    for ws in (WS_A, WS_B):
        for peer in (ALICE, BOB):
            for session in (SESS_1, SESS_2):
                for method, url, body in _directed_requests(ws, peer, session):
                    tmpl, pp = _resolve(method, urlsplit(url).path)
                    query = dict(
                        pair.split("=")
                        for pair in urlsplit(url).query.split("&")
                        if pair
                    )
                    allowed = tmpl is not None and oracle.permitted(
                        token, method, tmpl, pp, query, body
                    )
                    status = _status(fuzz_client, token, method, url, body)
                    checks += 1
                    assert status < 500, (status, method, url)
                    oks += 200 <= status < 300
                    if 200 <= status < 300 and not allowed:
                        bad.append(f"{status} {method} {url} {body}")
    assert checks > 300
    assert oks, "directed matrix never reached a 2xx; the oracle is untested"
    assert not bad, f"{token.label} got unentitled 2xx:\n" + "\n".join(bad)
