# pyright: reportUnannotatedClassAttribute=false, reportUninitializedInstanceVariable=false
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownLambdaType=false, reportAttributeAccessIssue=false
"""Sequences of API calls never produce a 5xx.

The per-operation fuzzer in `test_api_no_server_errors.py` sends one request at
a time, so it cannot reach states that need a history: a session that was
deleted and is still pending cleanup, a peer removed mid-session, a workspace
that is no longer empty. This state machine drives those histories.

Names come from a tiny pool so that creates collide with, and resurrect,
resources that were deleted earlier in the run.

Background queue work (the deriver, deletion cleanup) does not run under test,
so "deletion pending" is the only deletion state reachable here.
"""

import os
from typing import Any

from fastapi.testclient import TestClient
from hypothesis import HealthCheck, settings
from hypothesis import strategies as st
from hypothesis.stateful import (
    Bundle,
    RuleBasedStateMachine,
    rule,
    run_state_machine_as_test,
)

# Raise locally to dig for rarer failures, e.g. HONCHO_FUZZ_EXAMPLES=300
MAX_EXAMPLES = int(os.environ.get("HONCHO_FUZZ_EXAMPLES", "25"))
STEP_COUNT = int(os.environ.get("HONCHO_FUZZ_STEPS", "30"))

WORKSPACES = st.sampled_from(["ws-a", "ws-b"])
PEERS = st.sampled_from(["p1", "p2", "p3"])
SESSIONS = st.sampled_from(["s1", "s2"])
SCOPES = st.sampled_from(["sc1", "sc2"])
CONTENT = st.one_of(
    st.text(max_size=40),
    st.sampled_from(["\x00", "a\x00b", "\U0001f600", "", " ", "\u2028"]),
)
# A few keys shared across rows, so one key ends up holding numbers on some
# rows and text on others, which is what numeric filters must cope with.
_META_KEYS = st.one_of(
    st.sampled_from(["score", "tag"]), st.text(min_size=1, max_size=5)
)
_META_VALUES = st.one_of(
    st.integers(),
    st.text(max_size=5),
    st.sampled_from(["7", "high", " ", "1e200000"]),
    st.booleans(),
    st.none(),
)
METADATA = st.dictionaries(_META_KEYS, _META_VALUES, max_size=3)
OPT_BOOL = st.one_of(st.none(), st.booleans())
CONFIGURATION = st.fixed_dictionaries(
    {},
    optional={
        "reasoning": st.fixed_dictionaries({"enabled": st.booleans()}),
        "summary": st.fixed_dictionaries(
            {"enabled": st.booleans()},
            optional={"messages_per_short_summary": st.integers(-5, 40)},
        ),
        "peer_card": st.fixed_dictionaries({"use": st.booleans()}),
        "custom_instructions": st.text(max_size=10),
    },
)


def _filters(*leaves: st.SearchStrategy[Any]) -> st.SearchStrategy[Any]:
    """Filters made of the given leaf conditions, plus AND/OR/NOT/wildcards."""
    leaf = st.one_of(*leaves)
    return st.one_of(
        st.none(),
        st.just({}),
        leaf,
        st.fixed_dictionaries({"AND": st.lists(leaf, max_size=3)}),
        st.fixed_dictionaries({"OR": st.lists(leaf, max_size=3)}),
        st.fixed_dictionaries({"NOT": st.lists(leaf, max_size=2)}),
        st.fixed_dictionaries({"AND": st.lists(st.just({"*": "*"}), max_size=2)}),
    )


_META_COMPARISONS = st.dictionaries(
    st.sampled_from(["gt", "gte", "lt", "lte", "ne"]),
    _META_VALUES,
    min_size=1,
    max_size=2,
)
_META = st.fixed_dictionaries(
    {
        "metadata": st.dictionaries(
            _META_KEYS, st.one_of(_META_VALUES, _META_COMPARISONS), max_size=3
        )
    }
)
_CREATED = st.fixed_dictionaries(
    {"created_at": st.fixed_dictionaries({"gte": st.just("2020-01-01T00:00:00Z")})}
)
SESSION_FILTERS = _filters(
    _META, _CREATED, st.fixed_dictionaries({"is_active": st.booleans()})
)
PEER_FILTERS = _filters(_META, _CREATED, st.fixed_dictionaries({"id": PEERS}))
MESSAGE_FILTERS = _filters(_META, _CREATED, st.fixed_dictionaries({"peer_id": PEERS}))
CONCLUSION_FILTERS = _filters(
    st.fixed_dictionaries({"observer": PEERS}),
    st.fixed_dictionaries({"observed": PEERS}),
    st.fixed_dictionaries({"session_id": SESSIONS}),
)
WORKSPACE_FILTERS = _filters(_META, _CREATED)


class ApiMachine(RuleBasedStateMachine):
    """Drives the /v3 API; the TestClient is injected as a class attribute."""

    client: TestClient

    workspaces = Bundle("workspaces")
    peers = Bundle("peers")
    sessions = Bundle("sessions")
    messages = Bundle("messages")
    conclusions = Bundle("conclusions")

    def call(self, method: str, url: str, **kwargs: Any) -> Any:
        """Send a request; fail on a 5xx or an exception the app did not handle."""
        label = f"{method} {url} {kwargs.get('json') or kwargs.get('params') or ''}"
        try:
            response = self.client.request(method, url, **kwargs)
        except Exception as exc:
            raise AssertionError(f"{type(exc).__name__}: {exc}\n{label}") from exc
        assert response.status_code < 500, (
            f"{response.status_code} {label}\n{response.text}"
        )
        return response

    # -- workspaces ----------------------------------------------------------

    @rule(target=workspaces, ws=WORKSPACES, metadata=METADATA, cfg=CONFIGURATION)
    def create_workspace(self, ws, metadata, cfg):
        self.call(
            "POST",
            "/v3/workspaces",
            json={"id": ws, "metadata": metadata, "configuration": cfg},
        )
        return ws

    @rule(ws=workspaces, metadata=METADATA, cfg=CONFIGURATION)
    def update_workspace(self, ws, metadata, cfg):
        self.call(
            "PUT",
            f"/v3/workspaces/{ws}",
            json={"metadata": metadata, "configuration": cfg},
        )

    @rule(ws=workspaces, really=st.integers(0, 5))
    def delete_workspace(self, ws, really):
        # Deleting a workspace makes everything under it 404, so do it rarely.
        if really:
            return
        self.call("DELETE", f"/v3/workspaces/{ws}")

    @rule(filters=WORKSPACE_FILTERS)
    def list_workspaces(self, filters):
        self.call("POST", "/v3/workspaces/list", json={"filters": filters})

    @rule(ws=workspaces, query=CONTENT)
    def search_workspace(self, ws, query):
        self.call("POST", f"/v3/workspaces/{ws}/search", json={"query": query})

    @rule(ws=workspaces)
    def queue_status(self, ws):
        self.call("GET", f"/v3/workspaces/{ws}/queue/status")

    # -- peers ---------------------------------------------------------------

    @rule(target=peers, ws=workspaces, peer=PEERS, metadata=METADATA)
    def create_peer(self, ws, peer, metadata):
        self.call(
            "POST",
            f"/v3/workspaces/{ws}/peers",
            json={"id": peer, "metadata": metadata},
        )
        return (ws, peer)

    @rule(
        p=peers, metadata=METADATA, cfg=st.dictionaries(st.text(max_size=4), OPT_BOOL)
    )
    def update_peer(self, p, metadata, cfg):
        ws, peer = p
        self.call(
            "PUT",
            f"/v3/workspaces/{ws}/peers/{peer}",
            json={"metadata": metadata, "configuration": cfg},
        )

    @rule(ws=workspaces, filters=PEER_FILTERS)
    def list_peers(self, ws, filters):
        self.call("POST", f"/v3/workspaces/{ws}/peers/list", json={"filters": filters})

    @rule(
        p=peers,
        other=PEERS,
        card=st.lists(st.text(max_size=20), max_size=3),
        use_target=st.booleans(),
    )
    def set_peer_card(self, p, other, card, use_target):
        ws, peer = p
        params = {"target": other} if use_target else None
        self.call(
            "PUT",
            f"/v3/workspaces/{ws}/peers/{peer}/card",
            json={"peer_card": card},
            params=params,
        )
        self.call("GET", f"/v3/workspaces/{ws}/peers/{peer}/card", params=params)

    @rule(p=peers, other=PEERS, query=CONTENT)
    def peer_reads(self, p, other, query):
        ws, peer = p
        base = f"/v3/workspaces/{ws}/peers/{peer}"
        self.call("POST", f"{base}/sessions", json={})
        self.call("POST", f"{base}/search", json={"query": query})
        self.call("GET", f"{base}/context", params={"target": other})
        self.call("POST", f"{base}/representation", json={"target": other})

    # -- sessions ------------------------------------------------------------

    @rule(
        target=sessions,
        ws=workspaces,
        session=SESSIONS,
        peer_map=st.dictionaries(
            PEERS,
            st.fixed_dictionaries(
                {}, optional={"observe_me": OPT_BOOL, "observe_others": OPT_BOOL}
            ),
            max_size=3,
        ),
        metadata=METADATA,
        cfg=CONFIGURATION,
    )
    def create_session(self, ws, session, peer_map, metadata, cfg):
        self.call(
            "POST",
            f"/v3/workspaces/{ws}/sessions",
            json={
                "id": session,
                "peers": peer_map,
                "metadata": metadata,
                "configuration": cfg,
            },
        )
        return (ws, session)

    @rule(s=sessions, metadata=METADATA, cfg=CONFIGURATION)
    def update_session(self, s, metadata, cfg):
        ws, session = s
        self.call(
            "PUT",
            f"/v3/workspaces/{ws}/sessions/{session}",
            json={"metadata": metadata, "configuration": cfg},
        )

    @rule(s=sessions)
    def delete_session(self, s):
        ws, session = s
        self.call("DELETE", f"/v3/workspaces/{ws}/sessions/{session}")

    @rule(target=sessions, s=sessions, message_id=st.one_of(st.none(), st.just("nope")))
    def clone_session(self, s, message_id):
        ws, session = s
        response = self.call(
            "POST",
            f"/v3/workspaces/{ws}/sessions/{session}/clone",
            params={"message_id": message_id} if message_id else None,
        )
        return (
            (ws, response.json()["id"])
            if response.status_code == 201
            else (ws, session)
        )

    @rule(target=sessions, m=messages, s=sessions)
    def clone_session_at_message(self, m, s):
        """Clone at a message; `s` may be a different session than the message's."""
        ws, session = s
        message_id = m[2]
        response = self.call(
            "POST",
            f"/v3/workspaces/{ws}/sessions/{session}/clone",
            params={"message_id": message_id},
        )
        return (
            (ws, response.json()["id"])
            if response.status_code == 201
            else (ws, session)
        )

    @rule(
        s=sessions,
        peers_=st.dictionaries(
            PEERS,
            st.fixed_dictionaries({}, optional={"observe_me": OPT_BOOL}),
            max_size=3,
        ),
        mode=st.sampled_from(["add", "set"]),
    )
    def add_or_set_peers(self, s, peers_, mode):
        ws, session = s
        self.call(
            "POST" if mode == "add" else "PUT",
            f"/v3/workspaces/{ws}/sessions/{session}/peers",
            json=peers_,
        )

    @rule(s=sessions, peers_=st.lists(PEERS, max_size=3))
    def remove_peers(self, s, peers_):
        ws, session = s
        self.call(
            "DELETE", f"/v3/workspaces/{ws}/sessions/{session}/peers", json=peers_
        )

    @rule(s=sessions, peer=PEERS, observe_me=OPT_BOOL, observe_others=OPT_BOOL)
    def peer_config(self, s, peer, observe_me, observe_others):
        ws, session = s
        url = f"/v3/workspaces/{ws}/sessions/{session}/peers/{peer}/config"
        self.call(
            "PUT",
            url,
            json={"observe_me": observe_me, "observe_others": observe_others},
        )
        self.call("GET", url)

    @rule(ws=workspaces, filters=SESSION_FILTERS)
    def list_sessions(self, ws, filters):
        self.call(
            "POST", f"/v3/workspaces/{ws}/sessions/list", json={"filters": filters}
        )

    @rule(
        s=sessions,
        tokens=st.one_of(st.none(), st.integers(0, 5000)),
        summary=st.booleans(),
        peer_target=st.one_of(st.none(), PEERS),
        peer_perspective=st.one_of(st.none(), PEERS),
        search_query=st.one_of(st.none(), CONTENT),
    )
    def session_reads(
        self, s, tokens, summary, peer_target, peer_perspective, search_query
    ):
        ws, session = s
        base = f"/v3/workspaces/{ws}/sessions/{session}"
        params = {
            k: v
            for k, v in {
                "tokens": tokens,
                "summary": summary,
                "peer_target": peer_target,
                "peer_perspective": peer_perspective,
                "search_query": search_query,
            }.items()
            if v is not None
        }
        self.call("GET", f"{base}/context", params=params)
        self.call("GET", f"{base}/summaries")
        self.call("GET", f"{base}/peers")
        self.call("POST", f"{base}/search", json={"query": search_query or "x"})

    # -- messages ------------------------------------------------------------

    @rule(
        target=messages,
        s=sessions,
        batch=st.lists(
            st.tuples(
                PEERS,
                CONTENT,
                st.one_of(st.none(), METADATA),
                st.one_of(
                    st.none(),
                    st.sampled_from(
                        [
                            "2020-01-01T00:00:00Z",
                            "2999-01-01T00:00:00Z",
                            "1900-01-01T00:00:00+05:00",
                        ]
                    ),
                ),
            ),
            min_size=1,
            max_size=4,
        ),
    )
    def create_messages(self, s, batch):
        ws, session = s
        response = self.call(
            "POST",
            f"/v3/workspaces/{ws}/sessions/{session}/messages",
            json={
                "messages": [
                    {
                        "peer_id": p,
                        "content": c,
                        **({"metadata": m} if m is not None else {}),
                        **({"created_at": t} if t else {}),
                    }
                    for p, c, m, t in batch
                ]
            },
        )
        if response.status_code < 300 and response.json():
            return (ws, session, response.json()[-1]["id"])
        return (ws, session, "missing")

    @rule(m=messages, metadata=METADATA)
    def message_get_update(self, m, metadata):
        ws, session, mid = m
        url = f"/v3/workspaces/{ws}/sessions/{session}/messages/{mid}"
        self.call("GET", url)
        self.call("PUT", url, json={"metadata": metadata})

    @rule(
        s=sessions, filters=MESSAGE_FILTERS, reverse=st.booleans(), cursor=st.booleans()
    )
    def list_messages(self, s, filters, reverse, cursor):
        ws, session = s
        params: dict[str, Any] = {"reverse": reverse}
        if cursor:
            params["cursor"] = ""
        self.call(
            "POST",
            f"/v3/workspaces/{ws}/sessions/{session}/messages/list",
            json={"filters": filters},
            params=params,
        )

    # -- conclusions ---------------------------------------------------------

    @rule(
        target=conclusions,
        ws=workspaces,
        observer=PEERS,
        observed=PEERS,
        session=st.one_of(st.none(), SESSIONS),
        content=st.text(min_size=1, max_size=30),
    )
    def create_conclusion(self, ws, observer, observed, session, content):
        response = self.call(
            "POST",
            f"/v3/workspaces/{ws}/conclusions",
            json={
                "conclusions": [
                    {
                        "content": content,
                        "observer_id": observer,
                        "observed_id": observed,
                        "session_id": session,
                    }
                ]
            },
        )
        if response.status_code == 201:
            return (ws, response.json()[0]["id"], observer, observed)
        return (ws, "missing", observer, observed)

    @rule(c=conclusions)
    def delete_conclusion(self, c):
        ws, cid, _, _ = c
        self.call("GET", f"/v3/workspaces/{ws}/conclusions/{cid}")
        self.call("DELETE", f"/v3/workspaces/{ws}/conclusions/{cid}")

    @rule(c=conclusions, filters=CONCLUSION_FILTERS, query=CONTENT)
    def conclusion_reads(self, c, filters, query):
        ws, _, observer, observed = c
        self.call(
            "POST", f"/v3/workspaces/{ws}/conclusions/list", json={"filters": filters}
        )
        self.call(
            "POST",
            f"/v3/workspaces/{ws}/conclusions/query",
            json={
                "query": query,
                "filters": {"observer": observer, "observed": observed},
            },
        )

    # -- scopes, dreams, webhooks ---------------------------------------------

    @rule(ws=workspaces, scope=SCOPES, metadata=METADATA)
    def create_scope(self, ws, scope, metadata):
        self.call(
            "POST",
            f"/v3/workspaces/{ws}/scopes",
            json={"id": scope, "metadata": metadata},
        )
        self.call("GET", f"/v3/workspaces/{ws}/scopes/{scope}")
        self.call("GET", f"/v3/workspaces/{ws}/scopes/{scope}/status")
        self.call("POST", f"/v3/workspaces/{ws}/scopes/list", json={})

    @rule(s=sessions, scope=SCOPES, add=st.booleans())
    def scope_membership(self, s, scope, add):
        ws, session = s
        base = f"/v3/workspaces/{ws}/scopes/{scope}"
        if add:
            self.call("POST", f"{base}/sessions", json={"session_ids": [session]})
        else:
            self.call("DELETE", f"{base}/sessions/{session}")
        self.call("POST", f"{base}/sessions/list", json={})

    @rule(
        ws=workspaces,
        observer=PEERS,
        observed=st.one_of(st.none(), PEERS),
        session=st.one_of(st.none(), SESSIONS),
        dream_type=st.sampled_from(["omni", "card_refresh"]),
        rebuild=st.booleans(),
    )
    def schedule_dream(self, ws, observer, observed, session, dream_type, rebuild):
        self.call(
            "POST",
            f"/v3/workspaces/{ws}/schedule_dream",
            json={
                "observer": observer,
                "observed": observed,
                "session_id": session,
                "dream_type": dream_type,
                "rebuild": rebuild,
            },
        )

    @rule(ws=workspaces)
    def webhooks(self, ws):
        self.call(
            "POST",
            f"/v3/workspaces/{ws}/webhooks",
            json={"url": "https://example.com/hook"},
        )
        self.call("GET", f"/v3/workspaces/{ws}/webhooks")


def test_api_state_machine_no_server_errors(fuzz_client: TestClient) -> None:
    """Run the machine against the fuzz client (one DB session per request)."""
    machine = type("BoundApiMachine", (ApiMachine,), {"client": fuzz_client})
    run_state_machine_as_test(
        machine,
        settings=settings(
            max_examples=MAX_EXAMPLES,
            stateful_step_count=STEP_COUNT,
            deadline=None,
            database=None,
            suppress_health_check=[
                HealthCheck.function_scoped_fixture,
                HealthCheck.too_slow,
                HealthCheck.filter_too_much,
                HealthCheck.data_too_large,
            ],
        ),
    )
