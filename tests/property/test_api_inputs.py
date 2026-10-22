# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false, reportUnknownVariableType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnknownLambdaType=false, reportUnusedParameter=false
"""Targeted property tests for inputs the schema fuzzer reaches poorly.

Covers multipart uploads (the generic fuzzer always 422s before parsing), the
filter DSL on every list endpoint, and odd-but-valid JSON in metadata and
configuration bodies.
"""

import json
import os
from typing import Any

from fastapi.testclient import TestClient
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

MAX_EXAMPLES = int(os.environ.get("HONCHO_FUZZ_EXAMPLES", "25"))

FUZZ = settings(
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

WS = "/v3/workspaces/fuzz-ws"


def _no_5xx(response: Any, what: str) -> None:
    assert response.status_code < 500, f"{response.status_code} {what}\n{response.text}"


# --------------------------------------------------------------------------
# 1. Uploads
# --------------------------------------------------------------------------

CONTENT_TYPES = st.sampled_from(
    [
        "application/pdf",
        "text/plain",
        "text/markdown",
        "application/json",
        "application/octet-stream",
        "image/png",
        "",
        "text/",
        "APPLICATION/JSON",
        "application/json; charset=utf-8",
    ]
)

_json_scalars = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-(10**40), max_value=10**40),
    st.floats(allow_nan=False, allow_infinity=False),
    st.text(),
)
_json_values = st.recursive(
    _json_scalars,
    lambda c: st.one_of(st.lists(c, max_size=4), st.dictionaries(st.text(), c)),
    max_leaves=12,
)

_json_payloads = st.one_of(
    _json_values.map(lambda v: json.dumps(v).encode()),
    _json_values.map(lambda v: json.dumps(v, ensure_ascii=False).encode()),
    st.sampled_from(
        [b"", b"  ", b"NaN", b"[NaN, Infinity]", b'"\\u0000"', b'{"a\\u0000":1}']
    ),
    st.binary(max_size=200),
)

_pdf_payloads = st.one_of(
    st.binary(max_size=300),
    st.just(b"%PDF-1.4\n"),
    st.just(b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<</Root 1 0 R>>\n%%EOF"),
    st.binary(max_size=100).map(lambda b: b"%PDF-1.4\n" + b),
)

_text_payloads = st.one_of(
    st.binary(max_size=300),
    st.text().map(lambda s: s.encode("utf-16")),
    st.text().map(lambda s: s.encode("utf-8", "surrogatepass")),
    st.just(b"a\x00b"),
    st.just(b"\xff\xfe"),
    st.just(b""),
)

FILENAMES = st.one_of(
    st.sampled_from(["", "a.txt", "../../etc/passwd", "a" * 5000, "\x00.txt", "é.pdf"]),
    st.text(max_size=40),
)


@FUZZ
@given(
    payload=st.one_of(_json_payloads, _pdf_payloads, _text_payloads),
    content_type=CONTENT_TYPES,
    filename=FILENAMES,
    peer=st.sampled_from(["fuzz-alice", "nobody", "", "\x00", "a b"]),
    send_peer=st.booleans(),
    extra=st.dictionaries(
        st.sampled_from(["metadata", "configuration", "created_at"]),
        st.one_of(
            st.text(max_size=30),
            _json_values.map(json.dumps),
            st.sampled_from(["null", "[]", "1", '{"a":{"b":1}}', "2020-01-01", "x"]),
        ),
    ),
)
def test_upload_does_not_5xx(
    fuzz_client: TestClient,
    seeded_ids: dict[str, str],
    payload: bytes,
    content_type: str,
    filename: str,
    peer: str,
    send_peer: bool,
    extra: dict[str, str],
) -> None:
    data = dict(extra)
    if send_peer:
        data["peer_id"] = peer
    try:
        r = fuzz_client.post(
            f"{WS}/sessions/fuzz-session/messages/upload",
            files={"file": (filename, payload, content_type)},
            data=data,
        )
    except Exception as exc:  # TestClient re-raises unhandled server errors
        raise AssertionError(
            f"{type(exc).__name__}: {exc} ct={content_type!r} name={filename!r}"
            + f" payload={payload[:80]!r} data={data!r}"
        ) from exc
    _no_5xx(r, f"ct={content_type!r} name={filename!r} payload={payload[:80]!r}")


# --------------------------------------------------------------------------
# 2. Filter DSL
# --------------------------------------------------------------------------

_keys = st.sampled_from(["role", "dept", "n", "flag", "nested", "x.y", "", "a b"])
_vals = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-(10**25), max_value=10**25),
    st.floats(allow_nan=False, allow_infinity=False),
    st.sampled_from(["admin", "eng", "abc", "*", "2020-01-01", "2020-01-01T00:00:00Z"]),
    st.text(max_size=8),
    st.lists(st.one_of(st.integers(), st.text(max_size=4), st.none()), max_size=3),
)
_ops = st.sampled_from(
    ["gt", "gte", "lt", "lte", "ne", "in", "contains", "icontains", "bogus"]
)
_op_dict = st.dictionaries(_ops, _vals, min_size=1, max_size=2)
_meta = st.dictionaries(
    _keys,
    st.one_of(_vals, _op_dict, st.dictionaries(_keys, _vals, max_size=2)),
    max_size=3,
)

PEER_COLUMNS = [
    "id",
    "created_at",
    "is_active",
    "workspace_id",
    "metadata",
    "bogus",
    "token_count",
    "peer_id",
    "session_id",
    "level",
    "source_ids",
    "observer_id",
    "observed_id",
    "times_derived",
    "sync_state",
    "internal_metadata",
    "configuration",
]
_leaf = st.one_of(
    _meta.map(lambda m: {"metadata": m}),
    st.tuples(st.sampled_from(PEER_COLUMNS), st.one_of(_vals, _op_dict)).map(
        lambda t: {t[0]: t[1]}
    ),
    st.sampled_from(["*", None]).map(lambda w: {"id": w}),
    st.just({}),
)
filter_trees = st.recursive(
    _leaf,
    lambda c: st.one_of(
        st.dictionaries(
            st.sampled_from(["AND", "OR", "NOT"]),
            st.lists(c, max_size=3),
            min_size=1,
            max_size=1,
        ),
        st.dictionaries(
            st.sampled_from(["AND", "OR", "NOT"]), c, min_size=1, max_size=1
        ),
    ),
    max_leaves=10,
)


def _deep(depth: int, op: str) -> dict[str, Any]:
    f: dict[str, Any] = {"metadata": {"role": "admin"}}
    for _ in range(depth):
        f = {op: [f]}
    return f


LIST_ENDPOINTS = [
    "/v3/workspaces/list",
    f"{WS}/peers/list",
    f"{WS}/sessions/list",
    f"{WS}/scopes/list",
    f"{WS}/sessions/fuzz-session/messages/list",
    f"{WS}/conclusions/list",
    f"{WS}/peers/fuzz-alice/sessions",
]


@FUZZ
@given(f=filter_trees, endpoint=st.sampled_from(LIST_ENDPOINTS))
def test_filter_tree_does_not_5xx(
    fuzz_client: TestClient,
    seeded_ids: dict[str, str],
    f: dict[str, Any],
    endpoint: str,
) -> None:
    try:
        r = fuzz_client.post(endpoint, json={"filters": f})
    except Exception as exc:
        raise AssertionError(f"{type(exc).__name__}: {exc} {endpoint} {f!r}") from exc
    _no_5xx(r, f"{endpoint} {json.dumps(f)[:300]}")


def test_deep_filters_do_not_5xx(
    fuzz_client: TestClient, seeded_ids: dict[str, str]
) -> None:
    for depth in (1, 5, 6, 50, 500, 3000):
        for op in ("AND", "OR", "NOT"):
            for ep in LIST_ENDPOINTS:
                r = fuzz_client.post(ep, json={"filters": _deep(depth, op)})
                _no_5xx(r, f"{ep} depth={depth} {op}")


def _depth(f: Any) -> int:
    if isinstance(f, dict):
        inner = max((_depth(v) for v in f.values()), default=0)
        return inner + (1 if set(f) & {"AND", "OR", "NOT"} else 0)
    if isinstance(f, list):
        return max((_depth(v) for v in f), default=0)
    return 0


def _ids(fuzz_client: TestClient, f: dict[str, Any] | None) -> list[str] | int:
    body = {"filters": f} if f is not None else {}
    r = fuzz_client.post(f"{WS}/peers/list", json=body, params={"size": 100})
    if r.status_code >= 400:
        assert r.status_code < 500, r.text
        return r.status_code
    return sorted(i["id"] for i in r.json()["items"])


@FUZZ
@given(f=filter_trees)
def test_double_negation_and_wildcard_identities(
    fuzz_client: TestClient, seeded_ids: dict[str, str], f: dict[str, Any]
) -> None:
    for i, md in enumerate(
        [{"role": "admin", "n": 3}, {"role": "user", "n": 10, "flag": True}, {}]
    ):
        fuzz_client.post(f"{WS}/peers", json={"id": f"eq-{i}", "metadata": md})
    base = _ids(fuzz_client, f)
    if _depth(f) > 3:  # NOT(NOT f) adds two levels; the limit is 5
        return
    if isinstance(base, int):  # rejected: wrappers must also be rejected
        assert isinstance(_ids(fuzz_client, {"NOT": [{"NOT": [f]}]}), int)
        return
    nn = _ids(fuzz_client, {"NOT": [{"NOT": [f]}]})
    assert nn == base, f"NOT(NOT f) != f for {f!r}"
    aw = _ids(fuzz_client, {"AND": [f, {"id": "*"}]})
    assert aw == base, f"AND(f, *) != f for {f!r}"
    ow = _ids(fuzz_client, {"OR": [f, {"id": {"ne": "zzz-none"}}]})
    assert isinstance(ow, list)


# --------------------------------------------------------------------------
# 3. Metadata / configuration round trip
# --------------------------------------------------------------------------

_odd_text = st.one_of(
    st.text(max_size=20),
    st.text(
        alphabet=st.characters(codec="utf-8", categories=["Cc", "Cf", "Zs"]),
        max_size=10,
    ),
    st.sampled_from(
        ["\u0000", "a\u0000b", "‮RTL", "\ud800", "\udfff", "\U0001f600", "x" * 20000]
    ),
)
_odd_num = st.one_of(
    st.integers(min_value=-(10**30), max_value=10**30),
    st.sampled_from([2**63, -(2**63) - 1, 2**64, 10**400]),
    st.floats(allow_nan=False, allow_infinity=False),
    st.sampled_from([1e308, -1e308, -0.0, 5e-324, 1e-400]),
)
_odd_json = st.recursive(
    st.one_of(st.none(), st.booleans(), _odd_num, _odd_text),
    lambda c: st.one_of(
        st.lists(c, max_size=3), st.dictionaries(_odd_text, c, max_size=3)
    ),
    max_leaves=15,
)
odd_metadata = st.dictionaries(_odd_text, _odd_json, max_size=4)


def _norm(v: Any) -> Any:
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, (int, float)):
        try:
            return float(v)
        except OverflowError:
            return v
    if isinstance(v, list):
        return [_norm(i) for i in v]
    if isinstance(v, dict):
        return {k: _norm(i) for k, i in v.items()}
    return v


def _strip_nul(v: Any) -> Any:
    if isinstance(v, str):
        return v.replace("\x00", "")
    if isinstance(v, list):
        return [_strip_nul(i) for i in v]
    if isinstance(v, dict):
        return {_strip_nul(k): _strip_nul(i) for k, i in v.items()}
    return v


def _raw(body: dict[str, Any]) -> bytes:
    # allow_nan keeps str-of-float oddities; surrogates need ensure_ascii
    return json.dumps(body, ensure_ascii=True).encode()


@FUZZ
@given(md=odd_metadata, name=st.uuids().map(lambda u: f"rt-{u.hex[:12]}"))
def test_metadata_round_trip_peers(
    fuzz_client: TestClient, seeded_ids: dict[str, str], md: dict[str, Any], name: str
) -> None:
    try:
        r = fuzz_client.post(
            f"{WS}/peers",
            content=_raw({"id": name, "metadata": md}),
            headers={"content-type": "application/json"},
        )
    except Exception as exc:
        raise AssertionError(f"{type(exc).__name__}: {exc} md={md!r}") from exc
    _no_5xx(r, f"create md={md!r}")
    if r.status_code >= 400:
        return
    stored = r.json()["metadata"]
    got = fuzz_client.post(f"{WS}/peers/list", json={"filters": {"id": name}}).json()[
        "items"
    ][0]["metadata"]
    assert _norm(got) == _norm(stored), f"list != create: {got!r} vs {stored!r}"
    sent = json.loads(_raw({"m": md}))["m"]
    # metadata strips NUL by design; JSONB stores numbers as numeric
    assert _norm(stored) == _norm(_strip_nul(sent)), (
        f"stored differs from sent: sent={md!r} stored={stored!r}"
    )


@FUZZ
@given(md=odd_metadata, cfg=odd_metadata)
def test_session_and_message_metadata_no_5xx(
    fuzz_client: TestClient,
    seeded_ids: dict[str, str],
    md: dict[str, Any],
    cfg: dict[str, Any],
) -> None:
    hdr = {"content-type": "application/json"}
    for url, body in [
        (f"{WS}/sessions", {"id": "s-odd", "metadata": md, "configuration": cfg}),
        (f"{WS}/peers", {"id": "p-odd", "metadata": md, "configuration": cfg}),
        (
            f"{WS}/sessions/fuzz-session/messages",
            {
                "messages": [
                    {
                        "content": "x",
                        "peer_id": "fuzz-alice",
                        "metadata": md,
                        "configuration": cfg,
                    }
                ]
            },
        ),
        (f"{WS}/sessions/fuzz-session", {"metadata": md, "configuration": cfg}),
        (f"{WS}/peers/fuzz-alice", {"metadata": md, "configuration": cfg}),
        (f"{WS}/sessions/fuzz-session/peers", {"fuzz-alice": {"observe_me": True}}),
        (
            f"/v3/workspaces/{WS.rsplit('/', 1)[1]}",
            {"metadata": md, "configuration": cfg},
        ),
        ("/v3/workspaces", {"id": "w-odd", "metadata": md, "configuration": cfg}),
    ]:
        try:
            r = fuzz_client.request(
                "PUT"
                if body.get("id") is None
                and "messages" not in body
                and url.endswith(("fuzz-session", "fuzz-alice", "fuzz-ws"))
                else "POST",
                url,
                content=_raw(body),
                headers=hdr,
            )
        except Exception as exc:
            raise AssertionError(
                f"{type(exc).__name__}: {exc} {url} md={md!r} cfg={cfg!r}"
            ) from exc
        _no_5xx(r, f"{url} md={md!r} cfg={cfg!r}")
