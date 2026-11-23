"""HONCHOGR v2 bytes of the conclusion graph, decoded and checked against WIRE.md."""

import array
import copy
import struct
from collections.abc import Callable, Sequence
from typing import Any

import pytest

from src.crud.graph import (
    DELETED,
    OUTSIDE,
    ROW_LINK,
    ROW_MISSING,
    ROW_NODE,
    ROW_SCOPE,
    UNKNOWN,
    build_graph_columns,
)
from src.utils import honchogr, honchogr_reference
from src.utils.graph_wire import encode_graph
from src.utils.honchogr_reference import (
    LINEAGE_DICTIONARY,
    NO_COPIED_FROM,
    NULL_SESSION,
    SECTION_NAMES,
    TYPE_DICTIONARY,
)

NOT_APPLICABLE, COMPLETE, PARTIAL, LINEAGE_UNKNOWN = range(4)


def decode(payload: bytes) -> dict[str, Any]:
    """Decode HONCHOGR v2 bytes and check every structural rule of WIRE.md."""
    header = struct.unpack_from("<8sHHIIIIIIIII", payload)
    magic, version, flags, section_count, node_count, edge_count = header[:6]
    perspective_count, out_total, del_total, unk_total, unknown_lineage, id_width = (
        header[6:]
    )
    assert (magic, version, flags, section_count) == (b"HONCHOGR", 2, 0, 18)

    sections: dict[str, bytes] = {}
    end = 48 + 18 * 32
    for i, name in enumerate(SECTION_NAMES):
        raw, offset, length = struct.unpack_from("<16sQQ", payload, 48 + 32 * i)
        assert raw == name.encode().ljust(16, b"\0")
        assert offset == (end + 7) & ~7
        assert payload[end:offset] == bytes(offset - end)
        sections[name] = payload[offset : offset + length]
        end = offset + length
    assert end == len(payload)

    def numbers(name: str, code: str) -> list[int]:
        body = sections[name]
        return list(struct.unpack(f"<{len(body) // struct.calcsize(code)}{code}", body))

    def strings(body: bytes) -> list[str]:
        (count,) = struct.unpack_from("<I", body)
        at, values = 4, list[str]()
        for _ in range(count):
            (length,) = struct.unpack_from("<I", body, at)
            values.append(body[at + 4 : at + 4 + length].decode())
            at += 4 + length
        assert at == len(body)
        return values

    def perspectives(body: bytes) -> list[tuple[str, str, str]]:
        count, reserved = struct.unpack_from("<II", body)
        assert (count, reserved) == (perspective_count, 0)
        at, values = 8, list[tuple[str, str, str]]()
        for _ in range(count):
            observer_len, observed_len, kind = struct.unpack_from("<IIB", body, at)
            assert body[at + 9 : at + 16] == bytes(7)
            at += 16
            observer = body[at : at + observer_len].decode()
            observed = body[at + observer_len : at + observer_len + observed_len]
            at += observer_len + observed_len
            assert body[at : at + (-at % 8)] == bytes(-at % 8)
            at += -at % 8
            values.append((observer, observed.decode(), ("peer", "scope")[int(kind)]))
        assert at == len(body)
        return values

    if id_width:
        raw_ids = sections["node_ids"]
        assert len(raw_ids) == node_count * id_width
        ids = [
            raw_ids[i * id_width : (i + 1) * id_width].decode("ascii")
            for i in range(node_count)
        ]
    else:
        ids = strings(sections["node_ids"])

    graph: dict[str, Any] = {
        "edge_count": edge_count,
        "unknown_lineage": unknown_lineage,
        "totals": (out_total, del_total, unk_total),
        "id_width": id_width,
        "ids": ids,
        "types": [TYPE_DICTIONARY[t] for t in sections["node_type"]],
        "sessions": strings(sections["session_dict"]),
        "perspectives": perspectives(sections["persp_dict"]),
        "node_session": numbers("node_session", "I"),
        "node_persp": numbers("node_persp", "I"),
        "created_us": numbers("created_us", "q"),
        "lineage": list(sections["lineage"]),
        "outside": numbers("unresolved_out", "I"),
        "deleted": numbers("unresolved_del", "I"),
        "unknown": numbers("unresolved_unk", "I"),
        "times_derived": numbers("times_derived", "I"),
        "flags": list(sections["node_flags"]),
        "copied_from": numbers("copied_from", "I"),
        "offsets": numbers("csr_offsets", "I"),
        "targets": numbers("csr_targets", "I"),
    }
    assert strings(sections["type_dict"]) == list(TYPE_DICTIONARY)
    assert strings(sections["lineage_dict"]) == list(LINEAGE_DICTIONARY)
    _check_columns(graph, node_count)
    return graph


def _check_columns(g: dict[str, Any], n: int) -> None:
    for name in ("types", "node_session", "node_persp", "created_us", "lineage"):
        assert len(g[name]) == n
    assert len(g["offsets"]) == n + 1
    assert g["offsets"][0] == 0 and g["offsets"][-1] == g["edge_count"]
    assert all(a <= b for a, b in zip(g["offsets"], g["offsets"][1:], strict=False))
    assert all(t < n for t in g["targets"])
    assert len(set(g["ids"])) == n
    order = [(us, i.encode()) for us, i in zip(g["created_us"], g["ids"], strict=True)]
    assert order == sorted(order)
    sessions = [s.encode() for s in g["sessions"]]
    assert sessions == sorted(set(sessions))
    keys = [(o.encode(), d.encode(), k) for o, d, k in g["perspectives"]]
    assert keys == sorted(set(keys))
    assert all(s == NULL_SESSION or s < len(sessions) for s in g["node_session"])
    assert all(p < len(keys) for p in g["node_persp"])
    for i in range(n):
        assert g["flags"][i] in (0, 1)
        if not g["flags"][i]:
            assert g["copied_from"][i] == NO_COPIED_FROM
        else:
            assert g["copied_from"][i] == NO_COPIED_FROM or g["copied_from"][i] < n
        unresolved = g["outside"][i] or g["deleted"][i] or g["unknown"][i]
        degree = g["offsets"][i + 1] - g["offsets"][i]
        if unresolved:
            expected = PARTIAL
        elif g["types"][i] == "explicit":
            expected = NOT_APPLICABLE
        elif degree:
            expected = COMPLETE
        else:
            expected = LINEAGE_UNKNOWN
        assert g["lineage"][i] == expected
    assert g["unknown_lineage"] == g["lineage"].count(LINEAGE_UNKNOWN)
    assert g["totals"] == (sum(g["outside"]), sum(g["deleted"]), sum(g["unknown"]))


def _wire_fixture(*, string_ids: bool) -> dict[str, Any]:
    """The synthetic fixture of honcho-viz spike/wire-v1 tests/generate-api-graph-fixture.py."""
    if string_ids:
        node_ids = [
            f"id-{letter}-{greek}"
            for letter, greek in zip("ABCDEFGHI", "αβγδεζηθι", strict=True)
        ]
    else:
        node_ids = [letter * 21 for letter in "ABCDEFGHI"]
    return {
        "node_ids": node_ids,
        "node_type": array.array("B", [0, 2, 1, 0, 2, 3, 1, 2, 2]),
        "node_session": array.array(
            "I", [0, NULL_SESSION, 2, 0, 2, 2, 1, NULL_SESSION, 1]
        ),
        "node_perspective": array.array("I", [0, 0, 0, 1, 0, 0, 1, 1, 1]),
        "created_us": array.array(
            "q",
            [
                -1_000_001,
                -1,
                1_704_067_200_123_456,
                1_704_067_200_123_456,
                1_704_153_600_000_002,
                1_704_153_600_000_003,
                1_704_240_000_000_004,
                1_704_326_400_000_005,
                1_704_412_800_000_006,
            ],
        ),
        "unresolved_outside": array.array("I", [0, 0, 1, 0, 0, 0, 0, 0, 0]),
        "unresolved_deleted": array.array("I", [0, 0, 0, 0, 0, 0, 0, 1, 0]),
        "unresolved_unknown": array.array("I", [0, 0, 0, 1, 0, 0, 0, 1, 0]),
        "times_derived": array.array("I", [0, 7, 2, 0, 4, 5, 9, 3, 11]),
        "node_flags": array.array("B", [0, 0, 0, 0, 0, 0, 1, 1, 0]),
        "copied_from": array.array(
            "I", [NO_COPIED_FROM] * 6 + [0, NO_COPIED_FROM, NO_COPIED_FROM]
        ),
        "csr_offsets": array.array("I", [0, 0, 0, 1, 1, 2, 3, 4, 4, 5]),
        "csr_targets": array.array("I", [0, 5, 4, 0, 0]),
        "session_dict": ["séance 🌱", "Ω", "研究/β"],
        "perspective_dict": [
            ("obsérver 🌱", "peer-α", "peer"),
            ("obsérver 🌱", "scope/研究", "scope"),
        ],
        "id_width": 0 if string_ids else 21,
    }


def _as_lists(values: dict[str, Any]) -> dict[str, Any]:
    return {
        k: v.tolist() if isinstance(v, array.array) else v for k, v in values.items()
    }


@pytest.mark.parametrize("string_ids", [False, True])
@pytest.mark.parametrize("convert", [dict, _as_lists], ids=["arrays", "lists"])
def test_fast_encoder_writes_the_reference_bytes(
    string_ids: bool, convert: Callable[[dict[str, Any]], dict[str, Any]]
) -> None:
    values = convert(_wire_fixture(string_ids=string_ids))
    fallbacks = honchogr.fallbacks
    fast = honchogr.encode_graph_columns(**copy.deepcopy(values))
    assert honchogr.fallbacks == fallbacks
    assert fast == honchogr_reference.encode_graph_columns(**copy.deepcopy(values))


def _set(name: str, index: int, value: object) -> Callable[[dict[str, Any]], None]:
    return lambda v: v[name].__setitem__(index, value)


# The invalid inputs of honcho-viz spike/wire-v1 tests/encoder-validation.py.
_INVALID: dict[str, Callable[[dict[str, Any]], object]] = {
    "inconsistent column lengths": lambda v: v["times_derived"].pop(),
    "negative times_derived": lambda v: v.update(times_derived=[-1] + [0] * 8),
    "overflowing times_derived": lambda v: v.update(times_derived=[2**32] + [0] * 8),
    "non-integer column values": lambda v: v.update(node_type=[False] + [0] * 8),
    "node types outside dictionary": _set("node_type", 0, 4),
    "duplicate IDs": lambda v: v["node_ids"].__setitem__(1, v["node_ids"][0]),
    "anonymous IDs": _set("node_ids", 0, ""),
    "fixed-width non-ASCII IDs": _set("node_ids", 0, "α" * 21),
    "invalid fixed ID width": lambda v: v.update(id_width=20),
    "node order regressions": _set("created_us", 1, -2_000_000),
    "unsorted sessions": lambda v: v["session_dict"].reverse(),
    "duplicate perspectives": lambda v: v["perspective_dict"].append(
        v["perspective_dict"][-1]
    ),
    "unknown perspective kind": _set("perspective_dict", 0, ("obs", "target", "other")),
    "invalid session references": _set("node_session", 0, 3),
    "invalid perspective references": _set("node_perspective", 0, 2),
    "reserved node flag bits": _set("node_flags", 0, 2),
    "originals on non-copies": _set("copied_from", 0, 1),
    "copy references outside response": lambda v: (
        v["node_flags"].__setitem__(0, 1),
        v["copied_from"].__setitem__(0, 9),
    ),
    "invalid CSR offsets": _set("csr_offsets", 1, 6),
    "invalid CSR targets": _set("csr_targets", 0, 9),
    "unresolved header-total overflow": lambda v: v.update(
        unresolved_unknown=array.array("I", [0xFFFFFFFF, 1] + [0] * 7)
    ),
    "created_us overflow": lambda v: v.update(created_us=[-(2**63) - 1] + [0] * 8),
    "missing perspectives for nodes": lambda v: v.update(
        perspective_dict=[], node_perspective=[0] * 9
    ),
}


@pytest.mark.parametrize("mutate", _INVALID.values(), ids=list(_INVALID))
def test_fast_encoder_raises_the_reference_error(
    mutate: Callable[[dict[str, Any]], object],
) -> None:
    values = _wire_fixture(string_ids=False)
    mutate(values)
    with pytest.raises((TypeError, ValueError)) as reference:
        honchogr_reference.encode_graph_columns(**copy.deepcopy(values))
    with pytest.raises(reference.type) as fast:
        honchogr.encode_graph_columns(**copy.deepcopy(values))
    assert str(fast.value) == str(reference.value)


def test_wire_fixture_decodes_to_the_spec_values() -> None:
    graph = decode(honchogr.encode_graph_columns(**_wire_fixture(string_ids=False)))
    assert graph["totals"] == (1, 1, 2)
    assert graph["unknown_lineage"] == 1
    assert graph["lineage"] == [
        NOT_APPLICABLE,
        LINEAGE_UNKNOWN,
        PARTIAL,
        PARTIAL,
        COMPLETE,
        COMPLETE,
        COMPLETE,
        PARTIAL,
        COMPLETE,
    ]
    assert graph["targets"] == [0, 5, 4, 0, 0]
    assert graph["sessions"] == ["séance 🌱", "Ω", "研究/β"]
    assert graph["copied_from"][6:] == [0, NO_COPIED_FROM, NO_COPIED_FROM]


def _id(name: str) -> str:
    return name.ljust(21, "_")


def _node(
    name: str,
    created_us: int,
    *,
    observer: str = "alice",
    observed: str = "bob",
    level: str = "explicit",
    session: str | None = None,
    copied_from: str | None = None,
    times_derived: int = 1,
) -> Sequence[Any]:
    is_copy = copied_from is not None
    original = _id(copied_from) if copied_from else None
    return (
        ROW_NODE,
        _id(name),
        observer,
        observed,
        level,
        session,
        original,
        created_us,
        times_derived,
        is_copy,
    )


def _link(derived: str, position: int, source: str) -> Sequence[Any]:
    return (
        ROW_LINK,
        _id(derived),
        None,
        None,
        None,
        None,
        _id(source),
        None,
        position,
        None,
    )


def _missing(source: str, source_class: int) -> Sequence[Any]:
    return (
        ROW_MISSING,
        None,
        None,
        None,
        None,
        None,
        _id(source),
        None,
        source_class,
        None,
    )


def _scope(observer: str) -> Sequence[Any]:
    return (ROW_SCOPE, None, observer, None, None, None, None, None, None, None)


def test_graph_rows_become_spec_columns() -> None:
    rows = [
        _missing("gone", DELETED),
        _node("tie-b", 200, level="deductive", session="s2", times_derived=3),
        _link("tie-b", 1, "root"),
        _link("tie-b", 0, "tie-a"),
        _node("root", 100, session="s1"),
        _node("tie-a", 200, level="inductive", session="s1"),
        _link("tie-a", 0, "elsewhere"),
        _link("tie-a", 1, "gone"),
        _link("tie-a", 2, "nowhere"),
        _missing("elsewhere", OUTSIDE),
        _missing("nowhere", UNKNOWN),
        _node("lonely", 300, level="contradiction"),
        _node("copy-here", 400, observer="scope.team", copied_from="root"),
        _node("copy-away", 401, observer="scope.team", copied_from="absent"),
        _link("copy-away", 0, "unclassified"),
        _scope("scope.team"),
    ]
    graph = decode(encode_graph(build_graph_columns(rows)))

    assert graph["ids"] == [
        _id(n) for n in ("root", "tie-a", "tie-b", "lonely", "copy-here", "copy-away")
    ]
    assert graph["id_width"] == 21
    assert graph["created_us"] == [100, 200, 200, 300, 400, 401]
    assert graph["types"] == [
        "explicit",
        "inductive",
        "deductive",
        "contradiction",
        "explicit",
        "explicit",
    ]
    assert graph["sessions"] == ["s1", "s2"]
    assert graph["node_session"] == [0, 0, 1, NULL_SESSION, NULL_SESSION, NULL_SESSION]
    assert graph["perspectives"] == [
        ("alice", "bob", "peer"),
        ("scope.team", "bob", "scope"),
    ]
    assert graph["node_persp"] == [0, 0, 0, 0, 1, 1]
    assert graph["times_derived"] == [1, 1, 3, 1, 1, 1]
    assert graph["flags"] == [0, 0, 0, 0, 1, 1]
    assert graph["copied_from"] == [NO_COPIED_FROM] * 4 + [0, NO_COPIED_FROM]
    # tie-b cites tie-a at position 0 and root at position 1
    assert graph["offsets"] == [0, 0, 0, 2, 2, 2, 2]
    assert graph["targets"] == [1, 0]
    assert graph["outside"] == [0, 1, 0, 0, 0, 0]
    assert graph["deleted"] == [0, 1, 0, 0, 0, 0]
    # a source with no class row counts as unknown
    assert graph["unknown"] == [0, 1, 0, 0, 0, 1]
    assert graph["lineage"] == [
        NOT_APPLICABLE,
        PARTIAL,
        COMPLETE,
        LINEAGE_UNKNOWN,
        NOT_APPLICABLE,
        PARTIAL,
    ]


def test_unresolved_source_ids_stay_out_of_the_bytes() -> None:
    rows = [
        _node("kept", 1, level="deductive"),
        _link("kept", 0, "private-source"),
        _missing("private-source", UNKNOWN),
        _node("copy", 2, copied_from="private-original"),
    ]
    payload = encode_graph(build_graph_columns(rows))
    assert _id("private-source").encode() not in payload
    assert _id("private-original").encode() not in payload


def test_an_empty_graph_encodes() -> None:
    graph = decode(encode_graph(build_graph_columns([])))
    assert graph["ids"] == [] and graph["perspectives"] == []


def test_a_level_outside_the_type_dictionary_is_rejected() -> None:
    with pytest.raises(ValueError, match="no type for conclusion levels"):
        build_graph_columns([_node("odd", 1, level="speculative")])
