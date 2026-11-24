"""Graph rows become columns, and columns become an Arrow stream, with no database."""

from array import array
from typing import Any, cast

import pytest
from arro3.core import DataType, Table
from arro3.io import read_ipc_stream

from src.crud.graph import (
    DELETED,
    OUTSIDE,
    ROW_LINK,
    ROW_MISSING,
    ROW_NODE,
    ROW_SCOPE,
    UNKNOWN,
    GraphColumns,
    build_graph_columns,
)
from src.exceptions import ValidationException
from src.routers.graph import etag_matches
from src.utils import graph_arrow
from src.utils.graph_arrow import MEDIA_TYPE, encode_graph

LIMIT = 1_000


def _node(
    doc_id: str,
    created_us: int,
    *,
    level: str = "explicit",
    observer: str = "alice",
    observed: str = "bob",
    session: str | None = None,
    copied_from: str | None = None,
    times_derived: int = 0,
) -> tuple[Any, ...]:
    is_copy = copied_from is not None
    return (
        ROW_NODE,
        doc_id,
        observer,
        observed,
        level,
        session,
        copied_from,
        created_us,
        times_derived,
        is_copy,
    )


def _link(derived: str, position: int, source: str) -> tuple[Any, ...]:
    return (ROW_LINK, derived, None, None, None, None, source, None, position, None)


def _missing(source: str, source_class: int) -> tuple[Any, ...]:
    return (ROW_MISSING, None, None, None, None, None, source, None, source_class, None)


def _scope(name: str) -> tuple[Any, ...]:
    return (ROW_SCOPE, None, name, None, None, None, None, None, None, None)


def _values(column: Any) -> list[Any]:
    # arro3's stub says to_pylist returns an ndarray; it returns a list
    values: object = column.to_pylist()
    assert isinstance(values, list)
    return cast("list[Any]", values)


def _decode(body: bytes) -> dict[str, list[Any]]:
    table = Table.from_batches(list(read_ipc_stream(body)))
    return {name: _values(table.column(name)) for name in table.column_names}


def test_conclusions_are_ordered_by_creation_time_then_id():
    rows = [_node("c", 20), _node("b", 10), _node("a", 10)]
    cols = build_graph_columns(rows, LIMIT)
    assert cols.ids == ["a", "b", "c"]
    assert list(cols.created_us) == [10, 10, 20]


def test_sources_follow_their_position_and_missing_sources_are_counted():
    rows = [
        _node("root", 1),
        _node("other", 2),
        _node("derived", 3, level="deductive"),
        _link("derived", 1, "root"),
        _link("derived", 0, "other"),
        _link("derived", 2, "outside"),
        _link("derived", 3, "gone"),
        _link("derived", 4, "never"),
        _link("derived", 5, "unclassed"),
        _missing("outside", OUTSIDE),
        _missing("gone", DELETED),
        _missing("never", UNKNOWN),
    ]
    cols = build_graph_columns(rows, LIMIT)
    d = cols.ids.index("derived")
    start, end = cols.source_offsets[d], cols.source_offsets[d + 1]
    assert list(cols.source_targets[start:end]) == [1, 0]
    # a missing source with no class row counts as unknown
    assert (
        cols.unresolved_outside[d],
        cols.unresolved_deleted[d],
        cols.unresolved_unknown[d],
    ) == (1, 1, 2)


def test_lineage_says_how_much_of_a_conclusions_sources_resolved():
    rows = [
        _node("root", 1),
        _node("complete", 2, level="deductive"),
        _node("partial", 3, level="inductive"),
        _node("lonely", 4, level="deductive"),
        _node("explicit-partial", 5),
        _link("complete", 0, "root"),
        _link("partial", 0, "root"),
        _link("partial", 1, "gone"),
        _link("explicit-partial", 0, "gone"),
        _missing("gone", DELETED),
    ]
    cols = build_graph_columns(rows, LIMIT)
    assert dict(zip(cols.ids, cols.lineage, strict=True)) == {
        "root": "not_applicable",
        "complete": "complete",
        "partial": "partial",
        "lonely": "unknown",
        "explicit-partial": "partial",
    }


def test_scope_observers_and_copies():
    rows = [
        _node("original", 1),
        _node("copy", 2, observer="scope.team", copied_from="original"),
        _node("orphan", 3, observer="scope.team", copied_from="missing"),
        _scope("scope.team"),
    ]
    cols = build_graph_columns(rows, LIMIT)
    assert cols.observer_kinds == ["peer", "scope", "scope"]
    assert cols.is_copy == [False, True, True]
    assert cols.copied_from == [None, 0, None]


def test_a_graph_at_the_limit_builds_and_one_past_it_is_refused():
    rows = [_node("a", 1), _node("b", 2), _node("c", 3)]
    assert len(build_graph_columns(rows, 3).ids) == 3
    with pytest.raises(ValidationException, match="more than 2 conclusions"):
        build_graph_columns(rows, 2)


def test_the_stream_round_trips_every_column():
    rows = [
        _node("root", 1_704_067_200_000_000, session="s1"),
        _node("derived", 1_704_067_260_000_000, level="deductive", times_derived=2),
        _node("copy", 1_704_067_320_000_000, observer="scope.team", copied_from="root"),
        _link("derived", 0, "root"),
        _link("derived", 1, "gone"),
        _missing("gone", DELETED),
        _scope("scope.team"),
    ]
    cols = build_graph_columns(rows, LIMIT)
    graph = _decode(encode_graph(cols))

    assert graph["id"] == ["root", "derived", "copy"]
    assert graph["level"] == ["explicit", "deductive", "explicit"]
    assert graph["observer"] == ["alice", "alice", "scope.team"]
    assert graph["observed"] == ["bob", "bob", "bob"]
    assert graph["observer_kind"] == ["peer", "peer", "scope"]
    assert graph["session"] == ["s1", None, None]
    assert [t.timestamp() for t in graph["created_at"]] == [
        1_704_067_200,
        1_704_067_260,
        1_704_067_320,
    ]
    assert graph["times_derived"] == [0, 2, 0]
    assert graph["lineage"] == ["not_applicable", "partial", "not_applicable"]
    assert graph["unresolved_outside"] == [0, 0, 0]
    assert graph["unresolved_deleted"] == [0, 1, 0]
    assert graph["unresolved_unknown"] == [0, 0, 0]
    assert graph["is_copy"] == [False, False, True]
    assert graph["copied_from"] == [None, None, 0]
    assert graph["sources"] == [[], [0], []]


def test_the_stream_has_dictionary_columns_and_a_version():
    body = encode_graph(build_graph_columns([_node("a", 1)], LIMIT))
    schema = read_ipc_stream(body).schema
    for name in ("level", "observer", "observed", "observer_kind", "session"):
        assert DataType.is_dictionary(schema.field(name).type), name
    assert schema.metadata_str == {"honcho.graph.version": graph_arrow.FORMAT_VERSION}
    assert MEDIA_TYPE == "application/vnd.apache.arrow.stream"


def test_the_stream_splits_into_batches_with_a_short_last_one(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(graph_arrow, "BATCH_ROWS", 2)
    rows = [_node(f"n{i}", i) for i in range(5)] + [_link("n4", 0, "n0")]
    body = encode_graph(build_graph_columns(rows, LIMIT))
    assert [batch.num_rows for batch in read_ipc_stream(body)] == [2, 2, 1]
    assert _decode(body)["sources"] == [[], [], [], [], [0]]


def test_an_empty_graph_encodes():
    body = encode_graph(GraphColumns(source_offsets=array("i", [0])))
    assert _decode(body)["id"] == []


@pytest.mark.parametrize(
    ("header", "matches"),
    [
        (None, False),
        ('"abc"', True),
        ('W/"abc"', True),
        ('"x", W/"abc"', True),
        ("*", True),
        ('"abcd"', False),
    ],
)
def test_if_none_match_uses_weak_comparison(header: str | None, matches: bool):
    assert etag_matches(header, '"abc"') is matches
