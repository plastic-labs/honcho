"""Arrow IPC stream bytes for the conclusion graph route.

Every arro3 call lives here, so moving to another Arrow library changes one file.
"""

from io import BytesIO

from arro3.core import (
    Array,
    DataType,
    Field,
    RecordBatch,
    RecordBatchReader,
    list_array,
)
from arro3.io import write_ipc_stream

from src.crud.graph import GraphColumns

MEDIA_TYPE = "application/vnd.apache.arrow.stream"
# bumped when a column changes meaning or type; added columns don't bump it
FORMAT_VERSION = "1"
# lets a streaming reader draw the first conclusions before the rest arrive
BATCH_ROWS = 65_536

_STRING = DataType.string()
_UINT32 = DataType.uint32()
# level, lineage and observer_kind have a handful of values each
_LABEL = DataType.dictionary(DataType.int8(), _STRING)
# a workspace can hold more peers and sessions than int16 indexes
_NAME = DataType.dictionary(DataType.int32(), _STRING)
# pyarrow names a list's values "item"; arro3 would leave the name empty
_SOURCES = DataType.list(Field("item", _UINT32, nullable=False))


def _strings(values: list[str] | list[str | None], kind: DataType) -> Array:
    return Array(values, _STRING).cast(kind)


def graph_batch(cols: GraphColumns) -> RecordBatch:
    """Build the graph as one Arrow record batch."""
    return RecordBatch.from_pydict(
        {
            "id": Array(cols.ids, _STRING),
            "level": _strings(cols.levels, _LABEL),
            "observer": _strings(cols.observers, _NAME),
            "observed": _strings(cols.observeds, _NAME),
            "observer_kind": _strings(cols.observer_kinds, _LABEL),
            "session": _strings(cols.sessions, _NAME),
            "created_at": Array.from_buffer(cols.created_us).cast(
                DataType.timestamp("us", tz="UTC")
            ),
            "times_derived": Array.from_buffer(cols.times_derived),
            "lineage": _strings(cols.lineage, _LABEL),
            "unresolved_outside": Array.from_buffer(cols.unresolved_outside),
            "unresolved_deleted": Array.from_buffer(cols.unresolved_deleted),
            "unresolved_unknown": Array.from_buffer(cols.unresolved_unknown),
            "is_copy": Array(cols.is_copy, DataType.bool()),
            "copied_from": Array(cols.copied_from, _UINT32),
            "sources": list_array(
                Array.from_buffer(cols.source_offsets),
                Array.from_buffer(cols.source_targets),
                type=_SOURCES,
            ),
        },
        metadata={"honcho.graph.version": FORMAT_VERSION},
    )


def encode_graph(cols: GraphColumns) -> bytes:
    """Encode the graph as an uncompressed Arrow IPC stream."""
    batch = graph_batch(cols)
    rows = batch.num_rows
    # arro3's slice panics on a length past the end instead of clamping it
    batches = [
        batch.slice(start, min(BATCH_ROWS, rows - start))
        for start in range(0, rows, BATCH_ROWS)
    ] or [batch]
    sink = BytesIO()
    # arro3's runtime default is LZ4, though its type stub says None
    write_ipc_stream(
        RecordBatchReader.from_batches(batch.schema, batches), sink, compression=None
    )
    return sink.getvalue()
