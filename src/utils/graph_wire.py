"""Provisional binary graph encoding for the DEV-3414 spike; to be replaced by the shared wire module."""

import json
from array import array

from src.crud.graph import LEVELS, GraphColumns

MAGIC = b"HGR0"
CONTENT_TYPE = "application/vnd.honcho.graph"


def _pad(buf: bytearray) -> None:
    buf.extend(b"\0" * (-len(buf) % 8))


def encode_graph(cols: GraphColumns, header: dict[str, object]) -> bytes:
    n = len(cols.ids)
    meta = json.dumps(
        {
            **header,
            "node_count": n,
            "link_count": len(cols.source_index),
            "levels": LEVELS,
            "perspectives": [list(p) for p in cols.perspectives],
            "sections": [
                "ids:ascii21",
                "created_at:f64",
                "source_offsets:u32",
                "source_index:u32",
                "times_derived:u32",
                "unresolved:u16",
                "perspective:u16",
                "level:u8",
            ],
        },
        separators=(",", ":"),
    ).encode()

    out = bytearray(MAGIC)
    out.extend(len(meta).to_bytes(4, "little"))
    out.extend(meta)
    _pad(out)
    out.extend("".join(cols.ids).encode("ascii"))
    _pad(out)
    for typecode, values in (
        ("d", cols.created_at),
        ("I", cols.source_offsets),
        ("I", cols.source_index),
        ("I", cols.times_derived),
        ("H", [min(u, 0xFFFF) for u in cols.unresolved]),
        ("H", cols.perspective),
        ("B", cols.levels),
    ):
        out.extend(array(typecode, values).tobytes())
        _pad(out)
    return bytes(out)
