"""HONCHOGR v2 reference encoder: honcho-viz spike/wire-v1 wire/encoder.py verbatim, changed only where lint requires."""

from __future__ import annotations

import array
import struct
import sys
from collections.abc import Sequence
from typing import Final

GRAPH_VERSION: Final = 2
MAGIC: Final = b"HONCHOGR"
HEADER_BYTES: Final = 48
DIRECTORY_ENTRY_BYTES: Final = 32
NULL_SESSION: Final = 0xFFFFFFFF
NO_COPIED_FROM: Final = 0xFFFFFFFF
NODE_FLAG_IS_COPY: Final = 1
NODE_FLAG_ALLOWED: Final = NODE_FLAG_IS_COPY

TYPE_DICTIONARY: Final = (
    "explicit",
    "deductive",
    "inductive",
    "contradiction",
)
LINEAGE_DICTIONARY: Final = (
    "not_applicable",
    "complete",
    "partial",
    "unknown",
)

TYPE_EXPLICIT: Final = 0
LINEAGE_NOT_APPLICABLE: Final = 0
LINEAGE_COMPLETE: Final = 1
LINEAGE_PARTIAL: Final = 2
LINEAGE_UNKNOWN: Final = 3

PERSPECTIVE_KIND: Final = {"peer": 0, "scope": 1}

SECTION_NAMES: Final = (
    "type_dict",
    "lineage_dict",
    "session_dict",
    "persp_dict",
    "node_type",
    "node_session",
    "node_persp",
    "created_us",
    "lineage",
    "unresolved_out",
    "unresolved_del",
    "unresolved_unk",
    "times_derived",
    "node_flags",
    "copied_from",
    "csr_offsets",
    "csr_targets",
    "node_ids",
)

_HEADER: Final = struct.Struct("<8sHHIIIIIIIII")
_DIRECTORY_ENTRY: Final = struct.Struct("<16sQQ")
_PERSPECTIVE_ENTRY: Final = struct.Struct("<IIB7x")
_U32_MAX: Final = 0xFFFFFFFF
_I64_MIN: Final = -(2**63)
_I64_MAX: Final = 2**63 - 1


def _align8(value: int) -> int:
    return (value + 7) & ~7


def _require_sequence(name: str, values: object) -> Sequence[object]:
    if isinstance(values, (str, bytes, bytearray, memoryview)) or not (
        hasattr(values, "__len__") and hasattr(values, "__getitem__")
    ):
        raise TypeError(f"{name} must be a list, tuple, or array-like sequence")
    return values  # pyright: ignore[reportReturnType]


def _integer(name: str, value: object, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < minimum or value > maximum:
        raise ValueError(f"{name} must be in {minimum}..{maximum}")
    return value


def _column(name: str, values: object, length: int) -> Sequence[object]:
    column = _require_sequence(name, values)
    if len(column) != length:
        raise ValueError(
            f"{name} length {len(column)} does not match node count {length}"
        )
    return column


def _u32_column(name: str, values: object, length: int) -> list[int]:
    column = _column(name, values, length)
    return [
        _integer(f"{name}[{index}]", column[index], 0, _U32_MAX)
        for index in range(length)
    ]


def _little_endian_array(typecode: str, values: Sequence[int]) -> bytes:
    packed = array.array(typecode, values)
    expected_size = {"I": 4, "q": 8}[typecode]
    if packed.itemsize != expected_size:
        raise RuntimeError(
            f"platform {typecode!r} width is {packed.itemsize}, expected {expected_size}"
        )
    if sys.byteorder != "little":
        packed.byteswap()
    return packed.tobytes()


def _string_table(values: Sequence[str]) -> bytes:
    payload = bytearray(struct.pack("<I", len(values)))
    for value in values:
        encoded = value.encode("utf-8")
        if len(encoded) > _U32_MAX:
            raise ValueError("a string-table value exceeds the u32 byte-length limit")
        payload.extend(struct.pack("<I", len(encoded)))
        payload.extend(encoded)
    return bytes(payload)


def _validate_sorted_strings(name: str, values: object) -> tuple[str, ...]:
    sequence = _require_sequence(name, values)
    if len(sequence) > _U32_MAX:
        raise ValueError(f"{name} exceeds the u32 dictionary-count limit")
    result: list[str] = []
    previous: bytes | None = None
    for index in range(len(sequence)):
        value = sequence[index]
        if not isinstance(value, str):
            raise TypeError(f"{name}[{index}] must be a string")
        if not value:
            raise ValueError(f"{name}[{index}] must not be empty")
        encoded = value.encode("utf-8")
        if previous is not None and encoded <= previous:
            raise ValueError(
                f"{name} must be unique and strictly sorted by UTF-8 bytes"
            )
        previous = encoded
        result.append(value)
    return tuple(result)


def _validate_perspectives(values: object) -> tuple[tuple[str, str, str], ...]:
    sequence = _require_sequence("perspective_dict", values)
    if len(sequence) > _U32_MAX:
        raise ValueError("perspective_dict exceeds the u32 dictionary-count limit")
    result: list[tuple[str, str, str]] = []
    previous: tuple[bytes, bytes, int] | None = None
    for index in range(len(sequence)):
        entry = _require_sequence(f"perspective_dict[{index}]", sequence[index])
        if len(entry) != 3:
            raise ValueError(
                f"perspective_dict[{index}] must be (observer, observed, kind)"
            )
        observer, observed, kind = entry[0], entry[1], entry[2]
        if not isinstance(observer, str) or not observer:
            raise ValueError(
                f"perspective_dict[{index}].observer must be a non-empty string"
            )
        if not isinstance(observed, str) or not observed:
            raise ValueError(
                f"perspective_dict[{index}].observed must be a non-empty string"
            )
        if not isinstance(kind, str) or kind not in PERSPECTIVE_KIND:
            raise ValueError(
                f"perspective_dict[{index}].kind must be 'peer' or 'scope'"
            )
        key = (
            observer.encode("utf-8"),
            observed.encode("utf-8"),
            PERSPECTIVE_KIND[kind],
        )
        if previous is not None and key <= previous:
            raise ValueError(
                "perspective_dict must be unique and sorted by observer UTF-8, "
                + "observed UTF-8, then kind"
            )
        previous = key
        result.append((observer, observed, kind))
    return tuple(result)


def _perspective_table(values: Sequence[tuple[str, str, str]]) -> bytes:
    payload = bytearray(struct.pack("<II", len(values), 0))
    for observer, observed, kind in values:
        observer_bytes = observer.encode("utf-8")
        observed_bytes = observed.encode("utf-8")
        if len(observer_bytes) > _U32_MAX or len(observed_bytes) > _U32_MAX:
            raise ValueError("a perspective name exceeds the u32 byte-length limit")
        payload.extend(
            _PERSPECTIVE_ENTRY.pack(
                len(observer_bytes), len(observed_bytes), PERSPECTIVE_KIND[kind]
            )
        )
        payload.extend(observer_bytes)
        payload.extend(observed_bytes)
        payload.extend(b"\0" * (_align8(len(payload)) - len(payload)))
    return bytes(payload)


def _sum_u32(name: str, values: Sequence[int]) -> int:
    total = 0
    for value in values:
        total += value
        if total > _U32_MAX:
            raise ValueError(f"{name} total exceeds the u32 header limit")
    return total


def encode_graph_columns(
    *,
    node_ids: object,
    node_type: object,
    node_session: object,
    node_perspective: object,
    created_us: object,
    unresolved_outside: object,
    unresolved_deleted: object,
    unresolved_unknown: object,
    times_derived: object,
    node_flags: object,
    copied_from: object,
    csr_offsets: object,
    csr_targets: object,
    session_dict: object,
    perspective_dict: object,
    id_width: int = 0,
) -> bytes:
    """Validate and encode supplied-order HONCHOGR v2 columns.

    Node indices and every CSR/copy reference use the caller's order. The
    encoder never creates graph objects, synthesizes nodes, queries sources, or
    sorts nodes.
    """
    ids_input = _require_sequence("node_ids", node_ids)
    node_count = len(ids_input)
    if node_count > _U32_MAX:
        raise ValueError("node count exceeds the u32 index space")
    id_width = _integer("id_width", id_width, 0, _U32_MAX)

    ids: list[str] = []
    encoded_ids: list[bytes] = []
    seen_ids: set[str] = set()
    for index in range(node_count):
        value = ids_input[index]
        if not isinstance(value, str):
            raise TypeError(f"node_ids[{index}] must be a string")
        if not value:
            raise ValueError(f"node_ids[{index}] must not be empty")
        if value in seen_ids:
            raise ValueError(f"node_ids contains duplicate ID {value!r}")
        seen_ids.add(value)
        if id_width:
            try:
                encoded = value.encode("ascii")
            except UnicodeEncodeError as exc:
                raise ValueError("fixed-width node IDs must be ASCII") from exc
            if len(encoded) != id_width or any(
                byte < 0x21 or byte > 0x7E for byte in encoded
            ):
                raise ValueError(
                    f"node_ids[{index}] must contain exactly {id_width} printable ASCII bytes"
                )
        else:
            encoded = value.encode("utf-8")
            if len(encoded) > _U32_MAX:
                raise ValueError(f"node_ids[{index}] exceeds the u32 byte-length limit")
        ids.append(value)
        encoded_ids.append(encoded)

    sessions = _validate_sorted_strings("session_dict", session_dict)
    perspectives = _validate_perspectives(perspective_dict)
    if node_count and not perspectives:
        raise ValueError("perspective_dict must not be empty when nodes are present")

    types = _u32_column("node_type", node_type, node_count)
    node_sessions = _u32_column("node_session", node_session, node_count)
    node_perspectives = _u32_column("node_perspective", node_perspective, node_count)
    created_input = _column("created_us", created_us, node_count)
    created = [
        _integer(f"created_us[{index}]", created_input[index], _I64_MIN, _I64_MAX)
        for index in range(node_count)
    ]
    unresolved_out = _u32_column("unresolved_outside", unresolved_outside, node_count)
    unresolved_del = _u32_column("unresolved_deleted", unresolved_deleted, node_count)
    unresolved_unk = _u32_column("unresolved_unknown", unresolved_unknown, node_count)
    derived_counts = _u32_column("times_derived", times_derived, node_count)
    flags = _u32_column("node_flags", node_flags, node_count)
    originals = _u32_column("copied_from", copied_from, node_count)

    offsets_input = _require_sequence("csr_offsets", csr_offsets)
    if len(offsets_input) != node_count + 1:
        raise ValueError(
            f"csr_offsets length {len(offsets_input)} does not equal node count + 1"
        )
    offsets = [
        _integer(f"csr_offsets[{index}]", offsets_input[index], 0, _U32_MAX)
        for index in range(node_count + 1)
    ]
    targets_input = _require_sequence("csr_targets", csr_targets)
    edge_count = len(targets_input)
    if edge_count > _U32_MAX:
        raise ValueError("edge count exceeds the u32 CSR offset space")
    targets = [
        _integer(f"csr_targets[{index}]", targets_input[index], 0, _U32_MAX)
        for index in range(edge_count)
    ]
    if offsets[0] != 0:
        raise ValueError("csr_offsets[0] must be zero")
    for index in range(node_count):
        if offsets[index + 1] < offsets[index] or offsets[index + 1] > edge_count:
            raise ValueError(f"csr_offsets is not monotone and bounded at node {index}")
    if offsets[-1] != edge_count:
        raise ValueError("final csr_offsets value must equal edge count")
    for index, target in enumerate(targets):
        if target >= node_count:
            raise ValueError(f"csr_targets[{index}] is outside node bounds")

    lineage: list[int] = []
    unknown_lineage = 0
    previous_order: tuple[int, bytes] | None = None
    for node in range(node_count):
        if types[node] >= len(TYPE_DICTIONARY):
            raise ValueError(f"node_type[{node}] is outside type_dict")
        if node_sessions[node] != NULL_SESSION and node_sessions[node] >= len(sessions):
            raise ValueError(f"node_session[{node}] is outside session_dict")
        if node_perspectives[node] >= len(perspectives):
            raise ValueError(f"node_perspective[{node}] is outside perspective_dict")
        if flags[node] & ~NODE_FLAG_ALLOWED:
            raise ValueError(f"node_flags[{node}] contains reserved bits")
        is_copy = bool(flags[node] & NODE_FLAG_IS_COPY)
        original = originals[node]
        if not is_copy and original != NO_COPIED_FROM:
            raise ValueError(f"non-copy node {node} must use the copied_from sentinel")
        if is_copy and original != NO_COPIED_FROM and original >= node_count:
            raise ValueError(f"copied_from[{node}] is outside node bounds")

        order = (created[node], encoded_ids[node])
        if previous_order is not None and order <= previous_order:
            raise ValueError(
                "nodes must be unique and strictly ordered by created_us then ID bytes"
            )
        previous_order = order

        unresolved = (
            unresolved_out[node] or unresolved_del[node] or unresolved_unk[node]
        )
        degree = offsets[node + 1] - offsets[node]
        if unresolved:
            value = LINEAGE_PARTIAL
        elif types[node] == TYPE_EXPLICIT:
            value = LINEAGE_NOT_APPLICABLE
        elif degree:
            value = LINEAGE_COMPLETE
        else:
            value = LINEAGE_UNKNOWN
            unknown_lineage += 1
        lineage.append(value)

    outside_total = _sum_u32("unresolved_outside", unresolved_out)
    deleted_total = _sum_u32("unresolved_deleted", unresolved_del)
    unknown_total = _sum_u32("unresolved_unknown", unresolved_unk)

    node_id_body = b"".join(encoded_ids) if id_width else _string_table(ids)
    sections: tuple[tuple[str, bytes], ...] = (
        ("type_dict", _string_table(TYPE_DICTIONARY)),
        ("lineage_dict", _string_table(LINEAGE_DICTIONARY)),
        ("session_dict", _string_table(sessions)),
        ("persp_dict", _perspective_table(perspectives)),
        ("node_type", bytes(types)),
        ("node_session", _little_endian_array("I", node_sessions)),
        ("node_persp", _little_endian_array("I", node_perspectives)),
        ("created_us", _little_endian_array("q", created)),
        ("lineage", bytes(lineage)),
        ("unresolved_out", _little_endian_array("I", unresolved_out)),
        ("unresolved_del", _little_endian_array("I", unresolved_del)),
        ("unresolved_unk", _little_endian_array("I", unresolved_unk)),
        ("times_derived", _little_endian_array("I", derived_counts)),
        ("node_flags", bytes(flags)),
        ("copied_from", _little_endian_array("I", originals)),
        ("csr_offsets", _little_endian_array("I", offsets)),
        ("csr_targets", _little_endian_array("I", targets)),
        ("node_ids", node_id_body),
    )
    if tuple(name for name, _ in sections) != SECTION_NAMES:
        raise AssertionError("internal section order does not match SECTION_NAMES")

    section_offset = HEADER_BYTES + len(sections) * DIRECTORY_ENTRY_BYTES
    directory = bytearray()
    body = bytearray()
    for name, section in sections:
        aligned = _align8(section_offset)
        padding = aligned - section_offset
        if padding:
            body.extend(b"\0" * padding)
        section_offset = aligned
        encoded_name = name.encode("ascii")
        if len(encoded_name) > 15:
            raise AssertionError(f"section name is too long: {name}")
        directory.extend(
            _DIRECTORY_ENTRY.pack(
                encoded_name.ljust(16, b"\0"), section_offset, len(section)
            )
        )
        body.extend(section)
        section_offset += len(section)

    header = _HEADER.pack(
        MAGIC,
        GRAPH_VERSION,
        0,
        len(sections),
        node_count,
        edge_count,
        len(perspectives),
        outside_total,
        deleted_total,
        unknown_total,
        unknown_lineage,
        id_width,
    )
    if len(header) != HEADER_BYTES:
        raise AssertionError("internal header size mismatch")
    return b"".join((header, bytes(directory), bytes(body)))


__all__ = [
    "GRAPH_VERSION",
    "HEADER_BYTES",
    "LINEAGE_DICTIONARY",
    "MAGIC",
    "NO_COPIED_FROM",
    "NODE_FLAG_IS_COPY",
    "NULL_SESSION",
    "PERSPECTIVE_KIND",
    "SECTION_NAMES",
    "TYPE_DICTIONARY",
    "encode_graph_columns",
]
