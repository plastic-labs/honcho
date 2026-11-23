"""HONCHOGR v2 encoder that runs the reference encoder's checks over whole columns.

Output is byte-identical to honchogr_reference.encode_graph_columns. When any bulk
check fails, the reference encoder runs instead, so every rejection raises the
reference's exact exception type and message.
"""

# pyright: reportPrivateUsage=false
from __future__ import annotations

import array
import operator
from collections.abc import Iterable
from itertools import compress, islice
from typing import Any, Literal, cast

from src.utils import honchogr_reference as ref

_U32_MAX = 0xFFFFFFFF
_INT_TYPECODES = frozenset("bBhHiIlLqQ")
_PRINTABLE_ASCII = bytes(range(0x21, 0x7F))
# index = 4*unresolved + 2*explicit + has_sources, matching the reference's lineage branches
_LINEAGE_TABLE = bytes(
    [
        ref.LINEAGE_UNKNOWN,
        ref.LINEAGE_COMPLETE,
        ref.LINEAGE_NOT_APPLICABLE,
        ref.LINEAGE_NOT_APPLICABLE,
    ]
    + [ref.LINEAGE_PARTIAL] * 4
    + [0] * 248
)
assert ref.NODE_FLAG_ALLOWED == ref.NODE_FLAG_IS_COPY == 1, (
    "flag checks below assume one copy bit"
)
assert ref.TYPE_EXPLICIT == 0, "the explicit bit below tests node_type == 0"

fallbacks = 0


class _Fallback(Exception):
    pass


def _ints(
    typecode: Literal["I", "q"], values: Any, length: int | None = None
) -> array.array[int]:
    if isinstance(values, array.array):
        if values.typecode not in _INT_TYPECODES:
            raise _Fallback
        if values.typecode != typecode:
            values = values.tolist()
    elif set(map(type, values)) - {int}:
        raise _Fallback
    try:
        column = array.array(typecode, cast(Iterable[int], values))
    except (OverflowError, TypeError):
        raise _Fallback from None
    if length is not None and len(column) != length:
        raise _Fallback
    return column


def _nonzero_bytes(values: Iterable[int]) -> int:
    return int.from_bytes(bytes(map(bool, values)), "little")


def _encode(
    node_ids: Any,
    node_type: Any,
    node_session: Any,
    node_perspective: Any,
    created_us: Any,
    unresolved_outside: Any,
    unresolved_deleted: Any,
    unresolved_unknown: Any,
    times_derived: Any,
    node_flags: Any,
    copied_from: Any,
    csr_offsets: Any,
    csr_targets: Any,
    session_dict: Any,
    perspective_dict: Any,
    id_width: Any,
) -> bytes:
    for name, value in (
        ("node_ids", node_ids),
        ("node_type", node_type),
        ("node_session", node_session),
        ("node_perspective", node_perspective),
        ("created_us", created_us),
        ("unresolved_outside", unresolved_outside),
        ("unresolved_deleted", unresolved_deleted),
        ("unresolved_unknown", unresolved_unknown),
        ("times_derived", times_derived),
        ("node_flags", node_flags),
        ("copied_from", copied_from),
        ("csr_offsets", csr_offsets),
        ("csr_targets", csr_targets),
    ):
        ref._require_sequence(name, value)
    if type(id_width) is not int or not 0 <= id_width <= _U32_MAX:
        raise _Fallback

    ids = list(node_ids)
    n = len(ids)
    if n > _U32_MAX or set(map(type, ids)) - {str} or "" in ids or len(set(ids)) != n:
        raise _Fallback
    if id_width:
        body = "".join(ids).encode("ascii")
        if set(map(len, ids)) - {id_width} or body.translate(None, _PRINTABLE_ASCII):
            raise _Fallback
        node_id_body = body
    else:
        node_id_body = ref._string_table(ids)

    sessions = ref._validate_sorted_strings("session_dict", session_dict)
    perspectives = ref._validate_perspectives(perspective_dict)
    if n and not perspectives:
        raise _Fallback

    types = _ints("I", node_type, n)
    node_sessions = _ints("I", node_session, n)
    node_perspectives = _ints("I", node_perspective, n)
    created = _ints("q", created_us, n)
    unresolved_out = _ints("I", unresolved_outside, n)
    unresolved_del = _ints("I", unresolved_deleted, n)
    unresolved_unk = _ints("I", unresolved_unknown, n)
    derived_counts = _ints("I", times_derived, n)
    flags = _ints("I", node_flags, n)
    originals = _ints("I", copied_from, n)
    offsets = _ints("I", csr_offsets, n + 1)
    targets = _ints("I", csr_targets)
    edges = len(targets)
    if edges > _U32_MAX:
        raise _Fallback

    if offsets[0] != 0 or offsets[-1] != edges:
        raise _Fallback
    if not all(map(operator.le, offsets, islice(offsets, 1, None))):
        raise _Fallback
    if edges and max(targets) >= n:
        raise _Fallback

    if n:
        if max(types) >= len(ref.TYPE_DICTIONARY):
            raise _Fallback
        used_sessions = set(node_sessions)
        used_sessions.discard(ref.NULL_SESSION)
        if used_sessions and max(used_sessions) >= len(sessions):
            raise _Fallback
        if max(node_perspectives) >= len(perspectives):
            raise _Fallback
        if max(flags) > ref.NODE_FLAG_ALLOWED:
            raise _Fallback
        if set(compress(originals, map(operator.not_, flags))) - {ref.NO_COPIED_FROM}:
            raise _Fallback
        copy_targets = set(compress(originals, flags))
        copy_targets.discard(ref.NO_COPIED_FROM)
        if copy_targets and max(copy_targets) >= n:
            raise _Fallback
        # nodes are ordered by (created_us, ID bytes); str order equals UTF-8 byte order
        if not all(map(operator.le, created, islice(created, 1, None))):
            raise _Fallback
        ties = compress(
            range(n - 1), map(operator.eq, created, islice(created, 1, None))
        )
        if any(ids[i] >= ids[i + 1] for i in ties):
            raise _Fallback

    for column in (unresolved_out, unresolved_del, unresolved_unk):
        if sum(column) > _U32_MAX:
            raise _Fallback

    unresolved = _nonzero_bytes(
        map(
            operator.or_,
            map(operator.or_, unresolved_out, unresolved_del),
            unresolved_unk,
        )
    )
    explicit = int.from_bytes(bytes(map(operator.not_, types)), "little")
    has_sources = _nonzero_bytes(map(operator.sub, islice(offsets, 1, None), offsets))
    lineage = (
        ((unresolved << 2) | (explicit << 1) | has_sources)
        .to_bytes(n, "little")
        .translate(_LINEAGE_TABLE)
    )
    unknown_lineage = lineage.count(ref.LINEAGE_UNKNOWN)

    sections = (
        ("type_dict", ref._string_table(ref.TYPE_DICTIONARY)),
        ("lineage_dict", ref._string_table(ref.LINEAGE_DICTIONARY)),
        ("session_dict", ref._string_table(sessions)),
        ("persp_dict", ref._perspective_table(perspectives)),
        ("node_type", bytes(types.tolist())),
        ("node_session", ref._little_endian_array("I", node_sessions)),
        ("node_persp", ref._little_endian_array("I", node_perspectives)),
        ("created_us", ref._little_endian_array("q", created)),
        ("lineage", lineage),
        ("unresolved_out", ref._little_endian_array("I", unresolved_out)),
        ("unresolved_del", ref._little_endian_array("I", unresolved_del)),
        ("unresolved_unk", ref._little_endian_array("I", unresolved_unk)),
        ("times_derived", ref._little_endian_array("I", derived_counts)),
        ("node_flags", bytes(flags.tolist())),
        ("copied_from", ref._little_endian_array("I", originals)),
        ("csr_offsets", ref._little_endian_array("I", offsets)),
        ("csr_targets", ref._little_endian_array("I", targets)),
        ("node_ids", node_id_body),
    )
    if tuple(name for name, _ in sections) != ref.SECTION_NAMES:
        raise AssertionError("internal section order does not match SECTION_NAMES")

    section_offset = ref.HEADER_BYTES + len(sections) * ref.DIRECTORY_ENTRY_BYTES
    directory = bytearray()
    body = bytearray()
    for name, section in sections:
        aligned = ref._align8(section_offset)
        if aligned != section_offset:
            body.extend(b"\0" * (aligned - section_offset))
        section_offset = aligned
        directory.extend(
            ref._DIRECTORY_ENTRY.pack(
                name.encode("ascii").ljust(16, b"\0"), section_offset, len(section)
            )
        )
        body.extend(section)
        section_offset += len(section)

    header = ref._HEADER.pack(
        ref.MAGIC,
        ref.GRAPH_VERSION,
        0,
        len(sections),
        n,
        edges,
        len(perspectives),
        sum(unresolved_out),
        sum(unresolved_del),
        sum(unresolved_unk),
        unknown_lineage,
        id_width,
    )
    return b"".join((header, bytes(directory), bytes(body)))


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
    """Validate and encode supplied-order HONCHOGR v2 columns, as the reference does."""
    args: dict[str, Any] = dict(
        node_ids=node_ids,
        node_type=node_type,
        node_session=node_session,
        node_perspective=node_perspective,
        created_us=created_us,
        unresolved_outside=unresolved_outside,
        unresolved_deleted=unresolved_deleted,
        unresolved_unknown=unresolved_unknown,
        times_derived=times_derived,
        node_flags=node_flags,
        copied_from=copied_from,
        csr_offsets=csr_offsets,
        csr_targets=csr_targets,
        session_dict=session_dict,
        perspective_dict=perspective_dict,
        id_width=id_width,
    )
    try:
        return _encode(**args)
    except (_Fallback, TypeError, ValueError, OverflowError):
        global fallbacks
        fallbacks += 1
    return ref.encode_graph_columns(**args)
