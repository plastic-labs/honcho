"""HONCHOGR v2 response bytes for the conclusion graph route."""

from src.crud.graph import GraphColumns
from src.utils.honchogr import encode_graph_columns

# HONCHO_GRAPH_MEDIA_TYPE in the v2 decoder (honcho-viz lab/honcho-graph-format.mjs)
CONTENT_TYPE = "application/vnd.honcho.graph-csr"
# documents.id is exactly 21 characters of [A-Za-z0-9_-] (the id_length and id_format checks)
ID_WIDTH = 21


def encode_graph(cols: GraphColumns) -> bytes:
    return encode_graph_columns(
        node_ids=cols.ids,
        node_type=cols.node_type,
        node_session=cols.node_session,
        node_perspective=cols.node_perspective,
        created_us=cols.created_us,
        unresolved_outside=cols.unresolved_outside,
        unresolved_deleted=cols.unresolved_deleted,
        unresolved_unknown=cols.unresolved_unknown,
        times_derived=cols.times_derived,
        node_flags=cols.node_flags,
        copied_from=cols.copied_from,
        csr_offsets=cols.csr_offsets,
        csr_targets=cols.csr_targets,
        session_dict=cols.session_dict,
        perspective_dict=cols.perspective_dict,
        id_width=ID_WIDTH,
    )
