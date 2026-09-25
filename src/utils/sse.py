"""Server-sent event framing for streamed dialectic answers.

Shared by the peer and workspace chat routes so the two stay in step; the
frames they emit are part of the public API and the SDKs parse them.
"""

import json
from collections.abc import AsyncIterator

from src.schemas.api import Evidence
from src.utils.evidence import EvidenceAccumulator


async def format_dialectic_sse_stream(
    chunks: AsyncIterator[str],
    evidence: EvidenceAccumulator | None = None,
) -> AsyncIterator[str]:
    """Frame answer chunks as SSE events, then a terminal event.

    Evidence can only be known once the answer has finished streaming, so it
    rides on the terminal event rather than a frame of its own. `evidence` is
    the accumulator the agent filled in while answering; None means the caller
    did not ask for evidence and the terminal event carries only `done`.
    """
    async for chunk in chunks:
        yield f"data: {json.dumps({'delta': {'content': chunk}, 'done': False})}\n\n"

    final: dict[str, object] = {"done": True}
    if evidence is not None:
        final["evidence"] = _serializable(evidence.build())
    yield f"data: {json.dumps(final)}\n\n"


def _serializable(evidence: Evidence) -> object:
    """Round-trip through Pydantic's JSON encoder for the datetime fields."""
    return json.loads(evidence.model_dump_json())
