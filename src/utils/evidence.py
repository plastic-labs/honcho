"""Collation of what the dialectic agent read while answering.

Evidence is gathered passively: read paths hand the rows they already loaded to
an accumulator, and nothing is re-queried when the response is built. The model
is never asked to cite its sources, which keeps collection deterministic and
free of model load, at the cost of over-reporting -- the agent may read a
conclusion it does not end up leaning on.

Because nothing is re-queried, evidence inherits the scoping of the reads that
produced it: whatever the workspace, observer/observed pair and session
allowlist permitted the agent to see is exactly what can appear here.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, cast

from src import models
from src.schemas.api import (
    EVIDENCE_MESSAGE_PREVIEW_CHARS,
    Evidence,
    EvidenceMessageRef,
    EvidenceObservation,
    EvidenceToolCall,
)
from src.utils.representation import Representation


def _as_utc(timestamp: datetime) -> datetime:
    """Restore the UTC marker a representation timestamp dropped.

    `Representation` renders observations into prompts, so it strips tzinfo to
    keep those timestamps short. The underlying columns are stored UTC, so
    naming UTC again recovers the instant rather than shifting it -- and an API
    response should not hand a caller a bare local-looking timestamp to guess
    about.
    """
    return timestamp if timestamp.tzinfo is not None else timestamp.replace(tzinfo=UTC)


def _tool_call_from_log_entry(entry: dict[str, Any]) -> EvidenceToolCall:
    """Read one entry of a tool loop's call log.

    The log carries two shapes depending on where it came from: the accumulated
    loop history uses ``tool_name``/``tool_input``, a raw provider response uses
    ``name``/``input``. Accept either. Results are deliberately dropped -- they
    are already reflected in the conclusions and messages, and they are large.
    """
    name = entry.get("tool_name") or entry.get("name") or ""
    raw_input = entry.get("tool_input")
    if raw_input is None:
        raw_input = entry.get("input")
    return EvidenceToolCall(
        tool_name=str(name),
        tool_input=cast("dict[str, Any]", raw_input)
        if isinstance(raw_input, dict)
        else {},
    )


@dataclass
class EvidenceAccumulator:
    """Collects the rows a dialectic agent reads, for one chat call.

    Created by the router when the caller asks for evidence and threaded down
    through the agent into ``ToolContext``. Tool handlers reach it through the
    context, which they copy with ``dataclasses.replace`` -- a shallow copy, so
    every copy appends to this same instance.

    Conclusions and messages are keyed by ID so that a row two tools both
    returned is recorded once. Keying on ID rather than content matters:
    ``Representation``'s own deduplication ignores IDs, which would collapse
    distinct conclusions that happen to read the same.
    """

    conclusions: dict[str, models.Document] = field(default_factory=dict)
    messages: dict[str, models.Message] = field(default_factory=dict)
    tool_calls: list[EvidenceToolCall] = field(default_factory=list)

    def add_documents(self, documents: Iterable[models.Document]) -> None:
        """Record conclusions a read path returned."""
        for document in documents:
            self.conclusions.setdefault(document.id, document)

    def add_messages(self, messages: Iterable[models.Message]) -> None:
        """Record messages a read path returned."""
        for message in messages:
            self.messages.setdefault(message.public_id, message)

    def record_tool_calls(self, tool_calls_made: Sequence[dict[str, Any]]) -> None:
        """Replace the tool call log with a completed loop's history.

        The tool loop rewrites its log wholesale on every exit path, so this
        overwrites rather than appends. Failed calls never reach the log, so the
        result records successful invocations only.
        """
        self.tool_calls = [
            _tool_call_from_log_entry(entry) for entry in tool_calls_made
        ]

    def build(self) -> Evidence:
        """Flatten what was collected into the API shape."""
        return Evidence(
            conclusions=_flatten_conclusions(self.conclusions.values()),
            messages=sorted(
                (
                    EvidenceMessageRef(
                        id=message.public_id,
                        session_id=message.session_name,
                        peer_id=message.peer_name,
                        content_preview=message.content[
                            :EVIDENCE_MESSAGE_PREVIEW_CHARS
                        ],
                        created_at=message.created_at,
                    )
                    for message in self.messages.values()
                ),
                key=lambda ref: (ref.created_at, ref.id),
            ),
            tool_calls=list(self.tool_calls),
        )


def _flatten_conclusions(
    documents: Iterable[models.Document],
) -> list[EvidenceObservation]:
    """Turn conclusion rows into a flat, level-tagged list.

    Goes through ``Representation.from_documents`` rather than reading the rows
    directly so that evidence resolves ``source_ids`` and derivation timestamps
    the same way every other reader of a conclusion does -- both have fallbacks
    for older rows that are easy to get wrong twice.

    An observation's ``message_ids`` are dropped: they are internal row ids,
    and Honcho identifies messages by their public id everywhere it faces a
    caller.
    """
    representation = Representation.from_documents(list(documents))
    observations = [
        EvidenceObservation(
            id=observation.id,
            level="explicit",
            content=observation.content,
            created_at=_as_utc(observation.created_at),
            session_id=observation.session_name,
        )
        for observation in representation.explicit
    ]
    observations += [
        EvidenceObservation(
            id=observation.id,
            level="deductive",
            content=observation.conclusion,
            created_at=_as_utc(observation.created_at),
            session_id=observation.session_name,
            source_ids=observation.source_ids,
        )
        for observation in representation.deductive
    ]
    observations += [
        EvidenceObservation(
            id=observation.id,
            level="inductive",
            content=observation.conclusion,
            created_at=_as_utc(observation.created_at),
            session_id=observation.session_name,
            source_ids=observation.source_ids,
        )
        for observation in representation.inductive
    ]
    observations += [
        EvidenceObservation(
            id=observation.id,
            level="contradiction",
            content=observation.content,
            created_at=_as_utc(observation.created_at),
            session_id=observation.session_name,
            source_ids=observation.source_ids,
        )
        for observation in representation.contradiction
    ]
    observations.sort(key=lambda observation: (observation.created_at, observation.id))
    return observations
