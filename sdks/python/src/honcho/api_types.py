"""API types for Honcho SDK.

These types mirror the server's Pydantic schemas for API responses and requests.
"""

from __future__ import annotations

import datetime
import re
from collections.abc import Mapping
from typing import Annotated, Any, Literal, cast

from pydantic import AfterValidator, BaseModel, BeforeValidator, ConfigDict, Field

# Reasoning level of a conclusion. "explicit" conclusions are extracted directly
# from messages; the others are derived during dreaming.
ConclusionLevel = Literal["explicit", "deductive", "inductive", "contradiction"]

# ==============================================================================
# Configuration Types
# ==============================================================================


class ReasoningConfiguration(BaseModel):
    """Configuration for reasoning functionality."""

    model_config = ConfigDict(extra="forbid")  # pyright: ignore[reportUnannotatedClassAttribute]

    enabled: bool | None = None
    custom_instructions: str | None = None


class PeerCardConfiguration(BaseModel):
    """Configuration for peer card functionality."""

    model_config = ConfigDict(extra="forbid")  # pyright: ignore[reportUnannotatedClassAttribute]

    use: bool | None = None
    create: bool | None = None
    custom_instructions: str | None = None


class SummaryConfiguration(BaseModel):
    """Configuration for summary functionality."""

    model_config = ConfigDict(extra="forbid")  # pyright: ignore[reportUnannotatedClassAttribute]

    enabled: bool | None = None
    messages_per_short_summary: int | None = None
    messages_per_long_summary: int | None = None
    custom_instructions: str | None = None


class DreamConfiguration(BaseModel):
    """Configuration for dream functionality."""

    model_config = ConfigDict(extra="forbid")  # pyright: ignore[reportUnannotatedClassAttribute]

    enabled: bool | None = None
    custom_instructions: str | None = None


class DialecticConfiguration(BaseModel):
    """Configuration for the dialectic chat endpoints."""

    model_config = ConfigDict(extra="forbid")  # pyright: ignore[reportUnannotatedClassAttribute]

    custom_instructions: str | None = None


class WorkspaceConfiguration(BaseModel):
    """Workspace-level configuration options."""

    model_config = ConfigDict(extra="forbid")  # pyright: ignore[reportUnannotatedClassAttribute]

    custom_instructions: str | None = None
    reasoning: ReasoningConfiguration | None = None
    peer_card: PeerCardConfiguration | None = None
    summary: SummaryConfiguration | None = None
    dream: DreamConfiguration | None = None
    dialectic: DialecticConfiguration | None = None


class SessionConfiguration(WorkspaceConfiguration):
    """Session-level configuration options."""

    pass


class WorkspaceConfigurationResponse(WorkspaceConfiguration):
    """Workspace configuration for response parsing — tolerates unknown fields from newer servers."""

    model_config = ConfigDict(extra="ignore")  # pyright: ignore[reportUnannotatedClassAttribute]


class SessionConfigurationResponse(SessionConfiguration):
    """Session configuration for response parsing — tolerates unknown fields from newer servers."""

    model_config = ConfigDict(extra="ignore")  # pyright: ignore[reportUnannotatedClassAttribute]


class MessageConfiguration(BaseModel):
    """Message-level configuration options."""

    model_config = ConfigDict(extra="forbid")  # pyright: ignore[reportUnannotatedClassAttribute]

    reasoning: ReasoningConfiguration | None = None


# ==============================================================================
# Peer Config Types
# ==============================================================================


class PeerConfig(BaseModel):
    """Configuration for peer-level settings."""

    model_config = ConfigDict(extra="forbid")  # pyright: ignore[reportUnannotatedClassAttribute]

    observe_me: bool | None = None
    """Whether Honcho will use reasoning to form a representation of this peer."""


class SessionPeerConfig(BaseModel):
    """Configuration for a peer within a session."""

    model_config = ConfigDict(extra="forbid")  # pyright: ignore[reportUnannotatedClassAttribute]

    observe_others: bool | None = Field(
        None,
        description="Whether this peer should form a session-level theory-of-mind representation of other peers in the session",
    )
    observe_me: bool | None = Field(
        None,
        description="Whether other peers in this session should try to form a session-level theory-of-mind representation of this peer",
    )
    joined_at: datetime.datetime | None = Field(
        default=None,
        description="When this peer joined the session; defaults to now for a new membership. Write-only: it is sent when adding or setting session peers and when setting a peer's configuration, and is never returned. Set it when importing historical messages, since peer-perspective search only returns messages created at or after the peer's join time. A datetime without a timezone is read as UTC. A time more than five minutes in the future is rejected. Requires a Honcho release after v3.6.1",
    )


# ==============================================================================
# Workspace Types
# ==============================================================================


class WorkspaceResponse(BaseModel):
    """Workspace API response."""

    model_config = ConfigDict(populate_by_name=True)  # pyright: ignore[reportUnannotatedClassAttribute]

    id: str
    uid: str | None = None
    """Immutable server-assigned identifier. ``None`` on servers that predate it."""
    metadata: dict[str, Any] = Field(default_factory=dict)
    configuration: WorkspaceConfigurationResponse = Field(
        default_factory=WorkspaceConfigurationResponse
    )
    created_at: datetime.datetime
    auto: bool = False
    """Whether this is an auto workspace, which the ingest API writes to."""


class WorkspaceCreateParams(BaseModel):
    """Parameters for creating a workspace."""

    model_config = ConfigDict(extra="forbid")  # pyright: ignore[reportUnannotatedClassAttribute]

    id: str = Field(min_length=1, max_length=512)
    metadata: dict[str, Any] = Field(default_factory=dict)
    configuration: WorkspaceConfiguration = Field(
        default_factory=WorkspaceConfiguration
    )


class WorkspaceUpdateParams(BaseModel):
    """Parameters for updating a workspace."""

    model_config = ConfigDict(extra="forbid")  # pyright: ignore[reportUnannotatedClassAttribute]

    metadata: dict[str, Any] | None = None
    configuration: WorkspaceConfiguration | None = None


class WorkspaceListParams(BaseModel):
    """Parameters for listing workspaces."""

    model_config = ConfigDict(extra="forbid")  # pyright: ignore[reportUnannotatedClassAttribute]

    filters: dict[str, Any] | None = None


# ==============================================================================
# Peer Types
# ==============================================================================


class PeerResponse(BaseModel):
    """Peer API response."""

    model_config = ConfigDict(populate_by_name=True)  # pyright: ignore[reportUnannotatedClassAttribute]

    id: str
    uid: str | None = None
    """Immutable server-assigned identifier. ``None`` on servers that predate it."""
    workspace_id: str
    created_at: datetime.datetime
    metadata: dict[str, Any] = Field(default_factory=dict)
    configuration: PeerConfig = Field(default_factory=PeerConfig)


class PeerCreateParams(BaseModel):
    """Parameters for creating a peer."""

    model_config = ConfigDict(extra="forbid")  # pyright: ignore[reportUnannotatedClassAttribute]

    id: str = Field(min_length=1, max_length=512)
    metadata: dict[str, Any] | None = None
    configuration: PeerConfig | None = None


class PeerUpdateParams(BaseModel):
    """Parameters for updating a peer."""

    model_config = ConfigDict(extra="forbid")  # pyright: ignore[reportUnannotatedClassAttribute]

    metadata: dict[str, Any] | None = None
    configuration: PeerConfig | None = None


class PeerListParams(BaseModel):
    """Parameters for listing peers."""

    model_config = ConfigDict(extra="forbid")  # pyright: ignore[reportUnannotatedClassAttribute]

    filters: dict[str, Any] | None = None


class PeerRepresentationParams(BaseModel):
    """Parameters for getting peer representation."""

    model_config = ConfigDict(extra="forbid")  # pyright: ignore[reportUnannotatedClassAttribute]

    session_id: str | None = None
    target: str | None = None
    search_query: str | None = None
    search_top_k: int | None = Field(default=None, ge=1, le=100)
    search_max_distance: float | None = Field(default=None, ge=0.0, le=1.0)
    include_most_frequent: bool | None = None
    max_conclusions: int | None = Field(default=25, ge=1, le=100)


class RepresentationResponse(BaseModel):
    """Representation API response."""

    representation: str


class PeerCardResponse(BaseModel):
    """Peer card API response."""

    peer_card: list[str] | None = None


class PeerContextResponse(BaseModel):
    """Peer context API response."""

    peer_id: str
    target_id: str
    representation: str | None = None
    peer_card: list[str] | None = None


# ==============================================================================
# Session Types
# ==============================================================================


class SessionResponse(BaseModel):
    """Session API response."""

    model_config = ConfigDict(populate_by_name=True)  # pyright: ignore[reportUnannotatedClassAttribute]

    id: str
    uid: str | None = None
    """Immutable server-assigned identifier. ``None`` on servers that predate it."""
    is_active: bool
    workspace_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    configuration: SessionConfigurationResponse = Field(
        default_factory=SessionConfigurationResponse
    )
    created_at: datetime.datetime
    scopes: list[str] | None = None
    """Active scope names. Only returned by the session list with ``include_scopes``."""


class SessionCreateParams(BaseModel):
    """Parameters for creating a session."""

    model_config = ConfigDict(extra="forbid")  # pyright: ignore[reportUnannotatedClassAttribute]

    id: str = Field(min_length=1, max_length=512)
    metadata: dict[str, Any] | None = None
    peers: dict[str, SessionPeerConfig] | None = None
    configuration: SessionConfiguration | None = None
    scopes: list[str] | None = None


class SessionUpdateParams(BaseModel):
    """Parameters for updating a session."""

    model_config = ConfigDict(extra="forbid")  # pyright: ignore[reportUnannotatedClassAttribute]

    metadata: dict[str, Any] | None = None
    configuration: SessionConfiguration | None = None


class SessionListParams(BaseModel):
    """Parameters for listing sessions."""

    model_config = ConfigDict(extra="forbid")  # pyright: ignore[reportUnannotatedClassAttribute]

    filters: dict[str, Any] | None = None


# ==============================================================================
# Scope Types
# ==============================================================================


class ScopeResponse(BaseModel):
    """Scope API response."""

    model_config = ConfigDict(populate_by_name=True)  # pyright: ignore[reportUnannotatedClassAttribute]

    id: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime.datetime
    session_count: int | None = None
    """Active member sessions. Only returned by the scope list with ``include_session_count``."""


class ScopeBackfillJob(BaseModel):
    """Backfill job state for one session in a scope.

    ``docs_copied`` is present only once the backfill for that session completes.
    """

    model_config = ConfigDict(extra="ignore")  # pyright: ignore[reportUnannotatedClassAttribute]

    state: Literal["pending", "completed", "failed"]
    updated_at: datetime.datetime
    docs_copied: int | None = None


class ScopeStatusResponse(BaseModel):
    """Scope backfill/reconciliation status API response.

    ``backfill_status`` is keyed by session ID and only contains sessions that
    have had a backfill enqueued.
    """

    backfill_status: dict[str, ScopeBackfillJob] = Field(default_factory=dict)


# ==============================================================================
# Summary Types
# ==============================================================================


class Summary(BaseModel):
    """Summary model."""

    content: str
    message_id: str
    summary_type: str
    created_at: str
    token_count: int


class SessionSummariesResponse(BaseModel):
    """Session summaries API response."""

    model_config = ConfigDict(populate_by_name=True)  # pyright: ignore[reportUnannotatedClassAttribute]

    id: str
    short_summary: Summary | None = None
    long_summary: Summary | None = None


# ==============================================================================
# Session Context Types
# ==============================================================================


class SessionContextResponse(BaseModel):
    """Session context API response."""

    model_config = ConfigDict(populate_by_name=True)  # pyright: ignore[reportUnannotatedClassAttribute]

    id: str
    messages: list["MessageResponse"]
    summary: Summary | None = None
    peer_representation: str | None = None
    peer_card: list[str] | None = None


# ==============================================================================
# Message Types
# ==============================================================================


class MessageResponse(BaseModel):
    """Message API response."""

    model_config = ConfigDict(populate_by_name=True)  # pyright: ignore[reportUnannotatedClassAttribute]

    id: str
    content: str
    peer_id: str
    session_id: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime.datetime
    workspace_id: str
    token_count: int


class MessageCreateParams(BaseModel):
    """Parameters for creating a message."""

    model_config = ConfigDict(extra="forbid")  # pyright: ignore[reportUnannotatedClassAttribute]

    content: str
    peer_id: str
    metadata: dict[str, Any] | None = None
    configuration: MessageConfiguration | None = None
    created_at: datetime.datetime | None = None


class MessageBatchCreateParams(BaseModel):
    """Parameters for batch message creation."""

    model_config = ConfigDict(extra="forbid")  # pyright: ignore[reportUnannotatedClassAttribute]

    messages: list[MessageCreateParams] = Field(min_length=1, max_length=100)


class MessageUpdateParams(BaseModel):
    """Parameters for updating a message."""

    model_config = ConfigDict(extra="forbid")  # pyright: ignore[reportUnannotatedClassAttribute]

    metadata: dict[str, Any] | None = None


class MessageListParams(BaseModel):
    """Parameters for listing messages."""

    model_config = ConfigDict(extra="forbid")  # pyright: ignore[reportUnannotatedClassAttribute]

    filters: dict[str, Any] | None = None


class MessageSearchParams(BaseModel):
    """Parameters for searching messages."""

    model_config = ConfigDict(extra="forbid")  # pyright: ignore[reportUnannotatedClassAttribute]

    query: str
    filters: dict[str, Any] | None = None
    limit: int = Field(default=10, ge=1, le=100)


# ==============================================================================
# Conclusion Types
# ==============================================================================


class ConclusionResponse(BaseModel):
    """Conclusion API response."""

    model_config = ConfigDict(populate_by_name=True)  # pyright: ignore[reportUnannotatedClassAttribute]

    id: str
    content: str
    observer_id: str
    observed_id: str
    session_id: str | None = None
    level: ConclusionLevel = "explicit"
    source_ids: list[str] | None = None
    source_message_ids: list[str] | None = None
    times_derived: int = 1
    created_at: datetime.datetime


class ConclusionCreateParams(BaseModel):
    """Parameters for creating a conclusion."""

    model_config = ConfigDict(extra="forbid")  # pyright: ignore[reportUnannotatedClassAttribute]

    content: str = Field(min_length=1, max_length=65535)
    observer_id: str
    observed_id: str
    session_id: str | None = None


class ConclusionBatchCreateParams(BaseModel):
    """Parameters for batch conclusion creation."""

    model_config = ConfigDict(extra="forbid")  # pyright: ignore[reportUnannotatedClassAttribute]

    conclusions: list[ConclusionCreateParams] = Field(min_length=1, max_length=100)


class ConclusionListParams(BaseModel):
    """Parameters for listing conclusions."""

    model_config = ConfigDict(extra="forbid")  # pyright: ignore[reportUnannotatedClassAttribute]

    filters: dict[str, Any] | None = None


class ConclusionQueryParams(BaseModel):
    """Parameters for querying conclusions."""

    model_config = ConfigDict(extra="forbid")  # pyright: ignore[reportUnannotatedClassAttribute]

    query: str
    top_k: int = Field(default=10, ge=1, le=100)
    distance: float | None = Field(default=None, ge=0.0, le=1.0)
    filters: dict[str, Any] | None = None


# ==============================================================================
# Queue Status Types
# ==============================================================================


class SessionQueueStatus(BaseModel):
    """Status for a specific session in the queue."""

    session_id: str | None = None
    total_work_units: int
    completed_work_units: int
    in_progress_work_units: int
    pending_work_units: int


class QueueStatusResponse(BaseModel):
    """Queue status API response."""

    total_work_units: int
    completed_work_units: int
    in_progress_work_units: int
    pending_work_units: int
    sessions: dict[str, SessionQueueStatus] | None = None


# ==============================================================================
# Ingest Types
# ==============================================================================


# The rules ingest applies to a record, so a record validated here is one the
# server accepts (`honcho ingest import --check` relies on it). The differences
# are deliberate: unknown fields are rejected here, where the server ignores
# them, and two limits are server settings not known here: `content`'s length
# (MAX_MESSAGE_SIZE) and the token budget for custom instructions.
_NAME_PATTERN = r"^[a-zA-Z0-9_-]+$"
# NUL and lone surrogates: the characters Postgres can't store in text.
_UNSTORABLE = re.compile("[\x00\ud800-\udfff]")


def _storable(value: str) -> str:
    if _UNSTORABLE.search(value):
        raise ValueError("must not contain NUL or unpaired surrogate characters")
    return value


def _strip_nul(value: Any) -> Any:
    """NUL characters dropped from a string, or from a dict's or list's strings."""
    if isinstance(value, str):
        return value.replace("\x00", "")
    if isinstance(value, dict):
        return {_strip_nul(k): _strip_nul(v) for k, v in value.items()}  # pyright: ignore[reportUnknownVariableType]
    if isinstance(value, list):
        return [_strip_nul(item) for item in value]  # pyright: ignore[reportUnknownVariableType]
    return value


_METADATA_MAX_KEYS = 100
_METADATA_MAX_DEPTH = 5


def _metadata(value: Any) -> Any:
    """The server's limits on metadata, then NUL dropped from its strings."""
    if not isinstance(value, dict):
        return value

    def check(data: dict[Any, Any], depth: int) -> None:
        if depth > _METADATA_MAX_DEPTH:
            raise ValueError(
                f"Metadata nesting exceeds maximum depth of {_METADATA_MAX_DEPTH}"
            )
        for item in data.values():
            if isinstance(item, dict):
                check(item, depth + 1)  # pyright: ignore[reportUnknownArgumentType]

    if len(value) > _METADATA_MAX_KEYS:  # pyright: ignore[reportUnknownArgumentType]
        raise ValueError(
            f"Metadata exceeds maximum of {_METADATA_MAX_KEYS} top-level keys"
        )
    check(value, 1)  # pyright: ignore[reportUnknownArgumentType]
    return _strip_nul(value)


# An identifier is a lookup key: one the server can't store is refused, not
# changed. A display name has NUL characters dropped, as the server does.
_ExternalId = Annotated[
    str, AfterValidator(_storable), Field(min_length=1, max_length=512)
]
_DisplayName = Annotated[
    str, BeforeValidator(_strip_nul), Field(min_length=1, max_length=512)
]


_EMAIL = re.compile(r"[^@\s]+@[^@\s]+")


def _email_or_none(value: Any) -> str | None:
    """An email address, trimmed; anything else (a placeholder, blank) is None.

    Mirrors the server, which ignores such a value rather than refusing it.
    """
    if not isinstance(value, str):
        return None
    address = value.strip()
    if (
        len(address) > 320
        or not _EMAIL.fullmatch(address)
        or _UNSTORABLE.search(address)
    ):
        return None
    return address


class IngestParticipant(BaseModel):
    """A person on a source record. ``id`` is what connects them across records."""

    model_config = ConfigDict(extra="forbid")  # pyright: ignore[reportUnannotatedClassAttribute]

    id: _ExternalId | None = None
    """The person's ID in the source, if it has one."""
    name: _DisplayName
    """The person's display name."""
    handle: _DisplayName | None = None
    """The account's username in the source, unique there (a GitHub login, a
    Discord username). Used only to suggest which accounts are one person."""
    aliases: list[_DisplayName] = Field(default_factory=list, max_length=10)
    """Other names the account goes by, such as a profile name or a server
    nickname. Used only to suggest which accounts are one person."""
    email: Annotated[str | None, BeforeValidator(_email_or_none)] = None
    """The account's email address. The server keeps only a digest of it,
    scoped to the workspace, to suggest which accounts are one person. A value
    that isn't an address is ignored, as the server ignores it."""
    bot: bool = False
    """The source marks this account as a bot. Its peer is created with
    ``observe_me`` off, so nothing is derived about it."""


class IngestContainer(BaseModel):
    """Where a record lives in its source: a channel, a thread, a repository."""

    model_config = ConfigDict(extra="forbid")  # pyright: ignore[reportUnannotatedClassAttribute]

    id: _ExternalId
    name: _DisplayName | None = None
    parent_id: _ExternalId | None = None
    """The parent container's ID, e.g. a thread's channel."""


class IngestRecord(BaseModel):
    """One source record to ingest: a chat message, a meeting note, an event.

    Sending it again with a later ``revised_at`` replaces it during the
    workspace's dry run.

    Experimental: ingest may change or stop working without notice.
    """

    model_config = ConfigDict(extra="forbid")  # pyright: ignore[reportUnannotatedClassAttribute]

    op: Literal["upsert"] = "upsert"
    """Ingest the record, or a newer version of it."""
    source: str = Field(min_length=1, max_length=64, pattern=_NAME_PATTERN)
    """The source type, e.g. ``discord``."""
    source_instance: str = Field(min_length=1, max_length=128, pattern=_NAME_PATTERN)
    """The server, team or account inside the source."""
    created_at: datetime.datetime
    """The source's timestamp."""
    content: Annotated[str, BeforeValidator(_strip_nul)]
    source_id: _ExternalId | None = None
    """The record's ID in that instance. Without it, the record is identified by
    a hash of its content, so only identical content is deduplicated."""
    revised_at: datetime.datetime | None = None
    """When the source last changed this record."""
    container: IngestContainer | None = None
    """Where the record lives; omit for a standalone record."""
    author: IngestParticipant | None = None
    """Who wrote it; omit when there is no single author."""
    participants: list[IngestParticipant] = Field(default_factory=list, max_length=100)
    metadata: Annotated[dict[str, Any], BeforeValidator(_metadata)] | None = None
    configuration: MessageConfiguration | None = None
    session_id: (
        Annotated[
            str,
            AfterValidator(_storable),
            Field(min_length=1, max_length=512, pattern=_NAME_PATTERN),
        ]
        | None
    ) = None
    """An explicit session, used instead of mapping the container."""
    peer_id: _ExternalId | None = None
    """An explicit author peer, used instead of mapping the author."""


class IngestRecordDelete(BaseModel):
    """A record the source deleted.

    During the workspace's dry run its message is removed and a tombstone kept,
    so the record never comes back; after the dry run it's refused. A delete is
    final: a later version of the record is refused.

    Experimental: ingest may change or stop working without notice.
    """

    model_config = ConfigDict(extra="forbid")  # pyright: ignore[reportUnannotatedClassAttribute]

    op: Literal["delete"] = "delete"
    source: str = Field(min_length=1, max_length=64, pattern=_NAME_PATTERN)
    source_instance: str = Field(min_length=1, max_length=128, pattern=_NAME_PATTERN)
    source_id: _ExternalId
    """The deleted record's ID in that instance."""
    revised_at: datetime.datetime | None = None
    """When the source deleted it. A delete older than the stored version
    changes nothing."""


def parse_ingest_record(
    record: IngestRecord | IngestRecordDelete | Mapping[str, Any],
) -> IngestRecord | IngestRecordDelete:
    """A record as one of the two forms, by its ``op`` (an upsert may leave it out)."""
    if isinstance(record, IngestRecord | IngestRecordDelete):
        return record
    # A JSONL line can hold any JSON value, whatever the annotation says.
    if not isinstance(cast(object, record), Mapping):
        raise ValueError("A record is a JSON object")
    if record.get("op", "upsert") == "delete":
        return IngestRecordDelete.model_validate(record)
    return IngestRecord.model_validate(record)


class IngestRecordResult(BaseModel):
    """What happened to one ingested record."""

    index: int
    """The record's position among the records passed to ``ingest``."""
    outcome: str
    """``created``, ``updated``, ``deleted``, ``duplicate`` or ``rejected``."""
    reason: str | None = None
    """Why a record was rejected, e.g. ``content_conflict``."""
    record_id: str | None = None
    message_id: str | None = None
    session_id: str | None = None
    peer_id: str | None = None
    derivation: str | None = None
    """``pending``, ``queued``, ``deferred``, or ``deleted`` for a tombstone."""


class IngestResponse(BaseModel):
    """One result per ingested record, in the order the records were passed."""

    records: list[IngestRecordResult]


class IngestRecordCounts(BaseModel):
    """The workspace's ingested records by derivation state."""

    deferred: int
    """Stored, waiting for ``ingest_derive()`` to release them."""
    pending: int
    """Released, about to be queued."""
    queued: int
    """Queued for derivation, or derived."""
    deleted: int = 0
    """Tombstones: records the source deleted, kept so they don't come back."""


class IngestSuggestionStageCounts(BaseModel):
    """Open suggestions one stage made, by confidence."""

    high: int = 0
    low: int = 0


class IngestSuggestionCounts(BaseModel):
    """Open suggested links by stage, and whether a refresh is under way."""

    rules: IngestSuggestionStageCounts = Field(
        default_factory=IngestSuggestionStageCounts
    )
    """Made by the identity rules."""
    refreshing: bool = False


class IngestSyncSource(BaseModel):
    """How a connector following one source is doing."""

    source: str
    source_instance: str
    last_synced_at: datetime.datetime
    """When the connector last saved its position."""
    last_record_at: datetime.datetime | None = None
    """The source time of the newest record the connector has sent."""
    error: str | None = None
    """The connector's last error; None once it's fine."""


class IngestSyncState(IngestSyncSource):
    """Where a connector following one source has got to.

    Experimental: ingest may change or stop working without notice.
    """

    position: dict[str, Any]
    """The connector's own cursor, as it saved it."""
    revision: str
    """Changes on every save: pass it back to save only if nothing else has."""


class IngestStatusResponse(BaseModel):
    """What the workspace's imports produced, for review before deriving."""

    records: IngestRecordCounts
    deferred_tokens: int
    """Tokens in the messages of deferred records."""
    people: int
    """Peers created for imported accounts."""
    people_not_modeled: int
    """Of those, peers with ``observe_me`` off: nothing is derived about them."""
    sessions: int
    """Sessions created for imported containers."""
    sessions_not_modeled: int
    """Of those, sessions whose reasoning is off."""
    sessions_being_deleted: int = 0
    """Sessions a reset is still deleting; 0 once it has finished."""
    suggestions: IngestSuggestionCounts = Field(default_factory=IngestSuggestionCounts)
    """Open suggested links between accounts."""
    sources: list[IngestSyncSource] = Field(default_factory=list)
    """The sources connectors follow, with each one's last sync."""


class IngestAccount(BaseModel):
    """One source account mapped to a peer."""

    source: str
    source_instance: str
    id: str | None = None
    """The account's ID in the source, when it has one."""
    display_name: str | None = None


class IngestPerson(BaseModel):
    """A peer created for imported accounts."""

    peer_id: str
    display_name: str | None = None
    """The first account's display name."""
    accounts: list[IngestAccount]
    messages: int
    """Records its accounts wrote."""
    modeled: bool
    """False when ``observe_me`` is off: nothing is derived about it."""


class IngestSession(BaseModel):
    """A session created for an imported container."""

    session_id: str
    display_name: str | None = None
    """The container's display name in the source."""
    source: str
    source_instance: str
    messages: int
    """Records in the session."""
    modeled: bool
    """False when reasoning is off for the session: only summaries are made."""


class IngestLinkResponse(BaseModel):
    """The kept person and every account that now maps to it."""

    peer_id: str
    merged: list[str]
    """The people linked into it."""
    accounts: list[IngestAccount]


class IngestSuggestionAccount(BaseModel):
    """One account of a suggestion, as it was when the suggestion was made."""

    source: str
    source_instance: str
    id: str | None = None
    display_name: str | None = None
    handle: str | None = None
    peer_id: str | None = None
    """The person it maps to now; None if it's gone."""
    records: int
    """Records it had written when the suggestion was made."""


class IngestSuggestionEvidence(BaseModel):
    """One reason for a suggestion."""

    signal: str
    level: str
    detail: str


class IngestSuggestion(BaseModel):
    """Two accounts that look like one person, and why. Nothing is linked until accepted.

    Experimental: ingest may change or stop working without notice.
    """

    id: str
    kind: str
    stage: Literal["rules"]
    """What made it: ``rules``, the identity rules. With ``confidence``, it names a
    group of suggestions, such as ``rules:high``."""
    confidence: Literal["high", "low"]
    """``high``: evidence to accept at a glance. ``low``: needs a closer look."""
    score: float
    status: Literal["open", "accepted", "dismissed"]
    linked: bool
    """The two accounts map to one person now, however it happened."""
    accounts: list[IngestSuggestionAccount]
    evidence: list[IngestSuggestionEvidence]
    refreshed_at: datetime.datetime


class IngestSuggestionAccepted(BaseModel):
    """The accepted suggestion, and the link it made (None when already one person)."""

    suggestion: IngestSuggestion
    link: IngestLinkResponse | None = None


class IngestUnlinkResponse(BaseModel):
    """The person a linked account maps to again, from the next import on."""

    peer_id: str
    account: IngestAccount


class IngestResetResponse(BaseModel):
    """The imported sessions whose deletion was started."""

    sessions: list[str]


class IngestDeriveResponse(BaseModel):
    """How many deferred records were released for derivation."""

    pending: int
    """Records released for derivation."""
    remaining: int
    """Records in the workspace still deferred."""


# ==============================================================================
# Representation History Types
# ==============================================================================


class DreamRunResponse(BaseModel):
    """One dream over a representation: the observer's model of the observed peer."""

    id: str
    """The dream's run ID."""
    observer_id: str
    observed_id: str
    dream_type: str | None = None
    trigger_reason: str | None = None
    status: Literal["running", "completed", "failed", "interrupted"]
    """``failed`` and ``interrupted`` runs can still have changes."""
    created_count: int
    removed_count: int
    peer_card_updated: bool
    started_at: datetime.datetime
    finished_at: datetime.datetime | None = None
    pruned_at: datetime.datetime | None = None
    """When retention first deleted some of this run's changes. After that the run
    keeps only the created conclusions that still exist; the counts are unchanged."""


class DreamConclusionRef(BaseModel):
    """A conclusion a dream referenced, resolved when the run is read."""

    id: str
    content: str | None = None
    """None when ``status`` is ``unavailable``."""
    level: str | None = None
    status: Literal["current", "removed", "unavailable"]
    removed_in_run: str | None = None
    """The dream that removed it, when ``status`` is ``removed``."""


class DreamCreatedConclusion(BaseModel):
    """A conclusion a dream created, with the conclusions it was derived from."""

    conclusion: DreamConclusionRef
    sources: list[DreamConclusionRef] = Field(default_factory=list)
    specialist: str | None = None


class DreamRemovedConclusion(BaseModel):
    """A conclusion a dream removed, as it read when it was removed."""

    id: str
    content: str
    level: str | None = None
    replaced_by: str | None = None
    """Set when a conclusion from the same dream superseded this one."""
    specialist: str | None = None


class DreamPeerCardChange(BaseModel):
    """The peer card before the dream's first update and after its last."""

    before: list[str] | None = None
    after: list[str]


class DreamRunDetailResponse(DreamRunResponse):
    """A dream run with every change it committed."""

    created: list[DreamCreatedConclusion] = Field(default_factory=list)
    removed: list[DreamRemovedConclusion] = Field(default_factory=list)
    peer_card: DreamPeerCardChange | None = None


# ==============================================================================
# Dialectic (Chat) Types
# ==============================================================================


ReasoningLevel = Literal["minimal", "low", "medium", "high", "max"]


class DialecticParams(BaseModel):
    """Parameters for dialectic chat."""

    model_config = ConfigDict(extra="forbid")  # pyright: ignore[reportUnannotatedClassAttribute]

    session_id: str | None = None
    target: str | None = None
    query: str = Field(min_length=1, max_length=10000)
    stream: bool = False
    reasoning_level: ReasoningLevel = "low"
    response_format: dict[str, Any] | None = None
    include_evidence: bool = False


class EvidenceObservation(BaseModel):
    """A conclusion the dialectic read while answering."""

    id: str
    level: ConclusionLevel
    content: str
    created_at: datetime.datetime
    session_id: str | None = None
    observer_id: str | None = None
    """The peer who made the conclusion. None from servers before 3.2.1."""
    observed_id: str | None = None
    """The peer the conclusion is about. None from servers before 3.2.1."""
    source_ids: list[str] = Field(default_factory=list)


class EvidenceMessageRef(BaseModel):
    """A message the dialectic read while answering.

    Identity and provenance only -- no content. Fetch the message by `id` when
    you need its text; evidence is for auditing what was read, not for reading
    messages in bulk.
    """

    id: str
    session_id: str
    peer_id: str
    created_at: datetime.datetime


class EvidenceToolCall(BaseModel):
    """A tool the dialectic invoked while answering."""

    tool_name: str
    tool_input: dict[str, Any] = Field(default_factory=dict)


class Evidence(BaseModel):
    """What the dialectic read and did while answering.

    Collated from what the agent accessed rather than reported by the model, so
    it over-reports: a listed conclusion was read, which is not proof the
    answer leaned on it. `tool_calls` omits results and failed calls.
    """

    conclusions: list[EvidenceObservation] = Field(default_factory=list)
    messages: list[EvidenceMessageRef] = Field(default_factory=list)
    tool_calls: list[EvidenceToolCall] = Field(default_factory=list)
    reasoning_trace_id: str | None = None


class DialecticResponse(BaseModel):
    """Dialectic chat API response."""

    content: str | None
    evidence: Evidence | None = None


class DialecticStreamDelta(BaseModel):
    """Delta for streaming dialectic responses."""

    content: str | None = None


class DialecticStreamChunk(BaseModel):
    """Chunk in a streaming dialectic response."""

    delta: DialecticStreamDelta
    done: bool = False
    evidence: Evidence | None = None


# ==============================================================================
# Pagination Types
# ==============================================================================


class PageResponse(BaseModel):
    """Generic paginated response."""

    items: list[Any]
    page: int
    size: int
    total: int
    pages: int


# ==============================================================================
# File Upload Types
# ==============================================================================


class MessageUploadParams(BaseModel):
    """Parameters for file upload message creation."""

    model_config = ConfigDict(extra="forbid")  # pyright: ignore[reportUnannotatedClassAttribute]

    peer_id: str
    metadata: dict[str, Any] | None = None
    configuration: MessageConfiguration | None = None
    created_at: datetime.datetime | None = None


# Update forward reference
SessionContextResponse.model_rebuild()
