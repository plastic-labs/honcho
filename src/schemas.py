import datetime
import ipaddress
from enum import Enum
from typing import Annotated, Any, Self
from urllib.parse import urlparse

import tiktoken
from pydantic import (
    AliasChoices,
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)

from src.config import ReasoningLevel, settings
from src.utils.types import DocumentLevel

RESOURCE_NAME_PATTERN = r"^[a-zA-Z0-9_-]+$"


class DreamType(str, Enum):
    """Types of dreams that can be triggered."""

    OMNI = "omni"
    REASONING = "reasoning"  # Top-down reasoning: hypotheses, predictions, falsification, induction


class ReasoningConfiguration(BaseModel):
    enabled: bool | None = Field(
        default=None,
        description="Whether to enable reasoning functionality.",
    )
    custom_instructions: str | None = Field(
        default=None,
        description="TODO: currently unused. Custom instructions to use for the reasoning system on this workspace/session/message.",
    )


class PeerCardConfiguration(BaseModel):
    use: bool | None = Field(
        default=None,
        description="Whether to use peer card related to this peer during reasoning process.",
    )
    create: bool | None = Field(
        default=None,
        description="Whether to generate peer card based on content.",
    )


class SummaryConfiguration(BaseModel):
    enabled: bool | None = Field(
        default=None,
        description="Whether to enable summary functionality.",
    )
    messages_per_short_summary: int | None = Field(
        default=None,
        ge=10,
        description="Number of messages per short summary. Must be positive, greater than or equal to 10, and less than messages_per_long_summary.",
    )
    messages_per_long_summary: int | None = Field(
        default=None,
        ge=20,
        description="Number of messages per long summary. Must be positive, greater than or equal to 20, and greater than messages_per_short_summary.",
    )

    @model_validator(mode="after")
    def validate_summary_thresholds(self) -> Self:
        """Validate that short summary threshold <= long summary threshold."""
        short = self.messages_per_short_summary
        long = self.messages_per_long_summary

        if short is not None and long is not None and short >= long:
            raise ValueError(
                "messages_per_short_summary must be less than messages_per_long_summary"
            )

        return self


class DreamConfiguration(BaseModel):
    enabled: bool | None = Field(
        default=None,
        description="Whether to enable dream functionality. If reasoning is disabled, dreams will also be disabled and this setting will be ignored.",
    )


class AbducerConfig(BaseModel):
    """Configuration for the Abducer agent (hypothesis generation)."""

    max_hypotheses_per_cycle: int | None = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of hypotheses to generate per cycle",
    )
    min_confidence_threshold: float | None = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Minimum confidence for hypothesis acceptance",
    )
    max_source_premises: int | None = Field(
        default=10,
        ge=1,
        description="Maximum number of source premises to consider",
    )


class PredictorConfig(BaseModel):
    """Configuration for the Predictor agent (prediction generation)."""

    predictions_per_hypothesis: int | None = Field(
        default=3,
        ge=1,
        le=10,
        description="Number of predictions to generate per hypothesis",
    )
    blind_prediction_ratio: float | None = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Ratio of blind predictions (made without searching for contradictions)",
    )


class FalsifierConfig(BaseModel):
    """Configuration for the Falsifier agent (prediction testing)."""

    max_searches_per_prediction: int | None = Field(
        default=7,
        ge=1,
        le=20,
        description="Maximum number of search attempts per prediction",
    )
    search_cache_ttl: int | None = Field(
        default=86400,
        ge=0,
        description="Search cache TTL in seconds (24 hours default)",
    )
    contradiction_threshold: float | None = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="Similarity threshold for contradiction detection",
    )


class InductorConfig(BaseModel):
    """Configuration for the Inductor agent (pattern extraction)."""

    min_predictions_for_induction: int | None = Field(
        default=3,
        ge=2,
        description="Minimum number of unfalsified predictions required for induction",
    )
    pattern_confidence_threshold: float | None = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum confidence threshold for pattern acceptance",
    )
    max_inductions_per_cycle: int | None = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of inductions per cycle",
    )


class TopDownConfiguration(BaseModel):
    """Configuration for Top-Down reasoning system."""

    enabled: bool | None = Field(
        default=False,
        description="Whether to enable Top-Down reasoning functionality",
    )
    unaccounted_threshold: int | None = Field(
        default=50,
        ge=1,
        description="Number of unaccounted premises (|U_t|) that triggers hypothesis generation",
    )
    surprise_threshold: float | None = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Surprise score S(p_new) threshold that triggers hypothesis generation",
    )
    max_active_hypotheses: int | None = Field(
        default=100,
        ge=1,
        description="Maximum number of active hypotheses to maintain",
    )
    max_unaccounted_ratio: float | None = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Maximum ratio of unaccounted premises before hypothesis cycle",
    )
    abducer_config: AbducerConfig | None = Field(
        default_factory=AbducerConfig,
        description="Configuration for the Abducer agent",
    )
    predictor_config: PredictorConfig | None = Field(
        default_factory=PredictorConfig,
        description="Configuration for the Predictor agent",
    )
    falsifier_config: FalsifierConfig | None = Field(
        default_factory=FalsifierConfig,
        description="Configuration for the Falsifier agent",
    )
    inductor_config: InductorConfig | None = Field(
        default_factory=InductorConfig,
        description="Configuration for the Inductor agent",
    )


class WorkspaceConfiguration(BaseModel):
    """
    The set of options that can be in a workspace DB-level configuration dictionary.

    All fields are optional. Session-level configuration overrides workspace-level configuration, which overrides global configuration.
    """

    model_config = ConfigDict(extra="allow")  # pyright: ignore

    reasoning: ReasoningConfiguration | None = Field(
        default=None,
        description="Configuration for reasoning functionality.",
    )
    peer_card: PeerCardConfiguration | None = Field(
        default=None,
        description="Configuration for peer card functionality. If reasoning is disabled, peer cards will also be disabled and these settings will be ignored.",
    )
    summary: SummaryConfiguration | None = Field(
        default=None,
        description="Configuration for summary functionality.",
    )
    dream: DreamConfiguration | None = Field(
        default=None,
        description="Configuration for dream functionality. If reasoning is disabled, dreams will also be disabled and these settings will be ignored.",
    )
    topdown: TopDownConfiguration | None = Field(
        default=None,
        description="Configuration for Top-Down reasoning functionality.",
    )


class SessionConfiguration(WorkspaceConfiguration):
    """
    The set of options that can be in a session DB-level configuration dictionary.

    All fields are optional. Session-level configuration overrides workspace-level configuration, which overrides global configuration.
    """

    pass


class MessageConfiguration(BaseModel):
    """
    The set of options that can be in a message DB-level configuration dictionary.

    All fields are optional. Message-level configuration overrides all other configurations.
    """

    reasoning: ReasoningConfiguration | None = Field(
        default=None,
        description="Configuration for reasoning functionality.",
    )


class ResolvedReasoningConfiguration(BaseModel):
    enabled: bool


class ResolvedPeerCardConfiguration(BaseModel):
    use: bool
    create: bool


class ResolvedSummaryConfiguration(BaseModel):
    enabled: bool
    messages_per_short_summary: int
    messages_per_long_summary: int


class ResolvedDreamConfiguration(BaseModel):
    enabled: bool


class ResolvedConfiguration(BaseModel):
    """
    The final resolved configuration for a given message.
    Hierarchy: message > session > workspace > global configuration
    """

    reasoning: ResolvedReasoningConfiguration
    peer_card: ResolvedPeerCardConfiguration
    summary: ResolvedSummaryConfiguration
    dream: ResolvedDreamConfiguration


class PeerConfig(BaseModel):
    # TODO: Update description - should say "Whether honcho forms a representation of the peer itself"
    observe_me: bool | None = Field(
        default=None,
        description="Whether Honcho will use reasoning to form a representation of this peer",
    )


class SessionPeerConfig(PeerConfig):
    # TODO: Update description - should say "Whether this peer forms representations of other peers in the session"
    observe_others: bool | None = Field(
        default=None,
        description="Whether this peer should form a session-level theory-of-mind representation of other peers in the session",
    )


class WorkspaceBase(BaseModel):
    pass


class WorkspaceCreate(WorkspaceBase):
    name: Annotated[
        str,
        Field(alias="id", min_length=1, max_length=100, pattern=RESOURCE_NAME_PATTERN),
    ]
    metadata: dict[str, Any] = {}
    configuration: WorkspaceConfiguration = Field(
        default_factory=WorkspaceConfiguration
    )

    model_config = ConfigDict(populate_by_name=True)  # pyright: ignore


class WorkspaceGet(WorkspaceBase):
    filters: dict[str, Any] | None = None


class WorkspaceUpdate(WorkspaceBase):
    metadata: dict[str, Any] | None = None
    configuration: WorkspaceConfiguration | None = None


class Workspace(WorkspaceBase):
    name: str = Field(serialization_alias="id")
    h_metadata: dict[str, Any] = Field(
        default_factory=dict, serialization_alias="metadata"
    )
    configuration: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime.datetime

    model_config = ConfigDict(  # pyright: ignore
        from_attributes=True, populate_by_name=True
    )


class PeerBase(BaseModel):
    pass


class PeerCreate(PeerBase):
    name: Annotated[
        str,
        Field(alias="id", min_length=1, max_length=100, pattern=RESOURCE_NAME_PATTERN),
    ]
    metadata: dict[str, Any] | None = None
    configuration: dict[str, Any] | None = None

    model_config = ConfigDict(populate_by_name=True)  # pyright: ignore


class PeerGet(PeerBase):
    filters: dict[str, Any] | None = None


class PeerUpdate(PeerBase):
    metadata: dict[str, Any] | None = None
    configuration: dict[str, Any] | None = None


class Peer(PeerBase):
    name: str = Field(serialization_alias="id")
    workspace_name: str = Field(serialization_alias="workspace_id")
    created_at: datetime.datetime
    h_metadata: dict[str, Any] = Field(
        default_factory=dict, serialization_alias="metadata"
    )
    configuration: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(  # pyright: ignore
        from_attributes=True, populate_by_name=True
    )


class PeerRepresentationGet(BaseModel):
    session_id: str | None = Field(
        None, description="Optional session ID within which to scope the representation"
    )
    target: str | None = Field(
        None,
        description="Optional peer ID to get the representation for, from the perspective of this peer",
    )
    search_query: str | None = Field(
        None,
        description="Optional input to curate the representation around semantic search results",
    )
    search_top_k: int | None = Field(
        None,
        ge=1,
        le=100,
        description="Only used if `search_query` is provided. Number of semantic-search-retrieved conclusions to include in the representation",
    )
    search_max_distance: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Only used if `search_query` is provided. Maximum distance to search for semantically relevant conclusions",
    )
    include_most_frequent: bool | None = Field(
        default=None,
        description="Only used if `search_query` is provided. Whether to include the most frequent conclusions in the representation",
    )
    max_conclusions: int | None = Field(
        default=25,
        ge=1,
        le=100,
        description="Only used if `search_query` is provided. Maximum number of conclusions to include in the representation",
    )


class RepresentationResponse(BaseModel):
    representation: str


class PeerCardResponse(BaseModel):
    peer_card: list[str] | None = Field(
        None, description="The peer card content, or None if not found"
    )


class PeerCardSet(BaseModel):
    peer_card: list[str] = Field(..., description="The peer card content to set")


class MessageBase(BaseModel):
    pass


class MessageCreate(MessageBase):
    content: Annotated[str, Field(min_length=0, max_length=settings.MAX_MESSAGE_SIZE)]
    peer_name: str = Field(alias="peer_id")
    metadata: dict[str, Any] | None = None
    configuration: MessageConfiguration | None = None
    created_at: datetime.datetime | None = None

    _encoded_message: list[int] = PrivateAttr(default=[])

    @property
    def encoded_message(self) -> list[int]:
        return self._encoded_message

    @model_validator(mode="after")
    def validate_and_set_token_count(self) -> Self:
        encoding = tiktoken.get_encoding("o200k_base")
        encoded_message = encoding.encode(self.content)

        self._encoded_message = encoded_message
        return self


class MessageGet(MessageBase):
    filters: dict[str, Any] | None = None


class MessageUpdate(MessageBase):
    metadata: dict[str, Any] | None = None


class Message(MessageBase):
    public_id: str = Field(serialization_alias="id")
    content: str
    peer_name: str = Field(serialization_alias="peer_id")
    session_name: str = Field(serialization_alias="session_id")
    h_metadata: dict[str, Any] = Field(
        default_factory=dict, serialization_alias="metadata"
    )
    created_at: datetime.datetime
    workspace_name: str = Field(serialization_alias="workspace_id")
    token_count: int

    model_config = ConfigDict(  # pyright: ignore
        from_attributes=True, populate_by_name=True
    )


class MessageBatchCreate(BaseModel):
    """Schema for batch message creation with a max of 100 messages"""

    messages: list[MessageCreate] = Field(..., min_length=1, max_length=100)


class MessageUploadCreate(BaseModel):
    """Schema for message creation from file uploads"""

    peer_id: str = Field(..., description="ID of the peer creating the message")
    metadata: dict[str, Any] | None = None
    configuration: MessageConfiguration | None = None
    created_at: datetime.datetime | None = None

    model_config = ConfigDict(populate_by_name=True)  # pyright: ignore


class SessionBase(BaseModel):
    pass


class SessionCreate(SessionBase):
    name: Annotated[
        str,
        Field(alias="id", min_length=1, max_length=100, pattern=RESOURCE_NAME_PATTERN),
    ]
    metadata: dict[str, Any] | None = None
    peer_names: dict[str, SessionPeerConfig] | None = Field(default=None, alias="peers")
    configuration: SessionConfiguration | None = None

    model_config = ConfigDict(populate_by_name=True)  # pyright: ignore


class SessionGet(SessionBase):
    filters: dict[str, Any] | None = None


class SessionUpdate(SessionBase):
    metadata: dict[str, Any] | None = None
    configuration: SessionConfiguration | None = None


class Session(SessionBase):
    name: str = Field(serialization_alias="id")
    is_active: bool
    workspace_name: str = Field(serialization_alias="workspace_id")
    h_metadata: dict[str, Any] = Field(
        default_factory=dict, serialization_alias="metadata"
    )
    configuration: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime.datetime

    model_config = ConfigDict(  # pyright: ignore
        from_attributes=True, populate_by_name=True
    )


class Summary(BaseModel):
    content: str = Field(description="The summary text")
    message_id: int = Field(
        description="The internal ID of the message that this summary covers up to",
        exclude=True,
    )
    message_public_id: str = Field(
        description="The public ID of the message that this summary covers up to",
        serialization_alias="message_id",
    )
    summary_type: str = Field(description="The type of summary (short or long)")
    created_at: str = Field(
        description="The timestamp of when the summary was created (ISO format)"
    )
    token_count: int = Field(description="The number of tokens in the summary text")


class SessionContext(SessionBase):
    name: str = Field(serialization_alias="id")
    messages: list[Message]
    summary: Summary | None = Field(
        default=None, description="The summary if available"
    )
    peer_representation: str | None = Field(
        default=None,
        description="A curated subset of a peer representation, if context is requested from a specific perspective",
    )
    peer_card: list[str] | None = Field(
        default=None,
        description="The peer card, if context is requested from a specific perspective",
    )

    model_config = ConfigDict(  # pyright: ignore
        from_attributes=True, populate_by_name=True
    )


class PeerContext(BaseModel):
    """Context for a peer, including representation and peer card."""

    peer_id: str = Field(description="The ID of the peer")
    target_id: str = Field(description="The ID of the target peer being observed")
    representation: str | None = Field(
        default=None,
        description="A curated subset of the representation of the target peer from the observer's perspective",
    )
    peer_card: list[str] | None = Field(
        default=None,
        description="The peer card for the target peer from the observer's perspective",
    )


class SessionSummaries(SessionBase):
    name: str = Field(serialization_alias="id")
    short_summary: Summary | None = Field(
        default=None, description="The short summary if available"
    )
    long_summary: Summary | None = Field(
        default=None, description="The long summary if available"
    )

    model_config = ConfigDict(  # pyright: ignore
        from_attributes=True, populate_by_name=True
    )


class DocumentBase(BaseModel):
    pass


class DocumentMetadata(BaseModel):
    message_ids: list[int] = Field(
        description="The ID range(s) of the messages that this document was derived from. Acts as a link to the primary source of the document. Note that as a document gets deduplicated, additional ranges will be added, because the same document could be derived from completely separate message ranges."
    )
    message_created_at: str = Field(
        description="The timestamp of the message that this document was derived from. Note that this is not the same as the created_at timestamp of the document. This timestamp is usually only saved with second-level precision."
    )
    source_ids: list[str] | None = Field(
        default=None,
        description="Document IDs of source documents for tree traversal -- required for deductive and inductive documents",
    )
    premises: list[str] | None = Field(
        default=None,
        description="Human-readable premise text for display -- only applicable for deductive documents",
    )
    sources: list[str] | None = Field(
        default=None,
        description="Human-readable source text for display -- only applicable for inductive documents",
    )
    pattern_type: str | None = Field(
        default=None,
        description="Type of pattern identified (preference, behavior, personality, tendency, correlation) -- only applicable for inductive documents",
    )
    confidence: str | None = Field(
        default=None,
        description="Confidence level (high, medium, low) -- only applicable for inductive documents",
    )


class DocumentCreate(DocumentBase):
    content: Annotated[str, Field(min_length=1, max_length=100000)]
    session_name: str = Field(
        description="The session from which the document was derived"
    )
    level: DocumentLevel = Field(
        default="explicit",
        description="The level of the document (explicit, deductive, inductive, or contradiction)",
    )
    times_derived: int = Field(
        default=1,
        ge=1,
        description="The number of times that a semantic duplicate document to this one has been derived",
    )
    metadata: DocumentMetadata = Field()
    embedding: list[float] = Field()
    # Tree linkage field
    source_ids: list[str] | None = Field(
        default=None,
        description="Document IDs of source/premise documents -- for deductive and inductive documents",
    )


class ConclusionGet(BaseModel):
    """Schema for listing conclusions with optional filters."""

    filters: dict[str, Any] | None = None


class Conclusion(BaseModel):
    """Conclusion response - external view of a document."""

    id: str
    content: str
    observer: str = Field(
        description="The peer who made the conclusion",
        serialization_alias="observer_id",
    )
    observed: str = Field(
        description="The peer the conclusion is about",
        serialization_alias="observed_id",
    )
    session_name: str = Field(serialization_alias="session_id")
    created_at: datetime.datetime

    model_config = ConfigDict(  # pyright: ignore
        from_attributes=True,
        populate_by_name=True,
    )


class ConclusionQuery(BaseModel):
    """Query parameters for semantic search of conclusions."""

    query: str = Field(..., description="Semantic search query")
    top_k: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of results to return",
    )
    distance: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Maximum cosine distance threshold for results",
    )
    filters: dict[str, Any] | None = Field(
        default=None,
        description="Additional filters to apply",
    )


class ConclusionCreate(BaseModel):
    """Schema for creating a single conclusion."""

    content: Annotated[str, Field(min_length=1, max_length=65535)]
    observer_id: str = Field(..., description="The peer making the conclusion")
    observed_id: str = Field(..., description="The peer the conclusion is about")
    session_id: str = Field(..., description="The session this conclusion relates to")

    _token_count: int = PrivateAttr(default=0)

    @model_validator(mode="after")
    def validate_token_count(self) -> Self:
        """Validate that content doesn't exceed embedding token limit."""
        encoding = tiktoken.get_encoding("o200k_base")
        tokens = encoding.encode(self.content)
        self._token_count = len(tokens)

        if self._token_count > settings.MAX_EMBEDDING_TOKENS:
            raise ValueError(
                f"Content exceeds maximum embedding token limit of {settings.MAX_EMBEDDING_TOKENS} "
                + f"(got {self._token_count} tokens)"
            )
        return self


class ConclusionBatchCreate(BaseModel):
    """Schema for batch conclusion creation with a max of 100 conclusions."""

    conclusions: list[ConclusionCreate] = Field(
        ...,
        min_length=1,
        max_length=100,
        validation_alias=AliasChoices("conclusions", "observations"),
    )


class MessageSearchOptions(BaseModel):
    query: str = Field(..., description="Search query")
    filters: dict[str, Any] | None = Field(
        default=None, description="Filters to scope the search"
    )
    limit: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Number of results to return",
    )


class DialecticOptions(BaseModel):
    session_id: str | None = Field(
        None, description="ID of the session to scope the representation to"
    )
    target: str | None = Field(
        None,
        description="Optional peer to get the representation for, from the perspective of this peer",
    )
    query: Annotated[
        str, Field(min_length=1, max_length=10000, description="Dialectic API Prompt")
    ]
    stream: bool = False
    reasoning_level: ReasoningLevel = Field(
        default="low",
        description="Level of reasoning to apply: minimal, low, medium, high, or max",
    )


class DialecticResponse(BaseModel):
    content: str


class DialecticStreamDelta(BaseModel):
    """Delta object for streaming dialectic responses."""

    content: str | None = None
    # Future fields can be added here:
    # premises: str | None = None
    # tokens: int | None = None
    # analytics: dict[str, Any] | None = None


class DialecticStreamChunk(BaseModel):
    """Chunk in a streaming dialectic response."""

    delta: DialecticStreamDelta
    done: bool = False


class SessionCounts(BaseModel):
    """Counts for a specific session in queue processing."""

    completed: int
    in_progress: int
    pending: int


class QueueCounts(BaseModel):
    """Aggregated counts for queue processing status."""

    total: int
    completed: int
    in_progress: int
    pending: int
    sessions: dict[str, SessionCounts]


class QueueStatusRow(BaseModel):
    """Represents a row from the queue status SQL query result."""

    session_id: str | None
    total: int
    completed: int
    in_progress: int
    pending: int
    session_total: int
    session_completed: int
    session_in_progress: int
    session_pending: int


class SessionPeerData(BaseModel):
    """Data for managing session peer relationships."""

    peer_names: dict[str, SessionPeerConfig]


class MessageBulkData(BaseModel):
    """Data for bulk message operations."""

    messages: list[MessageCreate]
    session_name: str
    workspace_name: str


class SessionQueueStatus(BaseModel):
    """Status for a specific session within the processing queue."""

    session_id: str | None = Field(
        default=None,
        description="Session ID if filtered by session",
    )
    total_work_units: int = Field(description="Total work units")
    completed_work_units: int = Field(description="Completed work units")
    in_progress_work_units: int = Field(
        description="Work units currently being processed"
    )
    pending_work_units: int = Field(description="Work units waiting to be processed")


class QueueStatus(BaseModel):
    """Aggregated processing queue status."""

    total_work_units: int = Field(description="Total work units")
    completed_work_units: int = Field(description="Completed work units")
    in_progress_work_units: int = Field(
        description="Work units currently being processed"
    )
    pending_work_units: int = Field(description="Work units waiting to be processed")
    sessions: dict[str, SessionQueueStatus] | None = Field(
        default=None,
        description="Per-session status when not filtered by session",
    )


class ScheduleDreamRequest(BaseModel):
    observer: str = Field(..., description="Observer peer name")
    observed: str | None = Field(
        None, description="Observed peer name (defaults to observer if not specified)"
    )
    dream_type: DreamType = Field(..., description="Type of dream to schedule")
    session_id: str = Field(..., description="Session ID to scope the dream to")


# Webhook endpoint schemas
class WebhookEndpointBase(BaseModel):
    pass


class WebhookEndpointCreate(WebhookEndpointBase):
    url: str

    @field_validator("url")
    @classmethod
    def validate_webhook_url(cls, v: str) -> str:
        parsed = urlparse(v)

        if not all([parsed.scheme, parsed.netloc]):
            raise ValueError("Invalid URL format")

        # Only allow HTTP/HTTPS
        if parsed.scheme not in ["http", "https"]:
            raise ValueError("Only HTTP and HTTPS URLs are allowed")

        # Block private/internal addresses
        if parsed.hostname:
            try:
                ip_address = ipaddress.ip_address(parsed.hostname)
                if ip_address.is_private:
                    raise ValueError("Private IP addresses are not allowed")
            except ValueError:  # Not an IP address, might be a hostname
                pass

        return v


class WebhookEndpoint(WebhookEndpointBase):
    id: str
    workspace_name: str | None = Field(serialization_alias="workspace_id")
    url: str
    created_at: datetime.datetime

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)  # pyright: ignore


# Top-Down Reasoning schemas


class HypothesisBase(BaseModel):
    """Base schema for Hypothesis."""

    content: str = Field(..., description="The hypothesis content/statement")
    observer: str = Field(..., description="Peer name who observes")
    observed: str = Field(..., description="Peer name being observed")


class HypothesisCreate(HypothesisBase):
    """Schema for creating a new Hypothesis."""

    status: str | None = Field(
        default="active",
        description="Status of the hypothesis (active | superseded | falsified)",
    )
    confidence: float | None = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence level (0.0 to 1.0)",
    )
    source_premise_ids: list[str] | None = Field(
        default=None,
        description="IDs of source premises that led to this hypothesis",
    )
    unaccounted_premises_count: int | None = Field(
        default=0,
        ge=0,
        description="Number of unaccounted premises",
    )
    search_coverage: int | None = Field(
        default=0,
        ge=0,
        description="Search coverage metric",
    )
    tier: int | None = Field(
        default=0,
        ge=0,
        description="Tier level for hypothesis prioritization",
    )
    reasoning_metadata: dict[str, Any] | None = Field(
        default_factory=dict,
        description="Additional metadata about reasoning process",
    )
    collection_id: str | None = Field(
        default=None,
        description="Collection ID for organization",
    )


class HypothesisUpdate(BaseModel):
    """Schema for updating an existing Hypothesis."""

    content: str | None = Field(default=None, description="The hypothesis content/statement")
    status: str | None = Field(
        default=None,
        description="Status of the hypothesis (active | superseded | falsified)",
    )
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence level (0.0 to 1.0)",
    )
    source_premise_ids: list[str] | None = Field(
        default=None,
        description="IDs of source premises that led to this hypothesis",
    )
    unaccounted_premises_count: int | None = Field(
        default=None,
        ge=0,
        description="Number of unaccounted premises",
    )
    search_coverage: int | None = Field(
        default=None,
        ge=0,
        description="Search coverage metric",
    )
    tier: int | None = Field(
        default=None,
        ge=0,
        description="Tier level for hypothesis prioritization",
    )
    reasoning_metadata: dict[str, Any] | None = Field(
        default=None,
        description="Additional metadata about reasoning process",
    )
    superseded_by_id: str | None = Field(
        default=None,
        description="ID of hypothesis that superseded this one",
    )
    supersedes_ids: list[str] | None = Field(
        default=None,
        description="IDs of hypotheses this one supersedes",
    )
    collection_id: str | None = Field(
        default=None,
        description="Collection ID for organization",
    )


class Hypothesis(HypothesisBase):
    """Response schema for Hypothesis."""

    id: str = Field(..., description="Unique hypothesis ID")
    status: str = Field(..., description="Status of the hypothesis")
    confidence: float = Field(..., description="Confidence level (0.0 to 1.0)")
    source_premise_ids: list[str] | None = Field(
        default=None, description="IDs of source premises"
    )
    unaccounted_premises_count: int = Field(..., description="Number of unaccounted premises")
    search_coverage: int = Field(..., description="Search coverage metric")
    tier: int = Field(..., description="Tier level")
    reasoning_metadata: dict[str, Any] = Field(
        default_factory=dict, description="Reasoning metadata"
    )
    workspace_name: str = Field(..., description="Workspace name", serialization_alias="workspace_id")
    collection_id: str | None = Field(default=None, description="Collection ID")
    created_at: datetime.datetime = Field(..., description="Creation timestamp")
    updated_at: datetime.datetime = Field(..., description="Last update timestamp")

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)  # pyright: ignore


# Alias for API responses
HypothesisResponse = Hypothesis


class HypothesisGenealogy(BaseModel):
    """Schema for hypothesis genealogy/evolution tree."""

    hypothesis: Hypothesis = Field(..., description="The hypothesis in question")
    parents: list[Hypothesis] = Field(
        default_factory=list, description="Parent hypotheses that this superseded or was derived from"
    )
    children: list[Hypothesis] = Field(
        default_factory=list, description="Child hypotheses that superseded or were derived from this one"
    )
    reasoning_metadata: dict[str, Any] = Field(
        default_factory=dict, description="Reasoning metadata explaining the evolution"
    )


class PredictionBase(BaseModel):
    """Base schema for Prediction."""

    content: str = Field(..., description="The prediction content/statement")
    hypothesis_id: str = Field(..., description="ID of the hypothesis this prediction tests")


class PredictionCreate(PredictionBase):
    """Schema for creating a new Prediction."""

    status: str | None = Field(
        default="untested",
        description="Status of the prediction (unfalsified | falsified | untested)",
    )
    source_hypothesis_ids: list[str] | None = Field(
        default=None,
        description="IDs of source hypotheses that contributed to this prediction",
    )
    is_blind: bool | None = Field(
        default=True,
        description="Whether this is a blind prediction (made without searching for contradictions)",
    )
    collection_id: str | None = Field(
        default=None,
        description="Collection ID for organization",
    )


class PredictionUpdate(BaseModel):
    """Schema for updating an existing Prediction."""

    status: str | None = Field(
        default=None,
        description="Status of the prediction (unfalsified | falsified | untested)",
    )


class Prediction(PredictionBase):
    """Response schema for Prediction."""

    id: str = Field(..., description="Unique prediction ID")
    status: str = Field(..., description="Status of the prediction")
    source_hypothesis_ids: list[str] | None = Field(
        default=None, description="IDs of source hypotheses"
    )
    is_blind: bool = Field(..., description="Whether this is a blind prediction")
    workspace_name: str = Field(..., description="Workspace name", serialization_alias="workspace_id")
    collection_id: str | None = Field(default=None, description="Collection ID")
    created_at: datetime.datetime = Field(..., description="Creation timestamp")

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)  # pyright: ignore


# Alias for API responses
PredictionResponse = Prediction


class FalsificationTraceBase(BaseModel):
    """Base schema for FalsificationTrace."""

    prediction_id: str = Field(..., description="ID of the prediction being tested")


class FalsificationTraceCreate(FalsificationTraceBase):
    """Schema for creating a new FalsificationTrace.

    Note: FalsificationTrace is immutable once created (append-only).
    """

    search_queries: list[str] | None = Field(
        default=None,
        description="List of search queries executed during falsification",
    )
    contradicting_premise_ids: list[str] | None = Field(
        default=None,
        description="IDs of premises that contradict the prediction",
    )
    reasoning_chain: dict[str, Any] | None = Field(
        default_factory=dict,
        description="Detailed trace of the reasoning process",
    )
    final_status: str | None = Field(
        default="untested",
        description="Final status (unfalsified | falsified | untested)",
    )
    search_count: int | None = Field(
        default=0,
        ge=0,
        description="Number of searches performed",
    )
    search_efficiency_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Search efficiency score (0.0 to 1.0)",
    )
    collection_id: str | None = Field(
        default=None,
        description="Collection ID for organization",
    )


class FalsificationTrace(FalsificationTraceBase):
    """Response schema for FalsificationTrace.

    Note: FalsificationTrace is immutable (no update operations).
    """

    id: str = Field(..., description="Unique trace ID")
    search_queries: list[str] | None = Field(
        default=None, description="List of search queries executed"
    )
    contradicting_premise_ids: list[str] | None = Field(
        default=None, description="IDs of contradicting premises"
    )
    reasoning_chain: dict[str, Any] = Field(
        default_factory=dict, description="Reasoning process trace"
    )
    final_status: str = Field(..., description="Final status")
    search_count: int = Field(..., description="Number of searches performed")
    search_efficiency_score: float | None = Field(
        default=None, description="Search efficiency score"
    )
    workspace_name: str = Field(..., description="Workspace name", serialization_alias="workspace_id")
    collection_id: str | None = Field(default=None, description="Collection ID")
    created_at: datetime.datetime = Field(..., description="Creation timestamp")

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)  # pyright: ignore


# Alias for API responses
FalsificationTraceResponse = FalsificationTrace
TraceResponse = FalsificationTrace  # Short alias


class InductionBase(BaseModel):
    """Base schema for Induction."""

    content: str = Field(..., description="The induction content/pattern description")
    observer: str = Field(..., description="Peer name who observes")
    observed: str = Field(..., description="Peer name being observed")
    pattern_type: str = Field(
        ...,
        description="Type of pattern (temporal | causal | correlational | structural)",
    )


class InductionCreate(InductionBase):
    """Schema for creating a new Induction."""

    source_prediction_ids: list[str] | None = Field(
        default=None,
        description="IDs of source predictions that led to this induction",
    )
    source_premise_ids: list[str] | None = Field(
        default=None,
        description="IDs of source premises that support this induction",
    )
    confidence: str | None = Field(
        default="medium",
        description="Confidence level (high | medium | low)",
    )
    stability_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Stability score (0.0 to 1.0)",
    )
    collection_id: str | None = Field(
        default=None,
        description="Collection ID for organization",
    )


class Induction(InductionBase):
    """Response schema for Induction."""

    id: str = Field(..., description="Unique induction ID")
    source_prediction_ids: list[str] | None = Field(
        default=None, description="IDs of source predictions"
    )
    source_premise_ids: list[str] | None = Field(
        default=None, description="IDs of source premises"
    )
    confidence: str = Field(..., description="Confidence level")
    stability_score: float | None = Field(default=None, description="Stability score")
    workspace_name: str = Field(..., description="Workspace name", serialization_alias="workspace_id")
    collection_id: str | None = Field(default=None, description="Collection ID")
    created_at: datetime.datetime = Field(..., description="Creation timestamp")

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)  # pyright: ignore

# Alias for API responses
InductionResponse = Induction


class InductionSources(BaseModel):
    """Schema for induction sources (predictions and premises)."""

    induction: Induction = Field(..., description="The induction in question")
    source_predictions: list[Prediction] = Field(
        default_factory=list, description="Predictions that formed this pattern"
    )
    source_premises: list[Any] = Field(
        default_factory=list, description="Original observations (premises)"
    )
