import datetime
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from src.config import ReasoningLevel
from src.schemas import (
    DreamType,
    MessageConfiguration,
    SessionConfiguration,
    SessionPeerConfig,
    WorkspaceConfiguration,
)


class TestStep(BaseModel):
    model_config = ConfigDict(extra="forbid")  # pyright: ignore

    description: str | None = None


# --- Configuration Actions ---


class SetWorkspaceConfigAction(TestStep):
    step_type: Literal["set_workspace_config"] = "set_workspace_config"
    config: WorkspaceConfiguration


class SetSessionConfigAction(TestStep):
    step_type: Literal["set_session_config"] = "set_session_config"
    session_id: str
    config: SessionConfiguration


# --- Interaction Actions ---


class CreateSessionAction(TestStep):
    step_type: Literal["create_session"] = "create_session"
    session_id: str
    peer_configs: dict[str, SessionPeerConfig] | None = None
    config: SessionConfiguration | None = None


class AddMessageAction(TestStep):
    step_type: Literal["add_message"] = "add_message"
    session_id: str
    peer_id: str
    content: str
    created_at: datetime.datetime | None = None
    config: MessageConfiguration | None = None


class MessageItem(BaseModel):
    peer_id: str
    content: str
    config: MessageConfiguration | None = None
    created_at: datetime.datetime | None = None


class AddMessagesAction(TestStep):
    step_type: Literal["add_messages"] = "add_messages"
    session_id: str
    messages: list[MessageItem]


class CreateScopeAction(TestStep):
    """Create a scope and optionally add member sessions.

    Driven over raw HTTP rather than the SDK: scopes are a new API surface the
    published SDK does not expose yet, and gating coverage on an SDK release
    would leave the feature untested at exactly the point it needs testing.
    """

    step_type: Literal["create_scope"] = "create_scope"
    scope_id: str = Field(..., description="Unprefixed scope name")
    session_ids: list[str] = Field(
        default_factory=list,
        description="Existing sessions to add as members of the scope",
    )


# --- Wait Actions ---


class WaitAction(TestStep):
    step_type: Literal["wait"] = "wait"
    duration: float | None = Field(
        None, description="Wait for a specific duration in seconds"
    )
    target: Literal["queue_empty"] = "queue_empty"
    timeout: int = 60


# --- Dream Actions ---


class ScheduleDreamAction(TestStep):
    step_type: Literal["schedule_dream"] = "schedule_dream"
    observer: str = Field(..., description="Observer peer name")
    observed: str | None = Field(
        None, description="Observed peer name (defaults to observer if not specified)"
    )
    session_id: str = Field(..., description="Session ID to scope the dream to")
    dream_type: DreamType = Field(..., description="Type of dream to schedule")


# --- Assertions ---


class Assertion(BaseModel):
    pass


class LLMJudgeAssertion(Assertion):
    assertion_type: Literal["llm_judge"] = "llm_judge"
    prompt: str
    pass_if: bool = True


class ContainsAssertion(Assertion):
    assertion_type: Literal["contains"] = "contains"
    text: str
    case_sensitive: bool = False


class NotContainsAssertion(Assertion):
    assertion_type: Literal["not_contains"] = "not_contains"
    text: str
    case_sensitive: bool = False


class ExactMatchAssertion(Assertion):
    assertion_type: Literal["exact_match"] = "exact_match"
    text: str


class JsonMatchAssertion(Assertion):
    assertion_type: Literal["json_match"] = "json_match"
    schema_path: str | None = None  # Optional JSON schema path
    key_value_pairs: dict[str, Any] | None = None


class EvidenceContainsAssertion(Assertion):
    """Assert on what a chat run read, not on what it wrote.

    Evaluated against the `evidence` a chat / workspace_chat query returns, so
    it proves retrieval happened independently of how the answer was phrased.
    Evidence over-reports (prefetched rows count as read), so this shows a row
    was reached, not that the answer used it. Every field set must hold.
    """

    assertion_type: Literal["evidence_contains"] = "evidence_contains"
    conclusions_match: str | None = Field(
        default=None,
        description="Case-insensitive substring some evidence conclusion must contain",
    )
    conclusions_from_peers: list[str] | None = Field(
        default=None,
        description="Peers (by `observed_id`) that must each have a conclusion in evidence",
    )
    min_count: int | None = Field(
        default=None,
        ge=1,
        description=(
            "How many of `conclusions_from_peers` must be present; defaults to all"
        ),
    )
    messages_match: str | None = Field(
        default=None,
        description=(
            "Case-insensitive substring some evidence message must contain. Evidence"
            " carries message ids only, so the runner fetches each message's content."
        ),
    )
    not_from_sessions: list[str] | None = Field(
        default=None,
        description=(
            "No evidence conclusion or message may belong to these sessions."
            " Conclusions without a session id are not attributable and are skipped."
        ),
    )

    @model_validator(mode="after")
    def _validate(self) -> Self:
        if (
            self.conclusions_match is None
            and self.conclusions_from_peers is None
            and self.messages_match is None
            and self.not_from_sessions is None
        ):
            raise ValueError("evidence_contains needs at least one condition")
        for name in ("conclusions_match", "messages_match"):
            value = getattr(self, name)
            if value is not None and not value.strip():
                raise ValueError(f"{name} must not be blank")
        if self.not_from_sessions is not None and not self.not_from_sessions:
            raise ValueError("not_from_sessions must not be empty")
        if self.conclusions_from_peers is not None:
            if not self.conclusions_from_peers:
                raise ValueError("conclusions_from_peers must not be empty")
            if len(set(self.conclusions_from_peers)) != len(
                self.conclusions_from_peers
            ):
                raise ValueError("conclusions_from_peers must not contain duplicates")
        if self.min_count is not None:
            if self.conclusions_from_peers is None:
                raise ValueError("min_count requires conclusions_from_peers")
            if self.min_count > len(self.conclusions_from_peers):
                raise ValueError("min_count exceeds len(conclusions_from_peers)")
        return self

    @property
    def required_peer_count(self) -> int:
        return self.min_count or len(self.conclusions_from_peers or [])


# --- Query/Assertion Actions ---


class QueryAction(TestStep):
    step_type: Literal["query"] = "query"
    target: Literal[
        "chat",
        "get_context",
        "get_peer_card",
        "get_representation",
        "workspace_chat",
    ]

    session_id: str | None = None

    input: str | None = None

    # for get_context
    summary: bool = False
    max_tokens: int | None = None

    observed_peer_id: str | None = None
    observer_peer_id: str | None = None

    # for chat - reasoning level
    reasoning_level: ReasoningLevel | None = None

    # for chat - optional JSON Schema the response must conform to
    response_format: dict[str, Any] | None = None

    # Confine the read to one scope (observer swap on peer chat) or to the
    # union of several scopes' member sessions. Peer-chat/representation/
    # context go over raw HTTP; workspace_chat uses the SDK `scope` argument.
    scope: str | list[str] | None = None

    assertions: list[
        LLMJudgeAssertion
        | ContainsAssertion
        | NotContainsAssertion
        | ExactMatchAssertion
        | JsonMatchAssertion
        | EvidenceContainsAssertion
    ]


# --- Unified Step Type ---


class TestDefinition(BaseModel):
    description: str | None = None
    workspace_config: WorkspaceConfiguration | None = None
    steps: list[
        Annotated[
            SetWorkspaceConfigAction
            | SetSessionConfigAction
            | CreateSessionAction
            | AddMessageAction
            | AddMessagesAction
            | CreateScopeAction
            | WaitAction
            | ScheduleDreamAction
            | QueryAction,
            Field(discriminator="step_type"),
        ]
    ]
