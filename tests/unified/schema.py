import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field

from src.config import ReasoningLevel
from src.schemas import (
    DreamType,
    MessageConfiguration,
    SessionConfiguration,
    SessionPeerConfig,
    WorkspaceConfiguration,
)


class TestStep(BaseModel):
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
    flush: bool = Field(
        False,
        description="Enable flush mode to bypass batch token threshold before waiting",
    )


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
