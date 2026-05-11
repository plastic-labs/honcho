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


class AddMessagesFromFixtureAction(TestStep):
    """Load messages from a pre-extracted fixture JSON file.

    Fixtures live at tests/unified/test_cases/data/<fixture_path>. Use this
    when test inputs come from an external dataset (e.g., a sampled subset of
    a HuggingFace dataset) rather than being authored inline. Keeps test JSONs
    readable and avoids embedding large message corpora in test files.

    The fixture JSON must have shape:
        {"sessions": [{"messages": [{"role": "user|assistant", "content": "...", ...}, ...]}, ...]}
    """

    step_type: Literal["add_messages_from_fixture"] = "add_messages_from_fixture"
    session_id: str
    fixture_path: str = Field(
        ..., description="Filename within tests/unified/test_cases/data/"
    )
    fixture_session_index: int = Field(
        ..., description="Index into fixture['sessions'] selecting which session to load"
    )
    user_peer_id: str = Field(
        ..., description="Honcho peer ID to attribute fixture role='user' messages to"
    )
    assistant_peer_id: str | None = Field(
        None,
        description="Honcho peer ID for fixture role='assistant' messages. If omitted, assistant messages are skipped.",
    )
    limit: int | None = Field(
        None, description="Cap on number of conversational turns to ingest"
    )


# --- Artifact Actions ---


class SaveArtifactAction(TestStep):
    """Save a query result to a JSON file under the test run's artifact dir.

    Mirrors QueryAction's targets but performs no assertion; instead, the raw
    result is serialized and written to
    `<artifacts_root>/<test_name>_<timestamp>/<filename>`. Use this to capture
    observation lists, peer cards, or session contexts for offline analysis
    without coupling the analysis to a pass/fail gate.

    Filenames must be flat (no slashes, no leading dot) and should end in .json.
    """

    step_type: Literal["save_artifact"] = "save_artifact"
    description: str | None = None
    target: Literal["get_representation", "get_peer_card", "get_context"]
    filename: str = Field(
        ...,
        description="Output filename within the per-test artifact dir; should end in .json",
    )

    session_id: str | None = None
    observer_peer_id: str | None = None
    observed_peer_id: str | None = None
    # for get_context
    summary: bool = False
    max_tokens: int | None = None
    # for representation search (matches QueryAction.input semantics for get_representation)
    search_query: str | None = None


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
    target: Literal["chat", "get_context", "get_peer_card", "get_representation"]

    session_id: str | None = None

    input: str | None = None

    # for get_context
    summary: bool = False
    max_tokens: int | None = None

    observed_peer_id: str | None = None
    observer_peer_id: str | None = None

    # for chat - reasoning level
    reasoning_level: ReasoningLevel | None = None

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
    continue_on_failure: bool = Field(
        False,
        description=(
            "If true, assertion failures (TestExecutionError) don't abort the "
            "test — all steps run, and the test fails iff any assertion failed. "
            "Infrastructure errors still abort. Useful for tests that aggregate "
            "per-session results across many similar steps (e.g., naturalistic "
            "tests with N independent scenarios)."
        ),
    )
    steps: list[
        Annotated[
            SetWorkspaceConfigAction
            | SetSessionConfigAction
            | CreateSessionAction
            | AddMessageAction
            | AddMessagesAction
            | AddMessagesFromFixtureAction
            | SaveArtifactAction
            | WaitAction
            | ScheduleDreamAction
            | QueryAction,
            Field(discriminator="step_type"),
        ]
    ]
