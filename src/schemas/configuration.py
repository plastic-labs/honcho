"""Configuration schemas for hierarchical settings resolution.

Covers workspace, session, and message-level configuration as well as
the fully-resolved variants used at runtime.
"""

from enum import Enum
from typing import Annotated, Any, Self, cast

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

from src.config import settings
from src.utils.tokens import estimate_tokens


class DreamType(str, Enum):
    """Types of dreams that can be triggered."""

    OMNI = "omni"
    # Lightweight card-only refresh: runs a single specialist restricted to
    # peer-card tools. Used for event-driven refreshes (scope membership
    # changes, cold starts) — never creates or deletes observations.
    CARD_REFRESH = "card_refresh"


def _validate_custom_instructions_budget(
    custom_instructions: str | None,
) -> str | None:
    if custom_instructions is None:
        return None
    if not custom_instructions.strip():
        return custom_instructions

    max_tokens = settings.DERIVER.MAX_CUSTOM_INSTRUCTIONS_TOKENS
    if max_tokens <= 0:
        raise ValueError("custom_instructions are not enabled for this deployment")

    if estimate_tokens(custom_instructions) > max_tokens:
        raise ValueError(
            f"custom_instructions exceeds DERIVER.MAX_CUSTOM_INSTRUCTIONS_TOKENS ({max_tokens} tokens)"
        )

    return custom_instructions


CustomInstructions = Annotated[
    str | None, AfterValidator(_validate_custom_instructions_budget)
]


class ReasoningConfiguration(BaseModel):
    enabled: bool | None = Field(
        default=None,
        description="Whether to enable reasoning functionality.",
    )
    custom_instructions: CustomInstructions = Field(
        default=None,
        description="Custom instructions for the deriver. Overrides the top-level custom_instructions at the same level. Set to an empty string to explicitly use none.",
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
    custom_instructions: CustomInstructions = Field(
        default=None,
        description="Custom instructions for peer card updates. Added on top of the dream instructions and applied within the peer card's allowed entry kinds. Does not fall back to the top-level custom_instructions.",
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
    custom_instructions: CustomInstructions = Field(
        default=None,
        description="Custom instructions for session summaries. Overrides the top-level custom_instructions at the same level. Set to an empty string to explicitly use none.",
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
    custom_instructions: CustomInstructions = Field(
        default=None,
        description="Custom instructions for dreams. Overrides the top-level custom_instructions at the same level. Set to an empty string to explicitly use none.",
    )


class DialecticConfiguration(BaseModel):
    custom_instructions: CustomInstructions = Field(
        default=None,
        description="Custom instructions for the dialectic chat endpoints. Overrides the top-level custom_instructions at the same level. Set to an empty string to explicitly use none.",
    )


class WorkspaceConfiguration(BaseModel):
    """
    The set of options that can be in a workspace DB-level configuration dictionary.

    All fields are optional. Session-level configuration overrides workspace-level configuration, which overrides global configuration.
    """

    model_config = ConfigDict(extra="allow")  # pyright: ignore

    custom_instructions: CustomInstructions = Field(
        default=None,
        description="Custom instructions shared by the deriver, summarizer, dialectic, and dreamer. A module's own custom_instructions at the same level takes precedence, and session-level values beat workspace-level ones. Messages can only override reasoning.custom_instructions.",
    )
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
    dialectic: DialecticConfiguration | None = Field(
        default=None,
        description="Configuration for the dialectic chat endpoints.",
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
    custom_instructions: CustomInstructions = None
    custom_instructions_source: str | None = None


class ResolvedPeerCardConfiguration(BaseModel):
    use: bool
    create: bool
    custom_instructions: CustomInstructions = None
    custom_instructions_source: str | None = None


class ResolvedSummaryConfiguration(BaseModel):
    enabled: bool
    messages_per_short_summary: int
    messages_per_long_summary: int
    custom_instructions: CustomInstructions = None
    custom_instructions_source: str | None = None


class ResolvedDreamConfiguration(BaseModel):
    enabled: bool
    custom_instructions: CustomInstructions = None
    custom_instructions_source: str | None = None


class ResolvedDialecticConfiguration(BaseModel):
    custom_instructions: CustomInstructions = None
    custom_instructions_source: str | None = None


class ResolvedConfiguration(BaseModel):
    """
    The final resolved configuration for a given message.
    Hierarchy: message > session > workspace > global configuration
    """

    reasoning: ResolvedReasoningConfiguration
    peer_card: ResolvedPeerCardConfiguration
    summary: ResolvedSummaryConfiguration
    dream: ResolvedDreamConfiguration
    dialectic: ResolvedDialecticConfiguration = Field(
        default_factory=ResolvedDialecticConfiguration
    )

    @model_validator(mode="before")
    @classmethod
    def migrate_deriver_to_reasoning(cls, data: Any) -> Any:
        """Handle v3.0.0 migration: 'deriver' was renamed to 'reasoning'."""
        if not isinstance(data, dict):
            return data

        config = cast(dict[str, Any], data)

        if "deriver" in config and "reasoning" not in config:
            config["reasoning"] = config.pop("deriver")

        return config


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
