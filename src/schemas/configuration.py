"""Configuration schemas for hierarchical settings resolution.

Covers workspace, session, and message-level configuration as well as
the fully-resolved variants used at runtime.
"""

from enum import Enum
from typing import Any, Self, cast

import tiktoken
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from src.config import settings

_TOKENIZER = tiktoken.get_encoding("o200k_base")


def _estimate_tokens(text: str) -> int:
    """Estimate token count for reasoning configuration validation."""
    try:
        return len(_TOKENIZER.encode(text))
    except Exception:
        return len(text) // 4


def _validate_custom_instructions_budget(value: str | None) -> str | None:
    """Reject custom instructions that exceed the configured deriver budget."""
    if value is None or not value.strip():
        return value

    token_count = _estimate_tokens(value)
    token_limit = settings.DERIVER.effective_max_custom_instructions_tokens
    if token_count > token_limit:
        raise ValueError(
            "reasoning.custom_instructions uses "
            f"{token_count} tokens and exceeds DERIVER.MAX_CUSTOM_INSTRUCTIONS_TOKENS "
            f"({token_limit})"
        )

    return value


class DreamType(str, Enum):
    """Types of dreams that can be triggered."""

    OMNI = "omni"


class ReasoningConfiguration(BaseModel):
    enabled: bool | None = Field(
        default=None,
        description="Whether to enable reasoning functionality.",
    )
    custom_instructions: str | None = Field(
        default=None,
        description="Optional custom instructions for the reasoning system on this workspace/session/message. May be omitted or set to a blank string. Non-blank values are rejected if they exceed the configured deriver custom-instructions token budget.",
    )

    _validate_custom_instructions = field_validator(
        "custom_instructions", mode="after"
    )(_validate_custom_instructions_budget)


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
    custom_instructions: str | None = None

    _validate_custom_instructions = field_validator(
        "custom_instructions", mode="after"
    )(_validate_custom_instructions_budget)


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
