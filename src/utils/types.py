from typing import Generic, Literal, NamedTuple, TypeVar

T = TypeVar("T")


class GetOrCreateResult(NamedTuple, Generic[T]):
    """Result of a get_or_create operation indicating whether the resource was created."""

    resource: T
    created: bool


SupportedProviders = Literal["anthropic", "openai", "google", "groq", "custom", "vllm"]
TaskType = Literal[
    "webhook",
    "summary",
    "representation",
    "dream",
    "dream_reasoning",  # Top-down reasoning during dreams
    "deletion",
    "hypothesis_generation",  # TODO: Remove after Phase 3.3 refactoring
    "prediction_testing",  # TODO: Remove after Phase 3.3 refactoring
    "falsification",  # TODO: Remove after Phase 3.3 refactoring
    "induction",  # TODO: Remove after Phase 3.3 refactoring
]
DocumentLevel = Literal["explicit", "inductive", "contradiction"]
