from typing import Generic, Literal, NamedTuple, TypeVar

T = TypeVar("T")


class GetOrCreateResult(NamedTuple, Generic[T]):
    """Result of a get_or_create operation indicating whether the resource was created."""

    resource: T
    created: bool


SupportedProviders = Literal["anthropic", "openai", "google", "groq", "custom", "vllm"]
TaskType = Literal["webhook", "summary", "representation", "dream", "deletion"]
DocumentLevel = Literal["explicit", "deductive", "inductive", "contradiction"]
