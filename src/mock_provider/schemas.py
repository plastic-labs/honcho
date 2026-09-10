"""Request models for the mock provider's OpenAI-compatible endpoints.

Validating the request envelope rather than hand-coercing it makes the mock
behave like the thing it mocks: real OpenAI answers a malformed request with a
400 and an error envelope, and ``openai_error_response`` in ``main`` turns
Pydantic's failure into exactly that.

Two deliberate choices:

- ``extra="allow"`` on every model, and every field optional. Validation should
  fire on a wrong *type* (a string where a list belongs), never on a field this
  mock has not heard of — otherwise a new upstream parameter turns a working
  setup into a hard failure.
- Open-ended payloads stay ``dict[str, Any]``. ``response_format`` carries an
  arbitrary caller-supplied JSON Schema, so only its envelope is worth typing;
  ``schema_gen`` walks the rest.
"""

from __future__ import annotations

from typing import Annotated, Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictInt


class MockRequest(BaseModel):
    """Permissive base: unknown fields pass through untouched."""

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="allow")


class ChatMessage(MockRequest):
    role: str | None = None
    # Multimodal requests send a list of content parts rather than a string, so
    # this cannot narrow further.
    content: Any = None


class StreamOptions(MockRequest):
    # Typed rather than left as a dict because the usage chunk is conditional on
    # it. StrictBool for the same reason `dimensions` is StrictInt: plain `bool`
    # coerces "yes"/"on"/"true"/"1", so a string would quietly decide the shape
    # of the stream instead of failing the way the real API does.
    include_usage: StrictBool = False


class ChatCompletionRequest(MockRequest):
    model: str | None = None
    messages: list[ChatMessage] = []
    response_format: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    # StrictBool because this one field decides between two response *shapes* —
    # a JSON body or an SSE stream — so coercing a string here is the difference
    # between a working client and one that hangs waiting for events.
    stream: StrictBool = False
    stream_options: StreamOptions | None = None


class EmbeddingsRequest(MockRequest):
    # Every input shape the OpenAI embeddings API accepts. Pydantic's smart
    # union keeps list[str] and list[int] apart instead of coercing one to the
    # other.
    input: str | list[str] | list[int] | list[list[int]] | None = None
    model: str | None = None
    # StrictInt because bool is an int subclass: a JSON `true` here would
    # otherwise silently become a one-dimensional vector. gt=0 because the real
    # API rejects a non-positive width, and substituting the default instead
    # would answer a bad request with a plausible-looking vector.
    dimensions: Annotated[StrictInt, Field(gt=0)] | None = None
    # The SDK omits this only when it wants base64, so absent means base64.
    encoding_format: Literal["float", "base64"] = "base64"
