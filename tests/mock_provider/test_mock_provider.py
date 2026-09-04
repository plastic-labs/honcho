"""Contract tests for the mock provider.

The failure this guards against is silent: when the mock answers a structured
request with something the deriver cannot parse, ``repair_response_model_json``
falls back to an empty ``PromptRepresentation`` and the run looks like "the
deriver found nothing" rather than "the mock is broken". So the assertions here
are about parseability against real Honcho models, not about response shape.
"""

from __future__ import annotations

import base64
import hashlib
import json
import struct
from collections.abc import Callable
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel, Field

from src.mock_provider.coerce import as_dict
from src.mock_provider.embeddings import content_to_embedding
from src.mock_provider.main import app
from src.mock_provider.schema_gen import HARD_MAX_DEPTH, MAX_DEPTH, generate
from src.utils.representation import PromptRepresentation

# A $ref/$defs schema, which is what Pydantic emits for any nested model and the
# indirection a naive generator silently drops.
PROBE_SCHEMA: dict[str, Any] = {
    "$defs": {
        "Item": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "count": {"type": "integer", "minimum": 1, "maximum": 5},
            },
            "required": ["name", "count"],
        }
    },
    "type": "object",
    "properties": {
        "label": {"type": "string"},
        "items": {"type": "array", "items": {"$ref": "#/$defs/Item"}},
    },
    "required": ["label", "items"],
}


class ProbeItem(BaseModel):
    name: str
    count: int = Field(ge=1, le=5)


class Probe(BaseModel):
    label: str
    items: list[ProbeItem]


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def _post_chat(client: TestClient, **payload: Any) -> dict[str, Any]:
    payload.setdefault("model", "mock-model")
    payload.setdefault("messages", [{"role": "user", "content": "hello"}])
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200, response.text
    return response.json()


def _json_schema_format(schema: dict[str, Any], name: str) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {"name": name, "schema": schema, "strict": True},
    }


# --- structured output ------------------------------------------------------


def test_json_schema_request_round_trips_into_its_pydantic_model(
    client: TestClient,
) -> None:
    body = _post_chat(
        client, response_format=_json_schema_format(PROBE_SCHEMA, "Probe")
    )
    content = body["choices"][0]["message"]["content"]

    probe = Probe.model_validate_json(content)
    assert probe.label
    assert probe.items, "$ref array must not come back empty"
    assert all(1 <= item.count <= 5 for item in probe.items)


def test_deriver_response_model_round_trips() -> None:
    """The real model the deriver parses, not a stand-in."""
    schema = PromptRepresentation.model_json_schema()
    content = json.dumps(generate(schema))

    representation = PromptRepresentation.model_validate_json(content)
    assert representation.explicit, (
        "an empty explicit list is exactly the silent failure this mock avoids"
    )


def test_json_schema_response_is_never_prose(client: TestClient) -> None:
    body = _post_chat(
        client, response_format=_json_schema_format(PROBE_SCHEMA, "Probe")
    )
    json.loads(body["choices"][0]["message"]["content"])


def test_unreadable_json_schema_still_returns_parseable_json(
    client: TestClient,
) -> None:
    body = _post_chat(
        client,
        response_format={"type": "json_schema", "json_schema": {"name": "Broken"}},
    )
    assert json.loads(body["choices"][0]["message"]["content"]) == {}


def test_json_object_mode_recovers_the_schema_from_the_prompt(
    client: TestClient,
) -> None:
    """json_object mode puts the schema in the prompt, not in response_format."""
    body = _post_chat(
        client,
        messages=[
            {"role": "user", "content": "Extract facts."},
            {
                "role": "user",
                "content": "Respond with valid JSON matching this schema:\n"
                + json.dumps(PROBE_SCHEMA),
            },
        ],
        response_format={"type": "json_object"},
    )
    Probe.model_validate_json(body["choices"][0]["message"]["content"])


def test_json_object_mode_without_a_schema_returns_an_empty_object(
    client: TestClient,
) -> None:
    body = _post_chat(client, response_format={"type": "json_object"})
    assert json.loads(body["choices"][0]["message"]["content"]) == {}


def test_plain_request_returns_prose(client: TestClient) -> None:
    body = _post_chat(client)
    content = body["choices"][0]["message"]["content"]
    assert "[mock]" in content
    with pytest.raises(json.JSONDecodeError):
        json.loads(content)


def test_tools_request_does_not_emit_tool_calls(client: TestClient) -> None:
    """The tool loop must terminate; a mock that calls tools would spin."""
    body = _post_chat(
        client,
        tools=[
            {
                "type": "function",
                "function": {"name": "search_memory", "parameters": {}},
            }
        ],
    )
    assert body["choices"][0]["message"]["tool_calls"] is None
    assert body["choices"][0]["finish_reason"] == "stop"


def test_identical_requests_are_byte_identical(client: TestClient) -> None:
    payload: dict[str, Any] = {
        "model": "mock-model",
        "messages": [{"role": "user", "content": "determinism"}],
        "response_format": _json_schema_format(PROBE_SCHEMA, "Probe"),
    }
    first = client.post("/v1/chat/completions", json=payload).json()
    second = client.post("/v1/chat/completions", json=payload).json()
    assert first == second


def test_usage_is_reported(client: TestClient) -> None:
    body = _post_chat(client)
    usage = body["usage"]
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
    assert usage["completion_tokens"] > 0


# --- schema generation edge cases -------------------------------------------


def test_recursive_schema_terminates() -> None:
    """Reasoning trees nest premises inside conclusions, so cycles are normal."""
    schema: dict[str, Any] = {
        "$defs": {
            "Node": {
                "type": "object",
                "properties": {
                    "value": {"type": "string"},
                    "child": {"anyOf": [{"$ref": "#/$defs/Node"}, {"type": "null"}]},
                },
                "required": ["value", "child"],
            }
        },
        "$ref": "#/$defs/Node",
    }
    node: dict[str, Any] | None = generate(schema)
    depth = 0
    while node is not None and node.get("child") is not None:
        node = node["child"]
        depth += 1
        assert depth < 50, "recursive schema did not terminate"


_SCALAR_CASES: list[tuple[str, dict[str, Any], Callable[[Any], bool]]] = [
    ("enum", {"type": "string", "enum": ["a", "b"]}, lambda v: v in ("a", "b")),
    ("const", {"const": 7}, lambda v: v == 7),
    ("boolean", {"type": "boolean"}, lambda v: isinstance(v, bool)),
    ("null", {"type": "null"}, lambda v: v is None),
    ("number", {"type": "number"}, lambda v: isinstance(v, float)),
    ("nullable-union", {"type": ["string", "null"]}, lambda v: isinstance(v, str)),
    ("pinned-int", {"type": "integer", "minimum": 3, "maximum": 3}, lambda v: v == 3),
    (
        "exclusive-bounds",
        {"type": "integer", "exclusiveMinimum": 1, "exclusiveMaximum": 3},
        lambda v: v == 2,
    ),
    (
        "date-time",
        {"type": "string", "format": "date-time"},
        lambda v: str(v).endswith("Z"),
    ),
    ("min-length", {"type": "string", "minLength": 400}, lambda v: len(v) >= 400),
    ("max-length", {"type": "string", "maxLength": 4}, lambda v: len(v) == 4),
    (
        "min-items",
        {"type": "array", "items": {"type": "string"}, "minItems": 3},
        lambda v: len(v) >= 3,
    ),
    (
        "max-items",
        {"type": "array", "items": {"type": "string"}, "maxItems": 1},
        lambda v: len(v) == 1,
    ),
]


@pytest.mark.parametrize(
    ("schema", "check"),
    [(schema, check) for _, schema, check in _SCALAR_CASES],
    ids=[name for name, _, _ in _SCALAR_CASES],
)
def test_scalar_schema_forms(
    schema: dict[str, Any], check: Callable[[Any], bool]
) -> None:
    assert check(generate(schema))


def test_all_of_is_flattened() -> None:
    schema: dict[str, Any] = {
        "allOf": [
            {
                "type": "object",
                "properties": {"a": {"type": "string"}},
                "required": ["a"],
            },
            {
                "type": "object",
                "properties": {"b": {"type": "integer"}},
                "required": ["b"],
            },
        ]
    }
    result = generate(schema)
    assert isinstance(result["a"], str)
    assert isinstance(result["b"], int)


def test_fixed_tuple_schema_round_trips() -> None:
    """Pydantic emits a fixed tuple as `prefixItems` with no `items`.

    Reading only `items` yields [], which fails the minItems the same schema
    carries — the silent-empty failure this module exists to avoid.
    """

    class Tupled(BaseModel):
        pair: tuple[str, int]

    schema = Tupled.model_json_schema()
    assert "prefixItems" in schema["properties"]["pair"]

    result = generate(schema)
    assert isinstance(result["pair"], list)
    Tupled.model_validate(result)


def test_prefix_items_are_followed_by_homogeneous_items() -> None:
    """A variadic tuple constrains leading positions and the rest by `items`."""
    schema: dict[str, Any] = {
        "type": "array",
        "prefixItems": [{"type": "string"}, {"type": "integer"}],
        "items": {"type": "boolean"},
        "minItems": 4,
    }
    result = generate(schema)

    assert len(result) == 4
    assert isinstance(result[0], str)
    assert isinstance(result[1], int)
    assert all(isinstance(value, bool) for value in result[2:])


def test_min_items_is_met_when_items_is_omitted() -> None:
    """Absent `items` leaves trailing positions unconstrained, not disallowed."""
    schema: dict[str, Any] = {
        "type": "array",
        "prefixItems": [{"type": "string"}],
        "minItems": 3,
    }
    result = generate(schema)

    assert len(result) == 3
    assert isinstance(result[0], str)


@pytest.mark.parametrize(
    ("constraints", "multiple"),
    [
        ({"minimum": 0, "maximum": 100, "multipleOf": 10}, 10),
        ({"minimum": 7, "maximum": 9, "multipleOf": 4}, 4),
        ({"minimum": -100, "maximum": 0, "multipleOf": 25}, 25),
        ({"minimum": 5, "multipleOf": 3}, 3),
        ({"maximum": -5, "multipleOf": 3}, 3),
        ({"multipleOf": 6}, 6),
    ],
)
def test_multiple_of_is_honoured_within_bounds(
    constraints: dict[str, Any], multiple: int
) -> None:
    """Path-seeded values land off the multiple unless snapped back onto it."""
    low = constraints.get("minimum")
    high = constraints.get("maximum")

    # Several paths, because a single one can satisfy the constraint by luck.
    for index in range(12):
        schema: dict[str, Any] = {
            "type": "object",
            "properties": {f"f{index}": {"type": "integer", **constraints}},
            "required": [f"f{index}"],
        }
        value = generate(schema)[f"f{index}"]

        assert value % multiple == 0, f"{value} is not a multiple of {multiple}"
        if low is not None:
            assert value >= low
        if high is not None:
            assert value <= high


def test_unsatisfiable_multiple_of_stays_within_bounds() -> None:
    """No multiple of 10 lies in [3, 7], so the bounds win over the multiple."""
    schema: dict[str, Any] = {
        "type": "integer",
        "minimum": 3,
        "maximum": 7,
        "multipleOf": 10,
    }
    result = generate(schema)
    assert 3 <= result <= 7


def test_required_recursive_ref_terminates_instead_of_overflowing() -> None:
    """A required, non-nullable cycle has no `default` or null branch to stop on.

    MAX_DEPTH alone does not save it — `_generate_object` keeps descending into
    required properties — so the absolute cap has to.
    """
    schema: dict[str, Any] = {
        "$defs": {
            "Node": {
                "type": "object",
                "properties": {"child": {"$ref": "#/$defs/Node"}},
                "required": ["child"],
            }
        },
        "$ref": "#/$defs/Node",
    }
    node = as_dict(generate(schema))

    depth = 0
    # The cap returns {}, so an empty dict is the terminator.
    while node:
        node = as_dict(node["child"])
        depth += 1
        assert depth <= HARD_MAX_DEPTH, "absolute depth cap did not hold"
    assert depth > MAX_DEPTH, "should descend past the soft cap before stopping"


def test_required_recursive_array_terminates_instead_of_overflowing() -> None:
    """minItems >= 1 keeps `_generate_array` from emptying out at the soft cap."""
    schema: dict[str, Any] = {
        "$defs": {
            "Node": {
                "type": "object",
                "properties": {
                    "kids": {
                        "type": "array",
                        "items": {"$ref": "#/$defs/Node"},
                        "minItems": 1,
                    }
                },
                "required": ["kids"],
            }
        },
        "$ref": "#/$defs/Node",
    }
    generate(schema)  # must not raise RecursionError


def test_generation_is_stable_across_calls() -> None:
    assert generate(PROBE_SCHEMA) == generate(PROBE_SCHEMA)


def test_sibling_fields_of_the_same_type_differ() -> None:
    """Path-seeded, so a schema of identical fields is not all one value."""
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "first": {"type": "string"},
            "second": {"type": "string"},
        },
        "required": ["first", "second"],
    }
    result = generate(schema)
    assert result["first"] != result["second"]


# --- embeddings -------------------------------------------------------------


def test_embeddings_default_to_1536_and_are_stable(client: TestClient) -> None:
    payload = {
        "model": "text-embedding-3-small",
        "input": "hello",
        "encoding_format": "float",
    }
    first = client.post("/v1/embeddings", json=payload)
    assert first.status_code == 200, first.text
    vector = first.json()["data"][0]["embedding"]

    assert len(vector) == 1536
    assert all(-1.0 <= value <= 1.0 for value in vector)
    assert client.post("/v1/embeddings", json=payload).json() == first.json()


def test_embeddings_honour_the_requested_dimension(client: TestClient) -> None:
    """A width mismatch raises in EmbeddingClient and blocks startup."""
    response = client.post(
        "/v1/embeddings",
        json={"input": "hello", "dimensions": 256, "encoding_format": "float"},
    )
    assert len(response.json()["data"][0]["embedding"]) == 256


def test_different_inputs_give_different_vectors(client: TestClient) -> None:
    response = client.post(
        "/v1/embeddings",
        json={"input": ["alpha", "beta"], "encoding_format": "float"},
    )
    data = response.json()["data"]
    assert len(data) == 2
    assert [item["index"] for item in data] == [0, 1]
    assert data[0]["embedding"] != data[1]["embedding"]


def test_batch_returns_one_embedding_per_input(client: TestClient) -> None:
    """EmbeddingClient._validate_embedding_count rejects any other count."""
    texts = [f"text-{index}" for index in range(17)]
    response = client.post(
        "/v1/embeddings", json={"input": texts, "encoding_format": "float"}
    )
    assert len(response.json()["data"]) == len(texts)


def test_base64_is_the_default_encoding_and_decodes_to_the_float_vector(
    client: TestClient,
) -> None:
    """The SDK omits encoding_format precisely when it wants base64."""
    response = client.post("/v1/embeddings", json={"input": "hello"})
    encoded = response.json()["data"][0]["embedding"]
    assert isinstance(encoded, str)

    raw = base64.b64decode(encoded)
    decoded = list(struct.unpack(f"<{len(raw) // 4}f", raw))
    assert len(decoded) == 1536
    expected = content_to_embedding("hello", 1536)
    assert decoded == pytest.approx(expected, abs=1e-6)  # pyright: ignore[reportUnknownMemberType]


def test_embedding_matches_the_test_suite_helper() -> None:
    """Kept in step with _content_to_embedding in tests/conftest.py.

    Both must derive the same vector from the same text, so a suite that mocks
    the embedding client in-process and one that talks to this provider over
    HTTP agree on what a given string embeds to.
    """
    digest = hashlib.sha256(b"hello").digest()
    expected = [(digest[i % len(digest)] / 255.0) * 2 - 1 for i in range(8)]

    assert content_to_embedding("hello", 8) == pytest.approx(expected)  # pyright: ignore[reportUnknownMemberType]


# --- routing ----------------------------------------------------------------


def test_routes_are_mounted_with_and_without_the_v1_prefix(
    client: TestClient,
) -> None:
    for path in ("/v1/chat/completions", "/chat/completions"):
        response = client.post(
            path, json={"model": "m", "messages": [{"role": "user", "content": "x"}]}
        )
        assert response.status_code == 200, path


def test_unimplemented_post_returns_405_not_a_plausible_200(
    client: TestClient,
) -> None:
    """A catch-all POST would make a missing endpoint look like it worked."""
    assert client.post("/v1/completions", json={}).status_code == 405


def test_health_and_catch_all_get(client: TestClient) -> None:
    assert client.get("/health").json()["status"] == "ok"
    assert client.get("/").status_code == 200


# --- request validation -----------------------------------------------------


def test_malformed_body_returns_an_openai_error_envelope(client: TestClient) -> None:
    """A bad request must look like the real API's, not like FastAPI's 422.

    Mid-run, a 422 in FastAPI's own error shape reads as a Honcho bug rather
    than a bad request, and no OpenAI client knows how to interpret it.
    """
    response = client.post(
        "/v1/embeddings", json={"input": "hello", "dimensions": "not-a-number"}
    )

    assert response.status_code == 400
    error = response.json()["error"]
    assert error["type"] == "invalid_request_error"
    assert error["message"]
    assert set(error) == {"message", "type", "param", "code"}


def test_boolean_dimensions_is_rejected_not_silently_coerced(
    client: TestClient,
) -> None:
    """bool is an int subclass, so `true` would otherwise mean 1 dimension."""
    response = client.post(
        "/v1/embeddings", json={"input": "hello", "dimensions": True}
    )

    assert response.status_code == 400


@pytest.mark.parametrize("dimensions", [0, -1])
def test_non_positive_dimensions_is_rejected_not_defaulted(
    client: TestClient, dimensions: int
) -> None:
    """Substituting 1536 would answer a bad request with a plausible vector."""
    response = client.post(
        "/v1/embeddings",
        json={"input": "hello", "dimensions": dimensions, "encoding_format": "float"},
    )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


def test_unknown_fields_are_accepted(client: TestClient) -> None:
    """Validation must fire on wrong types, never on unrecognised parameters.

    A new upstream parameter should not turn a working setup into a hard
    failure, so every model allows extras.
    """
    body = _post_chat(
        client,
        temperature=0.7,
        max_completion_tokens=256,
        reasoning_effort="minimal",
        some_parameter_invented_next_year=True,
    )
    assert body["choices"][0]["finish_reason"] == "stop"


def test_wrongly_typed_messages_are_rejected(client: TestClient) -> None:
    response = client.post(
        "/v1/chat/completions", json={"model": "m", "messages": "not-a-list"}
    )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"input": "solo"}, 1),
        ({"input": ["a", "b", "c"]}, 3),
        ({"input": [1, 2, 3]}, 1),
        ({"input": [[1, 2], [3, 4]]}, 2),
        ({"input": None}, 0),
    ],
    ids=["string", "list-of-strings", "token-array", "token-arrays", "null"],
)
def test_every_documented_input_shape_is_accepted(
    client: TestClient, payload: dict[str, Any], expected: int
) -> None:
    """A flat int list is one tokenized input, not many single-token ones."""
    response = client.post(
        "/v1/embeddings", json={**payload, "encoding_format": "float"}
    )

    assert response.status_code == 200, response.text
    assert len(response.json()["data"]) == expected


# --- streaming --------------------------------------------------------------


def _stream_chunks(
    client: TestClient, stream_options: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    """The SSE payloads of a streaming completion, `[DONE]` asserted and dropped."""
    body: dict[str, Any] = {
        "model": "mock-model",
        "messages": [{"role": "user", "content": "stream please"}],
        "stream": True,
    }
    if stream_options is not None:
        body["stream_options"] = stream_options

    with client.stream("POST", "/v1/chat/completions", json=body) as response:
        assert response.status_code == 200
        lines = [
            line[len("data: ") :]
            for line in response.iter_lines()
            if line.startswith("data: ")
        ]

    assert lines[-1] == "[DONE]"
    return [json.loads(line) for line in lines[:-1]]


def test_stream_emits_content_then_a_final_usage_chunk(client: TestClient) -> None:
    """The backend ends the stream on the usage chunk, so it must come last."""
    chunks = _stream_chunks(client, {"include_usage": True})

    content = "".join(
        chunk["choices"][0]["delta"].get("content", "")
        for chunk in chunks
        if chunk["choices"]
    )
    assert "[mock]" in content

    assert any(
        chunk["choices"] and chunk["choices"][0]["finish_reason"] == "stop"
        for chunk in chunks
    )

    usage_chunk = chunks[-1]
    assert usage_chunk["usage"]["completion_tokens"] > 0
    assert usage_chunk["choices"] == []


@pytest.mark.parametrize(
    "stream_options",
    [None, {}, {"include_usage": False}],
    ids=["absent", "empty", "false"],
)
def test_stream_without_include_usage_emits_no_usage_chunk(
    client: TestClient, stream_options: dict[str, Any] | None
) -> None:
    """The real API sends the usage chunk only when asked, so neither does this.

    A caller that did not opt in must not have to skip a trailing chunk with an
    empty `choices` array.
    """
    chunks = _stream_chunks(client, stream_options)

    assert all("usage" not in chunk for chunk in chunks)
    assert chunks[-1]["choices"][0]["finish_reason"] == "stop"

    content = "".join(
        chunk["choices"][0]["delta"].get("content", "") for chunk in chunks
    )
    assert "[mock]" in content


@pytest.mark.parametrize("value", ["definitely", "yes", "on", "true", "1", 1])
def test_non_boolean_include_usage_is_rejected(client: TestClient, value: Any) -> None:
    """The usage chunk is conditional on this, so a wrong type must 400.

    The truthy strings matter more than the nonsense one: plain `bool` coerces
    "yes"/"on"/"true"/"1", so without StrictBool a string would silently decide
    whether the stream carries usage.
    """
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "mock-model",
            "messages": [{"role": "user", "content": "x"}],
            "stream": True,
            "stream_options": {"include_usage": value},
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


@pytest.mark.parametrize("value", ["yes", "true", "1", 1])
def test_non_boolean_stream_is_rejected(client: TestClient, value: Any) -> None:
    """`stream` picks between a JSON body and an SSE stream, so it must be exact."""
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "mock-model",
            "messages": [{"role": "user", "content": "x"}],
            "stream": value,
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
