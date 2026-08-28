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

from src.mock_provider.embeddings import content_to_embedding
from src.mock_provider.main import app
from src.mock_provider.schema_gen import generate
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
    assert (
        representation.explicit
    ), "an empty explicit list is exactly the silent failure this mock avoids"


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


# --- streaming --------------------------------------------------------------


def test_stream_emits_content_then_a_final_usage_chunk(client: TestClient) -> None:
    """The backend ends the stream on the usage chunk, so it must come last."""
    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "mock-model",
            "messages": [{"role": "user", "content": "stream please"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        },
    ) as response:
        assert response.status_code == 200
        lines = [
            line[len("data: ") :]
            for line in response.iter_lines()
            if line.startswith("data: ")
        ]

    assert lines[-1] == "[DONE]"
    chunks = [json.loads(line) for line in lines[:-1]]

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
