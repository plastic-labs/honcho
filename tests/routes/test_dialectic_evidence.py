"""Tests for the `include_evidence` request option on both chat endpoints.

These cover the request/response contract: whether evidence is asked for,
whether an accumulator reaches the dialectic, and how the result is
serialized -- including onto the stream, where evidence can only ride on the
terminal event.

Note what these deliberately do NOT cover. The autouse
`mock_llm_call_functions` fixture replaces `agentic_chat` and `workspace_chat`
wholesale, so nothing here exercises collection; a test that asserted on
evidence content while the dialectic is mocked would only be reading back its
own fixture. Collection is covered in tests/dialectic/test_evidence.py and
tests/utils/test_agent_tools.py.
"""

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import pytest
from fastapi.testclient import TestClient
from nanoid import generate as generate_nanoid

from src.models import Peer, Workspace
from src.utils.evidence import EvidenceAccumulator

TOOL_CALL = {"tool_name": "search_memory", "tool_input": {"query": "coffee"}}


@dataclass(frozen=True)
class Endpoint:
    """One of the two chat endpoints, and the mock standing in for its agent."""

    name: str
    path: str
    query: str
    mock_key: str

    def url(self, workspace: Workspace, peer: Peer) -> str:
        return self.path.format(workspace=workspace.name, peer=peer.name)


PEER_CHAT = Endpoint(
    name="peer",
    path="/v3/workspaces/{workspace}/peers/{peer}/chat",
    query="What does this user drink?",
    mock_key="agentic_chat",
)
WORKSPACE_CHAT = Endpoint(
    name="workspace",
    path="/v3/workspaces/{workspace}/chat",
    query="What do people here drink?",
    mock_key="workspace_chat",
)
BOTH_ENDPOINTS = pytest.mark.parametrize(
    "endpoint", [PEER_CHAT, WORKSPACE_CHAT], ids=lambda e: e.name
)


def _record_sample_evidence(evidence: EvidenceAccumulator) -> None:
    """Stand in for what a real run would have collated."""
    evidence.record_tool_calls([TOOL_CALL])


def _stub_chat(mock: Any, content: str) -> None:
    """Have the mocked dialectic fill in the accumulator it was handed."""

    async def _chat(*_args: object, **kwargs: Any) -> str:
        evidence = kwargs.get("evidence")
        if evidence is not None:
            _record_sample_evidence(evidence)
        return content

    mock.side_effect = _chat


def _stub_chat_stream(mock: Any) -> None:
    def _chat_stream(*_args: object, **kwargs: Any) -> AsyncIterator[str]:
        evidence = kwargs.get("evidence")

        async def _chunks() -> AsyncIterator[str]:
            if evidence is not None:
                _record_sample_evidence(evidence)
            for chunk in ("Test ", "streaming ", "response"):
                yield chunk

        return _chunks()

    mock.side_effect = _chat_stream


def _sse_events(text: str) -> list[dict[str, Any]]:
    return [
        json.loads(line.removeprefix("data: "))
        for line in text.splitlines()
        if line.startswith("data: ")
    ]


@BOTH_ENDPOINTS
class TestBothEndpoints:
    """The two chat endpoints have to behave identically here."""

    def test_evidence_is_absent_by_default(
        self,
        endpoint: Endpoint,
        client: TestClient,
        sample_data: tuple[Workspace, Peer],
    ):
        workspace, peer = sample_data

        response = client.post(
            endpoint.url(workspace, peer), json={"query": endpoint.query}
        )

        assert response.status_code == 200
        assert response.json()["evidence"] is None

    def test_serializes_what_the_run_collated(
        self,
        endpoint: Endpoint,
        client: TestClient,
        sample_data: tuple[Workspace, Peer],
        mock_llm_call_functions: dict[str, Any],
    ):
        workspace, peer = sample_data
        _stub_chat(mock_llm_call_functions[endpoint.mock_key], "They drink coffee.")

        response = client.post(
            endpoint.url(workspace, peer),
            json={"query": endpoint.query, "include_evidence": True},
        )

        assert response.status_code == 200
        body = response.json()
        assert body["content"] == "They drink coffee."
        assert body["evidence"]["tool_calls"] == [TOOL_CALL]

    def test_evidence_rides_on_the_terminal_stream_event(
        self,
        endpoint: Endpoint,
        client: TestClient,
        sample_data: tuple[Workspace, Peer],
        mock_llm_call_functions: dict[str, Any],
    ):
        workspace, peer = sample_data
        _stub_chat_stream(mock_llm_call_functions[f"{endpoint.mock_key}_stream"])

        response = client.post(
            endpoint.url(workspace, peer),
            json={
                "query": endpoint.query,
                "stream": True,
                "include_evidence": True,
            },
        )

        assert response.status_code == 200
        events = _sse_events(response.text)
        content_events, final = events[:-1], events[-1]
        assert "".join(e["delta"]["content"] for e in content_events) == (
            "Test streaming response"
        )
        assert all(e["done"] is False for e in content_events)
        assert all("evidence" not in e for e in content_events)
        assert final["done"] is True
        assert final["evidence"]["tool_calls"] == [TOOL_CALL]

    def test_the_stream_is_unchanged_when_evidence_is_not_requested(
        self,
        endpoint: Endpoint,
        client: TestClient,
        sample_data: tuple[Workspace, Peer],
    ):
        workspace, peer = sample_data

        response = client.post(
            endpoint.url(workspace, peer),
            json={"query": endpoint.query, "stream": True},
        )

        assert _sse_events(response.text)[-1] == {"done": True}


class TestOptingIn:
    """Whether an accumulator reaches the dialectic at all."""

    @pytest.mark.parametrize(
        ("body", "expect_accumulator"),
        [
            ({}, False),
            ({"include_evidence": False}, False),
            ({"include_evidence": True}, True),
        ],
        ids=["omitted", "false", "true"],
    )
    def test_an_accumulator_is_created_only_on_request(
        self,
        client: TestClient,
        sample_data: tuple[Workspace, Peer],
        mock_llm_call_functions: dict[str, Any],
        body: dict[str, Any],
        expect_accumulator: bool,
    ):
        """Opting out has to cost nothing, so nothing is collected at all."""
        workspace, peer = sample_data

        client.post(
            PEER_CHAT.url(workspace, peer), json={"query": PEER_CHAT.query, **body}
        )

        await_args = mock_llm_call_functions["agentic_chat"].await_args
        assert await_args is not None
        accumulator = await_args.kwargs["evidence"]
        assert isinstance(accumulator, EvidenceAccumulator) is expect_accumulator

    def test_reports_empty_evidence_when_nothing_was_read(
        self, client: TestClient, sample_data: tuple[Workspace, Peer]
    ):
        """Asked-for-but-empty is not the same as not asked for.

        The default mocked dialectic reads nothing, so this is the shape a run
        that found nothing produces -- and it is distinguishable from `null`.
        """
        workspace, peer = sample_data

        response = client.post(
            PEER_CHAT.url(workspace, peer),
            json={"query": PEER_CHAT.query, "include_evidence": True},
        )

        assert response.json()["evidence"] == {
            "conclusions": [],
            "messages": [],
            "tool_calls": [],
            "reasoning_trace_id": None,
        }


class TestEvidenceSchemaContract:
    def test_evidence_is_documented_on_the_response_schema(self, client: TestClient):
        """The chat routes hand-inject their 200 schema, so it can drift."""
        schema = client.get("/openapi.json").json()
        path = "/v3/workspaces/{workspace_id}/peers/{peer_id}/chat"
        chat_schema = schema["paths"][path]["post"]["responses"]["200"]["content"][
            "application/json"
        ]["schema"]

        assert "evidence" in chat_schema["properties"]

    @pytest.mark.parametrize(
        "options_schema", ["DialecticOptions", "WorkspaceChatOptions"]
    )
    def test_the_request_schema_advertises_the_toggle(
        self, client: TestClient, options_schema: str
    ):
        schema = client.get("/openapi.json").json()
        properties = schema["components"]["schemas"][options_schema]["properties"]

        assert properties["include_evidence"]["default"] is False


def test_a_bad_request_still_fails_when_evidence_is_requested(
    client: TestClient, sample_data: tuple[Workspace, Peer]
):
    """Asking for evidence must not reorder the validation that runs first.

    The accumulator is built in the handler, so it must not be constructed
    ahead of the checks that reject the request outright.
    """
    workspace, _ = sample_data

    response = client.post(
        f"/v3/workspaces/{workspace.name}/peers/{generate_nanoid()}/chat",
        json={
            "query": "anything",
            "include_evidence": True,
            "response_format": {"type": "array"},
        },
    )

    assert response.status_code == 422
