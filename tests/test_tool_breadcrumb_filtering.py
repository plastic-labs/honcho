"""Tests for ingestion-time filtering of tool-run breadcrumb messages."""

import pytest
from fastapi.testclient import TestClient

from src import models
from src.utils.message_filter import (
    filter_tool_run_breadcrumbs,
    is_tool_run_breadcrumb,
)

# ---------------------------------------------------------------------------
# Unit tests for the predicate / filter helper (no DB required)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "content",
    [
        "[Tool] Ran: Bash(ls -la)",
        "[Tool] Ran: Read(/etc/hosts)",
        "  [Tool] Ran: Edit(file.py)",  # leading whitespace still matches
        "\n[Tool] Ran: Grep(pattern)",
    ],
)
def test_breadcrumbs_are_detected(content: str):
    assert is_tool_run_breadcrumb(content) is True


@pytest.mark.parametrize(
    "content",
    [
        "",
        "Can you run the tests for me?",
        "I used the [Tool] Ran: line as an example in my message.",  # not a prefix
        "The tool ran successfully and printed the output.",
        "[Tool] output: something",  # different, non-breadcrumb tag
        "Here's what `[Tool] Ran:` means in Claude Code.",
    ],
)
def test_legitimate_content_is_kept(content: str):
    assert is_tool_run_breadcrumb(content) is False


def test_filter_preserves_order_and_drops_only_breadcrumbs():
    items = [
        {"content": "real one"},
        {"content": "[Tool] Ran: Bash(echo hi)"},
        {"content": "real two"},
        {"content": "[Tool] Ran: Read(x)"},
    ]
    kept = filter_tool_run_breadcrumbs(items, content_of=lambda m: m["content"])
    assert [m["content"] for m in kept] == ["real one", "real two"]


def test_filter_all_breadcrumbs_returns_empty():
    items = [
        {"content": "[Tool] Ran: a"},
        {"content": "[Tool] Ran: b"},
    ]
    assert filter_tool_run_breadcrumbs(items, content_of=lambda m: m["content"]) == []


# ---------------------------------------------------------------------------
# Integration test against the message-create route
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_messages_skips_tool_breadcrumbs(
    client: TestClient, sample_data: tuple[models.Workspace, models.Peer]
):
    """A mixed batch persists only the real messages; breadcrumbs are dropped."""
    test_workspace, test_peer = sample_data
    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions/breadcrumb-session/messages/",
        json={
            "messages": [
                {"content": "Real user message", "peer_id": test_peer.name},
                {"content": "[Tool] Ran: Bash(ls)", "peer_id": test_peer.name},
                {"content": "Another real message", "peer_id": test_peer.name},
            ]
        },
    )
    assert response.status_code in (200, 201)
    data = response.json()
    contents = [m["content"] for m in data]
    assert contents == ["Real user message", "Another real message"]


@pytest.mark.asyncio
async def test_create_messages_all_breadcrumbs_persists_nothing(
    client: TestClient, sample_data: tuple[models.Workspace, models.Peer]
):
    """A batch that is entirely breadcrumbs persists nothing and returns []."""
    test_workspace, test_peer = sample_data
    response = client.post(
        f"/v3/workspaces/{test_workspace.name}/sessions/all-breadcrumb-session/messages/",
        json={
            "messages": [
                {"content": "[Tool] Ran: Bash(ls)", "peer_id": test_peer.name},
                {"content": "[Tool] Ran: Read(x)", "peer_id": test_peer.name},
            ]
        },
    )
    assert response.status_code in (200, 201)
    assert response.json() == []
