import sys
from collections.abc import Generator
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

# Add the SDK src to the path to allow imports
sdk_src_path = Path(__file__).parent.parent.parent / "sdks" / "python" / "src"
sys.path.insert(0, str(sdk_src_path))

from sdks.python.src.honcho.client import Honcho  # noqa: E402
from sdks.python.src.honcho.peer import Peer  # noqa: E402
from sdks.python.src.honcho.session import Session, SessionPeerConfig  # noqa: E402


@pytest.fixture
def honcho_test_client(client: TestClient) -> Generator[Honcho, None, None]:
    """
    Returns a Honcho SDK client configured to talk to the test API.
    """
    http_client = httpx.Client(
        transport=client._transport,  # pyright: ignore
        base_url=str(client.base_url),
        headers=client.headers,
    )

    honcho_client = Honcho(workspace_id="sdk-test-workspace", http_client=http_client)
    yield honcho_client


def test_sdk_init_and_workspace_creation(
    honcho_test_client: Honcho, client: TestClient
):
    """
    Tests that the Honcho SDK can be initialized and that it creates a workspace.
    """
    assert honcho_test_client.workspace_id == "sdk-test-workspace"

    res = client.post("/v2/workspaces/list", json={})
    assert res.status_code == 200
    assert res.json()["items"][0]["id"] == "sdk-test-workspace"


def test_peer_operations(honcho_test_client: Honcho):
    """
    Tests creation and metadata operations for peers.
    """
    peers_page = honcho_test_client.get_peers()
    assert len(list(peers_page)) == 0

    peer = honcho_test_client.peer(id="test-peer-1")
    assert isinstance(peer, Peer)

    # Peer is created on first use
    metadata = peer.get_metadata()
    assert metadata == {}

    peers_page = honcho_test_client.get_peers()
    assert len(list(peers_page)) == 1

    peer.set_metadata({"foo": "bar"})
    metadata = peer.get_metadata()
    assert metadata == {"foo": "bar"}


def test_session_operations(honcho_test_client: Honcho):
    """
    Tests creation, peer management, and metadata for sessions.
    """
    sessions_page = honcho_test_client.get_sessions()
    assert len(list(sessions_page)) == 0

    session = honcho_test_client.session(id="test-session-1")
    assert isinstance(session, Session)

    # Session is created on first use
    metadata = session.get_metadata()
    assert metadata == {}

    sessions_page = honcho_test_client.get_sessions()
    assert len(list(sessions_page)) == 1

    session.set_metadata({"bar": "baz"})
    metadata = session.get_metadata()
    assert metadata == {"bar": "baz"}

    assistant = honcho_test_client.peer(id="assistant")
    user = honcho_test_client.peer(id="user")

    session.add_peers(
        [assistant, (user, SessionPeerConfig(observe_others=False, observe_me=False))]
    )

    session_peers = session.get_peers()
    assert len(session_peers) == 2


def test_message_and_chat_operations(honcho_test_client: Honcho):
    """
    Tests adding messages to a session and using the chat functionality.
    """
    session = honcho_test_client.session(id="test-chat-session")
    assistant = honcho_test_client.peer(id="chat-assistant")
    user = honcho_test_client.peer(id="chat-user")

    session.add_messages(
        [
            user.message("What is the capital of France?"),
            assistant.message("The capital of France is Paris."),
        ]
    )

    messages = session.get_messages()
    assert len(list(messages)) == 2

    # This is a mock response from the agent
    _response = user.chat("What did I ask about?")
