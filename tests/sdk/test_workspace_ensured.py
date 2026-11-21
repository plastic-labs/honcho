"""Reads on a client marked as ensured must not create its workspace.

``honcho workspace replay --dry-run`` looks at a destination workspace without
creating it. To do that, ``honcho_cli.replay.read_only`` sets the SDK's private
``_workspace_ensured`` flag, which skips the get-or-create that every
workspace-scoped call otherwise runs first.

The CLI's own tests are not part of this suite, so this test pins the behaviour
the CLI depends on. If you rename the flag or change what it does, update
``honcho_cli.replay.read_only`` in the same change.
"""

import httpx
from fastapi.testclient import TestClient

from sdks.python.src.honcho.client import Honcho
from sdks.python.src.honcho.session import Session

WORKSPACE = "sdk-test-never-created"


def _exists(client: TestClient, workspace_id: str) -> bool:
    """Whether the server has this workspace, asked without the SDK."""
    response = client.post(
        "/v3/workspaces/list", json={"filters": {"id": workspace_id}}
    )
    assert response.status_code == 200
    return any(item["id"] == workspace_id for item in response.json()["items"])


def test_reads_on_an_ensured_client_do_not_create_the_workspace(
    client: TestClient,
) -> None:
    sdk = Honcho(
        workspace_id=WORKSPACE,
        base_url=str(client.base_url),
        http_client=httpx.Client(
            transport=client._transport,  # pyright: ignore
            base_url=str(client.base_url),
            headers=client.headers,
        ),
    )
    assert sdk._workspace_ensured is False  # pyright: ignore[reportPrivateUsage]

    # The reads a dry run makes on the destination.
    sdk._workspace_ensured = True  # pyright: ignore[reportPrivateUsage]
    assert sdk.sessions(size=1).total == 0
    assert list(sdk.peers(filters={"id": {"in": ["alice"]}})) == []
    assert list(Session("s1", sdk).messages()) == []
    assert not _exists(client, WORKSPACE)

    # Control: without the flag, the same read creates the workspace.
    sdk._workspace_ensured = False  # pyright: ignore[reportPrivateUsage]
    sdk.sessions(size=1)
    assert _exists(client, WORKSPACE)
