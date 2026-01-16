import sys
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

# Add the SDK src to the path to allow imports
sdk_src_path = Path(__file__).parent.parent.parent / "sdks" / "python" / "src"
sys.path.insert(0, str(sdk_src_path))

# This is a bit of a hack to make the main conftest discoverable
sys.path.insert(0, str(Path(__file__).parent.parent))

from sdks.python.src.honcho.client import Honcho  # noqa: E402


@pytest.fixture
def honcho_sync_test_client(client: TestClient) -> Honcho:
    """
    Returns a Honcho SDK client configured to talk to the test API.
    Uses sync operations directly.
    """
    http_client = httpx.Client(
        transport=client._transport,  # pyright: ignore
        base_url=str(client.base_url),
        headers=client.headers,
    )

    honcho_client = Honcho(
        workspace_id="sdk-test-workspace-sync",
        base_url=str(client.base_url),
        http_client=http_client,
    )
    return honcho_client


@pytest.fixture
def honcho_async_test_client(client: TestClient) -> Honcho:
    """
    Returns a Honcho SDK client configured to talk to the test API.
    Uses .aio accessor for async operations.
    """
    # For sync workspace creation
    http_client = httpx.Client(
        transport=client._transport,  # pyright: ignore
        base_url=str(client.base_url),
        headers=client.headers,
    )

    honcho_client = Honcho(
        workspace_id="sdk-test-workspace-async",
        base_url=str(client.base_url),
        http_client=http_client,
    )

    # Set up async HTTP client manually for the ASGI transport
    from sdks.python.src.honcho.http import AsyncHonchoHTTPClient

    async_http = AsyncHonchoHTTPClient(
        base_url=str(client.base_url),
        api_key=None,
        http_client=httpx.AsyncClient(
            transport=httpx.ASGITransport(app=client.app),
            base_url=str(client.base_url),
            headers=client.headers,
        ),
    )
    honcho_client._async_http = async_http  # pyright: ignore

    return honcho_client


@pytest.fixture(params=["sync", "async"])
def client_fixture(
    request: pytest.FixtureRequest,
    honcho_sync_test_client: Honcho,
    honcho_async_test_client: Honcho,
) -> tuple[Honcho, str]:
    if request.param == "sync":
        return honcho_sync_test_client, "sync"
    return honcho_async_test_client, "async"
