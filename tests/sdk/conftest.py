import sys
from pathlib import Path

import httpx
import pytest
import pytest_asyncio
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


@pytest_asyncio.fixture
async def honcho_async_test_client(client: TestClient):
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
    # Warm the app via the TestClient transport before using ASGITransport.
    # This avoids running the same ASGI app concurrently across two event loops,
    # without mutating the Honcho client's local metadata/config caches.
    res = client.post("/v3/workspaces", json={"id": honcho_client.workspace_id})
    assert res.status_code in (200, 201)

    # Set up async HTTP client manually for the ASGI transport
    from sdks.python.src.honcho.http import AsyncHonchoHTTPClient

    async_httpx_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=client.app),
        base_url=str(client.base_url),
        headers=client.headers,
    )
    async_http = AsyncHonchoHTTPClient(
        base_url=str(client.base_url),
        api_key=None,
        http_client=async_httpx_client,
    )
    honcho_client._async_http = async_http  # pyright: ignore

    try:
        yield honcho_client
    finally:
        await async_httpx_client.aclose()


@pytest.fixture(params=["sync", "async"])
def client_fixture(
    request: pytest.FixtureRequest,
    honcho_sync_test_client: Honcho,
    honcho_async_test_client: Honcho,
) -> tuple[Honcho, str]:
    if request.param == "sync":
        return honcho_sync_test_client, "sync"
    return honcho_async_test_client, "async"
