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

from sdks.python.src.honcho.async_client.client import AsyncHoncho  # noqa: E402
from sdks.python.src.honcho.client import Honcho  # noqa: E402


@pytest.fixture
def honcho_sync_test_client(client: TestClient) -> Honcho:
    """
    Returns a Honcho SDK client configured to talk to the test API.
    """
    http_client = httpx.Client(
        transport=client._transport,  # pyright: ignore
        base_url=str(client.base_url),
        headers=client.headers,
    )

    honcho_client = Honcho(
        workspace_id="sdk-test-workspace-sync", http_client=http_client
    )
    return honcho_client


@pytest_asyncio.fixture
async def honcho_async_test_client(
    client: TestClient,
) -> AsyncHoncho:
    """
    Returns an async Honcho SDK client configured to talk to the test API.
    """
    async_http_client = httpx.AsyncClient(
        transport=httpx.ASGITransport(app=client.app),
        base_url=str(client.base_url),
        headers=client.headers,
    )

    http_client = httpx.Client(
        transport=client._transport,  # pyright: ignore
        base_url=str(client.base_url),
        headers=client.headers,
    )

    honcho_client = AsyncHoncho(
        workspace_id="sdk-test-workspace-async",
        async_http_client=async_http_client,
        http_client=http_client,
    )
    return honcho_client


@pytest.fixture(params=["sync", "async"])
def client_fixture(
    request: pytest.FixtureRequest,
    honcho_sync_test_client: Honcho,
    honcho_async_test_client: AsyncHoncho,
) -> tuple[Honcho | AsyncHoncho, str]:
    if request.param == "sync":
        return honcho_sync_test_client, "sync"
    return honcho_async_test_client, "async"
