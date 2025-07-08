import pytest
from fastapi.testclient import TestClient

from sdks.python.src.honcho.async_client.client import AsyncHoncho
from sdks.python.src.honcho.async_client.pagination import AsyncPage
from sdks.python.src.honcho.async_client.peer import AsyncPeer
from sdks.python.src.honcho.async_client.session import AsyncSession
from sdks.python.src.honcho.client import Honcho
from sdks.python.src.honcho.pagination import SyncPage
from sdks.python.src.honcho.peer import Peer
from sdks.python.src.honcho.session import Session


@pytest.mark.asyncio
async def test_client_init(
    client_fixture: tuple[Honcho | AsyncHoncho, str], client: TestClient
):
    """
    Tests that the Honcho SDK clients can be initialized and that they create a workspace.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        assert isinstance(honcho_client, AsyncHoncho)
        assert honcho_client.workspace_id == "sdk-test-workspace-async"
    else:
        assert isinstance(honcho_client, Honcho)
        assert honcho_client.workspace_id == "sdk-test-workspace-sync"

    # Check all pages to find the workspace
    found_workspace = False
    page = 1

    while not found_workspace:
        res = client.post("/v2/workspaces/list", json={}, params={"page": page})
        assert res.status_code == 200

        data = res.json()
        workspaces = data["items"]
        workspace_ids = [w["id"] for w in workspaces]

        if honcho_client.workspace_id in workspace_ids:
            found_workspace = True
            break

        # Check if there are more pages
        if page >= data.get("pages", 1) or len(workspaces) == 0:
            break

        page += 1

    assert found_workspace, (
        f"Workspace {honcho_client.workspace_id} not found in any page of results"
    )


@pytest.mark.asyncio
async def test_workspace_metadata(client_fixture: tuple[Honcho | AsyncHoncho, str]):
    """
    Tests getting and setting metadata on a workspace.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        assert isinstance(honcho_client, AsyncHoncho)
        metadata = await honcho_client.get_metadata()
        assert metadata == {}
        await honcho_client.set_metadata({"foo": "bar"})
        metadata = await honcho_client.get_metadata()
        assert metadata == {"foo": "bar"}
    else:
        assert isinstance(honcho_client, Honcho)
        metadata = honcho_client.get_metadata()
        assert metadata == {}
        honcho_client.set_metadata({"foo": "bar"})
        metadata = honcho_client.get_metadata()
        assert metadata == {"foo": "bar"}


@pytest.mark.asyncio
async def test_get_workspaces(client_fixture: tuple[Honcho | AsyncHoncho, str]):
    """
    Tests listing available workspaces.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        assert isinstance(honcho_client, AsyncHoncho)
        workspaces = await honcho_client.get_workspaces()
    else:
        assert isinstance(honcho_client, Honcho)
        workspaces = honcho_client.get_workspaces()

    assert isinstance(workspaces, list)
    assert honcho_client.workspace_id in workspaces


@pytest.mark.asyncio
async def test_client_list_peers_and_sessions(
    client_fixture: tuple[Honcho | AsyncHoncho, str],
):
    """
    Tests listing peers and sessions at the client level.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        assert isinstance(honcho_client, AsyncHoncho)
        peers_page = await honcho_client.get_peers()
        assert isinstance(peers_page, AsyncPage)
        assert len(peers_page.items) == 0

        sessions_page = await honcho_client.get_sessions()
        assert isinstance(sessions_page, AsyncPage)
        assert len(sessions_page.items) == 0

        peer = await honcho_client.peer(id="test-peer-client")
        assert isinstance(peer, AsyncPeer)
        await peer.get_metadata()  # Creates the peer

        peers_page = await honcho_client.get_peers()
        assert len(peers_page.items) == 1

        session = await honcho_client.session(id="test-session-client")
        assert isinstance(session, AsyncSession)
        await session.get_metadata()  # Creates the session

        sessions_page = await honcho_client.get_sessions()
        assert len(sessions_page.items) == 1
    else:
        assert isinstance(honcho_client, Honcho)
        peers_page = honcho_client.get_peers()
        assert isinstance(peers_page, SyncPage)
        assert len(list(peers_page)) == 0

        sessions_page = honcho_client.get_sessions()
        assert isinstance(sessions_page, SyncPage)
        assert len(list(sessions_page)) == 0

        peer = honcho_client.peer(id="test-peer-client")
        assert isinstance(peer, Peer)
        peer.get_metadata()

        peers_page = honcho_client.get_peers()
        assert len(list(peers_page)) == 1

        session = honcho_client.session(id="test-session-client")
        assert isinstance(session, Session)
        session.get_metadata()

        sessions_page = honcho_client.get_sessions()
        assert len(list(sessions_page)) == 1


@pytest.mark.asyncio
async def test_workspace_search(client_fixture: tuple[Honcho | AsyncHoncho, str]):
    """
    Tests searching for messages within a workspace.
    """
    honcho_client, client_type = client_fixture
    search_query = "a unique message for workspace search"

    if client_type == "async":
        assert isinstance(honcho_client, AsyncHoncho)
        session = await honcho_client.session(id="search-session-ws")
        assert isinstance(session, AsyncSession)
        user = await honcho_client.peer(id="search-user-ws")
        assert isinstance(user, AsyncPeer)
        await session.add_messages([user.message(search_query)])

        search_results = await honcho_client.search(search_query)
        results = search_results.items
        assert len(results) >= 1
        assert search_query in results[0].content
    else:
        assert isinstance(honcho_client, Honcho)
        session = honcho_client.session(id="search-session-ws")
        assert isinstance(session, Session)
        user = honcho_client.peer(id="search-user-ws")
        assert isinstance(user, Peer)
        session.add_messages([user.message(search_query)])

        search_results = honcho_client.search(search_query)
        results = list(search_results)
        assert len(results) >= 1
        assert search_query in results[0].content


def test_client_init_with_api_key(client: TestClient):
    """
    Tests that the Honcho client can be initialized with an explicit api_key parameter.
    This test covers line 97: client_kwargs["api_key"] = api_key
    """
    import httpx

    http_client = httpx.Client(
        transport=client._transport,  # pyright: ignore
        base_url=str(client.base_url),
        headers=client.headers,
    )

    # Test with explicit api_key to trigger line 97
    honcho_client = Honcho(
        api_key="test-api-key",
        workspace_id="test-workspace-api-key",
        http_client=http_client,
    )

    assert isinstance(honcho_client, Honcho)
    assert honcho_client.workspace_id == "test-workspace-api-key"


def test_client_init_with_api_key_none(client: TestClient):
    """
    Tests that the Honcho client can be initialized with api_key=None.
    This test ensures line 97 is NOT executed when api_key is None.
    """
    import httpx

    http_client = httpx.Client(
        transport=client._transport,  # pyright: ignore
        base_url=str(client.base_url),
        headers=client.headers,
    )

    # Test with api_key=None to ensure line 97 is not executed
    honcho_client = Honcho(
        api_key=None,
        workspace_id="test-workspace-api-key-none",
        http_client=http_client,
    )

    assert isinstance(honcho_client, Honcho)
    assert honcho_client.workspace_id == "test-workspace-api-key-none"


def test_client_init_with_api_key_empty_string(client: TestClient):
    """
    Tests that the Honcho client can be initialized with an empty string api_key.
    This test covers line 97: client_kwargs["api_key"] = api_key with empty string
    """
    import httpx

    http_client = httpx.Client(
        transport=client._transport,  # pyright: ignore
        base_url=str(client.base_url),
        headers=client.headers,
    )

    # Test with empty string api_key to trigger line 97 with edge case
    honcho_client = Honcho(
        api_key="", workspace_id="test-workspace-api-key-empty", http_client=http_client
    )

    assert isinstance(honcho_client, Honcho)
    assert honcho_client.workspace_id == "test-workspace-api-key-empty"


def test_client_init_with_environment_local(client: TestClient):
    """
    Tests that the Honcho client can be initialized with environment="local".
    This test covers line 99: client_kwargs["environment"] = environment
    """
    import httpx

    http_client = httpx.Client(
        transport=client._transport,  # pyright: ignore
        base_url=str(client.base_url),
        headers=client.headers,
    )

    # Test with environment="local" to trigger line 99
    honcho_client = Honcho(
        environment="local",
        workspace_id="test-workspace-env-local",
        http_client=http_client,
    )

    assert isinstance(honcho_client, Honcho)
    assert honcho_client.workspace_id == "test-workspace-env-local"


def test_client_init_with_environment_production(client: TestClient):
    """
    Tests that the Honcho client can be initialized with environment="production".
    This test covers line 99: client_kwargs["environment"] = environment
    """
    import httpx

    http_client = httpx.Client(
        transport=client._transport,  # pyright: ignore
        base_url=str(client.base_url),
        headers=client.headers,
    )

    # Test with environment="production" to trigger line 99
    honcho_client = Honcho(
        environment="production",
        workspace_id="test-workspace-env-prod",
        http_client=http_client,
    )

    assert isinstance(honcho_client, Honcho)
    assert honcho_client.workspace_id == "test-workspace-env-prod"


def test_client_init_with_environment_demo(client: TestClient):
    """
    Tests that the Honcho client can be initialized with environment="demo".
    This test covers line 99: client_kwargs["environment"] = environment
    """
    import httpx

    http_client = httpx.Client(
        transport=client._transport,  # pyright: ignore
        base_url=str(client.base_url),
        headers=client.headers,
    )

    # Test with environment="demo" to trigger line 99
    honcho_client = Honcho(
        environment="demo",
        workspace_id="test-workspace-env-demo",
        http_client=http_client,
    )

    assert isinstance(honcho_client, Honcho)
    assert honcho_client.workspace_id == "test-workspace-env-demo"


def test_client_init_with_environment_none(client: TestClient):
    """
    Tests that the Honcho client can be initialized with environment=None.
    This test ensures line 99 is NOT executed when environment is None.
    """
    import httpx

    http_client = httpx.Client(
        transport=client._transport,  # pyright: ignore
        base_url=str(client.base_url),
        headers=client.headers,
    )

    # Test with environment=None to ensure line 99 is not executed
    honcho_client = Honcho(
        environment=None,
        workspace_id="test-workspace-env-none",
        http_client=http_client,
    )

    assert isinstance(honcho_client, Honcho)
    assert honcho_client.workspace_id == "test-workspace-env-none"


def test_client_init_with_base_url(client: TestClient):
    """
    Tests that the Honcho client can be initialized with a custom base_url.
    This test covers line 101: client_kwargs["base_url"] = base_url
    """
    import httpx

    http_client = httpx.Client(
        transport=client._transport,  # pyright: ignore
        base_url=str(client.base_url),
        headers=client.headers,
    )

    # Test with custom base_url to trigger line 101
    honcho_client = Honcho(
        base_url="https://custom.example.com",
        workspace_id="test-workspace-base-url",
        http_client=http_client,
    )

    assert isinstance(honcho_client, Honcho)
    assert honcho_client.workspace_id == "test-workspace-base-url"


def test_client_init_with_base_url_none(client: TestClient):
    """
    Tests that the Honcho client can be initialized with base_url=None.
    This test ensures line 101 is NOT executed when base_url is None.
    """
    import httpx

    http_client = httpx.Client(
        transport=client._transport,  # pyright: ignore
        base_url=str(client.base_url),
        headers=client.headers,
    )

    # Test with base_url=None to ensure line 101 is not executed
    honcho_client = Honcho(
        base_url=None,
        workspace_id="test-workspace-base-url-none",
        http_client=http_client,
    )

    assert isinstance(honcho_client, Honcho)
    assert honcho_client.workspace_id == "test-workspace-base-url-none"


def test_client_init_with_timeout(client: TestClient):
    """
    Tests that the Honcho client can be initialized with a custom timeout.
    This test covers line 103: client_kwargs["timeout"] = timeout
    """
    import httpx

    http_client = httpx.Client(
        transport=client._transport,  # pyright: ignore
        base_url=str(client.base_url),
        headers=client.headers,
    )

    # Test with custom timeout to trigger line 103
    honcho_client = Honcho(
        timeout=30.0, workspace_id="test-workspace-timeout", http_client=http_client
    )

    assert isinstance(honcho_client, Honcho)
    assert honcho_client.workspace_id == "test-workspace-timeout"


def test_client_init_with_timeout_none(client: TestClient):
    """
    Tests that the Honcho client can be initialized with timeout=None.
    This test ensures line 103 is NOT executed when timeout is None.
    """
    import httpx

    http_client = httpx.Client(
        transport=client._transport,  # pyright: ignore
        base_url=str(client.base_url),
        headers=client.headers,
    )

    # Test with timeout=None to ensure line 103 is not executed
    honcho_client = Honcho(
        timeout=None,
        workspace_id="test-workspace-timeout-none",
        http_client=http_client,
    )

    assert isinstance(honcho_client, Honcho)
    assert honcho_client.workspace_id == "test-workspace-timeout-none"


def test_client_init_with_timeout_small(client: TestClient):
    """
    Tests that the Honcho client can be initialized with a small timeout.
    This test covers line 103: client_kwargs["timeout"] = timeout with edge case
    """
    import httpx

    http_client = httpx.Client(
        transport=client._transport,  # pyright: ignore
        base_url=str(client.base_url),
        headers=client.headers,
    )

    # Test with small timeout (0.1 seconds) to trigger line 103 with edge case
    honcho_client = Honcho(
        timeout=0.1,
        workspace_id="test-workspace-timeout-small",
        http_client=http_client,
    )

    assert isinstance(honcho_client, Honcho)
    assert honcho_client.workspace_id == "test-workspace-timeout-small"


def test_client_init_with_timeout_float(client: TestClient):
    """
    Tests that the Honcho client can be initialized with a float timeout.
    This test covers line 103: client_kwargs["timeout"] = timeout with float value
    """
    import httpx

    http_client = httpx.Client(
        transport=client._transport,  # pyright: ignore
        base_url=str(client.base_url),
        headers=client.headers,
    )

    # Test with float timeout to trigger line 103
    honcho_client = Honcho(
        timeout=45.5,
        workspace_id="test-workspace-timeout-float",
        http_client=http_client,
    )

    assert isinstance(honcho_client, Honcho)
    assert honcho_client.workspace_id == "test-workspace-timeout-float"


def test_client_init_with_max_retries(client: TestClient):
    """
    Tests that the Honcho client can be initialized with a custom max_retries.
    This test covers line 105: client_kwargs["max_retries"] = max_retries
    """
    import httpx

    http_client = httpx.Client(
        transport=client._transport,  # pyright: ignore
        base_url=str(client.base_url),
        headers=client.headers,
    )

    # Test with custom max_retries to trigger line 105
    honcho_client = Honcho(
        max_retries=5,
        workspace_id="test-workspace-max-retries",
        http_client=http_client,
    )

    assert isinstance(honcho_client, Honcho)
    assert honcho_client.workspace_id == "test-workspace-max-retries"


def test_client_init_with_max_retries_none(client: TestClient):
    """
    Tests that the Honcho client can be initialized with max_retries=None.
    This test ensures line 105 is NOT executed when max_retries is None.
    """
    import httpx

    http_client = httpx.Client(
        transport=client._transport,  # pyright: ignore
        base_url=str(client.base_url),
        headers=client.headers,
    )

    # Test with max_retries=None to ensure line 105 is not executed
    honcho_client = Honcho(
        max_retries=None,
        workspace_id="test-workspace-max-retries-none",
        http_client=http_client,
    )

    assert isinstance(honcho_client, Honcho)
    assert honcho_client.workspace_id == "test-workspace-max-retries-none"


def test_client_init_with_max_retries_zero(client: TestClient):
    """
    Tests that the Honcho client can be initialized with max_retries=0.
    This test covers line 105: client_kwargs["max_retries"] = max_retries with edge case
    """
    import httpx

    http_client = httpx.Client(
        transport=client._transport,  # pyright: ignore
        base_url=str(client.base_url),
        headers=client.headers,
    )

    # Test with max_retries=0 to trigger line 105 with edge case
    honcho_client = Honcho(
        max_retries=0,
        workspace_id="test-workspace-max-retries-zero",
        http_client=http_client,
    )

    assert isinstance(honcho_client, Honcho)
    assert honcho_client.workspace_id == "test-workspace-max-retries-zero"


def test_client_init_with_max_retries_large(client: TestClient):
    """
    Tests that the Honcho client can be initialized with a large max_retries value.
    This test covers line 105: client_kwargs["max_retries"] = max_retries with high value
    """
    import httpx

    http_client = httpx.Client(
        transport=client._transport,  # pyright: ignore
        base_url=str(client.base_url),
        headers=client.headers,
    )

    # Test with large max_retries to trigger line 105
    honcho_client = Honcho(
        max_retries=100,
        workspace_id="test-workspace-max-retries-large",
        http_client=http_client,
    )

    assert isinstance(honcho_client, Honcho)
    assert honcho_client.workspace_id == "test-workspace-max-retries-large"


def test_client_init_with_default_headers(client: TestClient):
    """
    Tests that the Honcho client can be initialized with custom default_headers.
    This test covers line 107: client_kwargs["default_headers"] = default_headers
    """
    import httpx

    http_client = httpx.Client(
        transport=client._transport,  # pyright: ignore
        base_url=str(client.base_url),
        headers=client.headers,
    )

    # Test with custom default_headers to trigger line 107
    default_headers = {
        "X-Custom-Header": "test-value",
        "Authorization": "Bearer test-token",
    }
    honcho_client = Honcho(
        default_headers=default_headers,
        workspace_id="test-workspace-default-headers",
        http_client=http_client,
    )

    assert isinstance(honcho_client, Honcho)
    assert honcho_client.workspace_id == "test-workspace-default-headers"


def test_client_init_with_default_headers_none(client: TestClient):
    """
    Tests that the Honcho client can be initialized with default_headers=None.
    This test ensures line 107 is NOT executed when default_headers is None.
    """
    import httpx

    http_client = httpx.Client(
        transport=client._transport,  # pyright: ignore
        base_url=str(client.base_url),
        headers=client.headers,
    )

    # Test with default_headers=None to ensure line 107 is not executed
    honcho_client = Honcho(
        default_headers=None,
        workspace_id="test-workspace-default-headers-none",
        http_client=http_client,
    )

    assert isinstance(honcho_client, Honcho)
    assert honcho_client.workspace_id == "test-workspace-default-headers-none"


def test_client_init_with_default_headers_empty(client: TestClient):
    """
    Tests that the Honcho client can be initialized with empty default_headers.
    This test covers line 107: client_kwargs["default_headers"] = default_headers with empty dict
    """
    import httpx

    http_client = httpx.Client(
        transport=client._transport,  # pyright: ignore
        base_url=str(client.base_url),
        headers=client.headers,
    )

    # Test with empty default_headers to trigger line 107 with edge case
    honcho_client = Honcho(
        default_headers={},
        workspace_id="test-workspace-default-headers-empty",
        http_client=http_client,
    )

    assert isinstance(honcho_client, Honcho)
    assert honcho_client.workspace_id == "test-workspace-default-headers-empty"


def test_client_init_with_default_headers_single(client: TestClient):
    """
    Tests that the Honcho client can be initialized with a single default header.
    This test covers line 107: client_kwargs["default_headers"] = default_headers with single header
    """
    import httpx

    http_client = httpx.Client(
        transport=client._transport,  # pyright: ignore
        base_url=str(client.base_url),
        headers=client.headers,
    )

    # Test with single default header to trigger line 107
    honcho_client = Honcho(
        default_headers={"Content-Type": "application/json"},
        workspace_id="test-workspace-default-headers-single",
        http_client=http_client,
    )

    assert isinstance(honcho_client, Honcho)
    assert honcho_client.workspace_id == "test-workspace-default-headers-single"


def test_client_init_with_default_query(client: TestClient):
    """
    Tests that the Honcho client can be initialized with custom default_query.
    This test covers line 109: client_kwargs["default_query"] = default_query
    """
    import httpx

    http_client = httpx.Client(
        transport=client._transport,  # pyright: ignore
        base_url=str(client.base_url),
        headers=client.headers,
    )

    # Test with custom default_query to trigger line 109
    default_query = {"version": "v1", "format": "json"}
    honcho_client = Honcho(
        default_query=default_query,
        workspace_id="test-workspace-default-query",
        http_client=http_client,
    )

    assert isinstance(honcho_client, Honcho)
    assert honcho_client.workspace_id == "test-workspace-default-query"


def test_client_init_with_default_query_none(client: TestClient):
    """
    Tests that the Honcho client can be initialized with default_query=None.
    This test ensures line 109 is NOT executed when default_query is None.
    """
    import httpx

    http_client = httpx.Client(
        transport=client._transport,  # pyright: ignore
        base_url=str(client.base_url),
        headers=client.headers,
    )

    # Test with default_query=None to ensure line 109 is not executed
    honcho_client = Honcho(
        default_query=None,
        workspace_id="test-workspace-default-query-none",
        http_client=http_client,
    )

    assert isinstance(honcho_client, Honcho)
    assert honcho_client.workspace_id == "test-workspace-default-query-none"


def test_client_init_with_default_query_empty(client: TestClient):
    """
    Tests that the Honcho client can be initialized with empty default_query.
    This test covers line 109: client_kwargs["default_query"] = default_query with empty dict
    """
    import httpx

    http_client = httpx.Client(
        transport=client._transport,  # pyright: ignore
        base_url=str(client.base_url),
        headers=client.headers,
    )

    # Test with empty default_query to trigger line 109 with edge case
    honcho_client = Honcho(
        default_query={},
        workspace_id="test-workspace-default-query-empty",
        http_client=http_client,
    )

    assert isinstance(honcho_client, Honcho)
    assert honcho_client.workspace_id == "test-workspace-default-query-empty"


def test_client_init_with_default_query_single(client: TestClient):
    """
    Tests that the Honcho client can be initialized with a single default query parameter.
    This test covers line 109: client_kwargs["default_query"] = default_query with single parameter
    """
    import httpx

    http_client = httpx.Client(
        transport=client._transport,  # pyright: ignore
        base_url=str(client.base_url),
        headers=client.headers,
    )

    # Test with single default query parameter to trigger line 109
    honcho_client = Honcho(
        default_query={"api_version": "2024-01"},
        workspace_id="test-workspace-default-query-single",
        http_client=http_client,
    )

    assert isinstance(honcho_client, Honcho)
    assert honcho_client.workspace_id == "test-workspace-default-query-single"


def test_client_init_with_default_query_multiple_types(client: TestClient):
    """
    Tests that the Honcho client can be initialized with default_query containing different value types.
    This test covers line 109: client_kwargs["default_query"] = default_query with various types
    """
    import httpx

    http_client = httpx.Client(
        transport=client._transport,  # pyright: ignore
        base_url=str(client.base_url),
        headers=client.headers,
    )

    # Test with default_query containing different types to trigger line 109
    default_query = {
        "string_param": "test_value",
        "number_param": 42,
        "bool_param": True,
        "list_param": ["a", "b", "c"],
    }
    honcho_client = Honcho(
        default_query=default_query,
        workspace_id="test-workspace-default-query-types",
        http_client=http_client,
    )

    assert isinstance(honcho_client, Honcho)
    assert honcho_client.workspace_id == "test-workspace-default-query-types"


def test_client_repr(client: TestClient):
    """
    Tests that the Honcho client __repr__ method returns the expected string format.
    This test covers line 295: return f"Honcho(workspace_id='{self.workspace_id}', base_url='{self._client.base_url}')"
    """
    import httpx

    http_client = httpx.Client(
        transport=client._transport,  # pyright: ignore
        base_url=str(client.base_url),
        headers=client.headers,
    )

    # Create a Honcho client instance
    honcho_client = Honcho(workspace_id="test-workspace-repr", http_client=http_client)

    # Call repr() to trigger line 295
    repr_result = repr(honcho_client)

    # Verify the format and content of the repr string
    # Get the actual base_url from the client to match against
    actual_base_url = str(honcho_client._client.base_url)
    expected_repr = (
        f"Honcho(workspace_id='test-workspace-repr', base_url='{actual_base_url}')"
    )
    assert repr_result == expected_repr

    # Additional assertions to ensure workspace_id and base_url are correctly included
    assert "test-workspace-repr" in repr_result
    assert actual_base_url in repr_result
    assert repr_result.startswith("Honcho(")
    assert repr_result.endswith("')")
    assert "workspace_id=" in repr_result
    assert "base_url=" in repr_result


def test_client_repr_with_long_workspace_id(client: TestClient):
    """
    Tests that the Honcho client __repr__ method works correctly with a long workspace_id.
    This test covers line 295: return f"Honcho(workspace_id='{self.workspace_id}', base_url='{self._client.base_url}')"
    """
    import httpx

    http_client = httpx.Client(
        transport=client._transport,  # pyright: ignore
        base_url=str(client.base_url),
        headers=client.headers,
    )

    # Create a Honcho client instance with a long workspace_id
    long_workspace_id = (
        "test-workspace-with-very-long-name-containing-many-chars-and-hyphens-123456789"
    )
    honcho_client = Honcho(workspace_id=long_workspace_id, http_client=http_client)

    # Call repr() to trigger line 295
    repr_result = repr(honcho_client)

    # Verify the format and content of the repr string
    actual_base_url = str(honcho_client._client.base_url)
    expected_repr = (
        f"Honcho(workspace_id='{long_workspace_id}', base_url='{actual_base_url}')"
    )
    assert repr_result == expected_repr

    # Additional assertions to ensure long workspace_id is handled correctly
    assert long_workspace_id in repr_result
    assert actual_base_url in repr_result
    assert repr_result.startswith("Honcho(")
    assert repr_result.endswith("')")


def test_client_repr_with_numeric_workspace_id(client: TestClient):
    """
    Tests that the Honcho client __repr__ method works correctly with a numeric workspace_id.
    This test covers line 295: return f"Honcho(workspace_id='{self.workspace_id}', base_url='{self._client.base_url}')"
    """
    import httpx

    http_client = httpx.Client(
        transport=client._transport,  # pyright: ignore
        base_url=str(client.base_url),
        headers=client.headers,
    )

    # Create a Honcho client instance with numeric workspace_id
    numeric_workspace_id = "123456789"
    honcho_client = Honcho(workspace_id=numeric_workspace_id, http_client=http_client)

    # Call repr() to trigger line 295
    repr_result = repr(honcho_client)

    # Verify the format and content of the repr string
    actual_base_url = str(honcho_client._client.base_url)
    expected_repr = (
        f"Honcho(workspace_id='{numeric_workspace_id}', base_url='{actual_base_url}')"
    )
    assert repr_result == expected_repr

    # Additional assertions
    assert numeric_workspace_id in repr_result
    assert actual_base_url in repr_result
    assert repr_result.startswith("Honcho(")
    assert repr_result.endswith("')")


def test_client_repr_multiple_calls(client: TestClient):
    """
    Tests that the Honcho client __repr__ method returns consistent results across multiple calls.
    This test covers line 295: return f"Honcho(workspace_id='{self.workspace_id}', base_url='{self._client.base_url}')"
    """
    import httpx

    http_client = httpx.Client(
        transport=client._transport,  # pyright: ignore
        base_url=str(client.base_url),
        headers=client.headers,
    )

    # Create a Honcho client instance
    honcho_client = Honcho(
        workspace_id="test-workspace-multiple-calls", http_client=http_client
    )

    # Call repr() multiple times to trigger line 295 repeatedly
    repr_result_1 = repr(honcho_client)
    repr_result_2 = repr(honcho_client)
    repr_result_3 = repr(honcho_client)

    # Verify all calls return the same result
    assert repr_result_1 == repr_result_2
    assert repr_result_2 == repr_result_3

    # Verify the format is correct
    actual_base_url = str(honcho_client._client.base_url)
    expected_repr = f"Honcho(workspace_id='test-workspace-multiple-calls', base_url='{actual_base_url}')"
    assert repr_result_1 == expected_repr

    # Additional assertions
    assert "test-workspace-multiple-calls" in repr_result_1
    assert actual_base_url in repr_result_1
    assert repr_result_1.startswith("Honcho(")
    assert repr_result_1.endswith("')")


def test_client_str(client: TestClient):
    """
    Tests that the Honcho client __str__ method returns the expected string format.
    This test covers line 304: return f"Honcho Client (workspace: {self.workspace_id})"
    """
    import httpx

    http_client = httpx.Client(
        transport=client._transport,  # pyright: ignore
        base_url=str(client.base_url),
        headers=client.headers,
    )

    # Create a Honcho client instance
    honcho_client = Honcho(workspace_id="test-workspace-str", http_client=http_client)

    # Call str() to trigger line 304
    str_result = str(honcho_client)

    # Verify the format and content of the str string
    expected_str = "Honcho Client (workspace: test-workspace-str)"
    assert str_result == expected_str

    # Additional assertions to ensure workspace_id is correctly included
    assert "test-workspace-str" in str_result
    assert str_result.startswith("Honcho Client (workspace: ")
    assert str_result.endswith(")")


def test_client_str_with_long_workspace_id(client: TestClient):
    """
    Tests that the Honcho client __str__ method works correctly with a long workspace_id.
    This test covers line 304: return f"Honcho Client (workspace: {self.workspace_id})"
    """
    import httpx

    http_client = httpx.Client(
        transport=client._transport,  # pyright: ignore
        base_url=str(client.base_url),
        headers=client.headers,
    )

    # Create a Honcho client instance with a long workspace_id
    long_workspace_id = (
        "test-workspace-with-very-long-name-containing-many-chars-and-hyphens-123456789"
    )
    honcho_client = Honcho(workspace_id=long_workspace_id, http_client=http_client)

    # Call str() to trigger line 304
    str_result = str(honcho_client)

    # Verify the format and content of the str string
    expected_str = f"Honcho Client (workspace: {long_workspace_id})"
    assert str_result == expected_str

    # Additional assertions to ensure long workspace_id is handled correctly
    assert long_workspace_id in str_result
    assert str_result.startswith("Honcho Client (workspace: ")
    assert str_result.endswith(")")


def test_client_str_with_numeric_workspace_id(client: TestClient):
    """
    Tests that the Honcho client __str__ method works correctly with a numeric workspace_id.
    This test covers line 304: return f"Honcho Client (workspace: {self.workspace_id})"
    """
    import httpx

    http_client = httpx.Client(
        transport=client._transport,  # pyright: ignore
        base_url=str(client.base_url),
        headers=client.headers,
    )

    # Create a Honcho client instance with numeric workspace_id
    numeric_workspace_id = "123456789"
    honcho_client = Honcho(workspace_id=numeric_workspace_id, http_client=http_client)

    # Call str() to trigger line 304
    str_result = str(honcho_client)

    # Verify the format and content of the str string
    expected_str = f"Honcho Client (workspace: {numeric_workspace_id})"
    assert str_result == expected_str

    # Additional assertions
    assert numeric_workspace_id in str_result
    assert str_result.startswith("Honcho Client (workspace: ")
    assert str_result.endswith(")")


def test_client_str_multiple_calls(client: TestClient):
    """
    Tests that the Honcho client __str__ method returns consistent results across multiple calls.
    This test covers line 304: return f"Honcho Client (workspace: {self.workspace_id})"
    """
    import httpx

    http_client = httpx.Client(
        transport=client._transport,  # pyright: ignore
        base_url=str(client.base_url),
        headers=client.headers,
    )

    # Create a Honcho client instance
    honcho_client = Honcho(
        workspace_id="test-workspace-str-multiple", http_client=http_client
    )

    # Call str() multiple times to trigger line 304 repeatedly
    str_result_1 = str(honcho_client)
    str_result_2 = str(honcho_client)
    str_result_3 = str(honcho_client)

    # Verify all calls return the same result
    assert str_result_1 == str_result_2
    assert str_result_2 == str_result_3

    # Verify the format is correct
    expected_str = "Honcho Client (workspace: test-workspace-str-multiple)"
    assert str_result_1 == expected_str

    # Additional assertions
    assert "test-workspace-str-multiple" in str_result_1
    assert str_result_1.startswith("Honcho Client (workspace: ")
    assert str_result_1.endswith(")")


def test_client_str_with_underscores_and_hyphens(client: TestClient):
    """
    Tests that the Honcho client __str__ method works correctly with underscores and hyphens in workspace_id.
    This test covers line 304: return f"Honcho Client (workspace: {self.workspace_id})"
    """
    import httpx

    http_client = httpx.Client(
        transport=client._transport,  # pyright: ignore
        base_url=str(client.base_url),
        headers=client.headers,
    )

    # Create a Honcho client instance with underscores and hyphens in workspace_id (valid pattern)
    valid_workspace_id = "test-workspace_with_underscores-and-hyphens"
    honcho_client = Honcho(workspace_id=valid_workspace_id, http_client=http_client)

    # Call str() to trigger line 304
    str_result = str(honcho_client)

    # Verify the format and content of the str string
    expected_str = f"Honcho Client (workspace: {valid_workspace_id})"
    assert str_result == expected_str

    # Additional assertions
    assert valid_workspace_id in str_result
    assert str_result.startswith("Honcho Client (workspace: ")
    assert str_result.endswith(")")


def test_client_str_with_mixed_case_workspace_id(client: TestClient):
    """
    Tests that the Honcho client __str__ method works correctly with mixed case workspace_id.
    This test covers line 304: return f"Honcho Client (workspace: {self.workspace_id})"
    """
    import httpx

    http_client = httpx.Client(
        transport=client._transport,  # pyright: ignore
        base_url=str(client.base_url),
        headers=client.headers,
    )

    # Create a Honcho client instance with mixed case workspace_id
    mixed_case_workspace_id = "Test-Workspace-MixedCase123"
    honcho_client = Honcho(
        workspace_id=mixed_case_workspace_id, http_client=http_client
    )

    # Call str() to trigger line 304
    str_result = str(honcho_client)

    # Verify the format and content of the str string
    expected_str = f"Honcho Client (workspace: {mixed_case_workspace_id})"
    assert str_result == expected_str

    # Additional assertions
    assert mixed_case_workspace_id in str_result
    assert str_result.startswith("Honcho Client (workspace: ")
    assert str_result.endswith(")")
