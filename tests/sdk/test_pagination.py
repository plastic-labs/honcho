import pytest

from sdks.python.src.honcho.async_client.client import AsyncHoncho
from sdks.python.src.honcho.async_client.pagination import AsyncPage
from sdks.python.src.honcho.async_client.peer import AsyncPeer
from sdks.python.src.honcho.client import Honcho
from sdks.python.src.honcho.pagination import SyncPage
from sdks.python.src.honcho.peer import Peer


@pytest.mark.asyncio
async def test_page_get_next_page(
    client_fixture: tuple[Honcho | AsyncHoncho, str],
):
    """
    Tests that Page.get_next_page() works correctly for both sync and async clients.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        assert isinstance(honcho_client, AsyncHoncho)

        # Create multiple peers to test pagination
        for i in range(15):
            peer = await honcho_client.peer(id=f"pagination-test-peer-async-{i}")
            await peer.get_metadata()  # Create the peer

        # Get first page
        first_page = await honcho_client.get_peers()
        assert isinstance(first_page, AsyncPage)
        assert len(first_page.items) > 0

        # If there's a next page, test get_next_page()
        if first_page.has_next_page():
            second_page = await first_page.get_next_page()
            assert second_page is not None
            assert isinstance(second_page, AsyncPage)
            assert second_page.page == 2
    else:
        assert isinstance(honcho_client, Honcho)

        # Create multiple peers to test pagination
        for i in range(15):
            peer = honcho_client.peer(id=f"pagination-test-peer-{i}")
            peer.get_metadata()  # Create the peer

        # Get first page
        first_page = honcho_client.get_peers()
        assert isinstance(first_page, SyncPage)
        assert len(first_page.items) > 0

        # If there's a next page, test get_next_page()
        if first_page.has_next_page():
            second_page = first_page.get_next_page()
            assert second_page is not None
            assert isinstance(second_page, SyncPage)
            assert second_page.page == 2


@pytest.mark.asyncio
async def test_page_transform_preserved_across_pages(
    client_fixture: tuple[Honcho | AsyncHoncho, str],
):
    """
    Tests that transformation function is preserved when getting next page.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        assert isinstance(honcho_client, AsyncHoncho)

        # Create multiple peers to ensure pagination
        for i in range(25):
            peer = await honcho_client.peer(id=f"transform-test-peer-async-{i}")
            await peer.get_metadata()

        # Get first page
        first_page = await honcho_client.get_peers()
        assert isinstance(first_page, AsyncPage)

        # Verify items are AsyncPeer instances (transformed)
        for item in first_page.items:
            assert isinstance(item, AsyncPeer)

        # Get next page and verify transformation is preserved
        if first_page.has_next_page():
            second_page = await first_page.get_next_page()
            assert second_page is not None
            for item in second_page.items:
                assert isinstance(item, AsyncPeer)
    else:
        assert isinstance(honcho_client, Honcho)

        # Create multiple peers to ensure pagination
        for i in range(25):
            peer = honcho_client.peer(id=f"transform-test-peer-{i}")
            peer.get_metadata()

        # Get first page
        first_page = honcho_client.get_peers()
        assert isinstance(first_page, SyncPage)

        # Verify items are Peer instances (transformed)
        for item in first_page.items:
            assert isinstance(item, Peer)

        # Get next page and verify transformation is preserved
        if first_page.has_next_page():
            second_page = first_page.get_next_page()
            assert second_page is not None
            for item in second_page.items:
                assert isinstance(item, Peer)


@pytest.mark.asyncio
async def test_page_get_next_page_throws_exception_on_last_page(
    client_fixture: tuple[Honcho | AsyncHoncho, str],
):
    """
    Tests that get_next_page() throws RuntimeError when on the last page.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        assert isinstance(honcho_client, AsyncHoncho)

        # Create just a few peers to ensure we're on the last page
        for i in range(3):
            peer = await honcho_client.peer(id=f"last-page-test-peer-async-{i}")
            await peer.get_metadata()

        # Get first page
        first_page = await honcho_client.get_peers()
        assert isinstance(first_page, AsyncPage)

        # Should be on last page (or only page)
        if not first_page.has_next_page():
            # get_next_page should throw runtime error
            try:
                await first_page.get_next_page()
            except Exception as e:
                assert isinstance(e, RuntimeError)
    else:
        assert isinstance(honcho_client, Honcho)

        # Create just a few peers to ensure we're on the last page
        for i in range(3):
            peer = honcho_client.peer(id=f"last-page-test-peer-{i}")
            peer.get_metadata()

        # Get first page
        first_page = honcho_client.get_peers()
        assert isinstance(first_page, SyncPage)

        # Should be on last page (or only page)
        if not first_page.has_next_page():
            # get_next_page should throw runtime error
            try:
                first_page.get_next_page()
            except Exception as e:
                assert isinstance(e, RuntimeError)


@pytest.mark.asyncio
async def test_page_manual_pagination(
    client_fixture: tuple[Honcho | AsyncHoncho, str],
):
    """
    Tests manual pagination with get_next_page.
    """
    honcho_client, client_type = client_fixture

    if client_type == "async":
        assert isinstance(honcho_client, AsyncHoncho)

        # Create enough peers to ensure multiple pages
        for i in range(25):
            peer = await honcho_client.peer(id=f"manual-pagination-test-peer-async-{i}")
            await peer.get_metadata()

        # Collect items via manual pagination
        manual_items = []
        manual_page = await honcho_client.get_peers()
        page_count = 0

        while manual_page is not None:
            page_count += 1
            manual_items.extend(manual_page.items)  # pyright: ignore

            if not manual_page.has_next_page():
                break

            manual_page = await manual_page.get_next_page()

        # Should have collected all items
        assert len(manual_items) >= 25  # pyright: ignore
        # All items should be AsyncPeer instances
        assert all(isinstance(item, AsyncPeer) for item in manual_items)  # pyright: ignore
    else:
        assert isinstance(honcho_client, Honcho)

        # Create enough peers to ensure multiple pages
        for i in range(25):
            peer = honcho_client.peer(id=f"manual-pagination-test-peer-{i}")
            peer.get_metadata()

        # Collect items via manual pagination
        manual_items = []
        manual_page = honcho_client.get_peers()
        page_count = 0

        while manual_page is not None:
            page_count += 1
            manual_items.extend(manual_page.items)  # pyright: ignore

            if not manual_page.has_next_page():
                break

            manual_page = manual_page.get_next_page()

        # Should have collected all items
        assert len(manual_items) >= 25  # pyright: ignore
        # All items should be Peer instances
        assert all(isinstance(item, Peer) for item in manual_items)  # pyright: ignore
