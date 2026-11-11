"""SDK methods for Representation History: `dreams()` and `get_dream()`."""

from nanoid import generate as generate_nanoid
from sqlalchemy.ext.asyncio import AsyncSession

from sdks.python.src.honcho.api_types import DreamRunDetailResponse, DreamRunResponse
from sdks.python.src.honcho.client import Honcho
from sdks.python.src.honcho.http import NotFoundError
from src import crud


async def test_list_and_get_dreams(
    client_fixture: tuple[Honcho, str], db_session: AsyncSession
):
    honcho_client, client_type = client_fixture
    if client_type == "async":
        maya = await honcho_client.aio.peer(id="dreams-maya")
    else:
        maya = honcho_client.peer(id="dreams-maya")
    workspace_name = honcho_client.workspace_id
    run = await crud.open_run(
        run_id=generate_nanoid(),
        workspace_name=workspace_name,
        observer=maya.id,
        observed=maya.id,
        dream_type="omni",
        trigger_reason="manual",
    )
    assert run is not None
    await crud.set_peer_card(
        db_session,
        workspace_name,
        ["IDENTITY: Name: Maya"],
        observer=maya.id,
        observed=maya.id,
        history=crud.DreamHistoryTarget(run_id=run.run_id, specialist="deduction"),
    )
    await crud.close_run(run.run_id, succeeded=True)

    if client_type == "async":
        runs = (await honcho_client.aio.dreams(observed=maya)).items
        detail = await honcho_client.aio.get_dream(run.run_id)
        other = (await honcho_client.aio.dreams("nobody")).items
    else:
        runs = list(honcho_client.dreams(observed=maya))
        detail = honcho_client.get_dream(run.run_id)
        other = list(honcho_client.dreams("nobody"))

    [listed] = runs
    assert isinstance(listed, DreamRunResponse)
    assert (listed.id, listed.observer_id, listed.status) == (
        run.run_id,
        maya.id,
        "completed",
    )
    assert listed.peer_card_updated
    assert isinstance(detail, DreamRunDetailResponse)
    assert detail.peer_card is not None
    assert detail.peer_card.after == ["IDENTITY: Name: Maya"]
    assert other == []


def test_get_unknown_dream_raises(honcho_sync_test_client: Honcho):
    honcho_sync_test_client.peer(id="dreams-anyone")
    try:
        honcho_sync_test_client.get_dream(generate_nanoid())
    except NotFoundError:
        return
    raise AssertionError("expected NotFoundError")
