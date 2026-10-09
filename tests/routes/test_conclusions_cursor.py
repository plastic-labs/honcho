import fastapi_pagination.ext.sqlalchemy as fp_sqlalchemy
import pytest
from fastapi.testclient import TestClient
from nanoid import generate as generate_nanoid
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.models import Peer, Workspace


async def _seed_conclusions(
    db_session: AsyncSession,
    sample_data: tuple[Workspace, Peer],
    count: int,
) -> tuple[str, str]:
    """Create `count` conclusions in one commit, so they share created_at."""
    test_workspace, test_peer = sample_data
    observed = models.Peer(
        name=str(generate_nanoid()), workspace_name=test_workspace.name
    )
    session = models.Session(
        name=str(generate_nanoid()), workspace_name=test_workspace.name
    )
    db_session.add_all([observed, session])
    await db_session.flush()
    db_session.add(
        models.Collection(
            workspace_name=test_workspace.name,
            observer=test_peer.name,
            observed=observed.name,
        )
    )
    await db_session.flush()
    for i in range(count):
        db_session.add(
            models.Document(
                workspace_name=test_workspace.name,
                observer=test_peer.name,
                observed=observed.name,
                content=f"Conclusion {i}",
                embedding=[0.1] * 1536,
                session_name=session.name,
            )
        )
    await db_session.commit()
    return test_workspace.name, session.name


def _walk_cursor(
    client: TestClient, workspace: str, session: str, size: int, reverse: bool
) -> list[list[str]]:
    pages: list[list[str]] = []
    cursor = ""
    while cursor is not None:
        response = client.post(
            f"/v3/workspaces/{workspace}/conclusions/list",
            params={"cursor": cursor, "size": size, "reverse": reverse},
            json={"filters": {"session_id": session}},
        )
        assert response.status_code == 200, response.text
        data = response.json()
        assert data["total"] is None
        assert "pages" not in data
        pages.append([item["id"] for item in data["items"]])
        cursor = data["next_page"]
        assert len(pages) <= 10, "cursor walk did not terminate"
    return pages


def _walk_offset(
    client: TestClient, workspace: str, session: str, size: int, reverse: bool
) -> list[str]:
    ids: list[str] = []
    page = 1
    while True:
        response = client.post(
            f"/v3/workspaces/{workspace}/conclusions/list",
            params={"page": page, "size": size, "reverse": reverse},
            json={"filters": {"session_id": session}},
        )
        data = response.json()
        ids += [item["id"] for item in data["items"]]
        if page >= data["pages"]:
            return ids
        page += 1


class TestConclusionCursorPagination:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("reverse", [False, True])
    async def test_cursor_walk_matches_offset_walk(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
        reverse: bool,
        monkeypatch: pytest.MonkeyPatch,
    ):
        """Every row once, in offset order, even when created_at ties."""
        workspace, session = await _seed_conclusions(db_session, sample_data, 15)

        def no_count(*_args: object, **_kwargs: object):
            raise AssertionError("cursor mode ran a COUNT query")

        with monkeypatch.context() as m:
            m.setattr(fp_sqlalchemy, "_total_flow", no_count)
            pages = _walk_cursor(client, workspace, session, size=10, reverse=reverse)

        assert [len(p) for p in pages] == [10, 5]
        cursor_ids = [i for p in pages for i in p]
        assert len(set(cursor_ids)) == 15
        assert cursor_ids == _walk_offset(
            client, workspace, session, size=10, reverse=reverse
        )

    @pytest.mark.asyncio
    async def test_cursor_empty_collection(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        workspace, session = await _seed_conclusions(db_session, sample_data, 0)

        assert _walk_cursor(client, workspace, session, size=10, reverse=False) == [[]]

    @pytest.mark.asyncio
    async def test_offset_response_shape_unchanged(
        self,
        client: TestClient,
        db_session: AsyncSession,
        sample_data: tuple[Workspace, Peer],
    ):
        """Without `cursor`, installed SDKs still get page/pages/total."""
        workspace, session = await _seed_conclusions(db_session, sample_data, 3)

        response = client.post(
            f"/v3/workspaces/{workspace}/conclusions/list",
            json={"filters": {"session_id": session}},
        )

        assert response.status_code == 200
        data = response.json()
        assert set(data) == {"items", "total", "page", "size", "pages"}
        assert (data["total"], data["page"], data["size"], data["pages"]) == (
            3,
            1,
            50,
            1,
        )

    @pytest.mark.asyncio
    async def test_invalid_cursor_rejected(
        self,
        client: TestClient,
        sample_data: tuple[Workspace, Peer],
    ):
        test_workspace, _ = sample_data

        response = client.post(
            f"/v3/workspaces/{test_workspace.name}/conclusions/list",
            params={"cursor": "%%%not-a-cursor"},
        )

        assert response.status_code == 400
