import datetime

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models
from src.crud.document import DocumentLevelSummary
from src.utils.memory_brief import (
    format_memory_counts,
    load_memory_counts,
    memory_layout_section,
)


def test_layout_describes_only_offered_tools():
    section = memory_layout_section(
        available_tools=["search_memory", "search_messages"], cite_ids=False
    )
    assert "**contradiction**" in section
    assert "`search_memory`" in section
    assert "`search_messages`" in section
    assert "`get_recent_observations`" not in section
    assert "[id:" not in section


def test_layout_explains_ids_when_the_agent_can_cite():
    section = memory_layout_section(available_tools=[], cite_ids=True)
    assert "[id:abc123]" in section
    assert "What each search tool covers" not in section


def test_layout_is_stable_for_the_same_arguments():
    tools = ["get_recent_observations", "search_memory"]
    assert memory_layout_section(
        available_tools=tools, cite_ids=True
    ) == memory_layout_section(available_tools=reversed(tools), cite_ids=True)


def test_counts_line_lists_every_level_in_order():
    summary = DocumentLevelSummary(
        counts_by_level={"deductive": 3, "explicit": 10},
        earliest=datetime.datetime(2026, 3, 2, 8, tzinfo=datetime.UTC),
        latest=datetime.datetime(2026, 10, 13, 22, tzinfo=datetime.UTC),
    )
    assert format_memory_counts(summary) == (
        "Memory counts: 10 explicit, 3 deductive, 0 inductive, 0 contradiction "
        + "(13 conclusions, created 2026-03-02 to 2026-10-13)."
    )


def test_counts_line_for_empty_memory():
    summary = DocumentLevelSummary(counts_by_level={}, earliest=None, latest=None)
    assert format_memory_counts(summary) == "Memory counts: no conclusions yet."


@pytest.mark.asyncio
async def test_summary_counts_live_documents_per_pair(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
):
    workspace, peer = sample_data
    db_session.add(
        models.Collection(
            workspace_name=workspace.name, observer=peer.name, observed=peer.name
        )
    )
    await db_session.flush()
    for content, level, deleted in [
        ("likes tea", "explicit", False),
        ("drinks tea daily", "explicit", False),
        ("is a tea drinker", "deductive", False),
        ("hates tea", "explicit", True),
    ]:
        db_session.add(
            models.Document(
                workspace_name=workspace.name,
                observer=peer.name,
                observed=peer.name,
                content=content,
                level=level,
                embedding=[0.1] * 1536,
                deleted_at=datetime.datetime.now(datetime.UTC) if deleted else None,
            )
        )
    await db_session.commit()

    summary = await crud.summarize_documents_by_level(
        db_session, workspace.name, observer=peer.name, observed=peer.name
    )
    assert summary.counts_by_level == {"explicit": 2, "deductive": 1}
    assert summary.earliest is not None and summary.latest is not None

    line = await load_memory_counts(
        workspace.name, observer=peer.name, observed=peer.name
    )
    assert line.startswith("Memory counts: 2 explicit, 1 deductive,")
