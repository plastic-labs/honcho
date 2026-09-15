"""Tests that a dialectic run collates what it read.

The accumulator is unit-tested in tests/utils/test_evidence.py and the tool
handlers in tests/utils/test_agent_tools.py. What these add is the agent: a
real run against a real database, with only the LLM call mocked, so that the
paths outside the tool loop are covered too.
"""

from collections.abc import AsyncIterator, Mapping
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from nanoid import generate as generate_nanoid
from sqlalchemy.ext.asyncio import AsyncSession

from src import crud, models, schemas
from src.config import settings
from src.dialectic.core import DialecticAgent
from src.dialectic.workspace import WorkspaceDialecticAgent
from src.llm import (
    HonchoLLMCallResponse,
    HonchoLLMCallStreamChunk,
    StreamingResponseWithMetadata,
)
from src.utils.evidence import EvidenceAccumulator
from src.utils.representation import Representation


def llm_call_kwargs(mock_llm_call: AsyncMock) -> Mapping[str, Any]:
    """The arguments of the last LLM call, once one has actually happened."""
    assert mock_llm_call.await_args is not None, "the agent never called the LLM"
    return mock_llm_call.await_args.kwargs


def make_llm_response(
    tool_calls_made: list[dict[str, Any]] | None = None,
) -> HonchoLLMCallResponse[str]:
    return HonchoLLMCallResponse(
        content="The user drinks coffee.",
        input_tokens=10,
        output_tokens=5,
        finish_reasons=["end_turn"],
        tool_calls_made=tool_calls_made or [],
    )


@pytest.fixture
async def dialectic_test_data(
    db_session: AsyncSession,
    sample_data: tuple[models.Workspace, models.Peer],
    monkeypatch: pytest.MonkeyPatch,
) -> Any:
    """A peer with conclusions and messages the agent can actually recall.

    Returns (workspace, observer, observed, session, messages, documents).
    """
    # The documents are written straight to postgres, so recall has to go
    # through pgvector rather than the migrated vector store.
    monkeypatch.setattr(settings.VECTOR_STORE, "MIGRATED", False)

    workspace, observer = sample_data
    observed = models.Peer(name=str(generate_nanoid()), workspace_name=workspace.name)
    db_session.add(observed)
    await db_session.flush()

    session = (
        await crud.get_or_create_session(
            db_session,
            schemas.SessionCreate(
                name=str(generate_nanoid()),
                peers={observed.name: schemas.SessionPeerConfig(observe_me=True)},
            ),
            workspace.name,
        )
    ).resource
    db_session.add(
        models.Collection(
            workspace_name=workspace.name,
            observer=observer.name,
            observed=observed.name,
        )
    )
    await db_session.flush()

    messages: list[models.Message] = []
    for i, content in enumerate(
        ["I drink a lot of coffee", "Mornings are my best time"]
    ):
        message = models.Message(
            workspace_name=workspace.name,
            session_name=session.name,
            peer_name=observed.name,
            content=content,
            seq_in_session=i + 1,
            token_count=10,
        )
        db_session.add(message)
        messages.append(message)
    await db_session.flush()

    documents: list[models.Document] = []
    for i, (content, level) in enumerate(
        [
            ("User drinks coffee", "explicit"),
            ("User is a morning person", "explicit"),
            ("User drinks coffee in the morning", "deductive"),
        ]
    ):
        document = models.Document(
            workspace_name=workspace.name,
            observer=observer.name,
            observed=observed.name,
            content=content,
            embedding=[0.1 * (i + 1)] * 1536,
            session_name=session.name,
            level=level,
        )
        db_session.add(document)
        documents.append(document)
    await db_session.flush()

    for row in (*messages, *documents):
        await db_session.refresh(row)
    await db_session.commit()

    return workspace, observer, observed, session, messages, documents


def make_agent(
    dialectic_test_data: Any, evidence: EvidenceAccumulator | None = None, **kwargs: Any
) -> DialecticAgent:
    """A dialectic agent pointed at the fixture's peer pair."""
    _, observer, observed, session, _, _ = dialectic_test_data
    return DialecticAgent(
        workspace_name=observer.workspace_name,
        session_name=session.name,
        observer=observer.name,
        observed=observed.name,
        evidence=evidence,
        **kwargs,
    )


async def run_answer(
    agent: DialecticAgent, response: Any = None
) -> tuple[str, AsyncMock]:
    """Answer a query with the LLM call mocked out.

    Returns the answer and the mock, so a test can reach the arguments the
    agent handed the LLM -- the tool executor especially.
    """
    mock_llm_call = AsyncMock(return_value=response or make_llm_response())
    with patch("src.dialectic.core.honcho_llm_call", new=mock_llm_call):
        answer = await agent.answer("What does the user drink?")
    return answer, mock_llm_call


@pytest.mark.asyncio
class TestPrefetchEvidence:
    """Prefetched conclusions never pass through the tool executor.

    The agent reads them before its first LLM call, so on a query that answers
    without calling a tool they are the whole of what it read. If evidence only
    watched the tool loop it would look empty on exactly those queries.
    """

    async def test_records_conclusions_the_prefetch_loaded(
        self, dialectic_test_data: Any
    ):
        *_, documents = dialectic_test_data
        evidence = EvidenceAccumulator()
        await run_answer(make_agent(dialectic_test_data, evidence))

        recorded = set(evidence.conclusions)
        assert recorded, "prefetched conclusions were not recorded"
        assert recorded <= {document.id for document in documents}

    async def test_prefetch_covers_derived_conclusions_too(
        self, dialectic_test_data: Any
    ):
        """Prefetch searches explicit and derived levels separately."""
        *_, documents = dialectic_test_data
        deductive_id = next(d.id for d in documents if d.level == "deductive")
        evidence = EvidenceAccumulator()
        await run_answer(make_agent(dialectic_test_data, evidence))

        assert deductive_id in evidence.conclusions

    async def test_built_evidence_reports_the_prefetched_conclusions(
        self, dialectic_test_data: Any
    ):
        """The API shape carries them, not just the accumulator."""
        evidence = EvidenceAccumulator()
        await run_answer(make_agent(dialectic_test_data, evidence))

        built = evidence.build()
        assert {c.content for c in built.conclusions} >= {"User drinks coffee"}
        assert all(c.id for c in built.conclusions)

    async def test_records_nothing_when_the_prefetch_block_is_dropped(
        self, dialectic_test_data: Any, monkeypatch: pytest.MonkeyPatch
    ):
        """A prefetch that fails after the search reaches the agent with nothing.

        `_prefetch_relevant_observations` swallows any failure and returns
        None, so `_prepare_query` builds a prompt with no prefetch block. The
        rows the search returned were never shown to the agent, and evidence
        has to say so.
        """

        def explode(*_args: object, **_kwargs: object) -> str:
            raise RuntimeError("formatting blew up")

        monkeypatch.setattr(Representation, "format_as_markdown", explode)
        evidence = EvidenceAccumulator()
        agent = make_agent(dialectic_test_data, evidence)

        await run_answer(agent)

        assert evidence.conclusions == {}
        # The agent still answered, just without the prefetched context.
        assert agent.messages[-1]["content"].startswith("Query:")
        assert "Relevant Observations" not in agent.messages[-1]["content"]


@pytest.mark.asyncio
class TestToolLoopEvidence:
    async def test_the_agents_tool_executor_records_into_the_same_accumulator(
        self, dialectic_test_data: Any
    ):
        """Capture the executor the agent handed the LLM and drive it.

        The mocked LLM call never invokes tools, so this checks the wiring the
        way the tool loop would use it.
        """
        *_, documents = dialectic_test_data
        evidence = EvidenceAccumulator()
        _, mock_llm_call = await run_answer(make_agent(dialectic_test_data, evidence))

        evidence.conclusions.clear()
        tool_executor = llm_call_kwargs(mock_llm_call)["tool_executor"]
        await tool_executor("search_memory", {"query": "coffee"})

        assert set(evidence.conclusions) <= {d.id for d in documents}
        assert evidence.conclusions, "the tool executor did not record anything"

    async def test_records_the_tool_calls_the_loop_made(self, dialectic_test_data: Any):
        evidence = EvidenceAccumulator()
        await run_answer(
            make_agent(dialectic_test_data, evidence),
            make_llm_response(
                tool_calls_made=[
                    {
                        "tool_name": "search_memory",
                        "tool_input": {"query": "coffee"},
                        "tool_result": "Found 3 observations",
                    }
                ]
            ),
        )

        built = evidence.build()
        assert [(c.tool_name, c.tool_input) for c in built.tool_calls] == [
            ("search_memory", {"query": "coffee"})
        ]
        assert "Found 3 observations" not in built.model_dump_json()

    async def test_records_the_tool_calls_a_streamed_answer_made(
        self, dialectic_test_data: Any
    ):
        """Evidence is only complete once the stream has drained."""
        evidence = EvidenceAccumulator()
        agent = make_agent(dialectic_test_data, evidence)

        async def chunks() -> AsyncIterator[HonchoLLMCallStreamChunk]:
            for text in ("The user ", "drinks coffee."):
                yield HonchoLLMCallStreamChunk(content=text, is_done=False)
            yield HonchoLLMCallStreamChunk(content="", is_done=True)

        streaming = StreamingResponseWithMetadata(
            chunks(),
            tool_calls_made=[{"tool_name": "search_messages", "tool_input": {}}],
            input_tokens=10,
            output_tokens=5,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            iterations=1,
        )

        with patch(
            "src.dialectic.core.honcho_llm_call",
            new=AsyncMock(return_value=streaming),
        ):
            streamed = [
                chunk
                async for chunk in agent.answer_stream("What does the user drink?")
            ]

        assert "".join(streamed) == "The user drinks coffee."
        assert [c.tool_name for c in evidence.build().tool_calls] == ["search_messages"]


@pytest.mark.asyncio
class TestEvidenceOptedOut:
    async def test_collects_nothing_and_answers_the_same(
        self, dialectic_test_data: Any
    ):
        agent = make_agent(dialectic_test_data)
        mock_llm_call = AsyncMock(return_value=make_llm_response())

        with patch("src.dialectic.core.honcho_llm_call", new=mock_llm_call):
            answer = await agent.answer("What does the user drink?")

        assert answer == "The user drinks coffee."
        assert agent.evidence is None
        tool_executor = llm_call_kwargs(mock_llm_call)["tool_executor"]
        # The executor still works; it just has nowhere to record.
        await tool_executor("search_memory", {"query": "coffee"})


@pytest.mark.asyncio
class TestScopedEvidence:
    async def test_stays_empty_when_recall_fails_closed(self, dialectic_test_data: Any):
        """An empty session allowlist recalls nothing, so it cites nothing.

        Evidence reports only rows a permitted read returned, so it cannot
        become a way around the allowlist.
        """
        evidence = EvidenceAccumulator()
        await run_answer(
            make_agent(dialectic_test_data, evidence, session_allowlist=[])
        )

        assert evidence.conclusions == {}
        assert evidence.messages == {}

    async def test_only_reports_conclusions_inside_the_allowlist(
        self, dialectic_test_data: Any, db_session: AsyncSession
    ):
        _, observer, observed, session, _, documents = dialectic_test_data
        other_session = (
            await crud.get_or_create_session(
                db_session,
                schemas.SessionCreate(name=str(generate_nanoid())),
                observer.workspace_name,
            )
        ).resource
        outsider = models.Document(
            workspace_name=observer.workspace_name,
            observer=observer.name,
            observed=observed.name,
            content="User dislikes tea",
            embedding=[0.9] * 1536,
            session_name=other_session.name,
            level="explicit",
        )
        db_session.add(outsider)
        await db_session.flush()
        await db_session.refresh(outsider)
        await db_session.commit()

        evidence = EvidenceAccumulator()
        await run_answer(
            make_agent(dialectic_test_data, evidence, session_allowlist=[session.name])
        )

        assert outsider.id not in evidence.conclusions
        assert set(evidence.conclusions) <= {d.id for d in documents}


@pytest.mark.asyncio
class TestWorkspaceAgentEvidence:
    async def test_its_tool_executor_records_into_the_same_accumulator(
        self, dialectic_test_data: Any
    ):
        """The workspace agent's prefetch is a stats overview, not conclusions.

        So its evidence has to come from the tool loop, through the workspace
        executor's delegating handlers.
        """
        workspace, _, _, session, messages, _ = dialectic_test_data
        evidence = EvidenceAccumulator()
        agent = WorkspaceDialecticAgent(
            workspace_name=workspace.name,
            session_name=session.name,
            evidence=evidence,
        )
        _, mock_llm_call = await run_answer(agent)

        assert evidence.conclusions == {}, "the stats prefetch has nothing to cite"

        tool_executor = llm_call_kwargs(mock_llm_call)["tool_executor"]
        await tool_executor("grep_messages", {"text": "coffee", "context_window": 0})

        assert messages[0].public_id in evidence.messages
