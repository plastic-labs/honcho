import asyncio
from collections.abc import Callable
from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from src import crud
from src.config import AppSettings
from src.embedding_client import embedding_client
from src.utils import agent_tools
from src.utils import representation as representation_module
from src.utils.agent_tools import ToolContext
from src.utils.representation import (
    ContradictionObservation,
    DeductiveObservation,
    ExplicitObservation,
    InductiveObservation,
    Representation,
)
from src.utils.types import ToolResult

DEFAULT_ORDER = ("explicit", "deductive", "inductive", "contradiction")
PATTERNS_FIRST_ORDER = ("inductive", "explicit", "deductive", "contradiction")


def _full_representation() -> Representation:
    created_at = datetime(2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc)
    return Representation(
        explicit=[
            ExplicitObservation(
                id="exp-1",
                content="explicit fact",
                created_at=created_at,
                message_ids=[1],
            )
        ],
        deductive=[
            DeductiveObservation(
                id="ded-1",
                conclusion="deductive fact",
                premises=["explicit fact"],
                created_at=created_at,
                message_ids=[1],
            )
        ],
        inductive=[
            InductiveObservation(
                id="ind-1",
                conclusion="inductive pattern",
                confidence="high",
                pattern_type="preference",
                sources=["explicit fact"],
                created_at=created_at,
                message_ids=[1],
            )
        ],
        contradiction=[
            ContradictionObservation(
                id="con-1",
                content="conflicting fact",
                sources=["a", "b"],
                created_at=created_at,
                message_ids=[1],
            )
        ],
    )


def _assert_headers_in_order(rendered: str, headers: tuple[str, ...]) -> None:
    positions = [rendered.index(header) for header in headers]
    assert positions == sorted(positions)


def _render_with_ids(representation: Representation) -> str:
    return representation.str_with_ids()


def _render_without_timestamps(representation: Representation) -> str:
    return representation.str_no_timestamps()


def _render_as_markdown(representation: Representation) -> str:
    return representation.format_as_markdown(include_ids=True)


def test_representation_injection_order_defaults_to_current_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("REPRESENTATION_INJECTION_ORDER", raising=False)

    assert AppSettings().REPRESENTATION_INJECTION_ORDER == DEFAULT_ORDER


def test_representation_injection_order_accepts_comma_separated_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "REPRESENTATION_INJECTION_ORDER",
        "inductive, explicit, deductive, contradiction",
    )

    assert AppSettings().REPRESENTATION_INJECTION_ORDER == PATTERNS_FIRST_ORDER


@pytest.mark.parametrize(
    "configured_order",
    [
        "explicit,deductive,inductive",
        "explicit,deductive,inductive,explicit",
        "explicit,deductive,inductive,unknown",
    ],
)
def test_representation_injection_order_rejects_invalid_permutations(
    monkeypatch: pytest.MonkeyPatch, configured_order: str
) -> None:
    monkeypatch.setenv("REPRESENTATION_INJECTION_ORDER", configured_order)

    with pytest.raises(
        ValueError,
        match="REPRESENTATION_INJECTION_ORDER must contain each supported section exactly once",
    ):
        AppSettings()


def test_default_rendering_is_byte_for_byte_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        representation_module,
        "settings",
        AppSettings(REPRESENTATION_INJECTION_ORDER=DEFAULT_ORDER),
        raising=False,
    )
    representation = _full_representation()

    assert str(representation) == (
        "EXPLICIT:\n\n"
        "1. [2025-01-02 03:04:05] explicit fact\n\n"
        "DEDUCTIVE:\n\n"
        "1. [2025-01-02 03:04:05] deductive fact\n"
        "    - explicit fact\n\n"
        "INDUCTIVE:\n\n"
        "1. [2025-01-02 03:04:05] [high] inductive pattern\n"
        "    - explicit fact\n\n"
        "CONTRADICTION:\n\n"
        "1. [2025-01-02 03:04:05] CONTRADICTION: conflicting fact\n"
        "    - a\n"
        "    - b\n"
    )
    assert representation.str_with_ids() == (
        "EXPLICIT:\n\n"
        "1. [id:exp-1] [2025-01-02 03:04:05] explicit fact\n\n"
        "DEDUCTIVE:\n\n"
        "1. [id:ded-1] [2025-01-02 03:04:05] deductive fact\n"
        "    - explicit fact\n\n"
        "INDUCTIVE:\n\n"
        "1. [id:ind-1] [2025-01-02 03:04:05] [high] inductive pattern\n"
        "    - explicit fact\n\n"
        "CONTRADICTION:\n\n"
        "1. [id:con-1] [2025-01-02 03:04:05] CONTRADICTION: conflicting fact\n"
        "    - a\n"
        "    - b\n"
    )
    assert representation.str_no_timestamps() == (
        "EXPLICIT:\n\n"
        "1. explicit fact\n\n"
        "DEDUCTIVE:\n\n"
        "1. deductive fact\n"
        "    - explicit fact\n\n"
        "INDUCTIVE:\n\n"
        "1. [high] inductive pattern\n"
        "    - explicit fact\n\n"
        "CONTRADICTION:\n\n"
        "1. CONTRADICTION: conflicting fact\n"
        "    - a\n"
        "    - b\n"
    )
    assert representation.format_as_markdown(include_ids=True) == (
        "## Explicit Observations\n\n"
        "[2025-01-02 03:04:05] explicit fact\n\n"
        "## Deductive Observations\n\n"
        "[id:ded-1] [2025-01-02 03:04:05] deductive fact\n"
        "   Premises:\n"
        "   - explicit fact\n\n\n"
        "## Inductive Observations\n\n"
        "[id:ind-1]  **Pattern** [high]: inductive pattern\n"
        "   **Type**: preference\n"
        "   **Sources**:\n"
        "   - explicit fact\n\n\n"
        "## Contradictions\n\n"
        "[id:con-1]  **CONTRADICTION**: conflicting fact\n"
        "   **Conflicting statements**:\n"
        "   - a\n"
        "   - b\n\n"
    )


@pytest.mark.parametrize(
    ("renderer", "ordered_headers"),
    [
        (
            str,
            ("INDUCTIVE:", "EXPLICIT:", "DEDUCTIVE:", "CONTRADICTION:"),
        ),
        (
            _render_with_ids,
            ("INDUCTIVE:", "EXPLICIT:", "DEDUCTIVE:", "CONTRADICTION:"),
        ),
        (
            _render_without_timestamps,
            ("INDUCTIVE:", "EXPLICIT:", "DEDUCTIVE:", "CONTRADICTION:"),
        ),
        (
            _render_as_markdown,
            (
                "## Inductive Observations",
                "## Explicit Observations",
                "## Deductive Observations",
                "## Contradictions",
            ),
        ),
    ],
)
def test_all_renderers_use_the_configured_section_order(
    monkeypatch: pytest.MonkeyPatch,
    renderer: Callable[[Representation], str],
    ordered_headers: tuple[str, ...],
) -> None:
    monkeypatch.setattr(
        representation_module,
        "settings",
        AppSettings(REPRESENTATION_INJECTION_ORDER=PATTERNS_FIRST_ORDER),
        raising=False,
    )

    rendered = renderer(_full_representation())

    _assert_headers_in_order(rendered, ordered_headers)


@pytest.mark.asyncio
async def test_search_memory_prompt_uses_the_configured_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    representation = _full_representation()
    monkeypatch.setattr(
        representation_module,
        "settings",
        AppSettings(REPRESENTATION_INJECTION_ORDER=PATTERNS_FIRST_ORDER),
        raising=False,
    )
    monkeypatch.setattr(
        embedding_client,
        "embed",
        AsyncMock(return_value=[0.1]),
    )
    monkeypatch.setattr(
        crud,
        "query_documents",
        AsyncMock(return_value=[object()]),
    )

    def return_representation(_documents: object) -> Representation:
        return representation

    monkeypatch.setattr(Representation, "from_documents", return_representation)
    context = ToolContext(
        workspace_name="workspace",
        observer="observer",
        observed="observed",
        session_name=None,
        current_messages=None,
        include_observation_ids=False,
        history_token_limit=8192,
        db_lock=asyncio.Lock(),
    )

    result = await agent_tools._handle_search_memory(  # pyright: ignore[reportPrivateUsage]
        context,
        {"query": "patterns", "top_k": 10},
    )

    assert isinstance(result, ToolResult)
    _assert_headers_in_order(
        result.content,
        ("INDUCTIVE:", "EXPLICIT:", "DEDUCTIVE:", "CONTRADICTION:"),
    )
