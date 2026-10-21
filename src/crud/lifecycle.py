"""Deletion lifecycle barrier.

A producer captures a ``LifecycleFence`` in the transaction that reads its input
and checks it in the transaction that writes its output. The check takes
``FOR SHARE`` on the session and collection rows, which serializes it against an
erasure holding ``FOR UPDATE`` on the same rows. A generation that moved means an
erasure committed in between, and the write is refused with
``StaleLifecycleError``.

Lock order is session, then collections by id, in both the check and the erasure.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field

from sqlalchemy import select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession

from src import models
from src.exceptions import StaleLifecycleError

logger = logging.getLogger(__name__)

__all__ = [
    "LifecycleFence",
    "capture_fence",
    "check_fence",
]


@dataclass(frozen=True, slots=True)
class LifecycleFence:
    """Generations a producer observed when it captured its input."""

    workspace_name: str
    session_name: str | None = None
    # None when the session is not part of the input
    session_generation: int | None = None
    # (observer, observed) -> generation; 0 for a collection that did not exist
    representations: dict[tuple[str, str], int] = field(default_factory=dict)


async def capture_fence(
    db: AsyncSession,
    workspace_name: str,
    *,
    session_name: str | None = None,
    representations: Iterable[tuple[str, str]] = (),
) -> LifecycleFence:
    """Read the current generations for the producer's input.

    Column selects only: a cached ORM object merged into the session carries no
    generation.
    """
    session_generation: int | None = None
    if session_name is not None:
        session_generation = await db.scalar(
            select(models.Session.deletion_generation).where(
                models.Session.workspace_name == workspace_name,
                models.Session.name == session_name,
            )
        )
        if session_generation is None:
            raise StaleLifecycleError(
                f"Session {session_name} is gone from workspace {workspace_name}"
            )

    pairs = sorted(set(representations))
    generations = dict.fromkeys(pairs, 0)
    if pairs:
        rows = await db.execute(
            select(
                models.Collection.observer,
                models.Collection.observed,
                models.Collection.deletion_generation,
            ).where(
                models.Collection.workspace_name == workspace_name,
                tuple_(models.Collection.observer, models.Collection.observed).in_(
                    pairs
                ),
            )
        )
        for observer, observed, generation in rows:
            generations[(observer, observed)] = generation

    return LifecycleFence(
        workspace_name=workspace_name,
        session_name=session_name,
        session_generation=session_generation,
        representations=generations,
    )


async def check_fence(db: AsyncSession, fence: LifecycleFence) -> None:
    """Lock the fenced rows ``FOR SHARE`` and refuse the write if any moved.

    Must run inside the write transaction that commits the producer's output.

    Raises:
        StaleLifecycleError: If an erasure committed after the fence was captured.
    """
    if fence.session_name is not None:
        current = await db.scalar(
            select(models.Session.deletion_generation)
            .where(
                models.Session.workspace_name == fence.workspace_name,
                models.Session.name == fence.session_name,
            )
            .with_for_update(read=True)
        )
        if current is None or current != fence.session_generation:
            raise StaleLifecycleError(
                f"Session {fence.session_name} in {fence.workspace_name} changed"
                + f" generation ({fence.session_generation} -> {current})"
            )

    if fence.representations:
        rows = await db.execute(
            select(
                models.Collection.observer,
                models.Collection.observed,
                models.Collection.deletion_generation,
            )
            .where(
                models.Collection.workspace_name == fence.workspace_name,
                tuple_(models.Collection.observer, models.Collection.observed).in_(
                    list(fence.representations)
                ),
            )
            .order_by(models.Collection.id)
            .with_for_update(read=True)
        )
        current_generations = {
            (observer, observed): generation for observer, observed, generation in rows
        }
        for pair, captured in fence.representations.items():
            current = current_generations.get(pair, 0)
            if current != captured:
                raise StaleLifecycleError(
                    f"Representation {pair[0]}/{pair[1]} in {fence.workspace_name}"
                    + f" changed generation ({captured} -> {current})"
                )
