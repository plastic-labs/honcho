"""Embedding oversize-survival benchmark — before/after quantification for #569.

Measures ``batch_survival_rate``: with one over-length observation in a batch of N,
the fraction of observations that still get embedded + saved through
``RepresentationManager.save_representation``.

It needs **no database and no real provider**: the embedding provider is faked and
the DB write is faked (`_save_representation_internal` counts the embeddings it is
handed). The same script runs unmodified against any checkout, so the before/after
contrast is produced by running it on ``main`` (before) and on the fix branch
(after):

    PYTHONPATH=. uv run python tests/bench/bench_embedding_oversize_survival.py
"""

from __future__ import annotations

import asyncio
import inspect
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

# Make `src` importable when run directly (uv run python tests/bench/<this>.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

BATCH_N = 10


async def measure_batch_survival(batch_n: int = BATCH_N) -> float:
    """Save a batch where exactly one observation is over-length.

    Returns the fraction of observations that survive (get embedded + saved).
    """
    from src.config import EmbeddingModelConfig
    from src.crud import representation as representation_module
    from src.crud.representation import RepresentationManager
    from src.embedding_client import (
        _EmbeddingClient,  # pyright: ignore[reportPrivateUsage]
    )
    from src.exceptions import ValidationException
    from src.utils.representation import ExplicitObservation, Representation

    class _FakeEmbeddingsAPI:
        async def create(
            self, *, input: str | list[str], **_kwargs: Any
        ) -> SimpleNamespace:
            items = input if isinstance(input, list) else [input]
            return SimpleNamespace(
                data=[SimpleNamespace(embedding=[0.1] * 8) for _ in items]
            )

    class _FakeOpenAIClient:
        def __init__(self, *, api_key: str | None, base_url: str | None) -> None:
            self.api_key: str | None = api_key
            self.base_url: str | None = base_url
            self.embeddings: _FakeEmbeddingsAPI = _FakeEmbeddingsAPI()

    @asynccontextmanager
    async def _fake_tracked_db(_name: str):
        yield object()

    saved: dict[str, int] = {"count": 0}

    async def _fake_internal(
        _db: Any,
        _all_observations: Any,
        embeddings: list[list[float]],
        *_args: Any,
        **_kwargs: Any,
    ) -> int:
        saved["count"] = len(embeddings)
        return len(embeddings)

    with patch("src.embedding_client.AsyncOpenAI", _FakeOpenAIClient):
        # max_input_tokens=5 makes any real sentence "over-length" without a giant string.
        client = _EmbeddingClient(
            EmbeddingModelConfig(
                transport="openai",
                model="text-embedding-3-small",
                api_key="test-key",
            ),
            vector_dimensions=8,
            max_input_tokens=5,
            max_tokens_per_request=300_000,
            send_dimensions=False,
        )

        observations = [
            ExplicitObservation(
                content="short",
                created_at=datetime.now(timezone.utc),
                message_ids=[1],
                session_name="session-1",
            )
            for _ in range(batch_n - 1)
        ]
        observations.append(
            ExplicitObservation(
                content="word " * 200,  # far over the 5-token cap
                created_at=datetime.now(timezone.utc),
                message_ids=[1],
                session_name="session-1",
            )
        )
        representation = Representation(explicit=observations)
        manager = RepresentationManager("workspace-1", observer="bob", observed="alice")

        with (
            patch.object(representation_module, "embedding_client", client),
            patch("src.crud.representation.tracked_db", _fake_tracked_db),
            patch.object(
                manager,
                "_save_representation_internal",
                new=AsyncMock(side_effect=_fake_internal),
            ),
        ):
            try:
                await manager.save_representation(
                    representation,
                    message_ids=[1],
                    session_name="session-1",
                    message_created_at=datetime.now(timezone.utc),
                    message_level_configuration=Mock(),
                )
                return saved["count"] / batch_n
            # BEFORE mode wraps the oversize failure in ValidationException and
            # drops the whole batch — that's 0% survival. Any other exception is
            # a harness bug and must escape, not be scored as a failed batch.
            except ValidationException:
                return 0.0


def _supports_truncate() -> bool:
    from src.embedding_client import (
        _EmbeddingClient,  # pyright: ignore[reportPrivateUsage]
    )

    params = inspect.signature(_EmbeddingClient.simple_batch_embed).parameters
    return "on_oversize" in params


async def main() -> None:
    survival = await measure_batch_survival()
    mode = (
        "AFTER (truncate supported)" if _supports_truncate() else "BEFORE (no truncate)"
    )

    print("=" * 60)
    print(f"Embedding oversize-survival benchmark — {mode}")
    print(f"  batch_n={BATCH_N}")
    print("-" * 60)
    print(f"  batch_survival_rate : {survival:6.1%}   (target: 100%)")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
