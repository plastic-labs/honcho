from __future__ import annotations

import os
from dataclasses import dataclass

from src.config import EmbeddingTransport


@dataclass(frozen=True)
class LiveEmbeddingFamily:
    transport: EmbeddingTransport
    family: str
    env_var: str
    dimensions: int
    default_models: tuple[str, ...] = ()
    docs_url: str | None = None


@dataclass(frozen=True)
class LiveEmbeddingSpec:
    transport: EmbeddingTransport
    family: str
    model: str
    env_var: str
    dimensions: int
    docs_url: str | None = None

    @property
    def id(self) -> str:
        return f"{self.transport}:{self.family}:{self.model}"


EMBEDDING_FAMILIES: tuple[LiveEmbeddingFamily, ...] = (
    # Every Gemini model that advertises embedContent. The -2* models are the
    # regression surface for #745: the SDK folds a list of bare strings into a
    # single document and returns one embedding for the whole batch.
    LiveEmbeddingFamily(
        transport="gemini",
        family="gemini_embedding",
        env_var="LIVE_EMBEDDING_GEMINI_MODELS",
        # Matryoshka dimension supported across the family; keeps vectors small.
        dimensions=768,
        default_models=(
            "gemini-embedding-001",
            "gemini-embedding-2-preview",
            "gemini-embedding-2",
        ),
        docs_url="https://ai.google.dev/gemini-api/docs/embeddings",
    ),
    LiveEmbeddingFamily(
        transport="openai",
        family="openai_embedding",
        env_var="LIVE_EMBEDDING_OPENAI_MODELS",
        dimensions=1536,
        default_models=("text-embedding-3-small",),
        docs_url="https://platform.openai.com/docs/guides/embeddings",
    ),
)


def _parse_env_models(value: str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    return tuple(model.strip() for model in value.split(",") if model.strip())


def get_live_embedding_specs(
    *, transport: EmbeddingTransport | None = None
) -> tuple[LiveEmbeddingSpec, ...]:
    specs: list[LiveEmbeddingSpec] = []
    for family in EMBEDDING_FAMILIES:
        if transport is not None and family.transport != transport:
            continue
        models = _parse_env_models(os.getenv(family.env_var)) or family.default_models
        for model in models:
            specs.append(
                LiveEmbeddingSpec(
                    transport=family.transport,
                    family=family.family,
                    model=model,
                    env_var=family.env_var,
                    dimensions=family.dimensions,
                    docs_url=family.docs_url,
                )
            )
    return tuple(specs)


def selected_embedding_summary_lines() -> list[str]:
    lines: list[str] = []
    for family in EMBEDDING_FAMILIES:
        models = _parse_env_models(os.getenv(family.env_var)) or family.default_models
        joined = ", ".join(models) if models else "(none configured)"
        lines.append(f"{family.env_var} [{family.transport}/{family.family}]: {joined}")
    return lines
