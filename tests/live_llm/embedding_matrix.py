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
    base_url: str | None = None
    # Falls back to the transport's own key when unset.
    api_key_env: str | None = None
    dimensions_env: str | None = None
    base_url_env: str | None = None
    send_dimensions: bool = True
    send_dimensions_env: str | None = None


@dataclass(frozen=True)
class LiveEmbeddingSpec:
    transport: EmbeddingTransport
    family: str
    model: str
    env_var: str
    dimensions: int
    docs_url: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    send_dimensions: bool = True

    @property
    def id(self) -> str:
        return f"{self.transport}:{self.family}:{self.model}"


EMBEDDING_FAMILIES: tuple[LiveEmbeddingFamily, ...] = (
    # gemini-embedding-2 is the regression surface for #745: the SDK folds a
    # list of bare strings into a single document and returns one embedding for
    # the whole batch. Its preview twin behaves identically and is reachable
    # through the env var when it needs checking.
    LiveEmbeddingFamily(
        transport="gemini",
        family="gemini_embedding",
        env_var="LIVE_EMBEDDING_GEMINI_MODELS",
        # Matryoshka dimension supported across the family; keeps vectors small.
        dimensions=768,
        default_models=(
            "gemini-embedding-001",
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
    # OpenAI transport pointed at an OpenAI-compatible provider. This is the
    # regression surface for #932: the openai SDK asks for base64 embeddings
    # unless told otherwise, and third-party providers reject or empty out that
    # request. Empty default_models → skipped unless set.
    LiveEmbeddingFamily(
        transport="openai",
        family="openai_compatible_embedding",
        env_var="LIVE_EMBEDDING_OPENAI_COMPATIBLE_MODELS",
        dimensions=3072,
        dimensions_env="LIVE_EMBEDDING_OPENAI_COMPATIBLE_DIMENSIONS",
        base_url="https://openrouter.ai/api/v1",
        base_url_env="LIVE_EMBEDDING_OPENAI_COMPATIBLE_BASE_URL",
        api_key_env="OPENROUTER_API_KEY",
        # Mirrors honcho's own behaviour once VECTOR_DIMENSIONS is set; turn off
        # for a provider that rejects the param.
        send_dimensions=True,
        send_dimensions_env="LIVE_EMBEDDING_OPENAI_COMPATIBLE_SEND_DIMENSIONS",
        docs_url="https://openrouter.ai/docs/api-reference/embeddings",
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
        dimensions = family.dimensions
        if family.dimensions_env:
            dimensions = int(os.getenv(family.dimensions_env) or family.dimensions)
        base_url = family.base_url
        if family.base_url_env:
            base_url = os.getenv(family.base_url_env) or family.base_url
        send_dimensions = family.send_dimensions
        if family.send_dimensions_env:
            raw = os.getenv(family.send_dimensions_env)
            if raw is not None:
                send_dimensions = raw.strip().lower() in {"1", "true", "yes"}
        for model in models:
            specs.append(
                LiveEmbeddingSpec(
                    transport=family.transport,
                    family=family.family,
                    model=model,
                    env_var=family.env_var,
                    dimensions=dimensions,
                    docs_url=family.docs_url,
                    base_url=base_url,
                    api_key_env=family.api_key_env,
                    send_dimensions=send_dimensions,
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
