"""Runtime configuration introspection routes."""

from fastapi import APIRouter, Depends

from src import schemas
from src.config import (
    ConfiguredEmbeddingModelSettings,
    ConfiguredModelSettings,
    EmbeddingModelConfig,
    ModelConfig,
    resolve_embedding_model_config,
    resolve_model_config,
    settings,
)
from src.llm.credentials import default_transport_api_key
from src.security import require_auth

router = APIRouter(
    prefix="/runtime",
    tags=["runtime"],
    dependencies=[Depends(require_auth(admin=True))],
)


def _default_transport_base_url(transport: str) -> str | None:
    if transport == "anthropic":
        return settings.LLM.ANTHROPIC_BASE_URL
    if transport == "openai":
        return settings.LLM.OPENAI_BASE_URL
    if transport == "gemini":
        return settings.LLM.GEMINI_BASE_URL
    return None


def _effective_base_url(config: ModelConfig | EmbeddingModelConfig) -> str | None:
    if isinstance(config, EmbeddingModelConfig):
        return config.base_url
    return config.base_url or _default_transport_base_url(config.transport)


def _provider_label(transport: str, base_url: str | None) -> str:
    if transport == "openai" and base_url and "openrouter" in base_url.lower():
        return "openrouter"
    return transport


def _auth_mechanism(config: ModelConfig | EmbeddingModelConfig) -> str:
    explicit_auth = getattr(config, "auth_mode", None)
    if isinstance(explicit_auth, str) and explicit_auth:
        return explicit_auth
    if config.api_key or default_transport_api_key(config.transport):
        return "api_key"
    return "none"


def _model_info(config: ModelConfig | EmbeddingModelConfig) -> schemas.RuntimeModelInfo:
    base_url = _effective_base_url(config)
    return schemas.RuntimeModelInfo(
        model=config.model,
        provider=_provider_label(config.transport, base_url),
        transport=config.transport,
        auth_mechanism=_auth_mechanism(config),
        base_url=base_url,
    )


def _llm_info(config: ConfiguredModelSettings) -> schemas.RuntimeModelInfo:
    return _model_info(resolve_model_config(config))


def _embedding_info(
    config: ConfiguredEmbeddingModelSettings,
) -> schemas.RuntimeModelInfo:
    return _model_info(resolve_embedding_model_config(config))


def _llm_models() -> dict[str, schemas.RuntimeModelInfo]:
    models = {
        f"dialectic.{level}": _llm_info(level_settings.MODEL_CONFIG)
        for level, level_settings in settings.DIALECTIC.LEVELS.items()
    }
    models["deriver"] = _llm_info(settings.DERIVER.MODEL_CONFIG)
    models["summary"] = _llm_info(settings.SUMMARY.MODEL_CONFIG)
    models["dream.deduction"] = _llm_info(settings.DREAM.DEDUCTION_MODEL_CONFIG)
    models["dream.induction"] = _llm_info(settings.DREAM.INDUCTION_MODEL_CONFIG)
    return models


@router.get("/llm", response_model=schemas.LLMRuntimeInfo)
async def get_llm_runtime() -> schemas.LLMRuntimeInfo:
    """Return the configured LLM models currently used by Honcho agents."""
    models = _llm_models()
    current_source = "dialectic.low"
    return schemas.LLMRuntimeInfo(
        current_source=current_source,
        current=models[current_source],
        models=models,
    )


@router.get("/embeddings", response_model=schemas.EmbeddingRuntimeInfo)
async def get_embedding_runtime() -> schemas.EmbeddingRuntimeInfo:
    """Return the configured embedding model currently used by Honcho."""
    config = resolve_embedding_model_config(settings.EMBEDDING.MODEL_CONFIG)
    info = _model_info(config)
    return schemas.EmbeddingRuntimeInfo(
        **info.model_dump(),
        vector_dimensions=settings.EMBEDDING.VECTOR_DIMENSIONS,
        dimensions_mode=settings.EMBEDDING.MODEL_CONFIG.dimensions_mode,
    )


@router.get("/auth", response_model=schemas.RuntimeAuthInfo)
async def get_runtime_auth() -> schemas.RuntimeAuthInfo:
    """Return provider/auth metadata for the current LLM and embedding paths."""
    llm_runtime = await get_llm_runtime()
    return schemas.RuntimeAuthInfo(
        llm_source=llm_runtime.current_source,
        llm=llm_runtime.current,
        embeddings=_embedding_info(settings.EMBEDDING.MODEL_CONFIG),
    )
