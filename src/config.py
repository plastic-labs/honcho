import logging
import os
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal, cast

import tomllib
from dotenv import load_dotenv
from pydantic import AliasChoices, BaseModel, Field, field_validator, model_validator
from pydantic.fields import FieldInfo
from pydantic_settings import (
    BaseSettings,
    DotEnvSettingsSource,
    EnvSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

# Load .env file for local development.
# Make sure this is called before AppSettings is instantiated if you rely on .env for AppSettings construction.
if not os.getenv("PYTHON_DOTENV_DISABLED"):
    load_dotenv(override=True)

logger = logging.getLogger(__name__)

ModelTransport = Literal["anthropic", "openai", "gemini"]
EmbeddingTransport = Literal["openai", "gemini"]


def _default_embedding_model_for_transport(transport: EmbeddingTransport) -> str:
    if transport == "gemini":
        return "gemini-embedding-001"
    return "text-embedding-3-small"


def load_toml_config(config_path: str = "config.toml") -> dict[str, Any]:
    """Load configuration from TOML file if it exists."""
    if config_path == "config.toml" and os.getenv("HONCHO_CONFIG_TOML_DISABLED"):
        return {}
    config_file = Path(config_path)
    if config_file.exists():
        try:
            with open(config_file, "rb") as f:
                return tomllib.load(f)
        except (tomllib.TOMLDecodeError, OSError) as exc:
            logger.warning("Failed to load %s: %s", config_path, exc)
            return {}
    return {}


# Load TOML config once
TOML_CONFIG = load_toml_config()


ThinkingEffortLevel = Literal[
    "none", "minimal", "low", "medium", "high", "xhigh", "max"
]


class ModelOverrideSettings(BaseModel):
    """Advanced module-level transport overrides."""

    api_key: str | None = None
    api_key_env: str | None = None
    base_url: str | None = None

    provider_params: dict[str, Any] = Field(default_factory=dict)


class PromptCachePolicy(BaseModel):
    """Per-call prompt-caching configuration.

    Lives in config.py (not src/llm/caching.py) so ModelConfig can reference
    it as a field without a circular import. src/llm/caching.py re-exports
    this class for existing import paths.
    """

    mode: Literal["none", "prefix", "gemini_cached_content"] = "none"
    ttl_seconds: int | None = None
    key_version: str = "v1"


def _normalize_model_transport(data: Any) -> Any:
    """Normalize 'provider/model' shorthand into separate transport + model fields."""
    if not isinstance(data, dict):
        return data
    raw_data = cast(dict[Any, Any], data)
    update: dict[str, Any] = {str(key): value for key, value in raw_data.items()}
    model_value = update.get("model")
    transport_value = update.get("transport")
    if isinstance(model_value, str) and "/" in model_value and transport_value is None:
        prefix, bare_model = model_value.split("/", 1)
        if prefix in {"anthropic", "openai", "gemini"}:
            update["transport"] = prefix
            update["model"] = bare_model
    return update


def _validate_thinking_constraints(
    transport: ModelTransport, thinking_budget_tokens: int | None
) -> None:
    """Enforce transport-specific thinking_budget_tokens rules.

    Anthropic requires a minimum of 1024 tokens when thinking is enabled.
    Gemini/OpenAI accept any non-negative value (including 0 to disable).
    """
    if (
        transport == "anthropic"
        and thinking_budget_tokens is not None
        and 0 < thinking_budget_tokens < 1024
    ):
        raise ValueError("thinking_budget_tokens must be >= 1024 for Anthropic models")


class FallbackModelSettings(BaseModel):
    """Independent fallback model configuration. No inheritance from primary."""

    model: str
    transport: ModelTransport

    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    seed: int | None = None

    thinking_effort: ThinkingEffortLevel | None = Field(
        default=None,
        validation_alias=AliasChoices("thinking_effort", "reasoning_effort"),
    )
    thinking_budget_tokens: int | None = None

    max_output_tokens: int | None = None
    stop_sequences: list[str] | None = None

    cache_policy: PromptCachePolicy | None = None

    overrides: ModelOverrideSettings = Field(default_factory=ModelOverrideSettings)

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_model_format(cls, data: Any) -> Any:
        return _normalize_model_transport(data)

    @property
    def reasoning_effort(self) -> ThinkingEffortLevel | None:
        return self.thinking_effort

    @model_validator(mode="after")
    def _validate_runtime_shape(self) -> "FallbackModelSettings":
        _validate_thinking_constraints(self.transport, self.thinking_budget_tokens)
        return self


class ConfiguredModelSettings(BaseModel):
    """Operator-configurable persisted model settings."""

    model: str
    transport: ModelTransport

    fallback: FallbackModelSettings | None = None

    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    seed: int | None = None

    thinking_effort: ThinkingEffortLevel | None = Field(
        default=None,
        validation_alias=AliasChoices("thinking_effort", "reasoning_effort"),
    )
    thinking_budget_tokens: int | None = None

    max_output_tokens: int | None = None
    stop_sequences: list[str] | None = None

    cache_policy: PromptCachePolicy | None = None

    overrides: ModelOverrideSettings = Field(default_factory=ModelOverrideSettings)

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_model_format(cls, data: Any) -> Any:
        return _normalize_model_transport(data)

    @property
    def reasoning_effort(self) -> ThinkingEffortLevel | None:
        """Backward-compatible alias for the generic thinking effort field."""
        return self.thinking_effort

    @model_validator(mode="after")
    def _validate_runtime_shape(self) -> "ConfiguredModelSettings":
        _validate_thinking_constraints(self.transport, self.thinking_budget_tokens)
        return self


class ResolvedFallbackConfig(BaseModel):
    """Runtime-resolved fallback config with credentials already resolved."""

    model: str
    transport: ModelTransport

    api_key: str | None = None
    base_url: str | None = None

    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    seed: int | None = None

    thinking_effort: ThinkingEffortLevel | None = Field(
        default=None,
        validation_alias=AliasChoices("thinking_effort", "reasoning_effort"),
    )
    thinking_budget_tokens: int | None = None
    provider_params: dict[str, Any] = Field(default_factory=dict)

    max_output_tokens: int | None = None
    stop_sequences: list[str] | None = None

    cache_policy: PromptCachePolicy | None = None

    @property
    def reasoning_effort(self) -> ThinkingEffortLevel | None:
        return self.thinking_effort


class ModelConfig(BaseModel):
    """Reusable model configuration for any non-embedding LLM caller."""

    model: str
    transport: ModelTransport

    fallback: ResolvedFallbackConfig | None = None

    api_key: str | None = None
    base_url: str | None = None

    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    seed: int | None = None

    thinking_effort: ThinkingEffortLevel | None = Field(
        default=None,
        validation_alias=AliasChoices("thinking_effort", "reasoning_effort"),
    )
    thinking_budget_tokens: int | None = None
    provider_params: dict[str, Any] = Field(default_factory=dict)

    max_output_tokens: int | None = None
    stop_sequences: list[str] | None = None

    cache_policy: PromptCachePolicy | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_model_format(cls, data: Any) -> Any:
        return _normalize_model_transport(data)

    @property
    def reasoning_effort(self) -> ThinkingEffortLevel | None:
        """Backward-compatible alias for the generic thinking effort field."""
        return self.thinking_effort

    @model_validator(mode="after")
    def _validate_thinking_constraints_on_self(self) -> "ModelConfig":
        _validate_thinking_constraints(self.transport, self.thinking_budget_tokens)
        return self

    def for_model(
        self,
        model_override: str,
        *,
        transport_override: ModelTransport | None = None,
    ) -> "ModelConfig":
        return self.model_copy(
            update={
                "model": model_override,
                "transport": transport_override or self.transport,
            }
        )


class ConfiguredEmbeddingModelSettings(BaseModel):
    """Operator-configurable persisted embedding settings."""

    model: str = "text-embedding-3-small"
    transport: EmbeddingTransport = "openai"
    overrides: ModelOverrideSettings = Field(default_factory=ModelOverrideSettings)

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_model_format(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        raw_data = cast(dict[Any, Any], data)
        update: dict[str, Any] = {str(key): value for key, value in raw_data.items()}
        model_value = update.get("model")
        transport_value = update.get("transport")
        if (
            isinstance(model_value, str)
            and "/" in model_value
            and transport_value is None
        ):
            prefix, bare_model = model_value.split("/", 1)
            if prefix in {"openai", "gemini"}:
                update["transport"] = prefix
                update["model"] = bare_model
        return update

    @model_validator(mode="after")
    def _default_model_for_transport(self) -> "ConfiguredEmbeddingModelSettings":
        if "model" not in self.model_fields_set:
            self.model = _default_embedding_model_for_transport(self.transport)
        return self


class EmbeddingModelConfig(BaseModel):
    """Runtime embedding configuration with resolved credentials."""

    model: str = "text-embedding-3-small"
    transport: EmbeddingTransport = "openai"
    api_key: str | None = None
    base_url: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_model_format(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        raw_data = cast(dict[Any, Any], data)
        update: dict[str, Any] = {str(key): value for key, value in raw_data.items()}
        model_value = update.get("model")
        transport_value = update.get("transport")
        if (
            isinstance(model_value, str)
            and "/" in model_value
            and transport_value is None
        ):
            prefix, bare_model = model_value.split("/", 1)
            if prefix in {"openai", "gemini"}:
                update["transport"] = prefix
                update["model"] = bare_model
        return update

    @model_validator(mode="after")
    def _default_model_for_transport(self) -> "EmbeddingModelConfig":
        if "model" not in self.model_fields_set:
            self.model = _default_embedding_model_for_transport(self.transport)
        return self


def _resolve_secret(value: str | None, env_name: str | None) -> str | None:
    if value is not None:
        return value
    if env_name is None:
        return None
    return os.getenv(env_name)


def _resolve_fallback_config(
    fallback: FallbackModelSettings,
) -> ResolvedFallbackConfig:
    """Resolve a FallbackModelSettings into a runtime ResolvedFallbackConfig."""
    return ResolvedFallbackConfig(
        model=fallback.model,
        transport=fallback.transport,
        api_key=_resolve_secret(
            fallback.overrides.api_key,
            fallback.overrides.api_key_env,
        ),
        base_url=fallback.overrides.base_url,
        temperature=fallback.temperature,
        top_p=fallback.top_p,
        top_k=fallback.top_k,
        frequency_penalty=fallback.frequency_penalty,
        presence_penalty=fallback.presence_penalty,
        seed=fallback.seed,
        thinking_effort=fallback.thinking_effort,
        thinking_budget_tokens=fallback.thinking_budget_tokens,
        provider_params=fallback.overrides.provider_params,
        max_output_tokens=fallback.max_output_tokens,
        stop_sequences=fallback.stop_sequences,
        cache_policy=fallback.cache_policy,
    )


def resolve_model_config(configured: ConfiguredModelSettings) -> ModelConfig:
    """Resolve persisted model settings into the runtime ModelConfig."""

    resolved_fallback = (
        _resolve_fallback_config(configured.fallback)
        if configured.fallback is not None
        else None
    )

    return ModelConfig(
        model=configured.model,
        transport=configured.transport,
        fallback=resolved_fallback,
        api_key=_resolve_secret(
            configured.overrides.api_key,
            configured.overrides.api_key_env,
        ),
        base_url=configured.overrides.base_url,
        temperature=configured.temperature,
        top_p=configured.top_p,
        top_k=configured.top_k,
        frequency_penalty=configured.frequency_penalty,
        presence_penalty=configured.presence_penalty,
        seed=configured.seed,
        thinking_effort=configured.thinking_effort,
        thinking_budget_tokens=configured.thinking_budget_tokens,
        provider_params=configured.overrides.provider_params,
        max_output_tokens=configured.max_output_tokens,
        stop_sequences=configured.stop_sequences,
        cache_policy=configured.cache_policy,
    )


def _default_embedding_api_key(transport: EmbeddingTransport) -> str | None:
    """Fall back to the global LLM API key for the matching transport."""
    if transport == "openai":
        return settings.LLM.OPENAI_API_KEY
    if transport == "gemini":
        return settings.LLM.GEMINI_API_KEY


def resolve_embedding_model_config(
    configured: ConfiguredEmbeddingModelSettings,
) -> EmbeddingModelConfig:
    """Resolve persisted embedding settings into the runtime config."""

    api_key = _resolve_secret(
        configured.overrides.api_key,
        configured.overrides.api_key_env,
    )
    if api_key is None:
        api_key = _default_embedding_api_key(configured.transport)

    return EmbeddingModelConfig(
        model=configured.model,
        transport=configured.transport,
        api_key=api_key,
        base_url=configured.overrides.base_url,
    )


_TRANSPORT_SPECIFIC_THINKING_KEYS: frozenset[str] = frozenset(
    {"thinking_budget_tokens", "thinking_effort"}
)


def _fill_defaults_for_nested_field(
    data: dict[str, Any],
    field_name: str,
    default_factory: Any,
) -> dict[str, Any]:
    """Fill missing keys in a partial nested dict from the field's defaults.

    When Pydantic's env_nested_delimiter splits an env var like
    ``DERIVER_MODEL_CONFIG__THINKING_BUDGET_TOKENS=2048`` it produces
    ``{"MODEL_CONFIG": {"THINKING_BUDGET_TOKENS": 2048}}``.  Without merging
    that partial dict would fail validation because required keys like
    ``model`` and ``transport`` are missing.  This helper fills them from
    the field's ``default_factory`` so partial overrides work.

    If the env override switches ``transport`` to a value that differs from
    the default's, transport-specific thinking params
    (``thinking_budget_tokens``, ``thinking_effort``) are dropped from the
    default before merging.  This prevents e.g. a Gemini default's
    ``thinking_budget_tokens=1024`` from leaking into an OpenAI override,
    which would then be rejected by the OpenAI backend (OpenAI uses
    ``reasoning.effort``, not a token budget). Explicit thinking params in
    the env override are preserved.
    """
    raw: Any = data.get(field_name) or data.get(field_name.lower())
    if not isinstance(raw, dict):
        return data

    default_obj = default_factory()
    if isinstance(default_obj, BaseModel):
        default_dict: dict[str, Any] = default_obj.model_dump(by_alias=True)
    else:
        default_dict = dict(default_obj)

    raw_dict = cast(dict[str, Any], raw)
    raw_lower = {k.lower(): v for k, v in raw_dict.items()}
    default_lower = {k.lower(): v for k, v in default_dict.items()}
    override_transport = raw_lower.get("transport")
    default_transport = default_lower.get("transport")
    if override_transport is not None and override_transport != default_transport:
        for k in list(default_dict.keys()):
            if k.lower() in _TRANSPORT_SPECIFIC_THINKING_KEYS:
                del default_dict[k]

    merged: dict[str, Any] = {**default_dict, **raw_dict}
    # Preserve the key casing used in data
    key = field_name if field_name in data else field_name.lower()
    data[key] = merged
    return data


class TomlConfigSettingsSource(PydanticBaseSettingsSource):
    """Custom settings source for loading from TOML file."""

    def __init__(self, settings_cls: type[BaseSettings]) -> None:
        super().__init__(settings_cls)

    SECTION_MAP: ClassVar[dict[str, str]] = {
        "DB": "db",
        "AUTH": "auth",
        "SENTRY": "sentry",
        "CACHE": "cache",
        "LLM": "llm",
        "EMBEDDING": "embedding",
        "DERIVER": "deriver",
        "PEER_CARD": "peer_card",
        "DIALECTIC": "dialectic",
        "SUMMARY": "summary",
        "WEBHOOK": "webhook",
        "DREAM": "dream",
        "VECTOR_STORE": "vector_store",
        "METRICS": "metrics",
        "TELEMETRY": "telemetry",
        "": "app",  # For AppSettings with no prefix
    }

    def get_field_value(
        self, field: FieldInfo, field_name: str
    ) -> tuple[Any, str, bool]:
        # Get the env_prefix from the model config
        prefix = self.settings_cls.model_config.get("env_prefix", "")
        if prefix.endswith("_"):
            prefix = prefix[:-1]

        # Map prefixes to TOML sections
        section = self.SECTION_MAP.get(prefix, prefix.lower())
        toml_data = TOML_CONFIG.get(section, {})

        # Try different case variations
        field_value = toml_data.get(field_name.lower())
        if field_value is None:
            field_value = toml_data.get(field_name.upper())
        if field_value is None:
            field_value = toml_data.get(field_name)

        return field_value, field_name, False

    def __call__(self) -> dict[str, Any]:
        # Get the env_prefix from the model config
        prefix = self.settings_cls.model_config.get("env_prefix", "")
        if prefix.endswith("_"):
            prefix = prefix[:-1]

        section = self.SECTION_MAP.get(prefix, prefix.lower())
        toml_data = TOML_CONFIG.get(section, {})

        # Convert keys to uppercase to match field names
        return {key.upper(): value for key, value in toml_data.items()}


class HonchoSettings(BaseSettings):
    """Base class for all settings models in Honcho.

    Defines the source precedence for loading settings.
    """

    @classmethod
    def settings_customise_sources(  # pyright: ignore
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: EnvSettingsSource,
        dotenv_settings: DotEnvSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Correct precedence: init > env > .env > toml > secrets > defaults
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            TomlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )


class DBSettings(HonchoSettings):
    model_config = SettingsConfigDict(env_prefix="DB_", extra="ignore")  # pyright: ignore

    CONNECTION_URI: str = (
        "postgresql+psycopg://postgres:postgres@localhost:5432/postgres"
    )
    SCHEMA: str = "public"
    POOL_CLASS: str = "default"
    POOL_PRE_PING: bool = True
    POOL_SIZE: Annotated[int, Field(default=10, gt=0, le=1000)] = 10
    MAX_OVERFLOW: Annotated[int, Field(default=20, ge=0, le=1000)] = 20
    POOL_TIMEOUT: Annotated[int, Field(default=30, gt=0, le=300)] = (
        30  # seconds (max 5 minutes)
    )
    POOL_RECYCLE: Annotated[int, Field(default=300, gt=0, le=7200)] = (
        300  # seconds (max 2 hours)
    )
    POOL_USE_LIFO: bool = True
    SQL_DEBUG: bool = False
    TRACING: bool = False


class AuthSettings(HonchoSettings):
    model_config = SettingsConfigDict(env_prefix="AUTH_", extra="ignore")  # pyright: ignore

    USE_AUTH: bool = False
    JWT_SECRET: str | None = None  # Must be set if USE_AUTH is true

    @model_validator(mode="after")  # type: ignore
    def _require_jwt_secret(self) -> "AuthSettings":
        if self.USE_AUTH and not self.JWT_SECRET:
            raise ValueError("JWT_SECRET must be set if USE_AUTH is true")
        return self


class SentrySettings(HonchoSettings):
    model_config = SettingsConfigDict(env_prefix="SENTRY_", extra="ignore")  # pyright: ignore

    ENABLED: bool = False
    DSN: str | None = None
    RELEASE: str | None = None  # TODO maybe centralize this with release number
    ENVIRONMENT: str = "development"
    TRACES_SAMPLE_RATE: Annotated[float, Field(default=0.1, ge=0.0, le=1.0)] = 0.1
    PROFILES_SAMPLE_RATE: Annotated[float, Field(default=0.1, ge=0.0, le=1.0)] = 0.1


class LLMSettings(HonchoSettings):
    model_config = SettingsConfigDict(env_prefix="LLM_", extra="ignore")  # pyright: ignore

    # API Keys for LLM providers
    ANTHROPIC_API_KEY: str | None = None
    OPENAI_API_KEY: str | None = None
    GEMINI_API_KEY: str | None = None

    # General LLM settings
    DEFAULT_MAX_TOKENS: Annotated[int, Field(default=1000, gt=0, le=100_000)] = 2500

    # Maximum characters for tool output to prevent token explosion.
    # Set to 10,000 chars (~2,500 tokens at 4 chars/token) to stay well under
    # typical context limits while providing substantial tool output.
    MAX_TOOL_OUTPUT_CHARS: Annotated[int, Field(default=10000, gt=0, le=100_000)] = (
        10000
    )

    # Maximum characters for individual message content in tool results.
    # Keeps each message preview concise while preserving key context.
    MAX_MESSAGE_CONTENT_CHARS: Annotated[int, Field(default=2000, gt=0, le=10_000)] = (
        2000
    )


class EmbeddingSettings(HonchoSettings):
    model_config = SettingsConfigDict(  # pyright: ignore
        env_prefix="EMBEDDING_", env_nested_delimiter="__", extra="ignore"
    )

    @staticmethod
    def _MODEL_CONFIG_DEFAULT() -> ConfiguredEmbeddingModelSettings:
        return ConfiguredEmbeddingModelSettings(
            transport="openai",
            model="text-embedding-3-small",
        )

    MODEL_CONFIG: ConfiguredEmbeddingModelSettings = Field(
        default_factory=_MODEL_CONFIG_DEFAULT
    )
    VECTOR_DIMENSIONS: Annotated[int, Field(default=1536, gt=0)] = 1536
    MAX_INPUT_TOKENS: Annotated[int, Field(default=8192, gt=0)] = 8192
    MAX_TOKENS_PER_REQUEST: Annotated[int, Field(default=300_000, gt=0)] = 300_000

    @model_validator(mode="before")
    @classmethod
    def _merge_model_config_defaults(cls, data: Any) -> Any:
        if isinstance(data, dict):
            _fill_defaults_for_nested_field(
                cast(dict[str, Any], data),
                "MODEL_CONFIG",
                cls._MODEL_CONFIG_DEFAULT,
            )
        return data  # pyright: ignore[reportUnknownVariableType]


class DeriverSettings(HonchoSettings):
    model_config = SettingsConfigDict(  # pyright: ignore
        env_prefix="DERIVER_", env_nested_delimiter="__", extra="ignore"
    )

    ENABLED: bool = True

    WORKERS: Annotated[int, Field(default=1, gt=0, le=100)] = 1
    POLLING_SLEEP_INTERVAL_SECONDS: Annotated[
        float, Field(default=1.0, gt=0.0, le=60.0)
    ] = 1.0
    STALE_SESSION_TIMEOUT_MINUTES: Annotated[int, Field(default=5, gt=0, le=1440)] = 5

    # Retention window (seconds) for keeping errored items in the queue
    QUEUE_ERROR_RETENTION_SECONDS: Annotated[
        int, Field(default=30 * 24 * 3600, gt=0)
    ] = 30 * 24 * 3600  # 30 days default

    @staticmethod
    def _MODEL_CONFIG_DEFAULT() -> ConfiguredModelSettings:
        # Minimal default: transport + model only. Any other knobs would merge
        # into operator-supplied env / config.toml overrides via
        # _fill_defaults_for_nested_field and clobber intent.
        return ConfiguredModelSettings(
            transport="openai",
            model="gpt-5.4-mini",
        )

    MODEL_CONFIG: ConfiguredModelSettings = Field(default_factory=_MODEL_CONFIG_DEFAULT)

    # Whether to deduplicate documents when creating them
    DEDUPLICATE: bool = True

    LOG_OBSERVATIONS: bool = False

    MAX_INPUT_TOKENS: Annotated[int, Field(default=23000, gt=0, le=23000)] = 23000

    # Maximum number of observations to return in working representation
    # This is applied to both explicit and deductive observations
    WORKING_REPRESENTATION_MAX_OBSERVATIONS: Annotated[
        int, Field(default=100, gt=0, le=1000)
    ] = 100

    REPRESENTATION_BATCH_MAX_TOKENS: Annotated[
        int,
        Field(default=1024, ge=128, le=16_384),
    ] = 1024

    # When enabled, bypasses the batch token threshold and processes work immediately
    FLUSH_ENABLED: bool = False

    @model_validator(mode="before")
    @classmethod
    def _merge_model_config_defaults(cls, data: Any) -> Any:
        if isinstance(data, dict):
            _fill_defaults_for_nested_field(
                cast(dict[str, Any], data),
                "MODEL_CONFIG",
                cls._MODEL_CONFIG_DEFAULT,
            )
        return data  # pyright: ignore[reportUnknownVariableType]

    @model_validator(mode="after")
    def validate_batch_tokens_vs_context_limit(self):
        if self.REPRESENTATION_BATCH_MAX_TOKENS > self.MAX_INPUT_TOKENS:
            raise ValueError(
                f"REPRESENTATION_BATCH_MAX_TOKENS ({self.REPRESENTATION_BATCH_MAX_TOKENS}) cannot exceed max deriver input tokens ({self.MAX_INPUT_TOKENS})"
            )
        return self


class PeerCardSettings(HonchoSettings):
    model_config = SettingsConfigDict(env_prefix="PEER_CARD_", extra="ignore")  # pyright: ignore

    ENABLED: bool = True


# Reasoning levels for dialectic - defined here to avoid circular imports with schemas
ReasoningLevel = Literal["minimal", "low", "medium", "high", "max"]
REASONING_LEVELS: list[ReasoningLevel] = [
    "minimal",
    "low",
    "medium",
    "high",
    "max",
]


class DialecticLevelSettings(BaseModel):
    """Settings for a specific reasoning level in the dialectic."""

    model_config = SettingsConfigDict(populate_by_name=True)  # pyright: ignore

    MODEL_CONFIG: Annotated[
        ConfiguredModelSettings,
        Field(validation_alias="model_config"),
    ]
    MAX_TOOL_ITERATIONS: Annotated[
        int, Field(ge=0, le=50, validation_alias="max_tool_iterations")
    ]
    MAX_OUTPUT_TOKENS: Annotated[
        int | None, Field(ge=1, le=100_000, validation_alias="max_output_tokens")
    ] = None  # None means use global DIALECTIC.MAX_OUTPUT_TOKENS
    TOOL_CHOICE: Annotated[str | None, Field(validation_alias="tool_choice")] = (
        None  # None/auto lets model decide, "any"/"required" forces tool use
    )

    @model_validator(mode="after")
    def _validate_anthropic_thinking_budget(self) -> "DialecticLevelSettings":
        """Ensure Anthropic thinking budget is >= 1024 when enabled."""
        if (
            self.MODEL_CONFIG.transport == "anthropic"
            and self.MODEL_CONFIG.thinking_budget_tokens is not None
            and self.MODEL_CONFIG.thinking_budget_tokens > 0
            and self.MODEL_CONFIG.thinking_budget_tokens < 1024
        ):
            raise ValueError(
                "MODEL_CONFIG.thinking_budget_tokens must be >= 1024 for "
                + "Anthropic models when enabled "
                + f"(got {self.MODEL_CONFIG.thinking_budget_tokens})"
            )
        return self


def _default_dialectic_levels() -> dict[ReasoningLevel, DialecticLevelSettings]:
    # Minimal defaults per level: transport + model only. Non-MODEL_CONFIG
    # level tuning (MAX_TOOL_ITERATIONS, MAX_OUTPUT_TOKENS, TOOL_CHOICE)
    # stays here because it's the per-level behavior, not a model knob —
    # operators still override any of it via
    # DIALECTIC_LEVELS__<level>__MODEL_CONFIG__* without conflict.
    def _default_model_config() -> ConfiguredModelSettings:
        return ConfiguredModelSettings(
            transport="openai",
            model="gpt-5.4-mini",
        )

    return {
        "minimal": DialecticLevelSettings(
            MODEL_CONFIG=_default_model_config(),
            MAX_TOOL_ITERATIONS=1,
            MAX_OUTPUT_TOKENS=250,
            TOOL_CHOICE="any",
        ),
        "low": DialecticLevelSettings(
            MODEL_CONFIG=_default_model_config(),
            MAX_TOOL_ITERATIONS=5,
            TOOL_CHOICE="any",
        ),
        "medium": DialecticLevelSettings(
            MODEL_CONFIG=_default_model_config(),
            MAX_TOOL_ITERATIONS=2,
        ),
        "high": DialecticLevelSettings(
            MODEL_CONFIG=_default_model_config(),
            MAX_TOOL_ITERATIONS=4,
        ),
        "max": DialecticLevelSettings(
            MODEL_CONFIG=_default_model_config(),
            MAX_TOOL_ITERATIONS=10,
        ),
    }


class DialecticSettings(HonchoSettings):
    model_config = SettingsConfigDict(  # pyright: ignore
        env_prefix="DIALECTIC_", env_nested_delimiter="__", extra="ignore"
    )

    LEVELS: dict[ReasoningLevel, DialecticLevelSettings] = Field(
        default_factory=_default_dialectic_levels
    )

    MAX_OUTPUT_TOKENS: Annotated[int, Field(default=8192, gt=0, le=100_000)] = 8192
    MAX_INPUT_TOKENS: Annotated[int, Field(default=100_000, gt=0, le=200_000)] = 100_000

    # Token limit for get_recent_history tool within the agent
    HISTORY_TOKEN_LIMIT: Annotated[int, Field(default=8192, gt=0, le=100_000)] = 8192

    # Session history injection: max tokens of recent messages to include when session_id is specified.
    # Set to 0 to disable automatic session history injection.
    SESSION_HISTORY_MAX_TOKENS: Annotated[
        int, Field(default=4_096, ge=0, le=16_384)
    ] = 4_096

    @model_validator(mode="before")
    @classmethod
    def _merge_level_defaults(cls, data: Any) -> Any:
        """Merge partial level overrides with built-in defaults."""
        if not isinstance(data, dict):
            return data
        typed_data = cast(dict[str, Any], data)
        levels_raw: dict[str, Any] | None = typed_data.get("LEVELS") or typed_data.get(
            "levels"
        )
        if not isinstance(levels_raw, dict):
            return data  # pyright: ignore[reportUnknownVariableType]
        defaults = _default_dialectic_levels()
        for level_name_key, level_override_val in levels_raw.items():
            level_name = str(level_name_key)
            if not isinstance(level_override_val, dict):
                continue
            level_override = cast(dict[str, Any], level_override_val)
            if level_name in defaults:
                base: dict[str, Any] = defaults[level_name].model_dump(by_alias=True)
                # Recursively merge nested MODEL_CONFIG / model_config too.
                # model_dump() always produces the Python field name
                # ("MODEL_CONFIG"), but TOML overrides arrive as lowercase
                # ("model_config").  Check both casings in the override and
                # resolve the base value from whichever casing is present.
                for mc_key in ("MODEL_CONFIG", "model_config"):
                    if mc_key in level_override and isinstance(
                        level_override[mc_key], dict
                    ):
                        base_mc: dict[str, Any] = dict(
                            base.get("MODEL_CONFIG") or base.get("model_config") or {}
                        )
                        override_mc = cast(dict[str, Any], level_override[mc_key])
                        override_lower = {k.lower(): v for k, v in override_mc.items()}
                        base_lower = {k.lower(): v for k, v in base_mc.items()}
                        override_transport = override_lower.get("transport")
                        base_transport = base_lower.get("transport")
                        if (
                            override_transport is not None
                            and override_transport != base_transport
                        ):
                            for k in list(base_mc.keys()):
                                if k.lower() in _TRANSPORT_SPECIFIC_THINKING_KEYS:
                                    del base_mc[k]
                        level_override[mc_key] = {**base_mc, **override_mc}
                levels_raw[level_name] = {**base, **level_override}
        return data  # pyright: ignore[reportUnknownVariableType]

    @model_validator(mode="after")
    def _validate_token_budgets(self) -> "DialecticSettings":
        """Ensure the output token limit exceeds all thinking budgets."""
        for level, level_settings in self.LEVELS.items():
            thinking_budget = level_settings.MODEL_CONFIG.thinking_budget_tokens or 0
            effective_max = (
                level_settings.MAX_OUTPUT_TOKENS
                if level_settings.MAX_OUTPUT_TOKENS is not None
                else self.MAX_OUTPUT_TOKENS
            )
            if thinking_budget > 0 and thinking_budget >= effective_max:
                raise ValueError(
                    "MAX_OUTPUT_TOKENS must be greater than MODEL_CONFIG."
                    + f"thinking_budget_tokens for level '{level}'"
                )
        return self

    @model_validator(mode="after")
    def _validate_all_levels_present(self) -> "DialecticSettings":
        """Ensure all reasoning levels are configured."""
        missing = set(REASONING_LEVELS) - set(self.LEVELS.keys())
        if missing:
            raise ValueError(f"Missing configuration for reasoning levels: {missing}")
        return self


class SummarySettings(HonchoSettings):
    model_config = SettingsConfigDict(  # pyright: ignore
        env_prefix="SUMMARY_", env_nested_delimiter="__", extra="ignore"
    )

    ENABLED: bool = True

    MESSAGES_PER_SHORT_SUMMARY: Annotated[int, Field(default=20, gt=0, le=100)] = 20
    MESSAGES_PER_LONG_SUMMARY: Annotated[int, Field(default=60, gt=0, le=500)] = 60

    @staticmethod
    def _MODEL_CONFIG_DEFAULT() -> ConfiguredModelSettings:
        # Minimal default; extra knobs would merge into env/TOML overrides.
        return ConfiguredModelSettings(
            transport="openai",
            model="gpt-5.4-mini",
        )

    MODEL_CONFIG: ConfiguredModelSettings = Field(default_factory=_MODEL_CONFIG_DEFAULT)

    @model_validator(mode="before")
    @classmethod
    def _merge_model_config_defaults(cls, data: Any) -> Any:
        if isinstance(data, dict):
            _fill_defaults_for_nested_field(
                cast(dict[str, Any], data),
                "MODEL_CONFIG",
                cls._MODEL_CONFIG_DEFAULT,
            )
        return data  # pyright: ignore[reportUnknownVariableType]

    MAX_TOKENS_SHORT: Annotated[int, Field(default=1000, gt=0, le=10_000)] = 1000
    MAX_TOKENS_LONG: Annotated[int, Field(default=4000, gt=0, le=20_000)] = 4000


class WebhookSettings(HonchoSettings):
    model_config = SettingsConfigDict(env_prefix="WEBHOOK_", extra="ignore")  # pyright: ignore

    SECRET: str | None = None  # Must be set if configuring webhooks
    MAX_WORKSPACE_LIMIT: int = 10


class MetricsSettings(HonchoSettings):
    model_config = SettingsConfigDict(env_prefix="METRICS_", extra="ignore")  # pyright: ignore
    ENABLED: bool = False
    NAMESPACE: str | None = None


class TelemetrySettings(HonchoSettings):
    """CloudEvents telemetry settings for analytics.

    These settings configure the CloudEvents emitter for pushing
    structured events to an analytics backend.
    """

    model_config = SettingsConfigDict(env_prefix="TELEMETRY_", extra="ignore")  # pyright: ignore

    # Master toggle for CloudEvents emission
    ENABLED: bool = False

    # CloudEvents HTTP endpoint (e.g., "https://telemetry.honcho.dev/v1/events")
    ENDPOINT: str | None = None

    # Optional headers for authentication
    HEADERS: dict[str, str] | None = None

    # Batching configuration
    BATCH_SIZE: Annotated[int, Field(default=100, gt=0, le=1000)] = 100
    FLUSH_INTERVAL_SECONDS: Annotated[float, Field(default=1.0, gt=0.0, le=60.0)] = 1.0
    FLUSH_THRESHOLD: Annotated[int, Field(default=50, gt=0, le=1000)] = 50

    # Retry configuration
    MAX_RETRIES: Annotated[int, Field(default=3, gt=0, le=10)] = 3

    # Buffer configuration
    MAX_BUFFER_SIZE: Annotated[int, Field(default=10000, gt=0, le=100000)] = 10000

    # Namespace for instance identification (propagated from top-level NAMESPACE if not set)
    NAMESPACE: str | None = None


class CacheSettings(HonchoSettings):
    model_config = SettingsConfigDict(env_prefix="CACHE_", extra="ignore")  # pyright: ignore

    ENABLED: bool = False
    URL: str = "redis://localhost:6379/0?suppress=true"
    NAMESPACE: str | None = None
    DEFAULT_TTL_SECONDS: Annotated[int, Field(default=300, ge=1, le=86_400)] = (
        300  # how long to keep items in cache
    )

    DEFAULT_LOCK_TTL_SECONDS: Annotated[int, Field(default=5, ge=1, le=86_400)] = (
        5  # how long to hold a lock on a resource when fetching DB after cache miss
    )


class SurprisalSettings(BaseModel):
    """Settings for tree-based surprisal sampling during dreams."""

    ENABLED: bool = False

    # Tree configuration
    TREE_TYPE: Literal[
        "kdtree", "balltree", "rptree", "covertree", "lsh", "graph", "prototype"
    ] = "kdtree"
    TREE_K: Annotated[int, Field(default=5, gt=0, le=20)] = 5  # k for kNN-based trees

    # Sampling strategy
    SAMPLING_STRATEGY: Literal["recent", "random", "all"] = "recent"
    SAMPLE_SIZE: Annotated[int, Field(default=200, gt=0, le=2000)] = 200

    # Surprisal filtering (normalized scores: 0.0 = lowest, 1.0 = highest)
    TOP_PERCENT_SURPRISAL: Annotated[float, Field(default=0.10, gt=0.0, le=1.0)] = (
        0.10  # Top 10% of observations
    )
    # Hybrid mode: min high-surprisal observations to replace standard questions
    MIN_HIGH_SURPRISAL_FOR_REPLACE: Annotated[int, Field(default=10, gt=0)] = 10

    # Observation level filtering
    INCLUDE_LEVELS: list[str] = ["explicit", "deductive"]


class DreamSettings(HonchoSettings):
    model_config = SettingsConfigDict(  # pyright: ignore
        env_prefix="DREAM_", env_nested_delimiter="__", extra="ignore"
    )

    ENABLED: bool = True
    DOCUMENT_THRESHOLD: Annotated[int, Field(default=50, gt=0, le=1000)] = 50
    IDLE_TIMEOUT_MINUTES: Annotated[int, Field(default=60, gt=0, le=1440)] = 60
    MIN_HOURS_BETWEEN_DREAMS: Annotated[int, Field(default=8, gt=0, le=72)] = 8
    ENABLED_TYPES: list[str] = ["omni"]

    # Agent iteration limit - increased for extended reasoning workflow
    MAX_TOOL_ITERATIONS: Annotated[int, Field(default=20, gt=0, le=50)] = 20

    # Token limit for get_recent_history tool within the agent
    HISTORY_TOKEN_LIMIT: Annotated[int, Field(default=16_384, gt=0, le=200_000)] = (
        16_384
    )

    @staticmethod
    def _DEDUCTION_MODEL_CONFIG_DEFAULT() -> ConfiguredModelSettings:
        # Minimal default; extra knobs would merge into env/TOML overrides.
        return ConfiguredModelSettings(
            transport="openai",
            model="gpt-5.4-mini",
        )

    DEDUCTION_MODEL_CONFIG: ConfiguredModelSettings = Field(
        default_factory=_DEDUCTION_MODEL_CONFIG_DEFAULT
    )

    @staticmethod
    def _INDUCTION_MODEL_CONFIG_DEFAULT() -> ConfiguredModelSettings:
        # Minimal default; extra knobs would merge into env/TOML overrides.
        return ConfiguredModelSettings(
            transport="openai",
            model="gpt-5.4-mini",
        )

    INDUCTION_MODEL_CONFIG: ConfiguredModelSettings = Field(
        default_factory=_INDUCTION_MODEL_CONFIG_DEFAULT
    )

    # Surprisal-based sampling subsystem
    SURPRISAL: SurprisalSettings = Field(default_factory=SurprisalSettings)

    @model_validator(mode="before")
    @classmethod
    def _merge_model_config_defaults(cls, data: Any) -> Any:
        if isinstance(data, dict):
            typed_data = cast(dict[str, Any], data)
            _fill_defaults_for_nested_field(
                typed_data,
                "DEDUCTION_MODEL_CONFIG",
                cls._DEDUCTION_MODEL_CONFIG_DEFAULT,
            )
            _fill_defaults_for_nested_field(
                typed_data,
                "INDUCTION_MODEL_CONFIG",
                cls._INDUCTION_MODEL_CONFIG_DEFAULT,
            )
        return data  # pyright: ignore[reportUnknownVariableType]

    @model_validator(mode="after")
    def _validate_specialist_token_budgets(self) -> "DreamSettings":
        """Ensure thinking_budget_tokens < max_output_tokens for each specialist."""
        for name, cfg in (
            ("DEDUCTION_MODEL_CONFIG", self.DEDUCTION_MODEL_CONFIG),
            ("INDUCTION_MODEL_CONFIG", self.INDUCTION_MODEL_CONFIG),
        ):
            if (
                cfg.max_output_tokens is not None
                and cfg.thinking_budget_tokens is not None
                and cfg.max_output_tokens <= cfg.thinking_budget_tokens
            ):
                raise ValueError(
                    f"dream.{name}.max_output_tokens must be greater than "
                    + f"dream.{name}.thinking_budget_tokens"
                )
        return self


class VectorStoreSettings(HonchoSettings):
    """Settings for vector store (pgvector, Turbopuffer, or LanceDB)."""

    model_config = SettingsConfigDict(env_prefix="VECTOR_STORE_", extra="ignore")  # pyright: ignore

    # Vector store type to use
    TYPE: Literal["pgvector", "turbopuffer", "lancedb"] = "pgvector"

    MIGRATED: bool = False

    # Global namespace prefix for all vector namespaces
    # Namespaces follow the pattern: {NAMESPACE}.{type}.{hash}
    # where hash is a base64url-encoded SHA-256 of the workspace/peer names
    # - Documents: {NAMESPACE}.doc.{hash(workspace, observer, observed)}
    # - Messages: {NAMESPACE}.msg.{hash(workspace)}
    NAMESPACE: str = "honcho"

    DIMENSIONS: Annotated[
        int,
        Field(
            default=1536,
            gt=0,
        ),
    ] = 1536

    # Turbopuffer-specific settings
    TURBOPUFFER_API_KEY: str | None = None
    TURBOPUFFER_REGION: str | None = None

    # LanceDB-specific settings (local embedded mode)
    LANCEDB_PATH: str = "./lancedb_data"

    RECONCILIATION_INTERVAL_SECONDS: Annotated[int, Field(default=300, gt=0)] = (
        300  # 5 minutes
    )

    @model_validator(mode="after")
    def _require_api_key_for_turbopuffer(self) -> "VectorStoreSettings":
        if self.TYPE == "turbopuffer" and not self.TURBOPUFFER_API_KEY:
            raise ValueError(
                "VECTOR_STORE_TURBOPUFFER_API_KEY must be set when TYPE is 'turbopuffer'"
            )
        return self


class AppSettings(HonchoSettings):
    # No env_prefix for app-level settings
    model_config = SettingsConfigDict(  # pyright: ignore
        env_prefix="", env_nested_delimiter="__", extra="ignore"
    )

    # Application-wide settings
    LOG_LEVEL: str = "INFO"
    SESSION_OBSERVERS_LIMIT: Annotated[int, Field(default=10, gt=0)] = 10
    MAX_FILE_SIZE: Annotated[int, Field(default=5_242_880, gt=0)] = 5_242_880  # 5MB
    GET_CONTEXT_MAX_TOKENS: Annotated[int, Field(default=100_000, gt=0, le=250_000)] = (
        100_000
    )

    MAX_MESSAGE_SIZE: Annotated[int, Field(default=25_000, gt=0)] = 25_000
    EMBED_MESSAGES: bool = True
    LANGFUSE_HOST: str | None = None
    LANGFUSE_PUBLIC_KEY: str | None = None

    COLLECT_METRICS_LOCAL: bool = False
    LOCAL_METRICS_FILE: str = "metrics.jsonl"
    REASONING_TRACES_FILE: str | None = None  # Path to JSONL file for reasoning traces

    NAMESPACE: str = "honcho"  # Top-level namespace for all settings, can be overridden by nested-model settings

    # Nested settings models
    DB: DBSettings = Field(default_factory=DBSettings)
    AUTH: AuthSettings = Field(default_factory=AuthSettings)
    SENTRY: SentrySettings = Field(default_factory=SentrySettings)
    LLM: LLMSettings = Field(default_factory=LLMSettings)
    EMBEDDING: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    DERIVER: DeriverSettings = Field(default_factory=DeriverSettings)
    DIALECTIC: DialecticSettings = Field(default_factory=DialecticSettings)
    PEER_CARD: PeerCardSettings = Field(default_factory=PeerCardSettings)
    SUMMARY: SummarySettings = Field(default_factory=SummarySettings)
    WEBHOOK: WebhookSettings = Field(default_factory=WebhookSettings)
    METRICS: MetricsSettings = Field(default_factory=MetricsSettings)
    TELEMETRY: TelemetrySettings = Field(default_factory=TelemetrySettings)
    CACHE: CacheSettings = Field(default_factory=CacheSettings)
    DREAM: DreamSettings = Field(default_factory=DreamSettings)
    VECTOR_STORE: VectorStoreSettings = Field(default_factory=VectorStoreSettings)

    @field_validator("LOG_LEVEL")
    def validate_log_level(cls, v: str) -> str:
        log_level = v.upper()
        if log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ValueError(f"Invalid log level: {v}")
        return log_level

    @model_validator(mode="after")
    def propagate_namespace(self) -> "AppSettings":
        """Propagate top-level NAMESPACE to nested settings if not explicitly set."""
        if "NAMESPACE" not in self.CACHE.model_fields_set:
            self.CACHE.NAMESPACE = self.NAMESPACE
        if "NAMESPACE" not in self.VECTOR_STORE.model_fields_set:
            self.VECTOR_STORE.NAMESPACE = self.NAMESPACE
        if "DIMENSIONS" not in self.VECTOR_STORE.model_fields_set:
            self.VECTOR_STORE.DIMENSIONS = self.EMBEDDING.VECTOR_DIMENSIONS
        elif self.VECTOR_STORE.DIMENSIONS != self.EMBEDDING.VECTOR_DIMENSIONS:
            raise ValueError(
                "VECTOR_STORE.DIMENSIONS must match EMBEDDING.VECTOR_DIMENSIONS"
            )
        if "NAMESPACE" not in self.TELEMETRY.model_fields_set:
            self.TELEMETRY.NAMESPACE = self.NAMESPACE
        if "NAMESPACE" not in self.METRICS.model_fields_set:
            self.METRICS.NAMESPACE = self.NAMESPACE

        if self.EMBEDDING.VECTOR_DIMENSIONS != 1536 and (
            self.VECTOR_STORE.TYPE == "pgvector" or not self.VECTOR_STORE.MIGRATED
        ):
            raise ValueError(
                "EMBEDDING.VECTOR_DIMENSIONS must remain 1536 while pgvector is "
                + "active or vector-store migration is incomplete"
            )

        return self


# Create a single global instance of the settings
settings: AppSettings = AppSettings()
