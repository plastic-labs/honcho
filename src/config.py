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
load_dotenv(override=True)

logger = logging.getLogger(__name__)


def load_toml_config(config_path: str = "config.toml") -> dict[str, Any]:
    """Load configuration from TOML file if it exists."""
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


class ModelOverrideSettings(BaseModel):
    """Advanced module-level transport overrides."""

    api_key: str | None = None
    api_key_env: str | None = None
    base_url: str | None = None

    fallback_api_key: str | None = None
    fallback_api_key_env: str | None = None
    fallback_base_url: str | None = None

    provider_params: dict[str, Any] = Field(default_factory=dict)


class ConfiguredModelSettings(BaseModel):
    """Operator-configurable persisted model settings."""

    model: str
    transport: Literal["provider_native", "openai_compatible"] = "provider_native"

    fallback_model: str | None = None
    fallback_transport: Literal["provider_native", "openai_compatible"] | None = None

    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    seed: int | None = None

    thinking_effort: (
        Literal[
            "none",
            "minimal",
            "low",
            "medium",
            "high",
            "xhigh",
            "max",
        ]
        | None
    ) = Field(
        default=None,
        validation_alias=AliasChoices("thinking_effort", "reasoning_effort"),
    )
    thinking_budget_tokens: int | None = None

    max_output_tokens: int | None = None
    stop_sequences: list[str] | None = None

    overrides: ModelOverrideSettings = Field(default_factory=ModelOverrideSettings)

    @property
    def reasoning_effort(
        self,
    ) -> Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"] | None:
        """Backward-compatible alias for the generic thinking effort field."""
        return self.thinking_effort

    @model_validator(mode="after")
    def _validate_runtime_shape(self) -> "ConfiguredModelSettings":
        if (
            self.transport == "provider_native"
            and self.model.startswith("anthropic/")
            and self.thinking_budget_tokens is not None
            and 0 < self.thinking_budget_tokens < 1024
        ):
            raise ValueError(
                "thinking_budget_tokens must be >= 1024 for Anthropic models"
            )
        return self


class ModelConfig(BaseModel):
    """Reusable model configuration for any non-embedding LLM caller."""

    model: str
    transport: Literal["provider_native", "openai_compatible"] = "provider_native"

    fallback_model: str | None = None
    fallback_transport: Literal["provider_native", "openai_compatible"] | None = None

    api_key: str | None = None
    base_url: str | None = None

    fallback_api_key: str | None = None
    fallback_base_url: str | None = None

    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    seed: int | None = None

    thinking_effort: (
        Literal[
            "none",
            "minimal",
            "low",
            "medium",
            "high",
            "xhigh",
            "max",
        ]
        | None
    ) = Field(
        default=None,
        validation_alias=AliasChoices("thinking_effort", "reasoning_effort"),
    )
    thinking_budget_tokens: int | None = None
    provider_params: dict[str, Any] = Field(default_factory=dict)

    max_output_tokens: int | None = None
    stop_sequences: list[str] | None = None

    @property
    def reasoning_effort(
        self,
    ) -> Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"] | None:
        """Backward-compatible alias for the generic thinking effort field."""
        return self.thinking_effort

    @model_validator(mode="after")
    def _validate_transport(self) -> "ModelConfig":
        effective_fallback_transport = self.fallback_transport or self.transport

        if self.transport == "openai_compatible" and not self.base_url:
            raise ValueError("base_url is required when transport='openai_compatible'")

        if self.transport == "provider_native" and self.base_url is not None:
            raise ValueError(
                "base_url is only valid when transport='openai_compatible'"
            )

        if (
            self.fallback_model is not None
            and effective_fallback_transport == "openai_compatible"
            and self.fallback_base_url is None
        ):
            raise ValueError(
                "fallback_base_url is required when fallback_transport is 'openai_compatible'"
            )

        return self

    @model_validator(mode="after")
    def _validate_anthropic_thinking_minimum(self) -> "ModelConfig":
        if (
            self.transport == "provider_native"
            and self.model.startswith("anthropic/")
            and self.thinking_budget_tokens is not None
            and 0 < self.thinking_budget_tokens < 1024
        ):
            raise ValueError(
                "thinking_budget_tokens must be >= 1024 for Anthropic models"
            )
        return self

    def for_model(
        self,
        model_override: str,
        *,
        transport_override: Literal["provider_native", "openai_compatible"]
        | None = None,
    ) -> "ModelConfig":
        return self.model_copy(
            update={
                "model": model_override,
                "transport": transport_override or self.transport,
            }
        )


def _resolve_secret(value: str | None, env_name: str | None) -> str | None:
    if value is not None:
        return value
    if env_name is None:
        return None
    return os.getenv(env_name)


def resolve_model_config(configured: ConfiguredModelSettings) -> ModelConfig:
    """Resolve persisted model settings into the runtime ModelConfig."""

    return ModelConfig(
        model=configured.model,
        transport=configured.transport,
        fallback_model=configured.fallback_model,
        fallback_transport=configured.fallback_transport,
        api_key=_resolve_secret(
            configured.overrides.api_key,
            configured.overrides.api_key_env,
        ),
        base_url=configured.overrides.base_url,
        fallback_api_key=_resolve_secret(
            configured.overrides.fallback_api_key,
            configured.overrides.fallback_api_key_env,
        ),
        fallback_base_url=configured.overrides.fallback_base_url,
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
    )


_LEGACY_PROVIDER_PREFIXES: dict[str, str] = {
    "anthropic": "anthropic",
    "openai": "openai",
    "google": "gemini",
    "groq": "groq",
    "custom": "openai",
    "vllm": "hosted_vllm",
}


def _legacy_value(data: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in data:
            return data[key]
    return None


def _has_legacy_value(data: dict[str, Any], *keys: str) -> bool:
    return any(key in data for key in keys)


def _legacy_transport(
    provider: str | None,
) -> Literal["provider_native", "openai_compatible"] | None:
    if provider is None:
        return None
    if provider in {"custom", "vllm"}:
        return "openai_compatible"
    return "provider_native"


def _qualify_legacy_model_name(provider: str | None, model: Any) -> str | None:
    if not isinstance(model, str) or not model:
        return None
    if "/" in model:
        return model
    if provider is None:
        raise ValueError(
            "Legacy flat model configuration requires PROVIDER when MODEL is not provider-qualified"
        )
    prefix = _LEGACY_PROVIDER_PREFIXES.get(provider)
    if prefix is None:
        raise ValueError(f"Unsupported legacy provider in configuration: {provider}")
    return f"{prefix}/{model}"


def _configured_model_from_legacy_values(
    *,
    provider: str | None,
    model: Any,
    fallback_provider: str | None = None,
    fallback_model: Any = None,
    temperature: Any = None,
    thinking_effort: Any = None,
    thinking_budget_tokens: Any = None,
    max_output_tokens: Any = None,
    stop_sequences: Any = None,
) -> ConfiguredModelSettings | None:
    qualified_model = _qualify_legacy_model_name(provider, model)
    if qualified_model is None:
        return None

    qualified_fallback_model = _qualify_legacy_model_name(
        fallback_provider,
        fallback_model,
    )

    return ConfiguredModelSettings(
        model=qualified_model,
        transport=_legacy_transport(provider) or "provider_native",
        fallback_model=qualified_fallback_model,
        fallback_transport=_legacy_transport(fallback_provider),
        temperature=temperature,
        thinking_effort=thinking_effort,
        thinking_budget_tokens=thinking_budget_tokens,
        max_output_tokens=max_output_tokens,
        stop_sequences=stop_sequences,
    )


def _model_config_from_legacy_input(
    data: Any,
    *,
    model_config_keys: tuple[str, ...] = ("MODEL_CONFIG", "model_config"),
    provider_keys: tuple[str, ...] = ("PROVIDER", "provider"),
    model_keys: tuple[str, ...] = ("MODEL", "model"),
    fallback_provider_keys: tuple[str, ...] = ("BACKUP_PROVIDER", "backup_provider"),
    fallback_model_keys: tuple[str, ...] = ("BACKUP_MODEL", "backup_model"),
    temperature_keys: tuple[str, ...] = ("TEMPERATURE", "temperature"),
    thinking_effort_keys: tuple[str, ...] = ("THINKING_EFFORT", "thinking_effort"),
    thinking_budget_keys: tuple[str, ...] = (
        "THINKING_BUDGET_TOKENS",
        "thinking_budget_tokens",
    ),
    max_output_token_keys: tuple[str, ...] = (
        "MAX_OUTPUT_TOKENS",
        "max_output_tokens",
    ),
    stop_sequence_keys: tuple[str, ...] = ("STOP_SEQUENCES", "stop_sequences"),
) -> Any:
    if not isinstance(data, dict):
        return data
    input_dict = cast(dict[str, Any], data)
    if any(key in input_dict for key in model_config_keys):
        return input_dict

    has_legacy_model_shape = any(
        _has_legacy_value(input_dict, *keys)
        for keys in (
            provider_keys,
            model_keys,
            fallback_provider_keys,
            fallback_model_keys,
            temperature_keys,
            thinking_effort_keys,
            thinking_budget_keys,
            max_output_token_keys,
            stop_sequence_keys,
        )
    )
    if not has_legacy_model_shape:
        return input_dict

    configured = _configured_model_from_legacy_values(
        provider=_legacy_value(input_dict, *provider_keys),
        model=_legacy_value(input_dict, *model_keys),
        fallback_provider=_legacy_value(input_dict, *fallback_provider_keys),
        fallback_model=_legacy_value(input_dict, *fallback_model_keys),
        temperature=_legacy_value(input_dict, *temperature_keys),
        thinking_effort=_legacy_value(input_dict, *thinking_effort_keys),
        thinking_budget_tokens=_legacy_value(input_dict, *thinking_budget_keys),
        max_output_tokens=_legacy_value(input_dict, *max_output_token_keys),
        stop_sequences=_legacy_value(input_dict, *stop_sequence_keys),
    )
    if configured is None:
        return input_dict

    migrated = dict(input_dict)
    migrated["MODEL_CONFIG"] = configured
    return migrated


def _qualify_relative_model(base_model: str, relative_model: Any) -> str | None:
    if not isinstance(relative_model, str) or not relative_model:
        return None
    if "/" in relative_model:
        return relative_model
    prefix = base_model.split("/", 1)[0]
    return f"{prefix}/{relative_model}"


def _model_name_from_config_like(value: Any) -> str | None:
    if isinstance(value, ConfiguredModelSettings):
        return value.model
    if isinstance(value, dict):
        config_dict = cast(dict[str, Any], value)
        model = config_dict.get("model") or config_dict.get("MODEL")
        return model if isinstance(model, str) else None
    return None


def _merge_override_defaults(
    configured: ModelOverrideSettings,
    defaults: ModelOverrideSettings,
) -> ModelOverrideSettings:
    update: dict[str, Any] = {}
    configured_fields = configured.model_fields_set

    for field in (
        "api_key",
        "api_key_env",
        "base_url",
        "fallback_api_key",
        "fallback_api_key_env",
        "fallback_base_url",
    ):
        if field not in configured_fields:
            default_value = getattr(defaults, field)
            if default_value is not None:
                update[field] = default_value

    merged_provider_params = {
        **defaults.provider_params,
        **configured.provider_params,
    }
    if merged_provider_params != configured.provider_params:
        update["provider_params"] = merged_provider_params

    return configured.model_copy(update=update) if update else configured


def _merge_configured_model_defaults(
    configured: ConfiguredModelSettings,
    defaults: ConfiguredModelSettings,
) -> ConfiguredModelSettings:
    update: dict[str, Any] = {}
    configured_fields = configured.model_fields_set

    for field in (
        "transport",
        "fallback_model",
        "fallback_transport",
        "temperature",
        "top_p",
        "top_k",
        "frequency_penalty",
        "presence_penalty",
        "seed",
        "thinking_effort",
        "thinking_budget_tokens",
        "max_output_tokens",
        "stop_sequences",
    ):
        if field not in configured_fields:
            default_value = getattr(defaults, field)
            if default_value != getattr(configured, field):
                update[field] = default_value

    merged_overrides = _merge_override_defaults(
        configured.overrides, defaults.overrides
    )
    if merged_overrides != configured.overrides:
        update["overrides"] = merged_overrides

    return configured.model_copy(update=update) if update else configured


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
    OPENAI_COMPATIBLE_API_KEY: str | None = None
    GEMINI_API_KEY: str | None = None
    GROQ_API_KEY: str | None = None
    OPENAI_COMPATIBLE_BASE_URL: str | None = None

    # Separate vLLM endpoint (for local models)
    VLLM_API_KEY: str | None = None
    VLLM_BASE_URL: str | None = None

    EMBEDDING_PROVIDER: Literal["openai", "gemini", "openrouter"] = "openai"

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

    MODEL_CONFIG: ConfiguredModelSettings = Field(
        default_factory=lambda: ConfiguredModelSettings(
            model="gemini/gemini-2.5-flash-lite",
            thinking_budget_tokens=1024,
            max_output_tokens=4096,
        )
    )

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
    def _migrate_legacy_model_config(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        return _model_config_from_legacy_input(data)

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

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_model_config(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        return _model_config_from_legacy_input(
            data,
            max_output_token_keys=(),
        )

    @model_validator(mode="after")
    def _validate_anthropic_thinking_budget(self) -> "DialecticLevelSettings":
        """Ensure Anthropic thinking budget is >= 1024 when enabled."""
        if (
            self.MODEL_CONFIG.model.startswith("anthropic/")
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


class DialecticSettings(HonchoSettings):
    model_config = SettingsConfigDict(  # pyright: ignore
        env_prefix="DIALECTIC_", env_nested_delimiter="__", extra="ignore"
    )

    # Per-level settings for provider, model, thinking budget, and tool iterations
    # TODO: Fill in appropriate values for each reasoning level
    LEVELS: dict[ReasoningLevel, DialecticLevelSettings] = Field(
        default_factory=lambda: {
            "minimal": DialecticLevelSettings(
                MODEL_CONFIG=ConfiguredModelSettings(
                    model="gemini/gemini-2.5-flash-lite",
                    thinking_budget_tokens=0,
                ),
                MAX_TOOL_ITERATIONS=1,
                MAX_OUTPUT_TOKENS=250,
                TOOL_CHOICE="any",
            ),
            "low": DialecticLevelSettings(
                MODEL_CONFIG=ConfiguredModelSettings(
                    model="gemini/gemini-2.5-flash-lite",
                    thinking_budget_tokens=0,
                ),
                MAX_TOOL_ITERATIONS=5,
                TOOL_CHOICE="any",
            ),
            "medium": DialecticLevelSettings(
                MODEL_CONFIG=ConfiguredModelSettings(
                    model="anthropic/claude-haiku-4-5",
                    thinking_budget_tokens=1024,
                ),
                MAX_TOOL_ITERATIONS=2,
            ),
            "high": DialecticLevelSettings(
                MODEL_CONFIG=ConfiguredModelSettings(
                    model="anthropic/claude-haiku-4-5",
                    thinking_budget_tokens=1024,
                ),
                MAX_TOOL_ITERATIONS=4,
            ),
            "max": DialecticLevelSettings(
                MODEL_CONFIG=ConfiguredModelSettings(
                    model="anthropic/claude-haiku-4-5",
                    thinking_budget_tokens=2048,
                ),
                MAX_TOOL_ITERATIONS=10,
            ),
        }
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

    @model_validator(mode="after")
    def _validate_token_budgets(self) -> "DialecticSettings":
        """Ensure the output token limit exceeds all thinking budgets."""
        for level, level_settings in self.LEVELS.items():
            thinking_budget = level_settings.MODEL_CONFIG.thinking_budget_tokens or 0
            if thinking_budget > 0 and thinking_budget >= self.MAX_OUTPUT_TOKENS:
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

    MODEL_CONFIG: ConfiguredModelSettings = Field(
        default_factory=lambda: ConfiguredModelSettings(
            model="gemini/gemini-2.5-flash",
            thinking_budget_tokens=512,
        )
    )
    MAX_TOKENS_SHORT: Annotated[int, Field(default=1000, gt=0, le=10_000)] = 1000
    MAX_TOKENS_LONG: Annotated[int, Field(default=4000, gt=0, le=20_000)] = 4000

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_model_config(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        return _model_config_from_legacy_input(data)


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

    MODEL_CONFIG: ConfiguredModelSettings = Field(
        default_factory=lambda: ConfiguredModelSettings(
            model="anthropic/claude-sonnet-4-20250514",
            thinking_budget_tokens=8192,
            max_output_tokens=16_384,
        )
    )

    # Agent iteration limit - increased for extended reasoning workflow
    MAX_TOOL_ITERATIONS: Annotated[int, Field(default=20, gt=0, le=50)] = 20

    # Token limit for get_recent_history tool within the agent
    HISTORY_TOKEN_LIMIT: Annotated[int, Field(default=16_384, gt=0, le=200_000)] = (
        16_384
    )

    DEDUCTION_MODEL_CONFIG: ConfiguredModelSettings = Field(
        default_factory=lambda: ConfiguredModelSettings(
            model="anthropic/claude-haiku-4-5",
        )
    )
    INDUCTION_MODEL_CONFIG: ConfiguredModelSettings = Field(
        default_factory=lambda: ConfiguredModelSettings(
            model="anthropic/claude-haiku-4-5",
        )
    )

    # Surprisal-based sampling subsystem
    SURPRISAL: SurprisalSettings = Field(default_factory=SurprisalSettings)

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_model_configs(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        migrated = _model_config_from_legacy_input(data)
        dream_model_config = migrated.get("MODEL_CONFIG") or migrated.get(
            "model_config"
        )
        base_model = _model_name_from_config_like(dream_model_config)

        if (
            "DEDUCTION_MODEL_CONFIG" not in migrated
            and "deduction_model_config" not in migrated
        ):
            deduction_model = _legacy_value(
                migrated, "DEDUCTION_MODEL", "deduction_model"
            )
            if deduction_model is not None and base_model is not None:
                qualified_model = _qualify_relative_model(base_model, deduction_model)
                if qualified_model is not None:
                    migrated["DEDUCTION_MODEL_CONFIG"] = ConfiguredModelSettings(
                        model=qualified_model,
                    )

        if (
            "INDUCTION_MODEL_CONFIG" not in migrated
            and "induction_model_config" not in migrated
        ):
            induction_model = _legacy_value(
                migrated, "INDUCTION_MODEL", "induction_model"
            )
            if induction_model is not None and base_model is not None:
                qualified_model = _qualify_relative_model(base_model, induction_model)
                if qualified_model is not None:
                    migrated["INDUCTION_MODEL_CONFIG"] = ConfiguredModelSettings(
                        model=qualified_model,
                    )

        return migrated

    @model_validator(mode="after")
    def _merge_specialist_model_defaults(self) -> "DreamSettings":
        object.__setattr__(
            self,
            "DEDUCTION_MODEL_CONFIG",
            _merge_configured_model_defaults(
                self.DEDUCTION_MODEL_CONFIG,
                self.MODEL_CONFIG,
            ),
        )
        object.__setattr__(
            self,
            "INDUCTION_MODEL_CONFIG",
            _merge_configured_model_defaults(
                self.INDUCTION_MODEL_CONFIG,
                self.MODEL_CONFIG,
            ),
        )
        return self

    @model_validator(mode="after")
    def _validate_token_budgets(self) -> "DreamSettings":
        """Ensure the output token limit exceeds the thinking budget."""
        max_output_tokens = self.MODEL_CONFIG.max_output_tokens
        thinking_budget_tokens = self.MODEL_CONFIG.thinking_budget_tokens
        if (
            max_output_tokens is not None
            and thinking_budget_tokens is not None
            and max_output_tokens <= thinking_budget_tokens
        ):
            raise ValueError(
                "dream.MODEL_CONFIG.max_output_tokens must be greater than "
                + "dream.MODEL_CONFIG.thinking_budget_tokens"
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
    MAX_EMBEDDING_TOKENS: Annotated[int, Field(default=8192, gt=0)] = 8192
    MAX_EMBEDDING_TOKENS_PER_REQUEST: Annotated[int, Field(default=300_000, gt=0)] = (
        300_000
    )
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
        if "NAMESPACE" not in self.TELEMETRY.model_fields_set:
            self.TELEMETRY.NAMESPACE = self.NAMESPACE
        if "NAMESPACE" not in self.METRICS.model_fields_set:
            self.METRICS.NAMESPACE = self.NAMESPACE

        return self


# Create a single global instance of the settings
settings: AppSettings = AppSettings()
