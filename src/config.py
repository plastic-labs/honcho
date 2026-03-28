import logging
import os
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal

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

from src.utils.types import SupportedProviders

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


def _qualify_model_name(provider: SupportedProviders, model: str) -> str:
    if "/" in model:
        return model

    prefix_map: dict[SupportedProviders, str] = {
        "anthropic": "anthropic",
        "openai": "openai",
        "google": "gemini",
        "groq": "groq",
        "custom": "openai",
        "vllm": "hosted_vllm",
    }
    return f"{prefix_map[provider]}/{model}"


def _transport_for_provider(
    provider: SupportedProviders | None,
) -> Literal["provider_native", "openai_compatible"] | None:
    if provider is None:
        return None
    if provider in {"custom", "vllm"}:
        return "openai_compatible"
    return "provider_native"


def _strip_model_prefix(model: str) -> str:
    return model.split("/", 1)[1] if "/" in model else model


def _provider_for_configured_model(
    model: str,
    transport: Literal["provider_native", "openai_compatible"],
) -> SupportedProviders:
    if transport == "openai_compatible":
        if model.startswith("hosted_vllm/"):
            return "vllm"
        return "custom"

    prefix = model.split("/", 1)[0]
    provider_map: dict[str, SupportedProviders] = {
        "anthropic": "anthropic",
        "openai": "openai",
        "gemini": "google",
        "groq": "groq",
        "hosted_vllm": "vllm",
    }
    if prefix not in provider_map:
        raise ValueError(f"Unsupported model prefix in config: {prefix}")
    return provider_map[prefix]


def _configured_model_from_legacy(
    *,
    provider: SupportedProviders,
    model: str,
    fallback_provider: SupportedProviders | None = None,
    fallback_model: str | None = None,
    temperature: float | None = None,
    thinking_effort: Literal[
        "none",
        "minimal",
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
    ]
    | None = None,
    thinking_budget_tokens: int | None = None,
    max_output_tokens: int | None = None,
    stop_sequences: list[str] | None = None,
) -> ConfiguredModelSettings:
    return ConfiguredModelSettings(
        model=_qualify_model_name(provider, model),
        transport=_transport_for_provider(provider) or "provider_native",
        fallback_model=_qualify_model_name(fallback_provider, fallback_model)
        if fallback_provider and fallback_model
        else None,
        fallback_transport=_transport_for_provider(fallback_provider),
        temperature=temperature,
        thinking_effort=thinking_effort,
        thinking_budget_tokens=thinking_budget_tokens,
        max_output_tokens=max_output_tokens,
        stop_sequences=stop_sequences,
    )


def _merge_legacy_model_defaults(
    configured: ConfiguredModelSettings,
    *,
    fallback_provider: SupportedProviders | None = None,
    fallback_model: str | None = None,
    temperature: float | None = None,
    thinking_effort: Literal[
        "none",
        "minimal",
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
    ]
    | None = None,
    thinking_budget_tokens: int | None = None,
    max_output_tokens: int | None = None,
    stop_sequences: list[str] | None = None,
) -> ConfiguredModelSettings:
    update: dict[str, Any] = {}

    if (
        configured.fallback_model is None
        and fallback_provider is not None
        and fallback_model is not None
    ):
        update["fallback_model"] = _qualify_model_name(
            fallback_provider, fallback_model
        )
        update["fallback_transport"] = _transport_for_provider(fallback_provider)

    if configured.temperature is None and temperature is not None:
        update["temperature"] = temperature
    if configured.thinking_effort is None and thinking_effort is not None:
        update["thinking_effort"] = thinking_effort
    if configured.thinking_budget_tokens is None and thinking_budget_tokens is not None:
        update["thinking_budget_tokens"] = thinking_budget_tokens
    if configured.max_output_tokens is None and max_output_tokens is not None:
        update["max_output_tokens"] = max_output_tokens
    if configured.stop_sequences is None and stop_sequences is not None:
        update["stop_sequences"] = stop_sequences

    return configured.model_copy(update=update) if update else configured


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


def _legacy_model_fields_from_configured(
    configured: ConfiguredModelSettings,
) -> tuple[SupportedProviders, str, SupportedProviders | None, str | None]:
    provider = _provider_for_configured_model(configured.model, configured.transport)
    fallback_provider: SupportedProviders | None = None
    fallback_model: str | None = None

    if configured.fallback_model is not None:
        fallback_provider = _provider_for_configured_model(
            configured.fallback_model,
            configured.fallback_transport or configured.transport,
        )
        fallback_model = _strip_model_prefix(configured.fallback_model)

    return (
        provider,
        _strip_model_prefix(configured.model),
        fallback_provider,
        fallback_model,
    )


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


def _set_model_field(instance: BaseModel, field_name: str, value: Any) -> None:
    """Set a Pydantic field during after-validation syncing."""

    object.__setattr__(instance, field_name, value)


def _require_model_config(
    configured: ConfiguredModelSettings | None,
    *,
    owner: str,
) -> ConfiguredModelSettings:
    if configured is None:
        raise ValueError(f"{owner} MODEL_CONFIG must be resolved before use")
    return configured


class BackupLLMSettingsMixin:
    """Mixin class for settings that support backup LLM provider configuration.

    Provides backup provider and model fields along with validation to ensure
    both fields are set together or both are None.
    """

    BACKUP_PROVIDER: SupportedProviders | None = None
    BACKUP_MODEL: str | None = None

    @model_validator(mode="after")
    def _validate_backup_configuration(self):
        """Ensure both backup fields are set together or both are None."""
        if (self.BACKUP_PROVIDER is None) != (self.BACKUP_MODEL is None):
            raise ValueError(
                "BACKUP_PROVIDER and BACKUP_MODEL must both be set or both be None"
            )
        return self


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


class DeriverSettings(BackupLLMSettingsMixin, HonchoSettings):
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

    MODEL_CONFIG: ConfiguredModelSettings | None = None
    PROVIDER: SupportedProviders = "google"
    MODEL: str = "gemini-2.5-flash-lite"
    TEMPERATURE: float | None = None

    # Whether to deduplicate documents when creating them
    DEDUPLICATE: bool = True

    MAX_OUTPUT_TOKENS: Annotated[int, Field(default=4096, gt=0, le=100_000)] = 4096
    THINKING_BUDGET_TOKENS: Annotated[int, Field(default=1024, gt=0, le=5000)] = 1024

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

    @model_validator(mode="after")
    def _sync_model_config(self) -> "DeriverSettings":
        model_config = self.MODEL_CONFIG or _configured_model_from_legacy(
            provider=self.PROVIDER,
            model=self.MODEL,
            fallback_provider=self.BACKUP_PROVIDER,
            fallback_model=self.BACKUP_MODEL,
            temperature=self.TEMPERATURE,
            thinking_budget_tokens=self.THINKING_BUDGET_TOKENS,
            max_output_tokens=self.MAX_OUTPUT_TOKENS,
        )
        model_config = _merge_legacy_model_defaults(
            model_config,
            fallback_provider=self.BACKUP_PROVIDER,
            fallback_model=self.BACKUP_MODEL,
            temperature=self.TEMPERATURE,
            thinking_budget_tokens=self.THINKING_BUDGET_TOKENS,
            max_output_tokens=self.MAX_OUTPUT_TOKENS,
        )
        _set_model_field(self, "MODEL_CONFIG", model_config)

        provider, model, fallback_provider, fallback_model = (
            _legacy_model_fields_from_configured(model_config)
        )
        _set_model_field(self, "PROVIDER", provider)
        _set_model_field(self, "MODEL", model)
        _set_model_field(self, "BACKUP_PROVIDER", fallback_provider)
        _set_model_field(self, "BACKUP_MODEL", fallback_model)
        if model_config.temperature is not None:
            _set_model_field(self, "TEMPERATURE", model_config.temperature)
        if model_config.thinking_budget_tokens is not None:
            _set_model_field(
                self,
                "THINKING_BUDGET_TOKENS",
                model_config.thinking_budget_tokens,
            )
        if model_config.max_output_tokens is not None:
            _set_model_field(self, "MAX_OUTPUT_TOKENS", model_config.max_output_tokens)
        return self

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
        ConfiguredModelSettings | None,
        Field(validation_alias="model_config"),
    ] = None
    PROVIDER: Annotated[SupportedProviders, Field(validation_alias="provider")] = (
        "google"
    )
    MODEL: Annotated[str, Field(validation_alias="model")] = "gemini-2.5-pro"
    BACKUP_PROVIDER: Annotated[
        SupportedProviders | None, Field(validation_alias="backup_provider")
    ] = None
    BACKUP_MODEL: Annotated[str | None, Field(validation_alias="backup_model")] = None
    THINKING_BUDGET_TOKENS: Annotated[
        int | None, Field(ge=0, le=100_000, validation_alias="thinking_budget_tokens")
    ] = None
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
    def _sync_model_config(self) -> "DialecticLevelSettings":
        if self.MODEL_CONFIG is None:
            model_config = _configured_model_from_legacy(
                provider=self.PROVIDER,
                model=self.MODEL,
                fallback_provider=self.BACKUP_PROVIDER,
                fallback_model=self.BACKUP_MODEL,
                thinking_budget_tokens=self.THINKING_BUDGET_TOKENS,
                max_output_tokens=self.MAX_OUTPUT_TOKENS,
            )
        else:
            model_config = self.MODEL_CONFIG

        model_config = _merge_legacy_model_defaults(
            model_config,
            fallback_provider=self.BACKUP_PROVIDER,
            fallback_model=self.BACKUP_MODEL,
            thinking_budget_tokens=self.THINKING_BUDGET_TOKENS,
            max_output_tokens=self.MAX_OUTPUT_TOKENS,
        )
        _set_model_field(self, "MODEL_CONFIG", model_config)

        provider, model, fallback_provider, fallback_model = (
            _legacy_model_fields_from_configured(model_config)
        )
        _set_model_field(self, "PROVIDER", provider)
        _set_model_field(self, "MODEL", model)
        _set_model_field(self, "BACKUP_PROVIDER", fallback_provider)
        _set_model_field(self, "BACKUP_MODEL", fallback_model)
        if model_config.thinking_budget_tokens is not None:
            _set_model_field(
                self,
                "THINKING_BUDGET_TOKENS",
                model_config.thinking_budget_tokens,
            )
        if model_config.max_output_tokens is not None:
            _set_model_field(self, "MAX_OUTPUT_TOKENS", model_config.max_output_tokens)
        return self

    @model_validator(mode="after")
    def _validate_backup_configuration(self) -> "DialecticLevelSettings":
        """Ensure both backup fields are set together or both are None."""
        if (self.BACKUP_PROVIDER is None) != (self.BACKUP_MODEL is None):
            raise ValueError(
                "BACKUP_PROVIDER and BACKUP_MODEL must both be set or both be None"
            )
        return self

    @model_validator(mode="after")
    def _validate_anthropic_thinking_budget(self) -> "DialecticLevelSettings":
        """Ensure Anthropic thinking budget is >= 1024 when enabled."""
        if (
            self.PROVIDER == "anthropic"
            and self.THINKING_BUDGET_TOKENS is not None
            and self.THINKING_BUDGET_TOKENS > 0
            and self.THINKING_BUDGET_TOKENS < 1024
        ):
            raise ValueError(
                f"THINKING_BUDGET_TOKENS must be >= 1024 for Anthropic provider when enabled (got {self.THINKING_BUDGET_TOKENS})"
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
                PROVIDER="google",
                MODEL="gemini-2.5-flash-lite",
                THINKING_BUDGET_TOKENS=0,
                MAX_TOOL_ITERATIONS=1,
                MAX_OUTPUT_TOKENS=250,
                TOOL_CHOICE="any",
            ),
            "low": DialecticLevelSettings(
                PROVIDER="google",
                MODEL="gemini-2.5-flash-lite",
                THINKING_BUDGET_TOKENS=0,
                MAX_TOOL_ITERATIONS=5,
                TOOL_CHOICE="any",
            ),
            "medium": DialecticLevelSettings(
                PROVIDER="anthropic",
                MODEL="claude-haiku-4-5",
                THINKING_BUDGET_TOKENS=1024,
                MAX_TOOL_ITERATIONS=2,
            ),
            "high": DialecticLevelSettings(
                PROVIDER="anthropic",
                MODEL="claude-haiku-4-5",
                THINKING_BUDGET_TOKENS=1024,
                MAX_TOOL_ITERATIONS=4,
            ),
            "max": DialecticLevelSettings(
                PROVIDER="anthropic",
                MODEL="claude-haiku-4-5",
                THINKING_BUDGET_TOKENS=2048,
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
            thinking_budget = level_settings.THINKING_BUDGET_TOKENS or 0
            if thinking_budget > 0 and thinking_budget >= self.MAX_OUTPUT_TOKENS:
                raise ValueError(
                    f"MAX_OUTPUT_TOKENS must be greater than THINKING_BUDGET_TOKENS for level '{level}'"
                )
        return self

    @model_validator(mode="after")
    def _validate_all_levels_present(self) -> "DialecticSettings":
        """Ensure all reasoning levels are configured."""
        missing = set(REASONING_LEVELS) - set(self.LEVELS.keys())
        if missing:
            raise ValueError(f"Missing configuration for reasoning levels: {missing}")
        return self


class SummarySettings(BackupLLMSettingsMixin, HonchoSettings):
    model_config = SettingsConfigDict(  # pyright: ignore
        env_prefix="SUMMARY_", env_nested_delimiter="__", extra="ignore"
    )

    ENABLED: bool = True

    MESSAGES_PER_SHORT_SUMMARY: Annotated[int, Field(default=20, gt=0, le=100)] = 20
    MESSAGES_PER_LONG_SUMMARY: Annotated[int, Field(default=60, gt=0, le=500)] = 60

    MODEL_CONFIG: ConfiguredModelSettings | None = None
    PROVIDER: SupportedProviders = "google"
    MODEL: str = "gemini-2.5-flash"
    MAX_TOKENS_SHORT: Annotated[int, Field(default=1000, gt=0, le=10_000)] = 1000
    MAX_TOKENS_LONG: Annotated[int, Field(default=4000, gt=0, le=20_000)] = 4000

    THINKING_BUDGET_TOKENS: Annotated[int, Field(default=512, gt=0, le=2000)] = 512

    @model_validator(mode="after")
    def _sync_model_config(self) -> "SummarySettings":
        model_config = self.MODEL_CONFIG or _configured_model_from_legacy(
            provider=self.PROVIDER,
            model=self.MODEL,
            fallback_provider=self.BACKUP_PROVIDER,
            fallback_model=self.BACKUP_MODEL,
            thinking_budget_tokens=self.THINKING_BUDGET_TOKENS,
        )
        model_config = _merge_legacy_model_defaults(
            model_config,
            fallback_provider=self.BACKUP_PROVIDER,
            fallback_model=self.BACKUP_MODEL,
            thinking_budget_tokens=self.THINKING_BUDGET_TOKENS,
        )
        _set_model_field(self, "MODEL_CONFIG", model_config)

        provider, model, fallback_provider, fallback_model = (
            _legacy_model_fields_from_configured(model_config)
        )
        _set_model_field(self, "PROVIDER", provider)
        _set_model_field(self, "MODEL", model)
        _set_model_field(self, "BACKUP_PROVIDER", fallback_provider)
        _set_model_field(self, "BACKUP_MODEL", fallback_model)
        if model_config.thinking_budget_tokens is not None:
            _set_model_field(
                self,
                "THINKING_BUDGET_TOKENS",
                model_config.thinking_budget_tokens,
            )
        return self


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


class DreamSettings(BackupLLMSettingsMixin, HonchoSettings):
    model_config = SettingsConfigDict(  # pyright: ignore
        env_prefix="DREAM_", env_nested_delimiter="__", extra="ignore"
    )

    ENABLED: bool = True
    DOCUMENT_THRESHOLD: Annotated[int, Field(default=50, gt=0, le=1000)] = 50
    IDLE_TIMEOUT_MINUTES: Annotated[int, Field(default=60, gt=0, le=1440)] = 60
    MIN_HOURS_BETWEEN_DREAMS: Annotated[int, Field(default=8, gt=0, le=72)] = 8
    ENABLED_TYPES: list[str] = ["omni"]

    MODEL_CONFIG: ConfiguredModelSettings | None = None
    PROVIDER: SupportedProviders = "anthropic"
    MODEL: str = "claude-sonnet-4-20250514"
    MAX_OUTPUT_TOKENS: Annotated[int, Field(default=16_384, gt=0, le=64_000)] = 16_384
    THINKING_BUDGET_TOKENS: Annotated[int, Field(default=8192, gt=0, le=32_000)] = 8192

    # Agent iteration limit - increased for extended reasoning workflow
    MAX_TOOL_ITERATIONS: Annotated[int, Field(default=20, gt=0, le=50)] = 20

    # Token limit for get_recent_history tool within the agent
    HISTORY_TOKEN_LIMIT: Annotated[int, Field(default=16_384, gt=0, le=200_000)] = (
        16_384
    )

    DEDUCTION_MODEL_CONFIG: ConfiguredModelSettings | None = None

    # Deduction Specialist: handles logical inference
    DEDUCTION_MODEL: str = "claude-haiku-4-5"
    INDUCTION_MODEL_CONFIG: ConfiguredModelSettings | None = None
    # Induction Specialist: identifies patterns across observations
    INDUCTION_MODEL: str = "claude-haiku-4-5"

    # Surprisal-based sampling subsystem
    SURPRISAL: SurprisalSettings = Field(default_factory=SurprisalSettings)

    @model_validator(mode="after")
    def _sync_model_config(self) -> "DreamSettings":
        model_config = self.MODEL_CONFIG or _configured_model_from_legacy(
            provider=self.PROVIDER,
            model=self.MODEL,
            fallback_provider=self.BACKUP_PROVIDER,
            fallback_model=self.BACKUP_MODEL,
            thinking_budget_tokens=self.THINKING_BUDGET_TOKENS,
            max_output_tokens=self.MAX_OUTPUT_TOKENS,
        )
        model_config = _merge_legacy_model_defaults(
            model_config,
            fallback_provider=self.BACKUP_PROVIDER,
            fallback_model=self.BACKUP_MODEL,
            thinking_budget_tokens=self.THINKING_BUDGET_TOKENS,
            max_output_tokens=self.MAX_OUTPUT_TOKENS,
        )
        _set_model_field(self, "MODEL_CONFIG", model_config)

        provider, model, fallback_provider, fallback_model = (
            _legacy_model_fields_from_configured(model_config)
        )
        _set_model_field(self, "PROVIDER", provider)
        _set_model_field(self, "MODEL", model)
        _set_model_field(self, "BACKUP_PROVIDER", fallback_provider)
        _set_model_field(self, "BACKUP_MODEL", fallback_model)
        if model_config.thinking_budget_tokens is not None:
            _set_model_field(
                self,
                "THINKING_BUDGET_TOKENS",
                model_config.thinking_budget_tokens,
            )
        if model_config.max_output_tokens is not None:
            _set_model_field(self, "MAX_OUTPUT_TOKENS", model_config.max_output_tokens)

        deduction_model_config = (
            self.DEDUCTION_MODEL_CONFIG
            or _configured_model_from_legacy(
                provider=self.PROVIDER,
                model=self.DEDUCTION_MODEL,
            )
        )
        deduction_model_config = _merge_configured_model_defaults(
            deduction_model_config,
            model_config,
        )
        _set_model_field(self, "DEDUCTION_MODEL_CONFIG", deduction_model_config)

        induction_model_config = (
            self.INDUCTION_MODEL_CONFIG
            or _configured_model_from_legacy(
                provider=self.PROVIDER,
                model=self.INDUCTION_MODEL,
            )
        )
        induction_model_config = _merge_configured_model_defaults(
            induction_model_config,
            model_config,
        )
        _set_model_field(self, "INDUCTION_MODEL_CONFIG", induction_model_config)

        _set_model_field(
            self,
            "DEDUCTION_MODEL",
            _strip_model_prefix(
                _require_model_config(
                    self.DEDUCTION_MODEL_CONFIG,
                    owner="DREAM DEDUCTION",
                ).model
            ),
        )
        _set_model_field(
            self,
            "INDUCTION_MODEL",
            _strip_model_prefix(
                _require_model_config(
                    self.INDUCTION_MODEL_CONFIG,
                    owner="DREAM INDUCTION",
                ).model
            ),
        )
        return self

    @model_validator(mode="after")
    def _validate_token_budgets(self) -> "DreamSettings":
        """Ensure the output token limit exceeds the thinking budget."""
        if self.MAX_OUTPUT_TOKENS <= self.THINKING_BUDGET_TOKENS:
            raise ValueError(
                "MAX_OUTPUT_TOKENS must be greater than THINKING_BUDGET_TOKENS"
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
