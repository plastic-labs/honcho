import logging
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal, Protocol

import tomllib
from dotenv import load_dotenv
from pydantic import Field, field_validator, model_validator
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


class LLMComponentSettings(Protocol):
    """Protocol for settings classes that use LLM providers with backup support."""

    PROVIDER: SupportedProviders
    MODEL: str
    BACKUP_PROVIDER: SupportedProviders | None
    BACKUP_MODEL: str | None


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

    EMBEDDING_PROVIDER: Literal["openai", "gemini"] = "openai"

    # General LLM settings
    DEFAULT_MAX_TOKENS: Annotated[int, Field(default=1000, gt=0, le=100_000)] = 2500


class DeriverSettings(BackupLLMSettingsMixin, HonchoSettings):
    model_config = SettingsConfigDict(env_prefix="DERIVER_", extra="ignore")  # pyright: ignore

    ENABLED: bool = True
    AGENTIC: bool = False
    USE_LEGACY: bool = False  # Set to True to use legacy deriver with peer card updates

    WORKERS: Annotated[int, Field(default=1, gt=0, le=100)] = 1
    POLLING_SLEEP_INTERVAL_SECONDS: Annotated[
        float, Field(default=1.0, gt=0.0, le=60.0)
    ] = 1.0
    STALE_SESSION_TIMEOUT_MINUTES: Annotated[int, Field(default=5, gt=0, le=1440)] = 5

    # Retention window (seconds) for keeping errored items in the queue
    QUEUE_ERROR_RETENTION_SECONDS: Annotated[
        int, Field(default=30 * 24 * 3600, gt=0)
    ] = 30 * 24 * 3600  # 30 days default

    PROVIDER: SupportedProviders = "google"
    MODEL: str = "gemini-2.5-flash-lite"

    # Whether to deduplicate documents when creating them
    DEDUPLICATE: bool = True

    MAX_OUTPUT_TOKENS: Annotated[int, Field(default=10_000, gt=0, le=100_000)] = 10_000
    THINKING_BUDGET_TOKENS: Annotated[int, Field(default=1024, gt=0, le=5000)] = 1024

    MAX_INPUT_TOKENS: Annotated[int, Field(default=23000, gt=0, le=23000)] = 23000

    WORKING_REPRESENTATION_MAX_OBSERVATIONS: Annotated[
        int, Field(default=100, gt=0, le=1000)
    ] = 100

    REPRESENTATION_BATCH_MAX_TOKENS: Annotated[
        int,
        Field(
            default=4096,
            ge=1,
        ),
    ] = 4096


class PeerCardSettings(HonchoSettings):
    model_config = SettingsConfigDict(env_prefix="PEER_CARD_", extra="ignore")  # pyright: ignore

    ENABLED: bool = True


class DialecticSettings(BackupLLMSettingsMixin, HonchoSettings):
    model_config = SettingsConfigDict(env_prefix="DIALECTIC_", extra="ignore")  # pyright: ignore

    PROVIDER: SupportedProviders = "anthropic"
    MODEL: str = "claude-haiku-4-5"

    MAX_OUTPUT_TOKENS: Annotated[int, Field(default=2500, gt=0, le=100_000)] = 2500
    MAX_INPUT_TOKENS: Annotated[int, Field(default=100_000, gt=0, le=200_000)] = 100_000

    THINKING_BUDGET_TOKENS: Annotated[int, Field(default=1024, gt=0, le=5000)] = 4096


class SummarySettings(BackupLLMSettingsMixin, HonchoSettings):
    model_config = SettingsConfigDict(env_prefix="SUMMARY_", extra="ignore")  # pyright: ignore

    ENABLED: bool = True

    MESSAGES_PER_SHORT_SUMMARY: Annotated[int, Field(default=20, gt=0, le=100)] = 20
    MESSAGES_PER_LONG_SUMMARY: Annotated[int, Field(default=60, gt=0, le=500)] = 60

    PROVIDER: SupportedProviders = "google"
    MODEL: str = "gemini-2.5-flash"
    MAX_TOKENS_SHORT: Annotated[int, Field(default=1000, gt=0, le=10_000)] = 1000
    MAX_TOKENS_LONG: Annotated[int, Field(default=4000, gt=0, le=20_000)] = 4000

    THINKING_BUDGET_TOKENS: Annotated[int, Field(default=512, gt=0, le=2000)] = 512


class WebhookSettings(HonchoSettings):
    model_config = SettingsConfigDict(env_prefix="WEBHOOK_", extra="ignore")  # pyright: ignore

    SECRET: str | None = None  # Must be set if configuring webhooks
    MAX_WORKSPACE_LIMIT: int = 10


class MetricsSettings(HonchoSettings):
    model_config = SettingsConfigDict(env_prefix="METRICS_", extra="ignore")  # pyright: ignore

    ENABLED: bool = False
    NAMESPACE: str | None = None


class CacheSettings(HonchoSettings):
    model_config = SettingsConfigDict(env_prefix="CACHE_", extra="ignore")  # pyright: ignore

    ENABLED: bool = False
    URL: str = "redis://localhost:6379/0?suppress=false"
    NAMESPACE: str | None = None
    DEFAULT_TTL_SECONDS: Annotated[int, Field(default=300, ge=1, le=86_400)] = (
        300  # how long to keep items in cache
    )

    DEFAULT_LOCK_TTL_SECONDS: Annotated[int, Field(default=5, ge=1, le=86_400)] = (
        5  # how long to hold a lock on a resource when fetching DB after cache miss
    )


class DreamSettings(BackupLLMSettingsMixin, HonchoSettings):
    model_config = SettingsConfigDict(env_prefix="DREAM_", extra="ignore")  # pyright: ignore

    ENABLED: bool = True
    DOCUMENT_THRESHOLD: Annotated[int, Field(default=50, gt=0, le=1000)] = 50
    IDLE_TIMEOUT_MINUTES: Annotated[int, Field(default=60, gt=0, le=1440)] = 60
    MIN_HOURS_BETWEEN_DREAMS: Annotated[int, Field(default=8, gt=0, le=72)] = 8
    ENABLED_TYPES: list[str] = ["consolidate"]

    # LLM settings for dream processing
    PROVIDER: SupportedProviders = "anthropic"
    MODEL: str = "claude-haiku-4-5"
    MAX_OUTPUT_TOKENS: Annotated[int, Field(default=4000, gt=0, le=10_000)] = 4000
    THINKING_BUDGET_TOKENS: Annotated[int, Field(default=2048, gt=0, le=8192)] = 2048

    # Agent iteration limit - controls how many tool calling rounds the agent gets
    MAX_TOOL_ITERATIONS: Annotated[int, Field(default=8, gt=0, le=30)] = 8


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
    CACHE: CacheSettings = Field(default_factory=CacheSettings)
    DREAM: DreamSettings = Field(default_factory=DreamSettings)

    @field_validator("LOG_LEVEL")
    def validate_log_level(cls, v: str) -> str:
        log_level = v.upper()
        if log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ValueError(f"Invalid log level: {v}")
        return log_level

    @model_validator(mode="after")
    def propagate_namespace(self) -> "AppSettings":
        """Propagate top-level NAMESPACE to nested settings if not explicitly set.

        After this validator runs, CACHE.NAMESPACE and METRICS.NAMESPACE are guaranteed
        to exist.
        """
        if self.CACHE.NAMESPACE is None:
            self.CACHE.NAMESPACE = self.NAMESPACE
        if self.METRICS.NAMESPACE is None:
            self.METRICS.NAMESPACE = self.NAMESPACE
        return self


# Create a single global instance of the settings
settings: AppSettings = AppSettings()
