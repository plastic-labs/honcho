import logging
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal, Protocol

import tomllib
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator, model_validator
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
        "VECTOR_STORE": "vector_store",
        "OTEL": "otel",
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
    model_config = SettingsConfigDict(env_prefix="DERIVER_", extra="ignore")  # pyright: ignore

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

    PROVIDER: Annotated[SupportedProviders, Field(validation_alias="provider")]
    MODEL: Annotated[str, Field(validation_alias="model")]
    BACKUP_PROVIDER: Annotated[
        SupportedProviders | None, Field(validation_alias="backup_provider")
    ] = None
    BACKUP_MODEL: Annotated[str | None, Field(validation_alias="backup_model")] = None
    THINKING_BUDGET_TOKENS: Annotated[
        int, Field(ge=0, le=100_000, validation_alias="thinking_budget_tokens")
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
            if self.MAX_OUTPUT_TOKENS <= level_settings.THINKING_BUDGET_TOKENS:
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


class OpenTelemetrySettings(HonchoSettings):
    """OpenTelemetry settings for push-based metrics via OTLP.

    These settings configure the OTel SDK to push metrics via OTLP HTTP
    to any compatible backend (Mimir, Grafana Cloud, etc.).
    """

    model_config = SettingsConfigDict(env_prefix="OTEL_", extra="ignore")  # pyright: ignore

    # Master toggle for OTel metrics
    ENABLED: bool = False

    # OTLP HTTP endpoint for metrics (e.g., "https://mimir.example.com/otlp/v1/metrics")
    # For Mimir, the endpoint is typically: <mimir-url>/otlp/v1/metrics
    # For Grafana Cloud: https://otlp-gateway-<region>.grafana.net/otlp/v1/metrics
    ENDPOINT: str | None = None

    HEADERS: dict[str, str] | None = None

    # Export interval in milliseconds (default: 60 seconds)
    EXPORT_INTERVAL_MILLIS: int = 60000

    # Service name for resource attributes (identifies what this service is)
    SERVICE_NAME: str = "honcho"

    # Service namespace for resource attributes (defaults to top-level NAMESPACE if not set)
    SERVICE_NAMESPACE: str | None = None


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

    ## NOTE: specialist models use the same provider as the main model

    # Deduction Specialist: handles logical inference
    DEDUCTION_MODEL: str = "claude-haiku-4-5"
    # Induction Specialist: identifies patterns across observations
    INDUCTION_MODEL: str = "claude-haiku-4-5"

    # Surprisal-based sampling subsystem
    SURPRISAL: SurprisalSettings = Field(default_factory=SurprisalSettings)

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
    OTEL: OpenTelemetrySettings = Field(default_factory=OpenTelemetrySettings)
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
        """Propagate top-level NAMESPACE to nested settings if not explicitly set.

        After this validator runs, CACHE.NAMESPACE,
        VECTOR_STORE.NAMESPACE, TELEMETRY.NAMESPACE, and OTEL.SERVICE_NAMESPACE
        are guaranteed to exist. Explicitly provided nested namespaces are preserved.
        """
        if "NAMESPACE" not in self.CACHE.model_fields_set:
            self.CACHE.NAMESPACE = self.NAMESPACE
        if "NAMESPACE" not in self.VECTOR_STORE.model_fields_set:
            self.VECTOR_STORE.NAMESPACE = self.NAMESPACE
        if "NAMESPACE" not in self.TELEMETRY.model_fields_set:
            self.TELEMETRY.NAMESPACE = self.NAMESPACE
        if "SERVICE_NAMESPACE" not in self.OTEL.model_fields_set:
            self.OTEL.SERVICE_NAMESPACE = self.NAMESPACE

        return self


# Create a single global instance of the settings
settings: AppSettings = AppSettings()
