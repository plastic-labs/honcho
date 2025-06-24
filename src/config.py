import logging
from pathlib import Path
from typing import Annotated, Any, ClassVar

import tomllib
from dotenv import load_dotenv
from mirascope import Provider
from pydantic import Field, field_validator, model_validator
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


class TomlConfigSettingsSource(PydanticBaseSettingsSource):
    """Custom settings source for loading from TOML file."""

    def __init__(self, settings_cls: type[BaseSettings]) -> None:
        super().__init__(settings_cls)

    SECTION_MAP: ClassVar[dict[str, str]] = {
        "DB": "db",
        "AUTH": "auth",
        "SENTRY": "sentry",
        "LLM": "llm",
        "AGENT": "agent",
        "DERIVER": "deriver",
        "HISTORY": "history",
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
    model_config = SettingsConfigDict(env_prefix="DB_")  # pyright: ignore

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
    model_config = SettingsConfigDict(env_prefix="AUTH_")  # pyright: ignore

    USE_AUTH: bool = False
    JWT_SECRET: str | None = None  # Must be set if USE_AUTH is true

    @model_validator(mode="after")  # type: ignore
    def _require_jwt_secret(self) -> "AuthSettings":
        if self.USE_AUTH and not self.JWT_SECRET:
            raise ValueError("JWT_SECRET must be set if USE_AUTH is true")
        return self


class SentrySettings(HonchoSettings):
    model_config = SettingsConfigDict(env_prefix="SENTRY_")  # pyright: ignore

    ENABLED: bool = False
    DSN: str | None = None
    TRACES_SAMPLE_RATE: Annotated[float, Field(default=0.1, ge=0.0, le=1.0)] = 0.1
    PROFILES_SAMPLE_RATE: Annotated[float, Field(default=0.1, ge=0.0, le=1.0)] = 0.1


class LLMSettings(HonchoSettings):
    model_config = SettingsConfigDict(env_prefix="LLM_")  # pyright: ignore

    # API Keys for LLM providers
    ANTHROPIC_API_KEY: str | None = None
    OPENAI_API_KEY: str | None = None
    OPENAI_COMPATIBLE_API_KEY: str | None = None
    GEMINI_API_KEY: str | None = None
    GROQ_API_KEY: str | None = None  # Added missing GROQ API key
    OPENAI_COMPATIBLE_BASE_URL: str | None = None

    # General LLM settings
    DEFAULT_MAX_TOKENS: Annotated[int, Field(default=1000, gt=0, le=100000)] = 1000
    DEFAULT_TEMPERATURE: Annotated[float, Field(default=0.0, ge=0.0, le=2.0)] = 0.0

    # Dialectic specific
    DIALECTIC_PROVIDER: str = "anthropic"
    DIALECTIC_MODEL: str = "claude-3-haiku-20240307"
    # DIALECTIC_SYSTEM_PROMPT_FILE: Optional[str] = "prompts/dialectic_system.txt" # Example for file-based

    # Query Generation specific
    # QUERY_GENERATION_PROVIDER: str = "gemini"
    # QUERY_GENERATION_MODEL: str = "llama3-8b-8192"
    QUERY_GENERATION_PROVIDER: str = "gemini"
    QUERY_GENERATION_MODEL: str = "gemini-2.0-flash-lite"
    # QUERY_GENERATION_SYSTEM_PROMPT_FILE: Optional[str] = "prompts/query_generation_system.txt"

    # Tom Inference specific
    # TOM_INFERENCE_PROVIDER: Provider = "groq"
    # TOM_INFERENCE_MODEL: str = "llama-3.3-70b-versatile"
    TOM_INFERENCE_PROVIDER: Provider = "anthropic"
    TOM_INFERENCE_MODEL: str = "claude-3-5-haiku-20241022"

    # Summarization specific
    SUMMARY_PROVIDER: str = "gemini"
    SUMMARY_MODEL: str = (
        "gemini-1.5-flash-latest"  # Consider specific model version if needed
    )
    SUMMARY_MAX_TOKENS_SHORT: Annotated[int, Field(default=1000, gt=0, le=10000)] = 1000
    SUMMARY_MAX_TOKENS_LONG: Annotated[int, Field(default=2000, gt=0, le=20000)] = 2000
    # SUMMARY_SYSTEM_PROMPT_SHORT_FILE: Optional[str] = "prompts/summary_short_system.txt"
    # SUMMARY_SYSTEM_PROMPT_LONG_FILE: Optional[str] = "prompts/summary_long_system.txt"


class AgentSettings(HonchoSettings):
    model_config = SettingsConfigDict(env_prefix="AGENT_")  # pyright: ignore

    SEMANTIC_SEARCH_TOP_K: Annotated[int, Field(default=10, gt=0, le=100)] = 10
    SEMANTIC_SEARCH_MAX_DISTANCE: Annotated[
        float, Field(default=0.85, ge=0.0, le=1.0)
    ] = 0.85  # Max distance for semantic search relevance
    TOM_INFERENCE_METHOD: str = "single_prompt"


class DeriverSettings(HonchoSettings):
    model_config = SettingsConfigDict(env_prefix="DERIVER_")  # pyright: ignore

    WORKERS: Annotated[int, Field(default=1, gt=0, le=100)] = 1
    STALE_SESSION_TIMEOUT_MINUTES: Annotated[int, Field(default=5, gt=0, le=1440)] = (
        5  # Max 24 hours
    )
    POLLING_SLEEP_INTERVAL_SECONDS: Annotated[
        float, Field(default=1.0, gt=0.0, le=60.0)
    ] = 1.0
    TOM_METHOD: str = "single_prompt"
    USER_REPRESENTATION_METHOD: str = "long_term"


class HistorySettings(HonchoSettings):
    model_config = SettingsConfigDict(env_prefix="HISTORY_")  # pyright: ignore

    MESSAGES_PER_SHORT_SUMMARY: Annotated[int, Field(default=20, gt=0, le=100)] = 20
    MESSAGES_PER_LONG_SUMMARY: Annotated[int, Field(default=60, gt=0, le=500)] = 60


class AppSettings(HonchoSettings):
    # No env_prefix for app-level settings
    model_config = SettingsConfigDict(  # pyright: ignore
        env_prefix="", env_nested_delimiter="__"
    )

    # Application-wide settings
    LOG_LEVEL: str = "INFO"
    FASTAPI_HOST: str = "0.0.0.0"
    FASTAPI_PORT: Annotated[int, Field(default=8000, gt=0, le=65535)] = 8000

    # Nested settings models
    DB: DBSettings = Field(default_factory=DBSettings)
    AUTH: AuthSettings = Field(default_factory=AuthSettings)
    SENTRY: SentrySettings = Field(default_factory=SentrySettings)
    LLM: LLMSettings = Field(default_factory=LLMSettings)
    AGENT: AgentSettings = Field(default_factory=AgentSettings)
    DERIVER: DeriverSettings = Field(default_factory=DeriverSettings)
    HISTORY: HistorySettings = Field(default_factory=HistorySettings)

    @field_validator("LOG_LEVEL")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        log_level = v.upper()
        if log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ValueError(f"Invalid log level: {v}")
        return log_level


# Create a single global instance of the settings
settings: AppSettings = AppSettings()
