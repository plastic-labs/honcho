import logging
from pathlib import Path
from typing import Any, Optional

import tomllib
from dotenv import load_dotenv
from pydantic.fields import Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

# Load .env file for local development.
# Make sure this is called before AppSettings is instantiated if you rely on .env for AppSettings construction.
load_dotenv()


def load_toml_config(config_path: str = "config.toml") -> dict[str, Any]:
    """Load configuration from TOML file if it exists."""
    config_file = Path(config_path)
    if config_file.exists():
        try:
            with open(config_file, "rb") as f:
                return tomllib.load(f)
        except (tomllib.TOMLDecodeError, OSError) as exc:
            logging.warning("Failed to load %s: %s", config_path, exc)
            return {}
    return {}


# Load TOML config once
TOML_CONFIG = load_toml_config()


class TomlConfigSettingsSource(PydanticBaseSettingsSource):
    """Custom settings source for loading from TOML file."""

    def __init__(self, settings_cls: type[BaseSettings]):
        super().__init__(settings_cls)

    SECTION_MAP: dict[str, str] = {
        "DB": "db",
        "AUTH": "auth",
        "SENTRY": "sentry",
        "OPENTELEMETRY": "opentelemetry",
        "LLM": "llm",
        "AGENT": "agent",
        "DERIVER": "deriver",
        "HISTORY": "history",
        "": "app",  # For AppSettings with no prefix
    }

    def get_field_value(self, field: Field) -> tuple[Any, str, bool]:
        field_name = field.name
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


class TomlSettings(BaseSettings):
    """Base settings class that loads from TOML config first, then env vars."""

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        # Return sources in priority order (first is lowest priority)
        return (
            init_settings,
            TomlConfigSettingsSource(settings_cls),
            dotenv_settings,
            env_settings,
            file_secret_settings,
        )


class DBSettings(TomlSettings):
    model_config = SettingsConfigDict(env_prefix="DB_")

    CONNECTION_URI: str = (
        "postgresql+psycopg://postgres:postgres@localhost:5432/postgres"
    )
    SCHEMA: str = "public"
    POOL_PRE_PING: bool = True
    POOL_SIZE: int = 10
    MAX_OVERFLOW: int = 20
    POOL_TIMEOUT: int = 30  # seconds
    POOL_RECYCLE: int = 300  # seconds
    POOL_USE_LIFO: bool = True
    SQL_DEBUG: bool = False
    TRACING: bool = False


class AuthSettings(TomlSettings):
    model_config = SettingsConfigDict(env_prefix="AUTH_")

    USE_AUTH: bool = True
    JWT_SECRET: Optional[str] = None  # Must be set if USE_AUTH is true


class SentrySettings(TomlSettings):
    model_config = SettingsConfigDict(env_prefix="SENTRY_")

    ENABLED: bool = False
    DSN: Optional[str] = None
    TRACES_SAMPLE_RATE: float = 0.1
    PROFILES_SAMPLE_RATE: float = 0.1


class OpenTelemetrySettings(TomlSettings):
    model_config = SettingsConfigDict(env_prefix="OPENTELEMETRY_")
    ENABLED: bool = False


class LLMSettings(TomlSettings):
    model_config = SettingsConfigDict(env_prefix="LLM_")
    # General LLM settings
    DEFAULT_MAX_TOKENS: int = 1000
    DEFAULT_TEMPERATURE: float = 0.0

    # Dialectic specific
    DIALECTIC_PROVIDER: str = "anthropic"
    DIALECTIC_MODEL: str = "claude-3-7-sonnet-20250219"
    # DIALECTIC_SYSTEM_PROMPT_FILE: Optional[str] = "prompts/dialectic_system.txt" # Example for file-based

    # Query Generation specific
    QUERY_GENERATION_PROVIDER: str = "groq"
    QUERY_GENERATION_MODEL: str = "llama-3.1-8b-instant"
    # QUERY_GENERATION_SYSTEM_PROMPT_FILE: Optional[str] = "prompts/query_generation_system.txt"

    # Summarization specific
    SUMMARY_PROVIDER: str = "gemini"
    SUMMARY_MODEL: str = (
        "gemini-2.0-flash-lite"  # Consider specific model version if needed
    )
    SUMMARY_MAX_TOKENS_SHORT: int = 1000
    SUMMARY_MAX_TOKENS_LONG: int = 2000
    # SUMMARY_SYSTEM_PROMPT_SHORT_FILE: Optional[str] = "prompts/summary_short_system.txt"
    # SUMMARY_SYSTEM_PROMPT_LONG_FILE: Optional[str] = "prompts/summary_long_system.txt"


class AgentSettings(TomlSettings):
    model_config = SettingsConfigDict(env_prefix="AGENT_")

    SEMANTIC_SEARCH_TOP_K: int = 10
    SEMANTIC_SEARCH_MAX_DISTANCE: float = (
        0.85  # Max distance for semantic search relevance
    )
    TOM_INFERENCE_METHOD: str = "single_prompt"
    USER_REPRESENTATION_METAMESSAGE_TYPE: str = "honcho_user_representation"


class DeriverSettings(TomlSettings):
    model_config = SettingsConfigDict(env_prefix="DERIVER_")

    WORKERS: int = 1
    STALE_SESSION_TIMEOUT_MINUTES: int = 5
    POLLING_SLEEP_INTERVAL_SECONDS: float = 1.0
    TOM_METHOD: str = "single_prompt"
    USER_REPRESENTATION_METHOD: str = "long_term"


class HistorySettings(TomlSettings):
    model_config = SettingsConfigDict(env_prefix="HISTORY_")

    MESSAGES_PER_SHORT_SUMMARY: int = 20
    MESSAGES_PER_LONG_SUMMARY: int = 60


class AppSettings(TomlSettings):
    # No env_prefix for app-level settings
    model_config = SettingsConfigDict(env_prefix="")

    # Application-wide settings
    LOG_LEVEL: str = "INFO"
    FASTAPI_HOST: str = "0.0.0.0"
    FASTAPI_PORT: int = 8000

    # Nested settings models
    DB: DBSettings = DBSettings()
    AUTH: AuthSettings = AuthSettings()
    SENTRY: SentrySettings = SentrySettings()
    OPENTELEMETRY: OpenTelemetrySettings = OpenTelemetrySettings()
    LLM: LLMSettings = LLMSettings()
    AGENT: AgentSettings = AgentSettings()
    DERIVER: DeriverSettings = DeriverSettings()
    HISTORY: HistorySettings = HistorySettings()


# Global settings instance
settings = AppSettings()
