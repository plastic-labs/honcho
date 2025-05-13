import os
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Load .env file for local development.
# Make sure this is called before AppSettings is instantiated if you rely on .env for AppSettings construction.
load_dotenv()

class DBSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix='DB_')

    CONNECTION_URI: str = "postgresql+psycopg://postgres:postgres@localhost:5432/postgres"
    SCHEMA: str = "public"
    POOL_PRE_PING: bool = True
    POOL_SIZE: int = 10
    MAX_OVERFLOW: int = 20
    POOL_TIMEOUT: int = 30 # seconds
    POOL_RECYCLE: int = 300 # seconds
    POOL_USE_LIFO: bool = True
    SQL_DEBUG: bool = False

class AuthSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix='AUTH_')

    USE_AUTH: bool = True
    JWT_SECRET: Optional[str] = None # Must be set if USE_AUTH is true

class SentrySettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix='SENTRY_')

    ENABLED: bool = False
    DSN: Optional[str] = None
    TRACES_SAMPLE_RATE: float = 0.1
    PROFILES_SAMPLE_RATE: float = 0.1

class OpenTelemetrySettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix='OPENTELEMETRY_')
    ENABLED: bool = False

class LLMSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix='LLM_')
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
    SUMMARY_MODEL: str = "gemini-2.0-flash-lite" # Consider specific model version if needed
    SUMMARY_MAX_TOKENS_SHORT: int = 1000
    SUMMARY_MAX_TOKENS_LONG: int = 2000
    # SUMMARY_SYSTEM_PROMPT_SHORT_FILE: Optional[str] = "prompts/summary_short_system.txt"
    # SUMMARY_SYSTEM_PROMPT_LONG_FILE: Optional[str] = "prompts/summary_long_system.txt"

class AgentSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix='AGENT_')

    SEMANTIC_SEARCH_TOP_K: int = 10
    SEMANTIC_SEARCH_MAX_DISTANCE: float = 0.85 # Max distance for semantic search relevance
    TOM_INFERENCE_METHOD: str = "single_prompt"

class DeriverSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix='DERIVER_')

    WORKERS: int = 1
    STALE_SESSION_TIMEOUT_MINUTES: int = 5
    POLLING_SLEEP_INTERVAL_SECONDS: float = 1.0
    TOM_METHOD: str = "single_prompt"
    USER_REPRESENTATION_METHOD: str = "long_term"

class HistorySettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix='HISTORY_')

    MESSAGES_PER_SHORT_SUMMARY: int = 20
    MESSAGES_PER_LONG_SUMMARY: int = 60

class AppSettings(BaseSettings):
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

    # For loading from a TOML file in the future:
    # model_config = SettingsConfigDict(env_file_encoding='utf-8', extra='ignore', toml_file='config.toml')

# Global settings instance
settings = AppSettings()

# Example for loading prompts from files (can be uncommented and adapted)
# def load_prompt_from_file(file_path: str, default_prompt: str = "") -> str:
#     expanded_path = os.path.expanduser(file_path) # Handles ~ for home directory
#     if not os.path.isabs(expanded_path):
#         # Assuming prompts directory is relative to the project root or a known location
#         # This might need adjustment based on your project structure
#         base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Project root
#         expanded_path = os.path.join(base_dir, file_path)

#     try:
#         with open(expanded_path, 'r') as f:
#             return f.read().strip()
#     except FileNotFoundError:
#         # You might want to log a warning here
#         # logger.warning(f"Prompt file not found: {expanded_path}. Using default.")
#         return default_prompt
#     except Exception as e:
#         # logger.error(f"Error loading prompt file {expanded_path}: {e}")
#         return default_prompt

# # Example of loading a specific prompt if its file path is set
# if settings.LLM.DIALECTIC_SYSTEM_PROMPT_FILE:
#     settings.LLM.DIALECTIC_SYSTEM_PROMPT = load_prompt_from_file(
#         settings.LLM.DIALECTIC_SYSTEM_PROMPT_FILE,
#         default_prompt="Default dialectic system prompt if file is missing." # Provide a fallback
#     )
# # Repeat for other file-based prompts 