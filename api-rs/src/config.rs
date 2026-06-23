use crate::auth::AuthConfig;
use crate::cache::{CacheConfig, default_cache_namespace, default_cache_url};
use std::collections::HashMap;
use std::net::SocketAddr;
use thiserror::Error;

const DEFAULT_BIND_ADDRESS: &str = "0.0.0.0:8001";
const DEFAULT_DB_SCHEMA: &str = "public";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppConfig {
    pub bind_address: SocketAddr,
    pub database_url: String,
    pub db_schema: String,
    pub auth: AuthConfig,
    pub write_enabled: bool,
    pub cache: CacheConfig,
    /// Whether message creation also persists pending `MessageEmbedding` chunk
    /// rows (Python `settings.EMBED_MESSAGES`, default true).
    pub embed_messages: bool,
    /// Per-chunk token budget for embedding chunking, derived like the Python
    /// `_EmbeddingClient`: Gemini caps at `min(MAX_INPUT_TOKENS, 2048)`, OpenAI
    /// (default provider) uses `MAX_INPUT_TOKENS` directly.
    pub embedding_max_tokens: usize,
    /// Embedding model name (`EMBEDDING_MODEL`, default `text-embedding-3-small`).
    pub embedding_model: String,
    /// Target embedding dimensionality (`EMBEDDING_VECTOR_DIMENSIONS`, default 1536).
    pub embedding_vector_dimensions: usize,
    /// Whether OpenAI embedding requests forward `dimensions=`, mirroring
    /// `EmbeddingSettings.resolve_send_dimensions` (`EMBEDDING_DIMENSIONS_MODE`
    /// `always`/`never`/`auto`; `auto` sends only when `EMBEDDING_VECTOR_DIMENSIONS`
    /// is explicitly set and the model isn't a known-rejecting one).
    pub embedding_send_dimensions: bool,
    /// OpenAI API key for the synchronous search-query embedding, defaulting to
    /// `LLM_OPENAI_API_KEY` (Python `_default_embedding_api_key`). `None` disables
    /// the semantic search leg (search falls back to full-text only).
    pub embedding_api_key: Option<String>,
    /// Optional embedding endpoint override (`EMBEDDING_BASE_URL`).
    pub embedding_base_url: Option<String>,
    /// Embedding provider (`EMBEDDING_PROVIDER`, `openai` [default] or `gemini`),
    /// selecting which `embed_*` backend the embedder dispatches to.
    pub embedding_transport: crate::llm::Provider,
    /// Whether dream scheduling is enabled (`DREAM_ENABLED`, Python
    /// `settings.DREAM.ENABLED`, default true). `schedule_dream` 400s when off.
    pub dream_enabled: bool,
    /// Per-transport LLM API keys (`LLM_{ANTHROPIC,OPENAI,GEMINI}_API_KEY`),
    /// used by the dialectic completion calls. `None` for an unconfigured
    /// transport (its calls then fail auth).
    pub llm_anthropic_api_key: Option<String>,
    pub llm_openai_api_key: Option<String>,
    pub llm_gemini_api_key: Option<String>,
    /// HMAC signing secret for webhook delivery (`WEBHOOK_SECRET`, Python
    /// `settings.WEBHOOK.SECRET`, default `None`). The deriver worker cannot sign
    /// — and therefore skips — deliveries when this is unset.
    pub webhook_secret: Option<String>,
}

impl AppConfig {
    /// The per-transport LLM keys for the dialectic credential resolver.
    pub fn llm_keys(&self) -> crate::llm::credentials::TransportApiKeys {
        crate::llm::credentials::TransportApiKeys {
            anthropic: self.llm_anthropic_api_key.clone(),
            openai: self.llm_openai_api_key.clone(),
            gemini: self.llm_gemini_api_key.clone(),
        }
    }
}

/// The resolved embedding configuration the synchronous search-query embedding
/// needs (the message-ingest embedding path is decoupled and only needs
/// `embedding_max_tokens`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbeddingConfig {
    pub model: String,
    pub vector_dimensions: usize,
    pub send_dimensions: bool,
    pub max_tokens: usize,
    pub api_key: Option<String>,
    pub base_url: Option<String>,
    pub transport: crate::llm::Provider,
}

impl AppConfig {
    /// Bundle the embedding-related fields for the search route.
    pub fn embedding_config(&self) -> EmbeddingConfig {
        EmbeddingConfig {
            model: self.embedding_model.clone(),
            vector_dimensions: self.embedding_vector_dimensions,
            send_dimensions: self.embedding_send_dimensions,
            max_tokens: self.embedding_max_tokens,
            api_key: self.embedding_api_key.clone(),
            base_url: self.embedding_base_url.clone(),
            transport: self.embedding_transport,
        }
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ConfigError {
    #[error("DB_CONNECTION_URI is required")]
    MissingDatabaseUrl,
    #[error("RUST_API_BIND_ADDRESS is invalid: {0}")]
    InvalidBindAddress(String),
    #[error("AUTH_JWT_SECRET is required when AUTH_USE_AUTH is true")]
    MissingJwtSecret,
    #[error("{name} is invalid: {value}")]
    InvalidBool { name: &'static str, value: String },
}

impl AppConfig {
    pub fn from_env() -> Result<Self, ConfigError> {
        Self::from_pairs(std::env::vars())
    }

    pub fn from_pairs<I, K, V>(pairs: I) -> Result<Self, ConfigError>
    where
        I: IntoIterator<Item = (K, V)>,
        K: AsRef<str>,
        V: AsRef<str>,
    {
        let values = pairs
            .into_iter()
            .map(|(key, value)| (key.as_ref().to_string(), value.as_ref().to_string()))
            .collect::<HashMap<_, _>>();

        let bind_address = values
            .get("RUST_API_BIND_ADDRESS")
            .map(String::as_str)
            .unwrap_or(DEFAULT_BIND_ADDRESS)
            .parse::<SocketAddr>()
            .map_err(|error| ConfigError::InvalidBindAddress(error.to_string()))?;
        let database_url = values
            .get("DB_CONNECTION_URI")
            .map(String::as_str)
            .filter(|value| !value.trim().is_empty())
            .map(normalize_python_postgres_url)
            .ok_or(ConfigError::MissingDatabaseUrl)?;
        let db_schema = values
            .get("DB_SCHEMA")
            .map(String::as_str)
            .filter(|value| !value.trim().is_empty())
            .unwrap_or(DEFAULT_DB_SCHEMA)
            .to_string();
        let use_auth = values
            .get("AUTH_USE_AUTH")
            .map(String::as_str)
            .map(parse_bool)
            .unwrap_or(false);
        let write_enabled = values
            .get("RUST_API_ENABLE_WRITES")
            .map(String::as_str)
            .map(parse_bool)
            .unwrap_or(false);
        let cache_enabled = values
            .get("CACHE_ENABLED")
            .map(String::as_str)
            .filter(|value| !value.trim().is_empty())
            .map(|value| parse_bool_strict("CACHE_ENABLED", value))
            .transpose()?
            .unwrap_or(false);
        let cache_url = values
            .get("CACHE_URL")
            .map(String::as_str)
            .filter(|value| !value.trim().is_empty())
            .unwrap_or(default_cache_url())
            .to_string();
        let cache_namespace = values
            .get("CACHE_NAMESPACE")
            .map(String::as_str)
            .filter(|value| !value.trim().is_empty())
            .or_else(|| {
                values
                    .get("NAMESPACE")
                    .map(String::as_str)
                    .filter(|value| !value.trim().is_empty())
            })
            .unwrap_or(default_cache_namespace())
            .to_string();
        let embed_messages = values
            .get("EMBED_MESSAGES")
            .map(String::as_str)
            .filter(|value| !value.trim().is_empty())
            .map(parse_bool)
            .unwrap_or(true);
        let embedding_provider = values
            .get("EMBEDDING_PROVIDER")
            .map(String::as_str)
            .filter(|value| !value.trim().is_empty())
            .unwrap_or("openai")
            .to_ascii_lowercase();
        let embedding_max_input_tokens = values
            .get("EMBEDDING_MAX_INPUT_TOKENS")
            .map(String::as_str)
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap_or(8192);
        let embedding_max_tokens = if embedding_provider == "gemini" {
            embedding_max_input_tokens.min(2048)
        } else {
            embedding_max_input_tokens
        };
        let embedding_transport = if embedding_provider == "gemini" {
            crate::llm::Provider::Gemini
        } else {
            crate::llm::Provider::Openai
        };
        let embedding_model = values
            .get("EMBEDDING_MODEL")
            .map(String::as_str)
            .filter(|value| !value.trim().is_empty())
            .unwrap_or("text-embedding-3-small")
            .to_string();
        let vector_dimensions_set = values
            .get("EMBEDDING_VECTOR_DIMENSIONS")
            .map(String::as_str)
            .is_some_and(|value| !value.trim().is_empty());
        let embedding_vector_dimensions = values
            .get("EMBEDDING_VECTOR_DIMENSIONS")
            .map(String::as_str)
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap_or(1536);
        // Mirror of `EmbeddingSettings.resolve_send_dimensions`: `auto` (default)
        // sends `dimensions=` only when VECTOR_DIMENSIONS was explicitly set and
        // the model isn't a known-rejecting one (only `text-embedding-ada-002`).
        let dimensions_mode = values
            .get("EMBEDDING_DIMENSIONS_MODE")
            .map(String::as_str)
            .filter(|value| !value.trim().is_empty())
            .unwrap_or("auto")
            .to_ascii_lowercase();
        let embedding_send_dimensions = match dimensions_mode.as_str() {
            "always" => true,
            "never" => false,
            _ => vector_dimensions_set && embedding_model != "text-embedding-ada-002",
        };
        let embedding_api_key = values
            .get("LLM_OPENAI_API_KEY")
            .map(String::as_str)
            .filter(|value| !value.trim().is_empty())
            .map(str::to_string);
        let embedding_base_url = values
            .get("EMBEDDING_BASE_URL")
            .map(String::as_str)
            .filter(|value| !value.trim().is_empty())
            .map(str::to_string);
        let read_key = |key: &str| {
            values
                .get(key)
                .map(String::as_str)
                .filter(|value| !value.trim().is_empty())
                .map(str::to_string)
        };
        let llm_anthropic_api_key = read_key("LLM_ANTHROPIC_API_KEY");
        let llm_openai_api_key = read_key("LLM_OPENAI_API_KEY");
        let llm_gemini_api_key = read_key("LLM_GEMINI_API_KEY");
        let webhook_secret = read_key("WEBHOOK_SECRET");
        let dream_enabled = values
            .get("DREAM_ENABLED")
            .map(String::as_str)
            .filter(|value| !value.trim().is_empty())
            .map(parse_bool)
            .unwrap_or(true);

        let jwt_secret = values
            .get("AUTH_JWT_SECRET")
            .map(String::as_str)
            .filter(|value| !value.trim().is_empty())
            .map(str::to_string);

        if use_auth && jwt_secret.is_none() {
            return Err(ConfigError::MissingJwtSecret);
        }

        Ok(Self {
            bind_address,
            database_url,
            db_schema,
            auth: AuthConfig {
                use_auth,
                jwt_secret,
            },
            write_enabled,
            cache: CacheConfig {
                enabled: cache_enabled,
                url: cache_url,
                namespace: cache_namespace,
            },
            embed_messages,
            embedding_max_tokens,
            embedding_model,
            embedding_vector_dimensions,
            embedding_send_dimensions,
            embedding_api_key,
            embedding_base_url,
            embedding_transport,
            dream_enabled,
            llm_anthropic_api_key,
            llm_openai_api_key,
            llm_gemini_api_key,
            webhook_secret,
        })
    }
}

fn parse_bool(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

fn parse_bool_strict(name: &'static str, value: &str) -> Result<bool, ConfigError> {
    match value.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => Err(ConfigError::InvalidBool {
            name,
            value: value.to_string(),
        }),
    }
}

fn normalize_python_postgres_url(value: &str) -> String {
    value
        .trim()
        .replacen("postgresql+psycopg://", "postgresql://", 1)
        .replacen("postgres+psycopg://", "postgres://", 1)
}
