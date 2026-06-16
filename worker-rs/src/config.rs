//! Minimal worker configuration.
//!
//! Mirrors the env names Python uses (`DB_CONNECTION_URI`, `DB_SCHEMA`,
//! `DERIVER__*`) so the Rust worker can be dropped into the same deployment
//! environment. Only the settings the no-op consumer milestone needs are parsed
//! today; the polling/backoff knobs documented in the queue-schema reference are
//! added as later steps wire up the full polling loop.

use std::collections::HashMap;
use std::time::Duration;

use thiserror::Error;

const DEFAULT_DB_SCHEMA: &str = "public";
const DEFAULT_WORKERS: i64 = 1;
const DEFAULT_STALE_SESSION_TIMEOUT_MINUTES: i64 = 5;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerConfig {
    pub database_url: String,
    pub db_schema: String,
    /// Max work units claimed per poll (Python `DERIVER.WORKERS`).
    pub workers: i64,
    /// Active-queue-session heartbeat age beyond which a work unit is reclaimable
    /// (Python `DERIVER.STALE_SESSION_TIMEOUT_MINUTES`).
    pub stale_session_timeout: Duration,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ConfigError {
    #[error("DB_CONNECTION_URI is required")]
    MissingDatabaseUrl,
    #[error("{name} is invalid: {value}")]
    InvalidInt { name: &'static str, value: String },
}

impl WorkerConfig {
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

        let workers = parse_int(
            &values,
            "DERIVER__WORKERS",
            DEFAULT_WORKERS,
        )?;
        let stale_minutes = parse_int(
            &values,
            "DERIVER__STALE_SESSION_TIMEOUT_MINUTES",
            DEFAULT_STALE_SESSION_TIMEOUT_MINUTES,
        )?;

        Ok(Self {
            database_url,
            db_schema,
            workers,
            stale_session_timeout: Duration::from_secs((stale_minutes.max(0) as u64) * 60),
        })
    }
}

fn parse_int(
    values: &HashMap<String, String>,
    name: &'static str,
    default: i64,
) -> Result<i64, ConfigError> {
    match values.get(name).map(String::as_str).filter(|v| !v.trim().is_empty()) {
        None => Ok(default),
        Some(value) => value
            .trim()
            .parse::<i64>()
            .map_err(|_| ConfigError::InvalidInt {
                name,
                value: value.to_string(),
            }),
    }
}

/// Convert a SQLAlchemy-style URL (`postgresql+psycopg://`) to one `sqlx`
/// accepts, matching `api-rs`'s `normalize_python_postgres_url`.
pub fn normalize_python_postgres_url(value: &str) -> String {
    value
        .trim()
        .replacen("postgresql+psycopg://", "postgresql://", 1)
        .replacen("postgres+psycopg://", "postgres://", 1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_apply_and_url_is_normalized() {
        let config = WorkerConfig::from_pairs([(
            "DB_CONNECTION_URI",
            "postgresql+psycopg://u:p@localhost:5432/db",
        )])
        .expect("config");
        assert_eq!(config.database_url, "postgresql://u:p@localhost:5432/db");
        assert_eq!(config.db_schema, "public");
        assert_eq!(config.workers, DEFAULT_WORKERS);
        assert_eq!(config.stale_session_timeout, Duration::from_secs(300));
    }

    #[test]
    fn overrides_are_parsed() {
        let config = WorkerConfig::from_pairs([
            ("DB_CONNECTION_URI", "postgres://h/db"),
            ("DB_SCHEMA", "honcho"),
            ("DERIVER__WORKERS", "4"),
            ("DERIVER__STALE_SESSION_TIMEOUT_MINUTES", "10"),
        ])
        .expect("config");
        assert_eq!(config.db_schema, "honcho");
        assert_eq!(config.workers, 4);
        assert_eq!(config.stale_session_timeout, Duration::from_secs(600));
    }

    #[test]
    fn missing_url_errors() {
        let err = WorkerConfig::from_pairs(Vec::<(String, String)>::new()).unwrap_err();
        assert_eq!(err, ConfigError::MissingDatabaseUrl);
    }

    #[test]
    fn invalid_int_errors() {
        let err = WorkerConfig::from_pairs([
            ("DB_CONNECTION_URI", "postgres://h/db"),
            ("DERIVER__WORKERS", "lots"),
        ])
        .unwrap_err();
        assert_eq!(
            err,
            ConfigError::InvalidInt {
                name: "DERIVER__WORKERS",
                value: "lots".to_string()
            }
        );
    }
}
