//! Polling/batching subset of Python `DeriverSettings` (`src/config.py`).
//!
//! Only the fields the queue manager's scheduling + batching logic reads are
//! ported here; model-config / observation knobs are out of scope for the
//! worker's deterministic core and are added as their consumers are ported.
//!
//! Python uses pydantic-settings with `env_prefix="DERIVER_"`, so every field
//! maps to `DERIVER_<UPPER_SNAKE>`. Unlike pydantic we do not reject
//! out-of-range values: a missing or unparseable env var falls back to the
//! Python default (matching how `AppConfig::from_pairs` treats numeric vars).

use std::collections::HashMap;
use std::str::FromStr;

/// Collect an env-pair iterator into an owned `HashMap` (the shape the
/// `*Settings::from_pairs` parsers and [`crate::llm::ModelConfig::with_env_overrides`]
/// read).
pub fn collect_env<I, K, V>(pairs: I) -> HashMap<String, String>
where
    I: IntoIterator<Item = (K, V)>,
    K: AsRef<str>,
    V: AsRef<str>,
{
    pairs
        .into_iter()
        .map(|(key, value)| (key.as_ref().to_string(), value.as_ref().to_string()))
        .collect()
}

/// Parse `key` as `T`, falling back to `default` when absent or unparseable
/// (matching how the worker settings treat out-of-range / malformed env vars).
pub fn parse_or<T: FromStr>(values: &HashMap<String, String>, key: &str, default: T) -> T {
    values
        .get(key)
        .map(String::as_str)
        .map(str::trim)
        .and_then(|value| value.parse::<T>().ok())
        .unwrap_or(default)
}

/// Parse `key` as a boolean (`1`/`true`/`yes`/`on`, case-insensitive), falling
/// back to `default` when absent or blank.
pub fn parse_bool_or(values: &HashMap<String, String>, key: &str, default: bool) -> bool {
    values
        .get(key)
        .map(String::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(|value| matches!(value.to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
        .unwrap_or(default)
}

/// Scheduling + batching configuration for the deriver queue manager.
#[derive(Debug, Clone, PartialEq)]
pub struct DeriverSettings {
    /// `DERIVER_WORKERS` — concurrent work units per instance (default 1).
    pub workers: u32,
    /// `DERIVER_POLLING_SLEEP_INTERVAL_SECONDS` — base idle poll interval.
    pub polling_sleep_interval_seconds: f64,
    /// `DERIVER_POLLING_BACKOFF_ENABLED` — grow the idle interval while empty.
    pub polling_backoff_enabled: bool,
    /// `DERIVER_POLLING_SLEEP_MAX_INTERVAL_SECONDS` — backoff ceiling.
    pub polling_sleep_max_interval_seconds: f64,
    /// `DERIVER_POLLING_BACKOFF_MULTIPLIER` — per-cycle growth factor.
    pub polling_backoff_multiplier: f64,
    /// `DERIVER_POLLING_STARTUP_JITTER_SECONDS` — first-poll random delay window.
    pub polling_startup_jitter_seconds: f64,
    /// `DERIVER_POLLING_JITTER_RATIO` — scatter every sleep by +/- this ratio.
    pub polling_jitter_ratio: f64,
    /// `DERIVER_STALE_SESSION_TIMEOUT_MINUTES` — age before an AQS row is stale.
    pub stale_session_timeout_minutes: u32,
    /// `DERIVER_STALE_WORK_UNIT_CLEANUP_INTERVAL_SECONDS` — min spacing between
    /// stale-cleanup runs.
    pub stale_work_unit_cleanup_interval_seconds: f64,
    /// `DERIVER_REPRESENTATION_BATCH_MAX_TOKENS` — forced-batching token cap.
    pub representation_batch_max_tokens: i64,
    /// `DERIVER_REPRESENTATION_BATCH_MAX_AGE_SECONDS` — sub-threshold
    /// representation batches become eligible after this age; 0 disables.
    pub representation_batch_max_age_seconds: i64,
    /// `DERIVER_FLUSH_ENABLED` — bypass the batch token threshold.
    pub flush_enabled: bool,
    /// `DERIVER_QUEUE_ERROR_RETENTION_SECONDS` — how long errored queue items are
    /// retained before the reconciler's cleanup_queue task deletes them
    /// (default 30 days). Successfully processed items are deleted immediately.
    pub queue_error_retention_seconds: i64,
}

impl Default for DeriverSettings {
    fn default() -> Self {
        Self {
            workers: 1,
            polling_sleep_interval_seconds: 1.0,
            polling_backoff_enabled: true,
            polling_sleep_max_interval_seconds: 30.0,
            polling_backoff_multiplier: 2.0,
            polling_startup_jitter_seconds: 30.0,
            polling_jitter_ratio: 0.5,
            stale_session_timeout_minutes: 5,
            stale_work_unit_cleanup_interval_seconds: 60.0,
            representation_batch_max_tokens: 1024,
            representation_batch_max_age_seconds: 1800,
            flush_enabled: false,
            queue_error_retention_seconds: 30 * 24 * 3600,
        }
    }
}

impl DeriverSettings {
    /// Read settings from the process environment.
    pub fn from_env() -> Self {
        Self::from_pairs(std::env::vars())
    }

    /// Read settings from an arbitrary key/value source (testable).
    pub fn from_pairs<I, K, V>(pairs: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: AsRef<str>,
        V: AsRef<str>,
    {
        let values = pairs
            .into_iter()
            .map(|(key, value)| (key.as_ref().to_string(), value.as_ref().to_string()))
            .collect::<HashMap<_, _>>();

        let defaults = Self::default();

        let parse_num = |key: &str, default: f64| -> f64 {
            values
                .get(key)
                .map(String::as_str)
                .and_then(|value| value.trim().parse::<f64>().ok())
                .unwrap_or(default)
        };
        let parse_u32 = |key: &str, default: u32| -> u32 {
            values
                .get(key)
                .map(String::as_str)
                .and_then(|value| value.trim().parse::<u32>().ok())
                .unwrap_or(default)
        };
        let parse_i64 = |key: &str, default: i64| -> i64 {
            values
                .get(key)
                .map(String::as_str)
                .and_then(|value| value.trim().parse::<i64>().ok())
                .unwrap_or(default)
        };
        let parse_nonnegative_i64 = |key: &str, default: i64| -> i64 {
            values
                .get(key)
                .map(String::as_str)
                .and_then(|value| value.trim().parse::<i64>().ok())
                .filter(|value| *value >= 0)
                .unwrap_or(default)
        };
        let parse_bool = |key: &str, default: bool| -> bool {
            values
                .get(key)
                .map(String::as_str)
                .filter(|value| !value.trim().is_empty())
                .map(|value| {
                    matches!(
                        value.trim().to_ascii_lowercase().as_str(),
                        "1" | "true" | "yes" | "on"
                    )
                })
                .unwrap_or(default)
        };

        Self {
            workers: parse_u32("DERIVER_WORKERS", defaults.workers),
            polling_sleep_interval_seconds: parse_num(
                "DERIVER_POLLING_SLEEP_INTERVAL_SECONDS",
                defaults.polling_sleep_interval_seconds,
            ),
            polling_backoff_enabled: parse_bool(
                "DERIVER_POLLING_BACKOFF_ENABLED",
                defaults.polling_backoff_enabled,
            ),
            polling_sleep_max_interval_seconds: parse_num(
                "DERIVER_POLLING_SLEEP_MAX_INTERVAL_SECONDS",
                defaults.polling_sleep_max_interval_seconds,
            ),
            polling_backoff_multiplier: parse_num(
                "DERIVER_POLLING_BACKOFF_MULTIPLIER",
                defaults.polling_backoff_multiplier,
            ),
            polling_startup_jitter_seconds: parse_num(
                "DERIVER_POLLING_STARTUP_JITTER_SECONDS",
                defaults.polling_startup_jitter_seconds,
            ),
            polling_jitter_ratio: parse_num(
                "DERIVER_POLLING_JITTER_RATIO",
                defaults.polling_jitter_ratio,
            ),
            stale_session_timeout_minutes: parse_u32(
                "DERIVER_STALE_SESSION_TIMEOUT_MINUTES",
                defaults.stale_session_timeout_minutes,
            ),
            stale_work_unit_cleanup_interval_seconds: parse_num(
                "DERIVER_STALE_WORK_UNIT_CLEANUP_INTERVAL_SECONDS",
                defaults.stale_work_unit_cleanup_interval_seconds,
            ),
            representation_batch_max_tokens: parse_i64(
                "DERIVER_REPRESENTATION_BATCH_MAX_TOKENS",
                defaults.representation_batch_max_tokens,
            ),
            representation_batch_max_age_seconds: parse_nonnegative_i64(
                "DERIVER_REPRESENTATION_BATCH_MAX_AGE_SECONDS",
                defaults.representation_batch_max_age_seconds,
            ),
            flush_enabled: parse_bool("DERIVER_FLUSH_ENABLED", defaults.flush_enabled),
            queue_error_retention_seconds: parse_i64(
                "DERIVER_QUEUE_ERROR_RETENTION_SECONDS",
                defaults.queue_error_retention_seconds,
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_python() {
        let s = DeriverSettings::default();
        assert_eq!(s.workers, 1);
        assert_eq!(s.polling_sleep_interval_seconds, 1.0);
        assert!(s.polling_backoff_enabled);
        assert_eq!(s.polling_sleep_max_interval_seconds, 30.0);
        assert_eq!(s.polling_backoff_multiplier, 2.0);
        assert_eq!(s.polling_startup_jitter_seconds, 30.0);
        assert_eq!(s.polling_jitter_ratio, 0.5);
        assert_eq!(s.stale_session_timeout_minutes, 5);
        assert_eq!(s.stale_work_unit_cleanup_interval_seconds, 60.0);
        assert_eq!(s.representation_batch_max_tokens, 1024);
        assert_eq!(s.representation_batch_max_age_seconds, 1800);
        assert!(!s.flush_enabled);
    }

    #[test]
    fn empty_env_yields_defaults() {
        let s = DeriverSettings::from_pairs(Vec::<(String, String)>::new());
        assert_eq!(s, DeriverSettings::default());
    }

    #[test]
    fn env_overrides_are_parsed() {
        let s = DeriverSettings::from_pairs([
            ("DERIVER_WORKERS", "4"),
            ("DERIVER_POLLING_SLEEP_INTERVAL_SECONDS", "2.5"),
            ("DERIVER_POLLING_BACKOFF_ENABLED", "false"),
            ("DERIVER_POLLING_BACKOFF_MULTIPLIER", "3"),
            ("DERIVER_POLLING_JITTER_RATIO", "0"),
            ("DERIVER_REPRESENTATION_BATCH_MAX_TOKENS", "2048"),
            ("DERIVER_REPRESENTATION_BATCH_MAX_AGE_SECONDS", "30"),
            ("DERIVER_FLUSH_ENABLED", "1"),
        ]);
        assert_eq!(s.workers, 4);
        assert_eq!(s.polling_sleep_interval_seconds, 2.5);
        assert!(!s.polling_backoff_enabled);
        assert_eq!(s.polling_backoff_multiplier, 3.0);
        assert_eq!(s.polling_jitter_ratio, 0.0);
        assert_eq!(s.representation_batch_max_tokens, 2048);
        assert_eq!(s.representation_batch_max_age_seconds, 30);
        assert!(s.flush_enabled);
    }

    #[test]
    fn unparseable_value_falls_back_to_default() {
        let s = DeriverSettings::from_pairs([("DERIVER_WORKERS", "not-a-number")]);
        assert_eq!(s.workers, DeriverSettings::default().workers);
        let s =
            DeriverSettings::from_pairs([("DERIVER_REPRESENTATION_BATCH_MAX_AGE_SECONDS", "-1")]);
        assert_eq!(
            s.representation_batch_max_age_seconds,
            DeriverSettings::default().representation_batch_max_age_seconds
        );
    }
}
