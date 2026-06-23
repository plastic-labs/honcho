//! Port of `dreamer.dream_scheduler.check_and_schedule_dream` — the
//! explicit-observation threshold gate that enqueues dream work units. Without
//! it nothing dreams in production (analogous to the `ReconcilerScheduler`).
//! Called from `save_representation` after observation documents are written.
//!
//! **Deviation — no idle-timeout debounce.** Python's `DreamScheduler` is an
//! in-process singleton holding a per-`work_unit_key` map of delayed asyncio
//! tasks: each batch cancels and reschedules a timer that fires after
//! `IDLE_TIMEOUT_MINUTES`, so a dream only runs once the conversation goes idle.
//! This port enqueues immediately on threshold instead. The bounding invariants
//! are preserved by the queue itself: the in-flight pending check (here) plus
//! `enqueue_scheduled_dream`'s per-key dedup prevent duplicate dreams, and
//! `MIN_HOURS_BETWEEN_DREAMS` plus the `record_dream_guard` baseline reset
//! prevent re-dreaming. The `delay_reason` payload field still reflects the
//! configured intent (`idle_timeout` when `IDLE_TIMEOUT_MINUTES > 0`) for
//! telemetry fidelity, even though the dream fires now rather than after a delay.
//! A per-worker in-process timer would not coordinate across worker processes
//! anyway, so the queue-based gate is the more robust choice for a fleet.

use chrono::{DateTime, Utc};
use serde_json::Value;
use sqlx::PgPool;

use crate::db;

/// The subset of Python `settings.DREAM` that `check_and_schedule_dream` reads.
/// Defaults match `DreamSettings` (config.py): enabled, threshold 50, idle
/// timeout 60min, min 8h between dreams, only the `omni` dream type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DreamScheduleSettings {
    pub enabled: bool,
    pub document_threshold: i64,
    pub idle_timeout_minutes: i64,
    pub min_hours_between_dreams: i64,
    pub enabled_types: Vec<String>,
}

impl Default for DreamScheduleSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            document_threshold: 50,
            idle_timeout_minutes: 60,
            min_hours_between_dreams: 8,
            enabled_types: vec!["omni".to_string()],
        }
    }
}

impl DreamScheduleSettings {
    /// Read the scheduling knobs from the process environment (Python
    /// `env_prefix="DREAM_"`). Out-of-range values are not rejected — a missing
    /// or unparseable var falls back to the default, like [`super::super::deriver::settings::DeriverSettings::from_env`].
    pub fn from_env() -> Self {
        Self::from_pairs(std::env::vars())
    }

    /// Read from an arbitrary key/value source (testable).
    ///
    /// `DREAM_ENABLED_TYPES` is parsed as a comma-separated list (trimmed,
    /// empties dropped); an empty/absent value keeps the default `["omni"]`.
    /// Deviation: pydantic-settings parses `list[str]` env vars as JSON — the
    /// comma form is the conventional shell shape and the only override anyone
    /// realistically sets here.
    pub fn from_pairs<I, K, V>(pairs: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: AsRef<str>,
        V: AsRef<str>,
    {
        let values = pairs
            .into_iter()
            .map(|(key, value)| (key.as_ref().to_string(), value.as_ref().to_string()))
            .collect::<std::collections::HashMap<_, _>>();

        let defaults = Self::default();

        let parse_i64 = |key: &str, default: i64| -> i64 {
            values
                .get(key)
                .map(String::as_str)
                .and_then(|value| value.trim().parse::<i64>().ok())
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

        let enabled_types = values
            .get("DREAM_ENABLED_TYPES")
            .map(|raw| {
                raw.split(',')
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(str::to_string)
                    .collect::<Vec<_>>()
            })
            .filter(|types| !types.is_empty())
            .unwrap_or(defaults.enabled_types);

        Self {
            enabled: parse_bool("DREAM_ENABLED", defaults.enabled),
            document_threshold: parse_i64("DREAM_DOCUMENT_THRESHOLD", defaults.document_threshold),
            idle_timeout_minutes: parse_i64(
                "DREAM_IDLE_TIMEOUT_MINUTES",
                defaults.idle_timeout_minutes,
            ),
            min_hours_between_dreams: parse_i64(
                "DREAM_MIN_HOURS_BETWEEN_DREAMS",
                defaults.min_hours_between_dreams,
            ),
            enabled_types,
        }
    }
}

/// The `collection.internal_metadata["dream"]` guard fields (written by
/// `record_dream_guard` on a completed dream).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DreamMetadata {
    pub last_dream_document_count: i64,
    pub last_dream_at: Option<String>,
}

impl DreamMetadata {
    /// Read the `dream` sub-object out of a collection's `internal_metadata`,
    /// defaulting `last_dream_document_count` to 0 and `last_dream_at` to absent
    /// (Python's `dream_metadata.get(..., 0)` / `.get("last_dream_at")`).
    pub fn from_collection_internal_metadata(internal_metadata: &Value) -> Self {
        let dream = internal_metadata.get("dream");
        let last_dream_document_count = dream
            .and_then(|d| d.get("last_dream_document_count"))
            .and_then(Value::as_i64)
            .unwrap_or(0);
        let last_dream_at = dream
            .and_then(|d| d.get("last_dream_at"))
            .and_then(Value::as_str)
            .map(str::to_string);
        Self {
            last_dream_document_count,
            last_dream_at,
        }
    }
}

/// Why a dream was not scheduled (the `return False` branches), kept for tests
/// and logging.
#[derive(Debug, Clone, PartialEq)]
pub enum DreamSkipReason {
    /// `settings.DREAM.ENABLED` is false.
    Disabled,
    /// Fewer than `DOCUMENT_THRESHOLD` explicit docs since the last dream.
    BelowThreshold {
        documents_since_last_dream: i64,
        document_threshold: i64,
    },
    /// A valid `last_dream_at` is more recent than `MIN_HOURS_BETWEEN_DREAMS`.
    MinHoursGate { hours_since_last_dream: f64 },
}

/// The pure decision of `check_and_schedule_dream` before the DB in-flight check:
/// either skip with a reason, or schedule with the resolved attribution fields.
#[derive(Debug, Clone, PartialEq)]
pub enum DreamScheduleDecision {
    Skip(DreamSkipReason),
    Schedule {
        /// Always `"document_threshold"`.
        trigger_reason: String,
        /// `"idle_timeout"` if `IDLE_TIMEOUT_MINUTES > 0`, else `"immediate"`.
        delay_reason: String,
        documents_since_last_dream: i64,
        document_threshold: i64,
    },
}

/// Pure port of the threshold/min-hours gate in `check_and_schedule_dream`
/// (everything except the DB count and the in-flight queue check). Deterministic
/// and unit-testable.
///
/// An unparseable `last_dream_at` is treated as Python does: log-and-proceed (the
/// min-hours gate is skipped, the schedule still fires).
pub fn evaluate_dream_schedule(
    settings: &DreamScheduleSettings,
    metadata: &DreamMetadata,
    current_explicit_count: i64,
    now: DateTime<Utc>,
) -> DreamScheduleDecision {
    if !settings.enabled {
        return DreamScheduleDecision::Skip(DreamSkipReason::Disabled);
    }

    let documents_since_last_dream = current_explicit_count - metadata.last_dream_document_count;

    if documents_since_last_dream < settings.document_threshold {
        return DreamScheduleDecision::Skip(DreamSkipReason::BelowThreshold {
            documents_since_last_dream,
            document_threshold: settings.document_threshold,
        });
    }

    let delay_reason = if settings.idle_timeout_minutes > 0 {
        "idle_timeout"
    } else {
        "immediate"
    };

    // A present-but-unparseable timestamp falls through to schedule
    // (Python catches ValueError/TypeError and proceeds).
    if let Some(last_dream_at) = &metadata.last_dream_at
        && let Ok(last_dream_time) = parse_isoformat_utc(last_dream_at)
    {
        // Python: (now - last_dream_time).total_seconds() / 3600.
        let hours_since_last_dream =
            (now - last_dream_time).num_milliseconds() as f64 / 3_600_000.0;
        if hours_since_last_dream < settings.min_hours_between_dreams as f64 {
            return DreamScheduleDecision::Skip(DreamSkipReason::MinHoursGate {
                hours_since_last_dream,
            });
        }
    }

    DreamScheduleDecision::Schedule {
        trigger_reason: "document_threshold".to_string(),
        delay_reason: delay_reason.to_string(),
        documents_since_last_dream,
        document_threshold: settings.document_threshold,
    }
}

/// Parse a `last_dream_at` string written by `record_dream_guard`
/// (`python_isoformat_utc`, always `…+00:00`). Returns the instant in UTC.
fn parse_isoformat_utc(value: &str) -> Result<DateTime<Utc>, chrono::ParseError> {
    DateTime::parse_from_rfc3339(value).map(|dt| dt.with_timezone(&Utc))
}

/// Port of `check_and_schedule_dream`: if the `(observer, observed)` collection
/// has reached the explicit-document threshold (and the min-hours gate and
/// in-flight check pass), enqueue one dream per `ENABLED_TYPES`. Returns whether
/// a dream was scheduled.
///
/// `session_name` is the session the observations were saved under — passed
/// straight through (Python's `execute_dream` re-derives it from the latest
/// explicit document at fire time; the immediate-enqueue port already has it in
/// hand from the save context).
#[allow(clippy::too_many_arguments)]
pub async fn check_and_schedule_dream(
    pool: &PgPool,
    settings: &DreamScheduleSettings,
    workspace_name: &str,
    observer: &str,
    observed: &str,
    collection_internal_metadata: &Value,
    session_name: &str,
    now: DateTime<Utc>,
) -> Result<bool, sqlx::Error> {
    if !settings.enabled {
        return Ok(false);
    }

    let metadata = DreamMetadata::from_collection_internal_metadata(collection_internal_metadata);
    let current_explicit_count =
        db::count_explicit_documents(pool, workspace_name, observer, observed).await?;

    let (trigger_reason, delay_reason, documents_since_last_dream, document_threshold) =
        match evaluate_dream_schedule(settings, &metadata, current_explicit_count, now) {
            DreamScheduleDecision::Schedule {
                trigger_reason,
                delay_reason,
                documents_since_last_dream,
                document_threshold,
            } => (
                trigger_reason,
                delay_reason,
                documents_since_last_dream,
                document_threshold,
            ),
            DreamScheduleDecision::Skip(_) => return Ok(false),
        };

    // Queue is source of truth for in-flight dreams (skip all if any enabled-type
    // key is already pending).
    let pending_keys: Vec<String> = settings
        .enabled_types
        .iter()
        .map(|dream_type| {
            format!("dream:{dream_type}:{workspace_name}:{observer}:{observed}")
        })
        .collect();
    if db::any_dream_pending(pool, &pending_keys).await? {
        return Ok(false);
    }

    let mut scheduled = false;
    for dream_type in &settings.enabled_types {
        db::enqueue_scheduled_dream(
            pool,
            workspace_name,
            observer,
            observed,
            dream_type,
            session_name,
            &trigger_reason,
            &delay_reason,
            documents_since_last_dream,
            document_threshold,
        )
        .await?;
        scheduled = true;
    }
    Ok(scheduled)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;
    use serde_json::json;

    fn now() -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 6, 23, 12, 0, 0).unwrap()
    }

    #[test]
    fn defaults_match_python() {
        let s = DreamScheduleSettings::default();
        assert!(s.enabled);
        assert_eq!(s.document_threshold, 50);
        assert_eq!(s.idle_timeout_minutes, 60);
        assert_eq!(s.min_hours_between_dreams, 8);
        assert_eq!(s.enabled_types, vec!["omni".to_string()]);
    }

    #[test]
    fn empty_env_yields_defaults() {
        let s = DreamScheduleSettings::from_pairs(Vec::<(String, String)>::new());
        assert_eq!(s, DreamScheduleSettings::default());
    }

    #[test]
    fn env_overrides_are_parsed() {
        let s = DreamScheduleSettings::from_pairs([
            ("DREAM_ENABLED", "false"),
            ("DREAM_DOCUMENT_THRESHOLD", "100"),
            ("DREAM_IDLE_TIMEOUT_MINUTES", "0"),
            ("DREAM_MIN_HOURS_BETWEEN_DREAMS", "24"),
            ("DREAM_ENABLED_TYPES", "omni, focus ,, "),
        ]);
        assert!(!s.enabled);
        assert_eq!(s.document_threshold, 100);
        assert_eq!(s.idle_timeout_minutes, 0);
        assert_eq!(s.min_hours_between_dreams, 24);
        assert_eq!(s.enabled_types, vec!["omni".to_string(), "focus".to_string()]);
    }

    #[test]
    fn unparseable_or_empty_falls_back() {
        let s = DreamScheduleSettings::from_pairs([
            ("DREAM_DOCUMENT_THRESHOLD", "not-a-number"),
            ("DREAM_ENABLED_TYPES", "   "),
        ]);
        assert_eq!(s.document_threshold, 50);
        assert_eq!(s.enabled_types, vec!["omni".to_string()]);
    }

    #[test]
    fn metadata_defaults_when_absent() {
        let meta = DreamMetadata::from_collection_internal_metadata(&json!({}));
        assert_eq!(meta.last_dream_document_count, 0);
        assert_eq!(meta.last_dream_at, None);

        let meta = DreamMetadata::from_collection_internal_metadata(&json!({
            "dream": {"last_dream_document_count": 7, "last_dream_at": "2026-06-23T00:00:00+00:00"}
        }));
        assert_eq!(meta.last_dream_document_count, 7);
        assert_eq!(
            meta.last_dream_at.as_deref(),
            Some("2026-06-23T00:00:00+00:00")
        );
    }

    #[test]
    fn disabled_skips() {
        let s = DreamScheduleSettings {
            enabled: false,
            ..Default::default()
        };
        let meta = DreamMetadata {
            last_dream_document_count: 0,
            last_dream_at: None,
        };
        assert_eq!(
            evaluate_dream_schedule(&s, &meta, 100, now()),
            DreamScheduleDecision::Skip(DreamSkipReason::Disabled)
        );
    }

    #[test]
    fn below_threshold_skips() {
        let s = DreamScheduleSettings::default();
        let meta = DreamMetadata {
            last_dream_document_count: 10,
            last_dream_at: None,
        };
        // 59 - 10 = 49 < 50.
        assert_eq!(
            evaluate_dream_schedule(&s, &meta, 59, now()),
            DreamScheduleDecision::Skip(DreamSkipReason::BelowThreshold {
                documents_since_last_dream: 49,
                document_threshold: 50,
            })
        );
    }

    #[test]
    fn at_threshold_schedules_idle_timeout() {
        let s = DreamScheduleSettings::default();
        let meta = DreamMetadata {
            last_dream_document_count: 10,
            last_dream_at: None,
        };
        // 60 - 10 = 50 >= 50.
        assert_eq!(
            evaluate_dream_schedule(&s, &meta, 60, now()),
            DreamScheduleDecision::Schedule {
                trigger_reason: "document_threshold".to_string(),
                delay_reason: "idle_timeout".to_string(),
                documents_since_last_dream: 50,
                document_threshold: 50,
            }
        );
    }

    #[test]
    fn immediate_delay_reason_when_idle_timeout_zero() {
        let s = DreamScheduleSettings {
            idle_timeout_minutes: 0,
            ..Default::default()
        };
        let meta = DreamMetadata {
            last_dream_document_count: 0,
            last_dream_at: None,
        };
        match evaluate_dream_schedule(&s, &meta, 50, now()) {
            DreamScheduleDecision::Schedule { delay_reason, .. } => {
                assert_eq!(delay_reason, "immediate")
            }
            other => panic!("expected schedule, got {other:?}"),
        }
    }

    #[test]
    fn min_hours_gate_blocks_recent_dream() {
        let s = DreamScheduleSettings::default();
        // Last dream 2h ago, min is 8h → skip.
        let meta = DreamMetadata {
            last_dream_document_count: 0,
            last_dream_at: Some("2026-06-23T10:00:00+00:00".to_string()),
        };
        match evaluate_dream_schedule(&s, &meta, 50, now()) {
            DreamScheduleDecision::Skip(DreamSkipReason::MinHoursGate {
                hours_since_last_dream,
            }) => assert!((hours_since_last_dream - 2.0).abs() < 1e-6),
            other => panic!("expected min-hours skip, got {other:?}"),
        }
    }

    #[test]
    fn old_dream_passes_min_hours_gate() {
        let s = DreamScheduleSettings::default();
        // Last dream 12h ago, min is 8h → schedule.
        let meta = DreamMetadata {
            last_dream_document_count: 0,
            last_dream_at: Some("2026-06-23T00:00:00+00:00".to_string()),
        };
        assert!(matches!(
            evaluate_dream_schedule(&s, &meta, 50, now()),
            DreamScheduleDecision::Schedule { .. }
        ));
    }

    #[test]
    fn unparseable_last_dream_at_proceeds() {
        let s = DreamScheduleSettings::default();
        let meta = DreamMetadata {
            last_dream_document_count: 0,
            last_dream_at: Some("not-a-timestamp".to_string()),
        };
        assert!(matches!(
            evaluate_dream_schedule(&s, &meta, 50, now()),
            DreamScheduleDecision::Schedule { .. }
        ));
    }
}
