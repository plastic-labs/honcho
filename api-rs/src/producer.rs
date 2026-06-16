//! Queue producer logic, ported from `src/deriver/enqueue.py`,
//! `src/utils/queue_payload.py`, `src/utils/work_unit.py`, and
//! `src/utils/config_helpers.py`.
//!
//! These are the deterministic, DB-free pieces of message-create enqueueing:
//! resolving the hierarchical `ResolvedConfiguration`, selecting observers,
//! deciding when a summary is due, building `work_unit_key`s, and building the
//! representation/summary/deletion payloads with Python-identical JSON
//! (`exclude_none`, `mode="json"` datetime formatting). The DB wiring that
//! inserts messages and queue rows is layered on top in the message-write route.
//!
//! All output here is validated by golden tests against captured Python output;
//! see the `tests` module.

use chrono::{DateTime, Utc};
use serde_json::{Map, Value, json};
use thiserror::Error;

// Python global defaults (settings) baked in. These match `get_configuration`'s
// default `config_dict` (DERIVER.ENABLED, PEER_CARD.ENABLED, SUMMARY.*,
// DREAM.ENABLED). Making them config-driven is a follow-up, like the
// custom_instructions token budget.
const DEFAULT_REASONING_ENABLED: bool = true;
const DEFAULT_PEER_CARD_USE: bool = true;
const DEFAULT_PEER_CARD_CREATE: bool = true;
const DEFAULT_SUMMARY_ENABLED: bool = true;
const DEFAULT_MESSAGES_PER_SHORT_SUMMARY: i64 = 20;
const DEFAULT_MESSAGES_PER_LONG_SUMMARY: i64 = 60;
const DEFAULT_DREAM_ENABLED: bool = true;

#[derive(Debug, Error, PartialEq, Eq)]
pub enum ProducerError {
    #[error("workspace_name and task_type are required to generate a work_unit_key")]
    MissingWorkUnitFields,
    #[error("dream_type is required for dream tasks")]
    MissingDreamType,
    #[error("deletion_type and resource_id are required for deletion tasks")]
    MissingDeletionFields,
    #[error("reconciler_type is required for reconciler tasks")]
    MissingReconcilerType,
    #[error("invalid task type: {0}")]
    InvalidTaskType(String),
}

/// The resolved per-message configuration. Field set mirrors Python's
/// `ResolvedConfiguration`; unknown override keys are dropped just as Pydantic
/// drops extras.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResolvedConfiguration {
    pub reasoning_enabled: bool,
    pub reasoning_custom_instructions: Option<String>,
    pub peer_card_use: bool,
    pub peer_card_create: bool,
    pub summary_enabled: bool,
    pub messages_per_short_summary: i64,
    pub messages_per_long_summary: i64,
    pub dream_enabled: bool,
}

impl Default for ResolvedConfiguration {
    fn default() -> Self {
        Self {
            reasoning_enabled: DEFAULT_REASONING_ENABLED,
            reasoning_custom_instructions: None,
            peer_card_use: DEFAULT_PEER_CARD_USE,
            peer_card_create: DEFAULT_PEER_CARD_CREATE,
            summary_enabled: DEFAULT_SUMMARY_ENABLED,
            messages_per_short_summary: DEFAULT_MESSAGES_PER_SHORT_SUMMARY,
            messages_per_long_summary: DEFAULT_MESSAGES_PER_LONG_SUMMARY,
            dream_enabled: DEFAULT_DREAM_ENABLED,
        }
    }
}

impl ResolvedConfiguration {
    /// Serialize to the exact JSON shape Python emits via
    /// `model_dump(mode="json", exclude_none=True)`: `custom_instructions` is
    /// omitted when `None`.
    pub fn to_payload_value(&self) -> Value {
        let mut reasoning = Map::new();
        reasoning.insert("enabled".into(), Value::Bool(self.reasoning_enabled));
        if let Some(ref instructions) = self.reasoning_custom_instructions {
            reasoning.insert(
                "custom_instructions".into(),
                Value::String(instructions.clone()),
            );
        }
        json!({
            "reasoning": Value::Object(reasoning),
            "peer_card": {"use": self.peer_card_use, "create": self.peer_card_create},
            "summary": {
                "enabled": self.summary_enabled,
                "messages_per_short_summary": self.messages_per_short_summary,
                "messages_per_long_summary": self.messages_per_long_summary,
            },
            "dream": {"enabled": self.dream_enabled},
        })
    }
}

/// Resolve configuration with Python's hierarchy: defaults → workspace → session
/// → message. Each override is a raw configuration object (as stored on the
/// workspace/session, or the message's `configuration`). Mirrors
/// `get_configuration` + `normalize_configuration_dict` + `deep_update`.
pub fn resolve_configuration(
    workspace: Option<&Value>,
    session: Option<&Value>,
    message: Option<&Value>,
) -> ResolvedConfiguration {
    let mut config = ResolvedConfiguration::default();
    for override_value in [workspace, session, message].into_iter().flatten() {
        apply_override(&mut config, override_value);
    }
    config
}

/// Apply one normalized override object onto `config`. Only known fields are
/// honored (extras dropped like Pydantic); `null` values are skipped (Python's
/// `deep_update` ignores `None`).
fn apply_override(config: &mut ResolvedConfiguration, raw: &Value) {
    let Some(object) = raw.as_object() else {
        return;
    };

    // normalize_configuration_dict: legacy deriver/skip_deriver -> reasoning.enabled,
    // but only when reasoning.enabled is not explicitly set.
    let reasoning_obj = object.get("reasoning").and_then(Value::as_object);
    let reasoning_enabled_explicit = reasoning_obj
        .and_then(|r| r.get("enabled"))
        .is_some_and(|v| !v.is_null());

    if !reasoning_enabled_explicit {
        if let Some(deriver_enabled) = object
            .get("deriver")
            .and_then(Value::as_object)
            .and_then(|d| d.get("enabled"))
            .and_then(Value::as_bool)
        {
            config.reasoning_enabled = deriver_enabled;
        }
        if object.get("skip_deriver") == Some(&Value::Bool(true)) {
            config.reasoning_enabled = false;
        }
    }

    if let Some(reasoning) = reasoning_obj {
        if let Some(enabled) = reasoning.get("enabled").and_then(Value::as_bool) {
            config.reasoning_enabled = enabled;
        }
        match reasoning.get("custom_instructions") {
            Some(Value::String(value)) => {
                config.reasoning_custom_instructions = Some(value.clone());
            }
            _ => {} // null/absent => no change (deep_update skips None)
        }
    }

    if let Some(peer_card) = object.get("peer_card").and_then(Value::as_object) {
        if let Some(use_value) = peer_card.get("use").and_then(Value::as_bool) {
            config.peer_card_use = use_value;
        }
        if let Some(create) = peer_card.get("create").and_then(Value::as_bool) {
            config.peer_card_create = create;
        }
    }

    if let Some(summary) = object.get("summary").and_then(Value::as_object) {
        if let Some(enabled) = summary.get("enabled").and_then(Value::as_bool) {
            config.summary_enabled = enabled;
        }
        if let Some(short) = summary
            .get("messages_per_short_summary")
            .and_then(Value::as_i64)
        {
            config.messages_per_short_summary = short;
        }
        if let Some(long) = summary
            .get("messages_per_long_summary")
            .and_then(Value::as_i64)
        {
            config.messages_per_long_summary = long;
        }
    }

    if let Some(dream) = object.get("dream").and_then(Value::as_object) {
        if let Some(enabled) = dream.get("enabled").and_then(Value::as_bool) {
            config.dream_enabled = enabled;
        }
    }
}

/// Format a UTC datetime exactly as Pydantic v2 `model_dump(mode="json")` does:
/// `Z` suffix, fractional seconds omitted when microseconds are zero, otherwise
/// 6-digit microseconds.
pub fn format_pydantic_datetime(value: DateTime<Utc>) -> String {
    let micros = value.timestamp_subsec_micros();
    if micros == 0 {
        value.format("%Y-%m-%dT%H:%M:%SZ").to_string()
    } else {
        format!("{}.{micros:06}Z", value.format("%Y-%m-%dT%H:%M:%S"))
    }
}

/// `observe_me` resolution for a sender (`get_effective_observe_me`): session-peer
/// config wins when it sets `observe_me`, else peer config, else default `true`.
pub fn effective_observe_me(
    peer_config: Option<&Value>,
    session_peer_config: Option<&Value>,
) -> bool {
    if let Some(value) = session_peer_config
        .and_then(Value::as_object)
        .and_then(|c| c.get("observe_me"))
        .and_then(Value::as_bool)
    {
        return value;
    }
    if let Some(value) = peer_config
        .and_then(Value::as_object)
        .and_then(|c| c.get("observe_me"))
        .and_then(Value::as_bool)
    {
        return value;
    }
    true
}

/// Whether a summary record is due at `seq` given the short/long thresholds.
pub fn summary_due(seq: i64, short: i64, long: i64) -> bool {
    (short > 0 && seq % short == 0) || (long > 0 && seq % long == 0)
}

/// Build the representation payload (`RepresentationPayload`, `exclude_none`).
pub fn representation_payload(
    session_name: &str,
    content: &str,
    observers: &[String],
    observed: &str,
    created_at: DateTime<Utc>,
    configuration: &Value,
) -> Value {
    json!({
        "task_type": "representation",
        "session_name": session_name,
        "content": content,
        "observers": observers,
        "observed": observed,
        "created_at": format_pydantic_datetime(created_at),
        "configuration": configuration,
    })
}

/// Build the summary payload (`SummaryPayload`, `exclude_none` — omit
/// `message_public_id` when absent).
pub fn summary_payload(
    session_name: &str,
    message_seq_in_session: i64,
    configuration: &Value,
    message_public_id: Option<&str>,
) -> Value {
    let mut object = Map::new();
    object.insert("task_type".into(), Value::String("summary".into()));
    object.insert("session_name".into(), Value::String(session_name.into()));
    object.insert(
        "message_seq_in_session".into(),
        Value::Number(message_seq_in_session.into()),
    );
    object.insert("configuration".into(), configuration.clone());
    if let Some(public_id) = message_public_id {
        object.insert(
            "message_public_id".into(),
            Value::String(public_id.into()),
        );
    }
    Value::Object(object)
}

/// Build the deletion payload (`DeletionPayload`).
pub fn deletion_payload(deletion_type: &str, resource_id: &str) -> Value {
    json!({
        "task_type": "deletion",
        "deletion_type": deletion_type,
        "resource_id": resource_id,
    })
}

/// Build a `work_unit_key` from a workspace and a payload object, mirroring
/// `construct_work_unit_key`. Missing `observer`/`observed`/`session_name`
/// serialize as the literal `"None"`.
pub fn construct_work_unit_key(
    workspace_name: &str,
    payload: &Value,
) -> Result<String, ProducerError> {
    let task_type = payload.get("task_type").and_then(Value::as_str);
    let (Some(task_type), false) = (task_type, workspace_name.is_empty()) else {
        return Err(ProducerError::MissingWorkUnitFields);
    };
    let task_type = if task_type.is_empty() {
        return Err(ProducerError::MissingWorkUnitFields);
    } else {
        task_type
    };

    let str_field = |key: &str| -> String {
        payload
            .get(key)
            .and_then(Value::as_str)
            .unwrap_or("None")
            .to_string()
    };

    match task_type {
        "representation" => Ok(format!(
            "representation:{workspace_name}:{}:{}",
            str_field("session_name"),
            str_field("observed"),
        )),
        "summary" => Ok(format!(
            "summary:{workspace_name}:{}:{}:{}",
            str_field("session_name"),
            str_field("observer"),
            str_field("observed"),
        )),
        "dream" => {
            let dream_type = payload.get("dream_type").and_then(Value::as_str);
            let Some(dream_type) = dream_type.filter(|value| !value.is_empty()) else {
                return Err(ProducerError::MissingDreamType);
            };
            Ok(format!(
                "dream:{dream_type}:{workspace_name}:{}:{}",
                str_field("observer"),
                str_field("observed"),
            ))
        }
        "webhook" => Ok(format!("webhook:{workspace_name}")),
        "deletion" => {
            let deletion_type = payload.get("deletion_type").and_then(Value::as_str);
            let resource_id = payload.get("resource_id").and_then(Value::as_str);
            match (
                deletion_type.filter(|v| !v.is_empty()),
                resource_id.filter(|v| !v.is_empty()),
            ) {
                (Some(deletion_type), Some(resource_id)) => Ok(format!(
                    "deletion:{workspace_name}:{deletion_type}:{resource_id}"
                )),
                _ => Err(ProducerError::MissingDeletionFields),
            }
        }
        "reconciler" => {
            let reconciler_type = payload.get("reconciler_type").and_then(Value::as_str);
            let Some(reconciler_type) = reconciler_type.filter(|value| !value.is_empty()) else {
                return Err(ProducerError::MissingReconcilerType);
            };
            Ok(format!("reconciler:{reconciler_type}"))
        }
        other => Err(ProducerError::InvalidTaskType(other.to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;
    use serde_json::json;

    fn dt(micros: u32) -> DateTime<Utc> {
        Utc.with_ymd_and_hms(2026, 6, 16, 12, 0, 0).unwrap()
            + chrono::Duration::microseconds(micros as i64)
    }

    #[test]
    fn default_configuration_matches_python() {
        let expected: Value = json!({
            "dream": {"enabled": true},
            "peer_card": {"create": true, "use": true},
            "reasoning": {"enabled": true},
            "summary": {"enabled": true, "messages_per_long_summary": 60, "messages_per_short_summary": 20}
        });
        assert_eq!(
            resolve_configuration(None, None, None).to_payload_value(),
            expected
        );
    }

    #[test]
    fn workspace_reasoning_override() {
        let ws = json!({"reasoning": {"enabled": false}});
        let value = resolve_configuration(Some(&ws), None, None).to_payload_value();
        assert_eq!(value["reasoning"], json!({"enabled": false}));
    }

    #[test]
    fn session_summary_override_merges() {
        let ws = json!({});
        let session = json!({"summary": {"messages_per_short_summary": 5}});
        let value = resolve_configuration(Some(&ws), Some(&session), None).to_payload_value();
        assert_eq!(
            value["summary"],
            json!({"enabled": true, "messages_per_long_summary": 60, "messages_per_short_summary": 5})
        );
    }

    #[test]
    fn legacy_deriver_maps_to_reasoning() {
        let ws = json!({"deriver": {"enabled": false}});
        let value = resolve_configuration(Some(&ws), None, None).to_payload_value();
        assert_eq!(value["reasoning"], json!({"enabled": false}));
    }

    #[test]
    fn custom_instructions_round_trip() {
        let ws = json!({"reasoning": {"custom_instructions": "be nice"}});
        let value = resolve_configuration(Some(&ws), None, None).to_payload_value();
        assert_eq!(
            value["reasoning"],
            json!({"custom_instructions": "be nice", "enabled": true})
        );
    }

    #[test]
    fn datetime_formatting_matches_pydantic() {
        assert_eq!(format_pydantic_datetime(dt(0)), "2026-06-16T12:00:00Z");
        assert_eq!(
            format_pydantic_datetime(dt(123456)),
            "2026-06-16T12:00:00.123456Z"
        );
        assert_eq!(
            format_pydantic_datetime(dt(120000)),
            "2026-06-16T12:00:00.120000Z"
        );
        assert_eq!(
            format_pydantic_datetime(dt(1)),
            "2026-06-16T12:00:00.000001Z"
        );
    }

    #[test]
    fn observe_me_truth_table() {
        assert!(effective_observe_me(None, None));
        assert!(!effective_observe_me(
            Some(&json!({"observe_me": false})),
            Some(&json!({}))
        ));
        assert!(effective_observe_me(
            Some(&json!({"observe_me": false})),
            Some(&json!({"observe_me": true}))
        ));
        assert!(effective_observe_me(
            Some(&json!({"observe_me": true})),
            Some(&json!({}))
        ));
        assert!(effective_observe_me(Some(&json!({})), Some(&json!({}))));
    }

    #[test]
    fn summary_due_thresholds() {
        assert!(summary_due(20, 20, 60));
        assert!(summary_due(60, 20, 60));
        assert!(!summary_due(19, 20, 60));
        assert!(summary_due(40, 20, 60));
    }

    #[test]
    fn work_unit_keys_match_python() {
        let rep = json!({"task_type": "representation", "session_name": "s1", "observed": "alice", "observers": ["alice"]});
        assert_eq!(
            construct_work_unit_key("ws1", &rep).unwrap(),
            "representation:ws1:s1:alice"
        );
        let summary = json!({"task_type": "summary", "session_name": "s1"});
        assert_eq!(
            construct_work_unit_key("ws1", &summary).unwrap(),
            "summary:ws1:s1:None:None"
        );
        let deletion = json!({"task_type": "deletion", "deletion_type": "session", "resource_id": "s1"});
        assert_eq!(
            construct_work_unit_key("ws1", &deletion).unwrap(),
            "deletion:ws1:session:s1"
        );
        assert_eq!(
            construct_work_unit_key("", &rep).unwrap_err(),
            ProducerError::MissingWorkUnitFields
        );
    }

    #[test]
    fn representation_payload_matches_python() {
        let config = resolve_configuration(None, None, None).to_payload_value();
        let observers = vec!["alice".to_string()];
        let payload = representation_payload("s1", "hello world", &observers, "alice", dt(0), &config);
        let expected: Value = json!({
            "configuration": {
                "dream": {"enabled": true},
                "peer_card": {"create": true, "use": true},
                "reasoning": {"enabled": true},
                "summary": {"enabled": true, "messages_per_long_summary": 60, "messages_per_short_summary": 20}
            },
            "content": "hello world",
            "created_at": "2026-06-16T12:00:00Z",
            "observed": "alice",
            "observers": ["alice"],
            "session_name": "s1",
            "task_type": "representation"
        });
        assert_eq!(payload, expected);
    }

    #[test]
    fn summary_payload_matches_python() {
        let config = resolve_configuration(None, None, None).to_payload_value();
        let payload = summary_payload("s1", 20, &config, Some("pub_abc"));
        let expected: Value = json!({
            "configuration": {
                "dream": {"enabled": true},
                "peer_card": {"create": true, "use": true},
                "reasoning": {"enabled": true},
                "summary": {"enabled": true, "messages_per_long_summary": 60, "messages_per_short_summary": 20}
            },
            "message_public_id": "pub_abc",
            "message_seq_in_session": 20,
            "session_name": "s1",
            "task_type": "summary"
        });
        assert_eq!(payload, expected);
        // message_public_id omitted when absent (exclude_none).
        let without = summary_payload("s1", 20, &config, None);
        assert!(without.get("message_public_id").is_none());
    }

    #[test]
    fn deletion_payload_matches_python() {
        assert_eq!(
            deletion_payload("session", "s1"),
            json!({"deletion_type": "session", "resource_id": "s1", "task_type": "deletion"})
        );
    }
}
