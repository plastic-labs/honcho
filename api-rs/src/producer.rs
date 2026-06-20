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
use std::collections::BTreeMap;
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
    #[error("invalid work_unit_key format: {0}")]
    InvalidWorkUnitKey(String),
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

    /// Parse a stored configuration object back into a `ResolvedConfiguration`,
    /// mirroring `ResolvedConfiguration.model_validate`. All sub-fields are
    /// required (as in the Python model); `reasoning.custom_instructions` is the
    /// only optional field. Applies the v3.0.0 `deriver` -> `reasoning`
    /// migration (`migrate_deriver_to_reasoning`) and ignores extra keys.
    ///
    /// The Python model also re-runs `_validate_custom_instructions_budget`,
    /// but that validator only raises or returns the value unchanged — it never
    /// transforms it — so it is a no-op for already-validated stored payloads
    /// and is not re-applied here.
    pub fn from_payload_value(raw: &Value) -> Result<Self, String> {
        let object = raw
            .as_object()
            .ok_or_else(|| "configuration must be an object".to_string())?;

        // migrate_deriver_to_reasoning: `deriver` was renamed to `reasoning`;
        // honor the legacy key only when `reasoning` is absent.
        let reasoning_src = if object.contains_key("deriver") && !object.contains_key("reasoning") {
            object.get("deriver")
        } else {
            object.get("reasoning")
        };
        let reasoning = reasoning_src
            .and_then(Value::as_object)
            .ok_or_else(|| "configuration.reasoning is required".to_string())?;
        let reasoning_enabled = reasoning
            .get("enabled")
            .and_then(Value::as_bool)
            .ok_or_else(|| "configuration.reasoning.enabled is required".to_string())?;
        let reasoning_custom_instructions = match reasoning.get("custom_instructions") {
            Some(Value::String(value)) => Some(value.clone()),
            None | Some(Value::Null) => None,
            Some(_) => {
                return Err("configuration.reasoning.custom_instructions must be a string".into());
            }
        };

        let peer_card = object
            .get("peer_card")
            .and_then(Value::as_object)
            .ok_or_else(|| "configuration.peer_card is required".to_string())?;
        let peer_card_use = peer_card
            .get("use")
            .and_then(Value::as_bool)
            .ok_or_else(|| "configuration.peer_card.use is required".to_string())?;
        let peer_card_create = peer_card
            .get("create")
            .and_then(Value::as_bool)
            .ok_or_else(|| "configuration.peer_card.create is required".to_string())?;

        let summary = object
            .get("summary")
            .and_then(Value::as_object)
            .ok_or_else(|| "configuration.summary is required".to_string())?;
        let summary_enabled = summary
            .get("enabled")
            .and_then(Value::as_bool)
            .ok_or_else(|| "configuration.summary.enabled is required".to_string())?;
        let messages_per_short_summary = summary
            .get("messages_per_short_summary")
            .and_then(Value::as_i64)
            .ok_or_else(|| "configuration.summary.messages_per_short_summary is required".to_string())?;
        let messages_per_long_summary = summary
            .get("messages_per_long_summary")
            .and_then(Value::as_i64)
            .ok_or_else(|| "configuration.summary.messages_per_long_summary is required".to_string())?;

        let dream = object
            .get("dream")
            .and_then(Value::as_object)
            .ok_or_else(|| "configuration.dream is required".to_string())?;
        let dream_enabled = dream
            .get("enabled")
            .and_then(Value::as_bool)
            .ok_or_else(|| "configuration.dream.enabled is required".to_string())?;

        Ok(Self {
            reasoning_enabled,
            reasoning_custom_instructions,
            peer_card_use,
            peer_card_create,
            summary_enabled,
            messages_per_short_summary,
            messages_per_long_summary,
            dream_enabled,
        })
    }
}

/// Keep only the initial homogeneous-configuration prefix of a representation
/// batch, mirroring `QueueManager._resolve_batch_configuration`.
///
/// `payloads` are the queue items' payload objects, in queue order. Returns the
/// number of leading items that share the first item's `configuration` (a
/// missing or `null` key resolves to `None`, matching `dict.get`), plus that
/// resolved configuration. An empty batch yields `(0, None)`. A malformed
/// configuration surfaces as `Err`, just as `model_validate` would raise.
pub fn resolve_batch_configuration_prefix<'a>(
    payloads: impl IntoIterator<Item = &'a Value>,
) -> Result<(usize, Option<ResolvedConfiguration>), String> {
    let parse = |payload: &Value| -> Result<Option<ResolvedConfiguration>, String> {
        match payload.get("configuration") {
            None | Some(Value::Null) => Ok(None),
            Some(raw) => Ok(Some(ResolvedConfiguration::from_payload_value(raw)?)),
        }
    };

    let mut iter = payloads.into_iter();
    let Some(first) = iter.next() else {
        return Ok((0, None));
    };
    let resolved = parse(first)?;

    let mut count = 1; // the first item is always part of the prefix
    for payload in iter {
        if parse(payload)? != resolved {
            break;
        }
        count += 1;
    }
    Ok((count, resolved))
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
        object.insert("message_public_id".into(), Value::String(public_id.into()));
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

/// Parsed components of a `work_unit_key`, mirroring Python `ParsedWorkUnit`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedWorkUnit {
    pub task_type: String,
    pub workspace_name: Option<String>,
    pub session_name: Option<String>,
    pub observer: Option<String>,
    pub observed: Option<String>,
    pub dream_type: Option<String>,
}

/// Parse a `work_unit_key` back into its components, mirroring
/// `parse_work_unit_key`. The colon-split is positional and per-task-type, with
/// strict segment-count checks (an unexpected count is an error, matching the
/// Python `ValueError`). `representation` accepts both the 4-segment current
/// form and the 5-segment legacy form (which carries an `observer`).
pub fn parse_work_unit_key(work_unit_key: &str) -> Result<ParsedWorkUnit, ProducerError> {
    let parts: Vec<&str> = work_unit_key.split(':').collect();
    let task_type = parts[0];

    let invalid = || ProducerError::InvalidWorkUnitKey(work_unit_key.to_string());
    let owned = |value: &str| Some(value.to_string());

    let base = |task_type: &str| ParsedWorkUnit {
        task_type: task_type.to_string(),
        workspace_name: None,
        session_name: None,
        observer: None,
        observed: None,
        dream_type: None,
    };

    match task_type {
        "representation" => match parts.len() {
            // New format: representation:{workspace}:{session}:{observed}
            4 => Ok(ParsedWorkUnit {
                workspace_name: owned(parts[1]),
                session_name: owned(parts[2]),
                observed: owned(parts[3]),
                ..base(task_type)
            }),
            // Legacy: representation:{workspace}:{session}:{observer}:{observed}
            5 => Ok(ParsedWorkUnit {
                workspace_name: owned(parts[1]),
                session_name: owned(parts[2]),
                observer: owned(parts[3]),
                observed: owned(parts[4]),
                ..base(task_type)
            }),
            _ => Err(invalid()),
        },
        "summary" => {
            if parts.len() != 5 {
                return Err(invalid());
            }
            Ok(ParsedWorkUnit {
                workspace_name: owned(parts[1]),
                session_name: owned(parts[2]),
                observer: owned(parts[3]),
                observed: owned(parts[4]),
                ..base(task_type)
            })
        }
        "dream" => {
            if parts.len() != 5 {
                return Err(invalid());
            }
            Ok(ParsedWorkUnit {
                workspace_name: owned(parts[2]),
                observer: owned(parts[3]),
                observed: owned(parts[4]),
                dream_type: owned(parts[1]),
                ..base(task_type)
            })
        }
        "webhook" => {
            if parts.len() != 2 {
                return Err(invalid());
            }
            Ok(ParsedWorkUnit {
                workspace_name: owned(parts[1]),
                ..base(task_type)
            })
        }
        "deletion" => {
            if parts.len() != 4 {
                return Err(invalid());
            }
            Ok(ParsedWorkUnit {
                workspace_name: owned(parts[1]),
                ..base(task_type)
            })
        }
        "reconciler" => {
            if parts.len() != 2 {
                return Err(invalid());
            }
            Ok(base(task_type))
        }
        other => Err(ProducerError::InvalidTaskType(other.to_string())),
    }
}

/// A peer's configuration row as returned by the observer-selection query
/// (`get_session_peer_configuration`). `peer_configuration` and
/// `session_peer_configuration` are the raw JSONB objects (empty object when the
/// column held `{}`); `is_active` is `left_at IS NULL`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PeerConfigEntry {
    pub peer_configuration: Value,
    pub session_peer_configuration: Value,
    pub is_active: bool,
}

/// The per-message fields needed to build queue records, mirroring the dict
/// elements `generate_queue_records` reads out of each enqueue payload.
#[derive(Debug, Clone)]
pub struct MessageForEnqueue<'a> {
    pub workspace_name: &'a str,
    pub session_name: &'a str,
    pub message_id: i64,
    pub message_public_id: &'a str,
    pub content: &'a str,
    pub peer_name: &'a str,
    pub created_at: DateTime<Utc>,
    pub seq_in_session: i64,
}

/// A fully-built queue row, mirroring the dict returned by
/// `create_representation_record` / `create_summary_record`: the payload plus
/// the dedicated `queue` columns (`work_unit_key`, `session_id`, `task_type`,
/// `workspace_name`, `message_id`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QueueRecord {
    pub work_unit_key: String,
    pub payload: Value,
    pub session_id: String,
    pub task_type: String,
    pub workspace_name: String,
    pub message_id: i64,
}

/// Generate the queue records for a single message, porting
/// `deriver.enqueue.generate_queue_records`.
///
/// `peers_with_configuration` maps every peer ever in the session (sorted by
/// peer name for deterministic observer ordering) to its peer-level and
/// session-level configuration plus active flag. A summary record is appended
/// when summaries are enabled and the sequence number trips a threshold; a
/// single representation record (with all observers) is appended when reasoning
/// is enabled and the sender should be observed.
pub fn generate_queue_records(
    message: &MessageForEnqueue<'_>,
    peers_with_configuration: &BTreeMap<String, PeerConfigEntry>,
    session_id: &str,
    conf: &ResolvedConfiguration,
) -> Result<Vec<QueueRecord>, ProducerError> {
    let observed = message.peer_name;
    let configuration = conf.to_payload_value();
    let mut records: Vec<QueueRecord> = Vec::new();

    if conf.summary_enabled
        && summary_due(
            message.seq_in_session,
            conf.messages_per_short_summary,
            conf.messages_per_long_summary,
        )
    {
        let payload = summary_payload(
            message.session_name,
            message.seq_in_session,
            &configuration,
            Some(message.message_public_id),
        );
        records.push(build_record(message, session_id, "summary", payload)?);
    }

    // Determine whether the sender should be observed. A sender absent from the
    // map (left the session after sending) defaults to observe_me = true.
    let sender_entry = peers_with_configuration.get(observed);
    let should_observe = effective_observe_me(
        sender_entry.map(|entry| &entry.peer_configuration),
        sender_entry.map(|entry| &entry.session_peer_configuration),
    );

    if !conf.reasoning_enabled {
        return Ok(records);
    }

    let mut observers: Vec<String> = Vec::new();
    if should_observe {
        // Self-observation: the sender observes themselves.
        observers.push(observed.to_string());

        for (peer_name, entry) in peers_with_configuration {
            if peer_name == observed {
                continue;
            }
            // Skip peers who have left the session.
            if !entry.is_active {
                continue;
            }
            let observe_others = entry
                .session_peer_configuration
                .as_object()
                .and_then(|config| config.get("observe_others"))
                .and_then(Value::as_bool)
                .unwrap_or(false);
            if !observe_others {
                continue;
            }
            observers.push(peer_name.clone());
        }
    }

    if !observers.is_empty() {
        let payload = representation_payload(
            message.session_name,
            message.content,
            &observers,
            observed,
            message.created_at,
            &configuration,
        );
        records.push(build_record(
            message,
            session_id,
            "representation",
            payload,
        )?);
    }

    Ok(records)
}

fn build_record(
    message: &MessageForEnqueue<'_>,
    session_id: &str,
    task_type: &str,
    payload: Value,
) -> Result<QueueRecord, ProducerError> {
    Ok(QueueRecord {
        work_unit_key: construct_work_unit_key(message.workspace_name, &payload)?,
        payload,
        session_id: session_id.to_string(),
        task_type: task_type.to_string(),
        workspace_name: message.workspace_name.to_string(),
        message_id: message.message_id,
    })
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
        let deletion =
            json!({"task_type": "deletion", "deletion_type": "session", "resource_id": "s1"});
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
        let payload =
            representation_payload("s1", "hello world", &observers, "alice", dt(0), &config);
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

    fn entry(peer: Value, session: Value, is_active: bool) -> PeerConfigEntry {
        PeerConfigEntry {
            peer_configuration: peer,
            session_peer_configuration: session,
            is_active,
        }
    }

    fn message<'a>(peer_name: &'a str, seq: i64) -> MessageForEnqueue<'a> {
        MessageForEnqueue {
            workspace_name: "ws1",
            session_name: "s1",
            message_id: 42,
            message_public_id: "pub_42",
            content: "hello",
            peer_name,
            created_at: dt(0),
            seq_in_session: seq,
        }
    }

    #[test]
    fn generate_records_self_observation_only() {
        let conf = ResolvedConfiguration::default();
        let mut peers = BTreeMap::new();
        peers.insert("alice".to_string(), entry(json!({}), json!({}), true));
        let records =
            generate_queue_records(&message("alice", 5), &peers, "sess_internal", &conf).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].task_type, "representation");
        assert_eq!(records[0].work_unit_key, "representation:ws1:s1:alice");
        assert_eq!(records[0].payload["observers"], json!(["alice"]));
        assert_eq!(records[0].session_id, "sess_internal");
        assert_eq!(records[0].message_id, 42);
    }

    #[test]
    fn generate_records_includes_observe_others_peers_sorted() {
        let conf = ResolvedConfiguration::default();
        let mut peers = BTreeMap::new();
        peers.insert("alice".to_string(), entry(json!({}), json!({}), true));
        peers.insert(
            "carol".to_string(),
            entry(json!({}), json!({"observe_others": true}), true),
        );
        peers.insert(
            "bob".to_string(),
            entry(json!({}), json!({"observe_others": true}), true),
        );
        // A peer who left the session is skipped even with observe_others.
        peers.insert(
            "dave".to_string(),
            entry(json!({}), json!({"observe_others": true}), false),
        );
        let records =
            generate_queue_records(&message("alice", 5), &peers, "sess_internal", &conf).unwrap();
        let rep = records
            .iter()
            .find(|r| r.task_type == "representation")
            .unwrap();
        assert_eq!(rep.payload["observers"], json!(["alice", "bob", "carol"]));
    }

    #[test]
    fn generate_records_skips_representation_when_observe_me_false() {
        let conf = ResolvedConfiguration::default();
        let mut peers = BTreeMap::new();
        peers.insert(
            "alice".to_string(),
            entry(json!({}), json!({"observe_me": false}), true),
        );
        let records =
            generate_queue_records(&message("alice", 5), &peers, "sess_internal", &conf).unwrap();
        assert!(records.is_empty());
    }

    #[test]
    fn generate_records_summary_and_representation_on_threshold() {
        let conf = ResolvedConfiguration::default();
        let mut peers = BTreeMap::new();
        peers.insert("alice".to_string(), entry(json!({}), json!({}), true));
        let records =
            generate_queue_records(&message("alice", 20), &peers, "sess_internal", &conf).unwrap();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].task_type, "summary");
        assert_eq!(records[0].payload["message_seq_in_session"], json!(20));
        assert_eq!(records[0].payload["message_public_id"], json!("pub_42"));
        assert_eq!(records[1].task_type, "representation");
    }

    #[test]
    fn generate_records_summary_only_when_reasoning_disabled() {
        let conf = ResolvedConfiguration {
            reasoning_enabled: false,
            ..ResolvedConfiguration::default()
        };
        let mut peers = BTreeMap::new();
        peers.insert("alice".to_string(), entry(json!({}), json!({}), true));
        let records =
            generate_queue_records(&message("alice", 20), &peers, "sess_internal", &conf).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].task_type, "summary");
    }

    #[test]
    fn generate_records_absent_sender_defaults_to_observed() {
        let conf = ResolvedConfiguration::default();
        // Sender left the session entirely; map has no entry for them.
        let peers = BTreeMap::new();
        let records =
            generate_queue_records(&message("ghost", 5), &peers, "sess_internal", &conf).unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].payload["observers"], json!(["ghost"]));
    }

    // --- ResolvedConfiguration::from_payload_value / batch-config prefix ---

    // --- parse_work_unit_key ---

    #[test]
    fn parse_representation_new_format() {
        let parsed = parse_work_unit_key("representation:ws1:sess1:bob").unwrap();
        assert_eq!(
            parsed,
            ParsedWorkUnit {
                task_type: "representation".into(),
                workspace_name: Some("ws1".into()),
                session_name: Some("sess1".into()),
                observer: None,
                observed: Some("bob".into()),
                dream_type: None,
            }
        );
    }

    #[test]
    fn parse_representation_legacy_format() {
        let parsed = parse_work_unit_key("representation:ws1:sess1:alice:bob").unwrap();
        assert_eq!(parsed.observer, Some("alice".into()));
        assert_eq!(parsed.observed, Some("bob".into()));
    }

    #[test]
    fn parse_summary_dream_webhook_deletion_reconciler() {
        let summary = parse_work_unit_key("summary:ws1:sess1:alice:bob").unwrap();
        assert_eq!(summary.session_name, Some("sess1".into()));
        assert_eq!(summary.observer, Some("alice".into()));

        let dream = parse_work_unit_key("dream:consolidate:ws1:alice:bob").unwrap();
        assert_eq!(dream.dream_type, Some("consolidate".into()));
        assert_eq!(dream.workspace_name, Some("ws1".into()));
        assert_eq!(dream.observer, Some("alice".into()));
        assert_eq!(dream.observed, Some("bob".into()));
        assert_eq!(dream.session_name, None);

        let webhook = parse_work_unit_key("webhook:ws1").unwrap();
        assert_eq!(webhook.workspace_name, Some("ws1".into()));

        let deletion = parse_work_unit_key("deletion:ws1:peer:p1").unwrap();
        assert_eq!(deletion.workspace_name, Some("ws1".into()));
        assert_eq!(deletion.session_name, None);

        let reconciler = parse_work_unit_key("reconciler:sync_vectors").unwrap();
        assert_eq!(reconciler.task_type, "reconciler");
        assert_eq!(reconciler.workspace_name, None);
    }

    #[test]
    fn parse_rejects_bad_segment_counts_and_types() {
        assert!(parse_work_unit_key("representation:ws1:sess1").is_err());
        assert!(parse_work_unit_key("summary:ws1:sess1:bob").is_err());
        assert!(parse_work_unit_key("dream:ws1:bob").is_err());
        assert!(parse_work_unit_key("webhook:ws1:extra").is_err());
        assert!(parse_work_unit_key("deletion:ws1:peer").is_err());
        assert!(parse_work_unit_key("reconciler").is_err());
        assert!(parse_work_unit_key("bogus:ws1").is_err());
    }

    #[test]
    fn construct_then_parse_round_trips() {
        let rep = json!({"task_type": "representation", "session_name": "s", "observed": "bob"});
        let key = construct_work_unit_key("ws1", &rep).unwrap();
        let parsed = parse_work_unit_key(&key).unwrap();
        assert_eq!(parsed.task_type, "representation");
        assert_eq!(parsed.workspace_name.as_deref(), Some("ws1"));
        assert_eq!(parsed.session_name.as_deref(), Some("s"));
        assert_eq!(parsed.observed.as_deref(), Some("bob"));

        let dream = json!({"task_type": "dream", "dream_type": "consolidate", "observer": "a", "observed": "b"});
        let key = construct_work_unit_key("ws1", &dream).unwrap();
        let parsed = parse_work_unit_key(&key).unwrap();
        assert_eq!(parsed.dream_type.as_deref(), Some("consolidate"));
        assert_eq!(parsed.workspace_name.as_deref(), Some("ws1"));
    }

    #[test]
    fn from_payload_value_round_trips() {
        let config = ResolvedConfiguration {
            reasoning_custom_instructions: Some("be terse".to_string()),
            peer_card_use: false,
            messages_per_short_summary: 7,
            ..ResolvedConfiguration::default()
        };
        let payload = config.to_payload_value();
        assert_eq!(
            ResolvedConfiguration::from_payload_value(&payload).unwrap(),
            config
        );
    }

    #[test]
    fn from_payload_value_default_round_trips_without_custom_instructions() {
        let config = ResolvedConfiguration::default();
        let payload = config.to_payload_value();
        // custom_instructions omitted by exclude_none; parsing yields None.
        let parsed = ResolvedConfiguration::from_payload_value(&payload).unwrap();
        assert_eq!(parsed, config);
        assert_eq!(parsed.reasoning_custom_instructions, None);
    }

    #[test]
    fn from_payload_value_applies_deriver_migration() {
        // Legacy payload using `deriver` instead of `reasoning`.
        let payload = json!({
            "deriver": {"enabled": false},
            "peer_card": {"use": true, "create": true},
            "summary": {"enabled": true, "messages_per_short_summary": 20, "messages_per_long_summary": 60},
            "dream": {"enabled": true},
        });
        let parsed = ResolvedConfiguration::from_payload_value(&payload).unwrap();
        assert!(!parsed.reasoning_enabled);
    }

    #[test]
    fn from_payload_value_prefers_reasoning_over_legacy_deriver() {
        let payload = json!({
            "reasoning": {"enabled": true},
            "deriver": {"enabled": false},
            "peer_card": {"use": true, "create": true},
            "summary": {"enabled": true, "messages_per_short_summary": 20, "messages_per_long_summary": 60},
            "dream": {"enabled": true},
        });
        let parsed = ResolvedConfiguration::from_payload_value(&payload).unwrap();
        assert!(parsed.reasoning_enabled);
    }

    #[test]
    fn from_payload_value_errors_on_missing_required_field() {
        let payload = json!({
            "reasoning": {"enabled": true},
            "peer_card": {"use": true, "create": true},
            "summary": {"enabled": true, "messages_per_short_summary": 20, "messages_per_long_summary": 60},
            // dream missing
        });
        assert!(ResolvedConfiguration::from_payload_value(&payload).is_err());
    }

    #[test]
    fn from_payload_value_ignores_extra_keys() {
        let payload = json!({
            "reasoning": {"enabled": true, "unknown": 1},
            "peer_card": {"use": true, "create": true},
            "summary": {"enabled": true, "messages_per_short_summary": 20, "messages_per_long_summary": 60},
            "dream": {"enabled": true},
            "extra_top_level": "ignored",
        });
        assert!(ResolvedConfiguration::from_payload_value(&payload).is_ok());
    }

    fn payload_with_config(config: Option<&ResolvedConfiguration>) -> Value {
        match config {
            Some(c) => json!({"configuration": c.to_payload_value()}),
            None => json!({"observers": ["a"]}), // no configuration key
        }
    }

    #[test]
    fn batch_prefix_empty_is_zero_none() {
        let payloads: [Value; 0] = [];
        assert_eq!(
            resolve_batch_configuration_prefix(payloads.iter()).unwrap(),
            (0, None)
        );
    }

    #[test]
    fn batch_prefix_all_none_config() {
        let payloads = [
            payload_with_config(None),
            payload_with_config(None),
            payload_with_config(None),
        ];
        assert_eq!(
            resolve_batch_configuration_prefix(payloads.iter()).unwrap(),
            (3, None)
        );
    }

    #[test]
    fn batch_prefix_homogeneous() {
        let config = ResolvedConfiguration::default();
        let payloads = [
            payload_with_config(Some(&config)),
            payload_with_config(Some(&config)),
        ];
        let (count, resolved) = resolve_batch_configuration_prefix(payloads.iter()).unwrap();
        assert_eq!(count, 2);
        assert_eq!(resolved, Some(config));
    }

    #[test]
    fn batch_prefix_breaks_on_config_change() {
        let config_a = ResolvedConfiguration::default();
        let config_b = ResolvedConfiguration {
            summary_enabled: false,
            ..ResolvedConfiguration::default()
        };
        let payloads = [
            payload_with_config(Some(&config_a)),
            payload_with_config(Some(&config_a)),
            payload_with_config(Some(&config_b)),
            payload_with_config(Some(&config_a)),
        ];
        let (count, resolved) = resolve_batch_configuration_prefix(payloads.iter()).unwrap();
        assert_eq!(count, 2);
        assert_eq!(resolved, Some(config_a));
    }

    #[test]
    fn batch_prefix_none_then_some_breaks() {
        let config = ResolvedConfiguration::default();
        let payloads = [
            payload_with_config(None),
            payload_with_config(Some(&config)),
        ];
        let (count, resolved) = resolve_batch_configuration_prefix(payloads.iter()).unwrap();
        assert_eq!(count, 1);
        assert_eq!(resolved, None);
    }
}
