//! Consumer-side queue-payload parsing, ported from `src/utils/queue_payload.py`
//! (the `*Payload` pydantic models) + the enums in `src/schemas`.
//!
//! These validate the `payload` JSONB the deriver pops off the queue before
//! dispatching. All payloads derive from `BasePayload` with `extra="forbid"`, so
//! an unexpected key is a validation error (mirrored here). Required fields,
//! optional defaults (`None`), and the `Literal` task-type/enum constraints
//! match the Python models. The producer-side builders already live in
//! `producer.rs`; this is the inverse used by `consumer.process_item`.

use serde_json::{Map, Value};

use crate::producer::ResolvedConfiguration;

/// `ReconcilerType` (`src/schemas/internal.py`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReconcilerType {
    SyncVectors,
    CleanupQueue,
}

impl ReconcilerType {
    fn parse(value: &str) -> Option<Self> {
        match value {
            "sync_vectors" => Some(Self::SyncVectors),
            "cleanup_queue" => Some(Self::CleanupQueue),
            _ => None,
        }
    }
}

/// `DreamType` (`src/schemas/configuration.py`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DreamType {
    Omni,
}

impl DreamType {
    fn parse(value: &str) -> Option<Self> {
        match value {
            "omni" => Some(Self::Omni),
            _ => None,
        }
    }

    /// The wire string (`DreamType.value`, threaded onto `DreamRunEvent.dream_type`).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Omni => "omni",
        }
    }
}

/// `DeletionPayload.deletion_type` (`Literal["session","observation","workspace"]`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeletionType {
    Session,
    Observation,
    Workspace,
}

impl DeletionType {
    fn parse(value: &str) -> Option<Self> {
        match value {
            "session" => Some(Self::Session),
            "observation" => Some(Self::Observation),
            "workspace" => Some(Self::Workspace),
            _ => None,
        }
    }

    /// The wire string for this deletion type (the value emitted on the
    /// `DeletionCompletedEvent.deletion_type` field).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Session => "session",
            Self::Observation => "observation",
            Self::Workspace => "workspace",
        }
    }
}

/// `SummaryPayload`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SummaryPayload {
    pub session_name: String,
    pub message_seq_in_session: i64,
    pub configuration: ResolvedConfiguration,
    pub message_public_id: Option<String>,
}

/// `WebhookPayload`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WebhookPayload {
    pub event_type: String,
    pub data: Value,
}

/// `DreamPayload`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DreamPayload {
    pub dream_type: DreamType,
    pub observer: String,
    pub observed: String,
    pub session_name: Option<String>,
    pub trigger_reason: Option<String>,
    pub delay_reason: Option<String>,
    pub documents_since_last_dream_at_schedule: Option<i64>,
    pub document_threshold: Option<i64>,
}

/// `DeletionPayload`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeletionPayload {
    pub deletion_type: DeletionType,
    pub resource_id: String,
}

/// `ReconcilerPayload`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReconcilerPayload {
    pub reconciler_type: ReconcilerType,
}

// --- shared parsing helpers (pydantic-faithful) ---

fn as_object(value: &Value) -> Result<&Map<String, Value>, String> {
    value
        .as_object()
        .ok_or_else(|| "payload must be an object".to_string())
}

/// `extra="forbid"`: every present key must be a declared field.
fn reject_unknown_keys(object: &Map<String, Value>, allowed: &[&str]) -> Result<(), String> {
    for key in object.keys() {
        if !allowed.contains(&key.as_str()) {
            return Err(format!("unexpected field: {key}"));
        }
    }
    Ok(())
}

/// The `Literal["<task_type>"]` field: if present it must equal `expected`;
/// absent is fine (the default applies).
fn check_task_type(object: &Map<String, Value>, expected: &str) -> Result<(), String> {
    match object.get("task_type") {
        None | Some(Value::Null) => Ok(()),
        Some(Value::String(value)) if value == expected => Ok(()),
        Some(_) => Err(format!("task_type must be \"{expected}\"")),
    }
}

fn required_str(object: &Map<String, Value>, key: &str) -> Result<String, String> {
    match object.get(key) {
        Some(Value::String(value)) => Ok(value.clone()),
        _ => Err(format!("{key} is required and must be a string")),
    }
}

/// Optional string: absent/null -> `None`; present non-string is an error.
fn optional_str(object: &Map<String, Value>, key: &str) -> Result<Option<String>, String> {
    match object.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) => Ok(Some(value.clone())),
        Some(_) => Err(format!("{key} must be a string")),
    }
}

fn required_i64(object: &Map<String, Value>, key: &str) -> Result<i64, String> {
    object
        .get(key)
        .and_then(Value::as_i64)
        .ok_or_else(|| format!("{key} is required and must be an integer"))
}

/// Optional integer: absent/null -> `None`; present non-integer is an error.
fn optional_i64(object: &Map<String, Value>, key: &str) -> Result<Option<i64>, String> {
    match object.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(value) => value
            .as_i64()
            .map(Some)
            .ok_or_else(|| format!("{key} must be an integer")),
    }
}

impl SummaryPayload {
    pub fn from_value(value: &Value) -> Result<Self, String> {
        let object = as_object(value)?;
        reject_unknown_keys(
            object,
            &[
                "task_type",
                "session_name",
                "message_seq_in_session",
                "configuration",
                "message_public_id",
            ],
        )?;
        check_task_type(object, "summary")?;
        let configuration = object
            .get("configuration")
            .ok_or_else(|| "configuration is required".to_string())
            .and_then(ResolvedConfiguration::from_payload_value)?;
        Ok(Self {
            session_name: required_str(object, "session_name")?,
            message_seq_in_session: required_i64(object, "message_seq_in_session")?,
            configuration,
            message_public_id: optional_str(object, "message_public_id")?,
        })
    }
}

impl WebhookPayload {
    pub fn from_value(value: &Value) -> Result<Self, String> {
        let object = as_object(value)?;
        reject_unknown_keys(object, &["task_type", "event_type", "data"])?;
        check_task_type(object, "webhook")?;
        let data = match object.get("data") {
            Some(value @ Value::Object(_)) => value.clone(),
            _ => return Err("data is required and must be an object".to_string()),
        };
        Ok(Self {
            event_type: required_str(object, "event_type")?,
            data,
        })
    }
}

impl DreamPayload {
    pub fn from_value(value: &Value) -> Result<Self, String> {
        let object = as_object(value)?;
        reject_unknown_keys(
            object,
            &[
                "task_type",
                "dream_type",
                "observer",
                "observed",
                "session_name",
                "trigger_reason",
                "delay_reason",
                "documents_since_last_dream_at_schedule",
                "document_threshold",
            ],
        )?;
        check_task_type(object, "dream")?;
        let dream_type = required_str(object, "dream_type")
            .and_then(|raw| DreamType::parse(&raw).ok_or_else(|| format!("invalid dream_type: {raw}")))?;
        Ok(Self {
            dream_type,
            observer: required_str(object, "observer")?,
            observed: required_str(object, "observed")?,
            session_name: optional_str(object, "session_name")?,
            trigger_reason: optional_str(object, "trigger_reason")?,
            delay_reason: optional_str(object, "delay_reason")?,
            documents_since_last_dream_at_schedule: optional_i64(
                object,
                "documents_since_last_dream_at_schedule",
            )?,
            document_threshold: optional_i64(object, "document_threshold")?,
        })
    }
}

impl DeletionPayload {
    pub fn from_value(value: &Value) -> Result<Self, String> {
        let object = as_object(value)?;
        reject_unknown_keys(object, &["task_type", "deletion_type", "resource_id"])?;
        check_task_type(object, "deletion")?;
        let deletion_type = required_str(object, "deletion_type").and_then(|raw| {
            DeletionType::parse(&raw).ok_or_else(|| format!("invalid deletion_type: {raw}"))
        })?;
        Ok(Self {
            deletion_type,
            resource_id: required_str(object, "resource_id")?,
        })
    }
}

impl ReconcilerPayload {
    pub fn from_value(value: &Value) -> Result<Self, String> {
        let object = as_object(value)?;
        reject_unknown_keys(object, &["task_type", "reconciler_type"])?;
        check_task_type(object, "reconciler")?;
        let reconciler_type = required_str(object, "reconciler_type").and_then(|raw| {
            ReconcilerType::parse(&raw).ok_or_else(|| format!("invalid reconciler_type: {raw}"))
        })?;
        Ok(Self { reconciler_type })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn config_value() -> Value {
        ResolvedConfiguration::default().to_payload_value()
    }

    #[test]
    fn summary_payload_parses_and_defaults_public_id() {
        let value = json!({
            "task_type": "summary",
            "session_name": "sess",
            "message_seq_in_session": 7,
            "configuration": config_value(),
        });
        let parsed = SummaryPayload::from_value(&value).unwrap();
        assert_eq!(parsed.session_name, "sess");
        assert_eq!(parsed.message_seq_in_session, 7);
        assert_eq!(parsed.message_public_id, None);
        assert_eq!(parsed.configuration, ResolvedConfiguration::default());
    }

    #[test]
    fn summary_payload_accepts_public_id() {
        let value = json!({
            "session_name": "sess",
            "message_seq_in_session": 1,
            "configuration": config_value(),
            "message_public_id": "pub123",
        });
        let parsed = SummaryPayload::from_value(&value).unwrap();
        assert_eq!(parsed.message_public_id.as_deref(), Some("pub123"));
    }

    #[test]
    fn extra_keys_are_forbidden() {
        let value = json!({
            "session_name": "sess",
            "message_seq_in_session": 1,
            "configuration": config_value(),
            "surprise": true,
        });
        assert!(SummaryPayload::from_value(&value).is_err());
    }

    #[test]
    fn wrong_task_type_literal_rejected() {
        let value = json!({
            "task_type": "representation",
            "session_name": "sess",
            "message_seq_in_session": 1,
            "configuration": config_value(),
        });
        assert!(SummaryPayload::from_value(&value).is_err());
    }

    #[test]
    fn missing_required_field_rejected() {
        let value = json!({
            "session_name": "sess",
            "configuration": config_value(),
        });
        assert!(SummaryPayload::from_value(&value).is_err());
    }

    #[test]
    fn webhook_payload_parses() {
        let value = json!({
            "task_type": "webhook",
            "event_type": "queue.empty",
            "data": {"k": "v"},
        });
        let parsed = WebhookPayload::from_value(&value).unwrap();
        assert_eq!(parsed.event_type, "queue.empty");
        assert_eq!(parsed.data, json!({"k": "v"}));
    }

    #[test]
    fn webhook_requires_object_data() {
        let value = json!({"event_type": "x", "data": "not-an-object"});
        assert!(WebhookPayload::from_value(&value).is_err());
    }

    #[test]
    fn dream_payload_parses_with_optionals() {
        let value = json!({
            "task_type": "dream",
            "dream_type": "omni",
            "observer": "alice",
            "observed": "bob",
            "trigger_reason": "manual",
            "document_threshold": 50,
        });
        let parsed = DreamPayload::from_value(&value).unwrap();
        assert_eq!(parsed.dream_type, DreamType::Omni);
        assert_eq!(parsed.observer, "alice");
        assert_eq!(parsed.observed, "bob");
        assert_eq!(parsed.session_name, None);
        assert_eq!(parsed.trigger_reason.as_deref(), Some("manual"));
        assert_eq!(parsed.document_threshold, Some(50));
    }

    #[test]
    fn dream_payload_rejects_unknown_dream_type() {
        let value = json!({"dream_type": "lucid", "observer": "a", "observed": "b"});
        assert!(DreamPayload::from_value(&value).is_err());
    }

    #[test]
    fn deletion_payload_parses_all_types() {
        for (raw, expected) in [
            ("session", DeletionType::Session),
            ("observation", DeletionType::Observation),
            ("workspace", DeletionType::Workspace),
        ] {
            let value = json!({"deletion_type": raw, "resource_id": "r1"});
            let parsed = DeletionPayload::from_value(&value).unwrap();
            assert_eq!(parsed.deletion_type, expected);
            assert_eq!(parsed.resource_id, "r1");
        }
        let bad = json!({"deletion_type": "peer", "resource_id": "r1"});
        assert!(DeletionPayload::from_value(&bad).is_err());
    }

    #[test]
    fn reconciler_payload_parses() {
        let sync = json!({"task_type": "reconciler", "reconciler_type": "sync_vectors"});
        assert_eq!(
            ReconcilerPayload::from_value(&sync).unwrap().reconciler_type,
            ReconcilerType::SyncVectors
        );
        let cleanup = json!({"reconciler_type": "cleanup_queue"});
        assert_eq!(
            ReconcilerPayload::from_value(&cleanup)
                .unwrap()
                .reconciler_type,
            ReconcilerType::CleanupQueue
        );
        let bad = json!({"reconciler_type": "nope"});
        assert!(ReconcilerPayload::from_value(&bad).is_err());
    }
}
