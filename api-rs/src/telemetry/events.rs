//! Telemetry event structs, ported from `src/telemetry/events/`.
//!
//! These mirror the pydantic event models' fields, class metadata
//! (`event_type`/`schema_version`/`category`), and `get_resource_id()` /
//! `generate_id()`. The CloudEvents envelope + emitter transport are a separate
//! follow-on unit; an orchestrator can construct and (later) emit these.

use chrono::{DateTime, Utc};
use serde::Serialize;

use super::{generate_event_id, python_isoformat_utc};

/// Port of `RepresentationCompletedEvent` (events/representation.py).
///
/// Field defaults match the pydantic model (the "additive" block defaults to 0/
/// false). `timestamp` is explicit here rather than `default_factory=now(UTC)`.
#[derive(Debug, Clone, Serialize)]
pub struct RepresentationCompletedEvent {
    pub timestamp: DateTime<Utc>,
    pub workspace_name: String,
    pub session_name: String,
    pub observed: String,
    pub queue_items_processed: i64,
    pub earliest_message_id: String,
    pub latest_message_id: String,
    pub message_count: i64,
    pub explicit_conclusion_count: i64,
    pub context_preparation_ms: f64,
    pub llm_call_ms: f64,
    pub total_duration_ms: f64,
    /// Queued-message tokens — the downstream metering key (do not rename).
    pub input_tokens: i64,
    pub total_input_tokens: i64,
    pub output_tokens: i64,
    // ---- Additive fields (pydantic defaults: 0 / false) ----
    pub queued_message_count: i64,
    pub prompt_message_count: i64,
    pub prompt_message_tokens: i64,
    pub extra_context_message_count: i64,
    pub extra_context_tokens: i64,
    pub prompt_scaffold_tokens: i64,
    pub batch_max_tokens: i64,
    pub max_input_tokens: i64,
    pub was_flush_enabled: bool,
    pub hit_batch_token_cap: bool,
    pub hit_input_token_cap: bool,
    pub observer_count: i64,
}

impl RepresentationCompletedEvent {
    pub const EVENT_TYPE: &'static str = "representation.completed";
    pub const SCHEMA_VERSION: i32 = 2;
    pub const CATEGORY: &'static str = "representation";

    /// Port of `get_resource_id`.
    pub fn get_resource_id(&self) -> String {
        format!(
            "{}:{}:{}",
            self.workspace_name, self.session_name, self.latest_message_id
        )
    }

    /// Port of `generate_id` (the package version is folded in by the caller —
    /// Python reads `HONCHO_VERSION`).
    pub fn generate_id(&self, honcho_version: Option<&str>) -> String {
        let iso = python_isoformat_utc(self.timestamp);
        generate_event_id(Self::EVENT_TYPE, &self.get_resource_id(), &iso, honcho_version)
    }
}

impl super::TelemetryEvent for RepresentationCompletedEvent {
    fn event_type(&self) -> &'static str {
        Self::EVENT_TYPE
    }
    fn schema_version(&self) -> i32 {
        Self::SCHEMA_VERSION
    }
    fn category(&self) -> &'static str {
        Self::CATEGORY
    }
    fn timestamp(&self) -> DateTime<Utc> {
        self.timestamp
    }
    fn get_resource_id(&self) -> String {
        RepresentationCompletedEvent::get_resource_id(self)
    }
    fn to_body(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }
}

/// Port of `DeletionCompletedEvent` (events/deletion.py): emitted when a deletion
/// task completes. `success` + cascade counts + optional `error_message` mirror
/// the pydantic model; the counts default to 0 / `None`.
#[derive(Debug, Clone, Serialize)]
pub struct DeletionCompletedEvent {
    pub timestamp: DateTime<Utc>,
    pub workspace_name: String,
    /// `"workspace"`, `"session"`, or `"conclusions"`.
    pub deletion_type: String,
    pub resource_id: String,
    pub success: bool,
    // ---- Cascade counts (pydantic defaults: 0) ----
    pub peers_deleted: i64,
    pub sessions_deleted: i64,
    pub messages_deleted: i64,
    pub conclusions_deleted: i64,
    /// Error detail when `success` is false (pydantic default: `None`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
}

impl DeletionCompletedEvent {
    pub const EVENT_TYPE: &'static str = "deletion.completed";
    pub const SCHEMA_VERSION: i32 = 1;
    pub const CATEGORY: &'static str = "deletion";

    /// Port of `get_resource_id`: `{workspace}:{type}:{resource}`.
    pub fn get_resource_id(&self) -> String {
        format!(
            "{}:{}:{}",
            self.workspace_name, self.deletion_type, self.resource_id
        )
    }

    /// Port of `generate_id` (package version folded in by the caller).
    pub fn generate_id(&self, honcho_version: Option<&str>) -> String {
        let iso = python_isoformat_utc(self.timestamp);
        generate_event_id(Self::EVENT_TYPE, &self.get_resource_id(), &iso, honcho_version)
    }
}

impl super::TelemetryEvent for DeletionCompletedEvent {
    fn event_type(&self) -> &'static str {
        Self::EVENT_TYPE
    }
    fn schema_version(&self) -> i32 {
        Self::SCHEMA_VERSION
    }
    fn category(&self) -> &'static str {
        Self::CATEGORY
    }
    fn timestamp(&self) -> DateTime<Utc> {
        self.timestamp
    }
    fn get_resource_id(&self) -> String {
        DeletionCompletedEvent::get_resource_id(self)
    }
    fn to_body(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn sample() -> RepresentationCompletedEvent {
        RepresentationCompletedEvent {
            timestamp: Utc.with_ymd_and_hms(2026, 6, 21, 12, 0, 0).unwrap(),
            workspace_name: "ws1".to_string(),
            session_name: "sess1".to_string(),
            observed: "peer".to_string(),
            queue_items_processed: 3,
            earliest_message_id: "msg_first".to_string(),
            latest_message_id: "msg_latest".to_string(),
            message_count: 5,
            explicit_conclusion_count: 2,
            context_preparation_ms: 1.5,
            llm_call_ms: 100.0,
            total_duration_ms: 120.0,
            input_tokens: 30,
            total_input_tokens: 42,
            output_tokens: 7,
            queued_message_count: 3,
            prompt_message_count: 5,
            prompt_message_tokens: 42,
            extra_context_message_count: 2,
            extra_context_tokens: 12,
            prompt_scaffold_tokens: 50,
            batch_max_tokens: 0,
            max_input_tokens: 25000,
            was_flush_enabled: false,
            hit_batch_token_cap: false,
            hit_input_token_cap: false,
            observer_count: 1,
        }
    }

    #[test]
    fn resource_id_and_metadata() {
        let event = sample();
        assert_eq!(event.get_resource_id(), "ws1:sess1:msg_latest");
        assert_eq!(RepresentationCompletedEvent::EVENT_TYPE, "representation.completed");
        assert_eq!(RepresentationCompletedEvent::SCHEMA_VERSION, 2);
        assert_eq!(RepresentationCompletedEvent::CATEGORY, "representation");
    }

    #[test]
    fn telemetry_event_trait_dispatch() {
        use crate::telemetry::{Emitter, NoopEmitter, TelemetryEvent};
        let event = sample();
        // Trait methods agree with the inherent impl + consts.
        assert_eq!(TelemetryEvent::event_type(&event), "representation.completed");
        assert_eq!(TelemetryEvent::schema_version(&event), 2);
        assert_eq!(TelemetryEvent::category(&event), "representation");
        assert_eq!(event.volume_class(), "ground_truth");
        assert_eq!(TelemetryEvent::get_resource_id(&event), "ws1:sess1:msg_latest");
        assert_eq!(
            TelemetryEvent::generate_id(&event, Some("9.9.9")),
            "evt_IYFU_IGcJmlPIF1Q8C2UgA"
        );
        // Body serializes the queued-message metering key.
        let body = event.to_body();
        assert_eq!(body["input_tokens"], 30);
        // NoopEmitter accepts the event as a trait object (compile + run check).
        let emitter = NoopEmitter;
        emitter.emit(&event);
    }

    #[test]
    fn generate_id_matches_python() {
        // Golden from generate_event_id("representation.completed", "ws1:sess1:msg_latest",
        //   "2026-06-21T12:00:00+00:00", "9.9.9").
        let event = sample();
        assert_eq!(event.generate_id(Some("9.9.9")), "evt_IYFU_IGcJmlPIF1Q8C2UgA");
    }

    #[test]
    fn deletion_event_metadata_and_golden_id() {
        use crate::telemetry::TelemetryEvent;
        let event = DeletionCompletedEvent {
            timestamp: Utc.with_ymd_and_hms(2026, 6, 21, 12, 0, 0).unwrap(),
            workspace_name: "ws1".to_string(),
            deletion_type: "workspace".to_string(),
            resource_id: "ws1".to_string(),
            success: true,
            peers_deleted: 2,
            sessions_deleted: 3,
            messages_deleted: 10,
            conclusions_deleted: 5,
            error_message: None,
        };
        assert_eq!(event.get_resource_id(), "ws1:workspace:ws1");
        assert_eq!(DeletionCompletedEvent::EVENT_TYPE, "deletion.completed");
        assert_eq!(DeletionCompletedEvent::SCHEMA_VERSION, 1);
        assert_eq!(DeletionCompletedEvent::CATEGORY, "deletion");
        // Golden from generate_event_id("deletion.completed", "ws1:workspace:ws1",
        //   "2026-06-21T12:00:00+00:00", "9.9.9").
        assert_eq!(event.generate_id(Some("9.9.9")), "evt_a-rvRoLarKuWrWn6RtmOrA");
        // error_message=None is omitted from the body (skip_serializing_if).
        let body = TelemetryEvent::to_body(&event);
        assert_eq!(body["peers_deleted"], 2);
        assert!(body.get("error_message").is_none());
    }
}
