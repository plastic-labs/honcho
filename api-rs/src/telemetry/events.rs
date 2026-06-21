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
    fn generate_id_matches_python() {
        // Golden from generate_event_id("representation.completed", "ws1:sess1:msg_latest",
        //   "2026-06-21T12:00:00+00:00", "9.9.9").
        let event = sample();
        assert_eq!(event.generate_id(Some("9.9.9")), "evt_IYFU_IGcJmlPIF1Q8C2UgA");
    }
}
