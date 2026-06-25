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

/// Port of `SyncVectorsCompletedEvent` (events/reconciliation.py): emitted when a
/// vector sync cycle completes. Global operation — no workspace context; the
/// resource id is the fixed string `"sync_vectors"`. Count fields default to 0.
#[derive(Debug, Clone, Serialize)]
pub struct SyncVectorsCompletedEvent {
    pub timestamp: DateTime<Utc>,
    pub documents_synced: i64,
    pub documents_failed: i64,
    pub documents_cleaned: i64,
    pub message_embeddings_synced: i64,
    pub message_embeddings_failed: i64,
    pub total_duration_ms: f64,
}

impl SyncVectorsCompletedEvent {
    pub const EVENT_TYPE: &'static str = "reconciliation.sync_vectors.completed";
    pub const SCHEMA_VERSION: i32 = 1;
    pub const CATEGORY: &'static str = "reconciliation";

    /// Port of `get_resource_id`: fixed for this global operation.
    pub fn get_resource_id(&self) -> String {
        "sync_vectors".to_string()
    }

    /// Port of `generate_id` (package version folded in by the caller).
    pub fn generate_id(&self, honcho_version: Option<&str>) -> String {
        let iso = python_isoformat_utc(self.timestamp);
        generate_event_id(Self::EVENT_TYPE, &self.get_resource_id(), &iso, honcho_version)
    }
}

impl super::TelemetryEvent for SyncVectorsCompletedEvent {
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
        SyncVectorsCompletedEvent::get_resource_id(self)
    }
    fn to_body(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }
}

/// Port of `CleanupStaleItemsCompletedEvent` (events/reconciliation.py): emitted
/// when a stale-items cleanup cycle completes. Global operation — no workspace
/// context; the resource id is the fixed string `"cleanup_stale_items"`.
#[derive(Debug, Clone, Serialize)]
pub struct CleanupStaleItemsCompletedEvent {
    pub timestamp: DateTime<Utc>,
    pub documents_cleaned: i64,
    pub queue_items_cleaned: i64,
    pub total_duration_ms: f64,
}

impl CleanupStaleItemsCompletedEvent {
    pub const EVENT_TYPE: &'static str = "reconciliation.cleanup_stale_items.completed";
    pub const SCHEMA_VERSION: i32 = 1;
    pub const CATEGORY: &'static str = "reconciliation";

    /// Port of `get_resource_id`: fixed for this global operation.
    pub fn get_resource_id(&self) -> String {
        "cleanup_stale_items".to_string()
    }

    /// Port of `generate_id` (package version folded in by the caller).
    pub fn generate_id(&self, honcho_version: Option<&str>) -> String {
        let iso = python_isoformat_utc(self.timestamp);
        generate_event_id(Self::EVENT_TYPE, &self.get_resource_id(), &iso, honcho_version)
    }
}

impl super::TelemetryEvent for CleanupStaleItemsCompletedEvent {
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
        CleanupStaleItemsCompletedEvent::get_resource_id(self)
    }
    fn to_body(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }
}

/// Port of `DreamRunEvent` (events/dream.py): the top-level event for a full
/// dream orchestration, with aggregate metrics across both specialists. Optional
/// fields (`session_name` + the additive scheduling-context block) serialize as
/// `null` when absent, matching pydantic `model_dump`. Schema version 2.
#[derive(Debug, Clone, Serialize)]
pub struct DreamRunEvent {
    pub timestamp: DateTime<Utc>,
    pub run_id: String,
    pub workspace_name: String,
    pub session_name: Option<String>,
    pub observer: String,
    pub observed: String,
    pub specialists_run: Vec<String>,
    pub deduction_success: bool,
    pub induction_success: bool,
    // ---- Surprisal sampling (pydantic defaults: false / 0) ----
    pub surprisal_enabled: bool,
    pub surprisal_conclusion_count: i64,
    // ---- Aggregated metrics ----
    pub total_iterations: i64,
    pub total_input_tokens: i64,
    pub total_output_tokens: i64,
    pub total_duration_ms: f64,
    // ---- Additive scheduling-context fields (pydantic defaults: None / 0) ----
    pub dream_type: Option<String>,
    pub enabled_types_count: i64,
    pub trigger_reason: Option<String>,
    pub delay_reason: Option<String>,
    pub documents_since_last_dream_at_schedule: Option<i64>,
    pub document_threshold: Option<i64>,
}

impl DreamRunEvent {
    pub const EVENT_TYPE: &'static str = "dream.run";
    pub const SCHEMA_VERSION: i32 = 2;
    pub const CATEGORY: &'static str = "dream";

    /// Port of `get_resource_id`: the run_id (unique per dream cycle).
    pub fn get_resource_id(&self) -> String {
        self.run_id.clone()
    }

    /// Port of `generate_id` (package version folded in by the caller).
    pub fn generate_id(&self, honcho_version: Option<&str>) -> String {
        let iso = python_isoformat_utc(self.timestamp);
        generate_event_id(Self::EVENT_TYPE, &self.get_resource_id(), &iso, honcho_version)
    }
}

impl super::TelemetryEvent for DreamRunEvent {
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
        DreamRunEvent::get_resource_id(self)
    }
    fn to_body(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }
}

/// Port of `DreamSpecialistEvent` (events/dream.py): emitted per specialist
/// (deduction / induction), correlated to the parent [`DreamRunEvent`] by
/// `run_id`. The additive denormalized rollups default to 0 / false / empty;
/// the per-level count maps serialize as JSON objects. Schema version 2.
#[derive(Debug, Clone, Serialize)]
pub struct DreamSpecialistEvent {
    pub timestamp: DateTime<Utc>,
    pub run_id: String,
    /// `"deduction"` or `"induction"`.
    pub specialist_type: String,
    pub workspace_name: String,
    pub observer: String,
    pub observed: String,
    pub iterations: i64,
    pub tool_calls_count: i64,
    pub input_tokens: i64,
    pub output_tokens: i64,
    pub duration_ms: f64,
    pub success: bool,
    // ---- Additive denormalized rollups (pydantic defaults: 0 / false / {}) ----
    pub created_observation_count: i64,
    pub deleted_observation_count: i64,
    pub created_counts_by_level: std::collections::BTreeMap<String, i64>,
    pub deleted_counts_by_level: std::collections::BTreeMap<String, i64>,
    pub peer_card_updated: bool,
    pub search_tool_calls_count: i64,
    /// Exception class name when `success` is false; `None` on success.
    pub error_class: Option<String>,
}

impl DreamSpecialistEvent {
    pub const EVENT_TYPE: &'static str = "dream.specialist";
    pub const SCHEMA_VERSION: i32 = 2;
    pub const CATEGORY: &'static str = "dream";

    /// Port of `get_resource_id`: `{run_id}:{specialist_type}`.
    pub fn get_resource_id(&self) -> String {
        format!("{}:{}", self.run_id, self.specialist_type)
    }

    /// Port of `generate_id` (package version folded in by the caller).
    pub fn generate_id(&self, honcho_version: Option<&str>) -> String {
        let iso = python_isoformat_utc(self.timestamp);
        generate_event_id(Self::EVENT_TYPE, &self.get_resource_id(), &iso, honcho_version)
    }
}

impl super::TelemetryEvent for DreamSpecialistEvent {
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
        DreamSpecialistEvent::get_resource_id(self)
    }
    fn to_body(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }
}

/// Port of `AgentToolConclusionsCreatedEvent` (events/agent.py): fired when a
/// `create_observations*` tool call persists conclusions, with the per-level
/// breakdown. Schema version 2.
#[derive(Debug, Clone, Serialize)]
pub struct AgentToolConclusionsCreatedEvent {
    pub timestamp: DateTime<Utc>,
    pub run_id: String,
    pub iteration: i64,
    pub parent_category: String,
    pub agent_type: String,
    pub workspace_name: String,
    pub observer: String,
    pub observed: String,
    pub conclusion_count: i64,
    /// Level of each created conclusion (pydantic default `[]`).
    pub levels: Vec<String>,
}

impl AgentToolConclusionsCreatedEvent {
    pub const EVENT_TYPE: &'static str = "agent.tool.conclusions.created";
    pub const SCHEMA_VERSION: i32 = 2;
    pub const CATEGORY: &'static str = "agent";

    /// Port of `get_resource_id`: `{run_id}:{iteration}:conclusions_created`.
    pub fn get_resource_id(&self) -> String {
        format!("{}:{}:conclusions_created", self.run_id, self.iteration)
    }

    pub fn generate_id(&self, honcho_version: Option<&str>) -> String {
        let iso = python_isoformat_utc(self.timestamp);
        generate_event_id(Self::EVENT_TYPE, &self.get_resource_id(), &iso, honcho_version)
    }
}

impl super::TelemetryEvent for AgentToolConclusionsCreatedEvent {
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
        AgentToolConclusionsCreatedEvent::get_resource_id(self)
    }
    fn to_body(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }
}

/// Port of `AgentToolConclusionsDeletedEvent` (events/agent.py): fired when a
/// `delete_observations` tool call soft-deletes conclusions. **Schema version 3**
/// (note: deleted is one version ahead of created/peer_card).
#[derive(Debug, Clone, Serialize)]
pub struct AgentToolConclusionsDeletedEvent {
    pub timestamp: DateTime<Utc>,
    pub run_id: String,
    pub iteration: i64,
    pub parent_category: String,
    pub agent_type: String,
    pub workspace_name: String,
    pub observer: String,
    pub observed: String,
    pub conclusion_count: i64,
    /// Level of each deleted conclusion (pydantic default `[]`).
    pub levels: Vec<String>,
}

impl AgentToolConclusionsDeletedEvent {
    pub const EVENT_TYPE: &'static str = "agent.tool.conclusions.deleted";
    pub const SCHEMA_VERSION: i32 = 3;
    pub const CATEGORY: &'static str = "agent";

    /// Port of `get_resource_id`: `{run_id}:{iteration}:conclusions_deleted`.
    pub fn get_resource_id(&self) -> String {
        format!("{}:{}:conclusions_deleted", self.run_id, self.iteration)
    }

    pub fn generate_id(&self, honcho_version: Option<&str>) -> String {
        let iso = python_isoformat_utc(self.timestamp);
        generate_event_id(Self::EVENT_TYPE, &self.get_resource_id(), &iso, honcho_version)
    }
}

impl super::TelemetryEvent for AgentToolConclusionsDeletedEvent {
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
        AgentToolConclusionsDeletedEvent::get_resource_id(self)
    }
    fn to_body(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap_or(serde_json::Value::Null)
    }
}

/// Port of `AgentToolPeerCardUpdatedEvent` (events/agent.py): fired when an
/// `update_peer_card` tool call replaces the peer card. Schema version 2.
#[derive(Debug, Clone, Serialize)]
pub struct AgentToolPeerCardUpdatedEvent {
    pub timestamp: DateTime<Utc>,
    pub run_id: String,
    pub iteration: i64,
    pub parent_category: String,
    pub agent_type: String,
    pub workspace_name: String,
    pub observer: String,
    pub observed: String,
    pub facts_count: i64,
}

impl AgentToolPeerCardUpdatedEvent {
    pub const EVENT_TYPE: &'static str = "agent.tool.peer_card.updated";
    pub const SCHEMA_VERSION: i32 = 2;
    pub const CATEGORY: &'static str = "agent";

    /// Port of `get_resource_id`: `{run_id}:{iteration}:peer_card_updated`.
    pub fn get_resource_id(&self) -> String {
        format!("{}:{}:peer_card_updated", self.run_id, self.iteration)
    }

    pub fn generate_id(&self, honcho_version: Option<&str>) -> String {
        let iso = python_isoformat_utc(self.timestamp);
        generate_event_id(Self::EVENT_TYPE, &self.get_resource_id(), &iso, honcho_version)
    }
}

impl super::TelemetryEvent for AgentToolPeerCardUpdatedEvent {
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
        AgentToolPeerCardUpdatedEvent::get_resource_id(self)
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

    #[test]
    fn sync_vectors_event_metadata_and_golden_id() {
        use crate::telemetry::TelemetryEvent;
        let event = SyncVectorsCompletedEvent {
            timestamp: Utc.with_ymd_and_hms(2026, 6, 21, 12, 0, 0).unwrap(),
            documents_synced: 4,
            documents_failed: 1,
            documents_cleaned: 2,
            message_embeddings_synced: 7,
            message_embeddings_failed: 0,
            total_duration_ms: 12.5,
        };
        assert_eq!(event.get_resource_id(), "sync_vectors");
        assert_eq!(
            SyncVectorsCompletedEvent::EVENT_TYPE,
            "reconciliation.sync_vectors.completed"
        );
        assert_eq!(SyncVectorsCompletedEvent::SCHEMA_VERSION, 1);
        assert_eq!(SyncVectorsCompletedEvent::CATEGORY, "reconciliation");
        assert_eq!(event.volume_class(), "ground_truth");
        // Golden from generate_event_id("reconciliation.sync_vectors.completed",
        //   ts=2026-06-21T12:00:00+00:00, "sync_vectors", "9.9.9").
        assert_eq!(event.generate_id(Some("9.9.9")), "evt_bC7QKIVrMuNEkEdK1b9Nfg");
        let body = TelemetryEvent::to_body(&event);
        assert_eq!(body["message_embeddings_synced"], 7);
        assert_eq!(body["documents_cleaned"], 2);
    }

    #[test]
    fn cleanup_stale_items_event_metadata_and_golden_id() {
        use crate::telemetry::TelemetryEvent;
        let event = CleanupStaleItemsCompletedEvent {
            timestamp: Utc.with_ymd_and_hms(2026, 6, 21, 12, 0, 0).unwrap(),
            documents_cleaned: 0,
            queue_items_cleaned: 9,
            total_duration_ms: 3.0,
        };
        assert_eq!(event.get_resource_id(), "cleanup_stale_items");
        assert_eq!(
            CleanupStaleItemsCompletedEvent::EVENT_TYPE,
            "reconciliation.cleanup_stale_items.completed"
        );
        assert_eq!(CleanupStaleItemsCompletedEvent::SCHEMA_VERSION, 1);
        assert_eq!(CleanupStaleItemsCompletedEvent::CATEGORY, "reconciliation");
        // Golden from generate_event_id("reconciliation.cleanup_stale_items.completed",
        //   ts=2026-06-21T12:00:00+00:00, "cleanup_stale_items", "9.9.9").
        assert_eq!(event.generate_id(Some("9.9.9")), "evt_2UxFc6lRt08lhWl9jznrVw");
        let body = TelemetryEvent::to_body(&event);
        assert_eq!(body["queue_items_cleaned"], 9);
    }

    #[test]
    fn dream_run_event_metadata_and_golden_id() {
        use crate::telemetry::TelemetryEvent;
        let event = DreamRunEvent {
            timestamp: Utc.with_ymd_and_hms(2026, 6, 21, 12, 0, 0).unwrap(),
            run_id: "run123".to_string(),
            workspace_name: "ws1".to_string(),
            session_name: None,
            observer: "alice".to_string(),
            observed: "bob".to_string(),
            specialists_run: vec!["deduction".to_string(), "induction".to_string()],
            deduction_success: true,
            induction_success: false,
            surprisal_enabled: false,
            surprisal_conclusion_count: 0,
            total_iterations: 7,
            total_input_tokens: 100,
            total_output_tokens: 40,
            total_duration_ms: 250.0,
            dream_type: Some("omni".to_string()),
            enabled_types_count: 1,
            trigger_reason: Some("document_threshold".to_string()),
            delay_reason: None,
            documents_since_last_dream_at_schedule: Some(55),
            document_threshold: Some(50),
        };
        assert_eq!(event.get_resource_id(), "run123");
        assert_eq!(DreamRunEvent::EVENT_TYPE, "dream.run");
        assert_eq!(DreamRunEvent::SCHEMA_VERSION, 2);
        assert_eq!(DreamRunEvent::CATEGORY, "dream");
        assert_eq!(event.volume_class(), "ground_truth");
        // Golden from generate_event_id("dream.run", ts=2026-06-21T12:00:00+00:00,
        //   "run123", "9.9.9").
        assert_eq!(event.generate_id(Some("9.9.9")), "evt_EWh1zmNaVOn1i79NeLHsyA");
        // Optional fields serialize as null (pydantic model_dump includes them).
        let body = TelemetryEvent::to_body(&event);
        assert_eq!(body["total_iterations"], 7);
        assert!(body.get("session_name").is_some_and(|v| v.is_null()));
        assert!(body.get("delay_reason").is_some_and(|v| v.is_null()));
        assert_eq!(body["dream_type"], "omni");
    }

    #[test]
    fn dream_specialist_event_metadata_and_golden_id() {
        use crate::telemetry::TelemetryEvent;
        let mut created = std::collections::BTreeMap::new();
        created.insert("deductive".to_string(), 3i64);
        let event = DreamSpecialistEvent {
            timestamp: Utc.with_ymd_and_hms(2026, 6, 21, 12, 0, 0).unwrap(),
            run_id: "run123".to_string(),
            specialist_type: "deduction".to_string(),
            workspace_name: "ws1".to_string(),
            observer: "alice".to_string(),
            observed: "bob".to_string(),
            iterations: 4,
            tool_calls_count: 6,
            input_tokens: 80,
            output_tokens: 30,
            duration_ms: 120.0,
            success: true,
            created_observation_count: 3,
            deleted_observation_count: 0,
            created_counts_by_level: created,
            deleted_counts_by_level: std::collections::BTreeMap::new(),
            peer_card_updated: true,
            search_tool_calls_count: 2,
            error_class: None,
        };
        assert_eq!(event.get_resource_id(), "run123:deduction");
        assert_eq!(DreamSpecialistEvent::EVENT_TYPE, "dream.specialist");
        assert_eq!(DreamSpecialistEvent::SCHEMA_VERSION, 2);
        assert_eq!(DreamSpecialistEvent::CATEGORY, "dream");
        // Golden from generate_event_id("dream.specialist",
        //   ts=2026-06-21T12:00:00+00:00, "run123:deduction", "9.9.9").
        assert_eq!(event.generate_id(Some("9.9.9")), "evt_zT7dVJ0YPBfTv1eGPfCq4g");
        let body = TelemetryEvent::to_body(&event);
        assert_eq!(body["created_observation_count"], 3);
        assert_eq!(body["created_counts_by_level"]["deductive"], 3);
        assert!(body.get("error_class").is_some_and(|v| v.is_null()));
    }

    #[test]
    fn agent_tool_conclusions_created_event_metadata_and_golden_id() {
        use crate::telemetry::TelemetryEvent;
        let event = AgentToolConclusionsCreatedEvent {
            timestamp: Utc.with_ymd_and_hms(2026, 6, 21, 12, 0, 0).unwrap(),
            run_id: "run123".to_string(),
            iteration: 4,
            parent_category: "dream".to_string(),
            agent_type: "deduction".to_string(),
            workspace_name: "ws1".to_string(),
            observer: "alice".to_string(),
            observed: "bob".to_string(),
            conclusion_count: 2,
            levels: vec!["deductive".to_string(), "deductive".to_string()],
        };
        assert_eq!(event.get_resource_id(), "run123:4:conclusions_created");
        assert_eq!(AgentToolConclusionsCreatedEvent::SCHEMA_VERSION, 2);
        assert_eq!(event.category(), "agent");
        assert_eq!(event.generate_id(Some("9.9.9")), "evt_mlGEXQWTht36-sslY1DwdQ");
        let body = TelemetryEvent::to_body(&event);
        assert_eq!(body["conclusion_count"], 2);
        assert_eq!(body["levels"][0], "deductive");
    }

    #[test]
    fn agent_tool_conclusions_deleted_event_metadata_and_golden_id() {
        let event = AgentToolConclusionsDeletedEvent {
            timestamp: Utc.with_ymd_and_hms(2026, 6, 21, 12, 0, 0).unwrap(),
            run_id: "run123".to_string(),
            iteration: 4,
            parent_category: "dream".to_string(),
            agent_type: "deduction".to_string(),
            workspace_name: "ws1".to_string(),
            observer: "alice".to_string(),
            observed: "bob".to_string(),
            conclusion_count: 1,
            levels: vec!["explicit".to_string()],
        };
        assert_eq!(event.get_resource_id(), "run123:4:conclusions_deleted");
        // Deleted is schema version 3 — one ahead of created/peer_card.
        assert_eq!(AgentToolConclusionsDeletedEvent::SCHEMA_VERSION, 3);
        assert_eq!(event.generate_id(Some("9.9.9")), "evt_hFpbTGqu2r8EcCvyyMyq_Q");
    }

    #[test]
    fn agent_tool_peer_card_updated_event_metadata_and_golden_id() {
        use crate::telemetry::TelemetryEvent;
        let event = AgentToolPeerCardUpdatedEvent {
            timestamp: Utc.with_ymd_and_hms(2026, 6, 21, 12, 0, 0).unwrap(),
            run_id: "run123".to_string(),
            iteration: 4,
            parent_category: "dream".to_string(),
            agent_type: "deduction".to_string(),
            workspace_name: "ws1".to_string(),
            observer: "alice".to_string(),
            observed: "bob".to_string(),
            facts_count: 7,
        };
        assert_eq!(event.get_resource_id(), "run123:4:peer_card_updated");
        assert_eq!(AgentToolPeerCardUpdatedEvent::SCHEMA_VERSION, 2);
        assert_eq!(event.generate_id(Some("9.9.9")), "evt_T_PfXbz2-ljh0CJw1GcLVw");
        let body = TelemetryEvent::to_body(&event);
        assert_eq!(body["facts_count"], 7);
    }
}
