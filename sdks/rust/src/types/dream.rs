//! Dream API types — background memory consolidation scheduling.

pub use super::common::{DreamConfiguration, ReasoningConfiguration};

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

/// Types of dreams that can be triggered.
#[non_exhaustive]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum DreamType {
    /// Omni dream — consolidate all observations.
    Omni,
}

/// Request to schedule a dream task.
///
/// Maps `ScheduleDreamRequest` from the `OpenAPI` spec.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, bon::Builder)]
#[non_exhaustive]
#[builder(derive(Debug), on(String, into))]
#[builder(finish_fn = build)]
pub struct ScheduleDreamRequest {
    /// Observer peer name.
    pub observer: String,
    /// Type of dream to schedule.
    pub dream_type: DreamType,
    /// Observed peer name (defaults to observer if not specified).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub observed: Option<String>,
    /// Session ID to scope the dream to.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
}

/// Status for a specific session within the processing queue.
///
/// Maps `SessionQueueStatus` from the `OpenAPI` spec.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct SessionQueueStatus {
    /// Session ID if filtered by session.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    /// Total work units.
    pub total_work_units: u64,
    /// Completed work units (since last periodic cleanup).
    pub completed_work_units: u64,
    /// Work units currently being processed.
    pub in_progress_work_units: u64,
    /// Work units waiting to be processed.
    pub pending_work_units: u64,
}

/// Aggregated processing queue status.
///
/// Tracks user-facing task types only: representation, summary, and dream.
/// Internal infrastructure tasks (reconciler, webhook, deletion) are excluded.
///
/// Maps `QueueStatus` from the `OpenAPI` spec.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct QueueStatus {
    /// Total work units.
    pub total_work_units: u64,
    /// Completed work units (since last periodic cleanup).
    pub completed_work_units: u64,
    /// Work units currently being processed.
    pub in_progress_work_units: u64,
    /// Work units waiting to be processed.
    pub pending_work_units: u64,
    /// Per-session status when not filtered by session.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sessions: Option<HashMap<String, SessionQueueStatus>>,
}
