//! Conclusion types for the Honcho API.
//!
//! Maps the `OpenAPI` schemas: `Conclusion`, `ConclusionCreate`,
//! `ConclusionBatchCreate`, `ConclusionGet`, `ConclusionQuery`, `Page[Conclusion]`.

use crate::types::common::JsonValue;
use crate::types::pagination::Page;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// A conclusion about a peer, produced by observation.
///
/// Maps `OpenAPI` `Conclusion`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[non_exhaustive]
pub struct Conclusion {
    /// Unique identifier.
    pub id: String,
    /// The conclusion content text.
    pub content: String,
    /// The peer who made the conclusion.
    pub observer_id: String,
    /// The peer the conclusion is about.
    pub observed_id: String,
    /// Optional session ID the conclusion belongs to.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    /// When the conclusion was created.
    pub created_at: DateTime<Utc>,
}

/// Request body for creating a single conclusion.
///
/// Maps `OpenAPI` `ConclusionCreate`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, bon::Builder)]
#[builder(on(String, into))]
#[builder(finish_fn = build)]
#[non_exhaustive]
pub struct ConclusionCreate {
    /// The conclusion content (1–65535 chars).
    pub content: String,
    /// The peer making the conclusion.
    pub observer_id: String,
    /// The peer the conclusion is about.
    pub observed_id: String,
    /// Optional session ID to associate the conclusion with.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
}

/// Request body for batch-creating conclusions (1–100 items).
///
/// Maps `OpenAPI` `ConclusionBatchCreate`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, bon::Builder)]
#[builder(finish_fn = build)]
#[non_exhaustive]
pub struct ConclusionBatchCreate {
    /// The conclusions to create.
    pub conclusions: Vec<ConclusionCreate>,
}

/// Request body for listing conclusions with optional filters.
///
/// Maps `OpenAPI` `ConclusionGet`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default, bon::Builder)]
#[builder(finish_fn = build)]
#[non_exhaustive]
pub struct ConclusionGet {
    /// Optional metadata filters.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filters: Option<JsonValue>,
}

/// Request body for semantic search over conclusions.
///
/// Maps `OpenAPI` `ConclusionQuery`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, bon::Builder)]
#[builder(on(String, into))]
#[builder(finish_fn = build)]
#[non_exhaustive]
pub struct ConclusionQuery {
    /// Semantic search query string.
    pub query: String,
    /// Number of results to return (1–100, default 10).
    #[serde(default = "default_top_k", skip_serializing_if = "is_default_top_k")]
    pub top_k: u32,
    /// Maximum cosine distance threshold (0.0–1.0).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub distance: Option<f64>,
    /// Additional metadata filters.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filters: Option<JsonValue>,
}

fn default_top_k() -> u32 {
    10
}

#[allow(clippy::trivially_copy_pass_by_ref)]
fn is_default_top_k(v: &u32) -> bool {
    *v == 10
}

/// A page of conclusion results.
///
/// Alias for `Page<Conclusion>`, maps `OpenAPI` `Page[Conclusion]`.
pub type ConclusionPage = Page<Conclusion>;
