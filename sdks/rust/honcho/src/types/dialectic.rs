//! Dialectic API types — chat/query with representation-backed responses.

use serde::{Deserialize, Serialize};

/// Reasoning effort level for dialectic queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ReasoningLevel {
    /// Minimal reasoning.
    Minimal,
    /// Low reasoning (default).
    #[default]
    Low,
    /// Medium reasoning.
    Medium,
    /// High reasoning.
    High,
    /// Maximum reasoning.
    Max,
}

#[allow(clippy::trivially_copy_pass_by_ref)]
fn is_default_reasoning_level(level: &ReasoningLevel) -> bool {
    matches!(level, ReasoningLevel::Low)
}

#[allow(clippy::trivially_copy_pass_by_ref)]
fn is_default_bool(val: &bool) -> bool {
    !val
}

/// Options for a dialectic chat request.
///
/// Maps `DialecticOptions` from the `OpenAPI` spec.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, bon::Builder)]
#[non_exhaustive]
#[builder(derive(Debug))]
pub struct DialecticOptions {
    /// Dialectic API prompt (1–10,000 chars).
    #[builder(into)]
    pub query: String,
    /// ID of the session to scope the representation to.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    /// Optional peer to get the representation for, from the perspective of this peer.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub target: Option<String>,
    /// Whether to stream the response.
    #[serde(default, skip_serializing_if = "is_default_bool")]
    #[builder(default = false)]
    pub stream: bool,
    /// Level of reasoning to apply.
    #[serde(default, skip_serializing_if = "is_default_reasoning_level")]
    #[builder(default = ReasoningLevel::Low)]
    pub reasoning_level: ReasoningLevel,
}

/// Response from the representation endpoint.
///
/// Maps `RepresentationResponse` from the `OpenAPI` spec.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct RepresentationResponse {
    /// The peer representation text.
    pub representation: String,
}
