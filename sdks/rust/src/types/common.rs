//! Common types shared across domain modules.

use serde::{Deserialize, Serialize};

/// JSON value alias for metadata and configuration fields.
pub type JsonValue = serde_json::Value;

/// Configuration for reasoning functionality.
#[non_exhaustive]
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ReasoningConfiguration {
    /// Whether to enable reasoning functionality.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    /// Custom instructions for the reasoning system.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub custom_instructions: Option<String>,
}

/// Configuration for automatic session summarization.
#[non_exhaustive]
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct SummaryConfiguration {
    /// Whether to enable summary functionality.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
    /// Number of messages per short summary (minimum 10).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub messages_per_short_summary: Option<u32>,
    /// Number of messages per long summary (minimum 20).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub messages_per_long_summary: Option<u32>,
}

/// Configuration for dream functionality.
#[non_exhaustive]
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct DreamConfiguration {
    /// Whether to enable dream functionality.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub enabled: Option<bool>,
}

/// Configuration for peer card generation and usage.
#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct PeerCardConfiguration {
    /// Whether to use the peer card during the reasoning process.
    #[serde(rename = "use", default, skip_serializing_if = "Option::is_none")]
    pub use_peer_card: Option<bool>,
    /// Whether to generate a peer card based on content.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub create: Option<bool>,
}
