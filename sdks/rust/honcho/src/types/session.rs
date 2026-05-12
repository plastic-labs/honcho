//! Session-related types for the Honcho API.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

pub use super::dream::{DreamConfiguration, ReasoningConfiguration, SessionQueueStatus};

/// A conversation session containing messages between peers.
#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Session {
    /// Unique session identifier.
    pub id: String,
    /// Whether the session is currently active.
    pub is_active: bool,
    /// The workspace this session belongs to.
    pub workspace_id: String,
    /// Arbitrary key-value metadata attached to the session.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, serde_json::Value>,
    /// Session-level configuration overrides.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub configuration: HashMap<String, serde_json::Value>,
    /// When the session was created.
    pub created_at: DateTime<Utc>,
}

/// Request body for creating a new session.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, bon::Builder)]
pub struct SessionCreate {
    /// Unique session identifier (alphanumeric, hyphens, underscores).
    pub id: String,
    /// Optional metadata to attach.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, serde_json::Value>>,
    /// Peer configurations keyed by peer ID.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peers: Option<HashMap<String, SessionPeerConfig>>,
    /// Optional session-level configuration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub configuration: Option<SessionConfiguration>,
}

/// Request body for updating a session.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, bon::Builder)]
pub struct SessionUpdate {
    /// Updated metadata (replaces existing).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, serde_json::Value>>,
    /// Updated session configuration (merges with existing).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub configuration: Option<SessionConfiguration>,
}

/// Query parameters for listing/getting sessions.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, bon::Builder)]
pub struct SessionGet {
    /// Filter criteria for sessions.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filters: Option<HashMap<String, serde_json::Value>>,
}

/// Session-level configuration overrides.
///
/// All fields are optional. Session-level configuration overrides
/// workspace-level configuration, which overrides global configuration.
#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SessionConfiguration {
    /// Configuration for reasoning functionality.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ReasoningConfiguration>,
    /// Configuration for peer card functionality.
    ///
    /// If reasoning is disabled, peer cards will also be disabled
    /// and these settings will be ignored.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peer_card: Option<PeerCardConfiguration>,
    /// Configuration for summary functionality.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<SummaryConfiguration>,
    /// Configuration for dream functionality.
    ///
    /// If reasoning is disabled, dreams will also be disabled
    /// and these settings will be ignored.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dream: Option<DreamConfiguration>,
}

/// Configuration for peer card generation and usage.
#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PeerCardConfiguration {
    /// Whether to use peer card during the reasoning process.
    #[serde(rename = "use", skip_serializing_if = "Option::is_none")]
    pub use_peer_card: Option<bool>,
    /// Whether to generate a peer card based on content.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub create: Option<bool>,
}

/// Configuration for automatic session summarization.
#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
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

/// Per-peer observation settings within a session.
#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SessionPeerConfig {
    /// Whether Honcho will use reasoning to form a representation of this peer.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub observe_me: Option<bool>,
    /// Whether this peer should form session-level representations of other peers.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub observe_others: Option<bool>,
}

/// Context returned when requesting session state.
#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SessionContext {
    /// Session identifier.
    pub id: String,
    /// Messages in the session.
    pub messages: Vec<super::message::Message>,
    /// The summary if available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<Summary>,
    /// Curated subset of a peer representation, if requested from a specific perspective.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peer_representation: Option<String>,
    /// The peer card, if requested from a specific perspective.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peer_card: Option<Vec<String>>,
}

/// Summaries for a session (short and/or long).
#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SessionSummaries {
    /// Session identifier.
    pub id: String,
    /// The short summary if available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub short_summary: Option<Summary>,
    /// The long summary if available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub long_summary: Option<Summary>,
}

/// The type of session summary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SummaryType {
    /// Short summary generated more frequently.
    Short,
    /// Long summary generated less frequently.
    Long,
}

/// A session summary covering messages up to a point.
#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Summary {
    /// The summary text.
    pub content: String,
    /// The public ID of the message this summary covers up to.
    pub message_id: String,
    /// The type of summary.
    pub summary_type: SummaryType,
    /// When the summary was created.
    pub created_at: DateTime<Utc>,
    /// Number of tokens in the summary text.
    pub token_count: u32,
}

/// A paginated list of sessions.
pub type SessionPage = super::pagination::Page<Session>;
