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
#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, bon::Builder)]
#[builder(on(String, into))]
#[builder(finish_fn = build)]
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
#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, bon::Builder)]
#[builder(on(String, into))]
#[builder(finish_fn = build)]
pub struct SessionUpdate {
    /// Updated metadata (replaces existing).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, serde_json::Value>>,
    /// Updated session configuration (merges with existing).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub configuration: Option<SessionConfiguration>,
}

/// Query parameters for listing/getting sessions.
#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, bon::Builder)]
#[builder(on(String, into))]
#[builder(finish_fn = build)]
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

/// Options for `Session::context_with_options`.
#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, bon::Builder)]
#[builder(on(String, into))]
#[builder(finish_fn = build)]
pub struct SessionContextOptions {
    /// Whether to include the session summary.
    #[serde(default = "default_true")]
    #[builder(default = true)]
    pub summary: bool,
    /// Whether to limit representation context to this session only.
    #[serde(default)]
    #[builder(default)]
    pub limit_to_session: bool,
    /// Maximum number of tokens to include in the context.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub tokens: Option<u32>,
    /// A peer ID to get context for.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peer_target: Option<String>,
    /// A peer ID to get context from the perspective of.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peer_perspective: Option<String>,
    /// A query string used to fetch semantically relevant conclusions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub search_query: Option<String>,
    /// Number of semantically relevant facts to return when searching.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub search_top_k: Option<u32>,
    /// Maximum semantic distance for search results (0.0–1.0).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub search_max_distance: Option<f64>,
    /// Whether to include the most frequent conclusions.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub include_most_frequent: Option<bool>,
    /// Maximum number of conclusions to include.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_conclusions: Option<u32>,
}

fn default_true() -> bool {
    true
}

/// Context returned when requesting session state.
#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SessionContext {
    /// Session identifier.
    pub id: String,
    /// Messages in the session.
    pub messages: Vec<super::message::MessageResponse>,
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
#[non_exhaustive]
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

/// Options for listing sessions with filters and pagination.
#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, bon::Builder)]
#[builder(on(String, into))]
#[builder(finish_fn = build)]
pub struct SessionListOptions {
    /// Filter criteria for sessions.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filters: Option<HashMap<String, serde_json::Value>>,
    /// Page number (1-based).
    #[serde(default = "default_page")]
    #[builder(default = default_page())]
    pub page: u64,
    /// Page size.
    #[serde(default = "default_size")]
    #[builder(default = default_size())]
    pub size: u64,
    /// Reverse order.
    #[serde(default)]
    #[builder(default)]
    pub reverse: bool,
}

fn default_page() -> u64 {
    1
}

fn default_size() -> u64 {
    50
}

/// A paginated list of sessions.
pub type SessionPage = super::pagination::Page<Session>;

impl SessionContext {
    /// Convert the context to OpenAI-compatible message format.
    ///
    /// System messages (`peer_representation`, `peer_card`, summary) are prepended.
    /// Assistant messages get `role: "assistant"`, all others get `role: "user"`.
    /// Each message also includes a `"name"` field set to the peer ID.
    #[must_use]
    pub fn to_openai(&self, assistant: &str) -> Vec<serde_json::Value> {
        let mut result: Vec<serde_json::Value> = Vec::new();

        if let Some(ref rep) = self.peer_representation {
            result.push(serde_json::json!({
                "role": "system",
                "content": format!("<peer_representation>{rep}</peer_representation>"),
            }));
        }

        if let Some(ref card) = self.peer_card {
            result.push(serde_json::json!({
                "role": "system",
                "content": format!("<peer_card>{}</peer_card>", serde_json::to_string(card).unwrap_or_default()),
            }));
        }

        if let Some(ref summary) = self.summary {
            result.push(serde_json::json!({
                "role": "system",
                "content": format!("<summary>{}</summary>", summary.content),
            }));
        }

        for message in &self.messages {
            if message.peer_id == assistant {
                result.push(serde_json::json!({
                    "role": "assistant",
                    "name": message.peer_id,
                    "content": message.content,
                }));
            } else {
                result.push(serde_json::json!({
                    "role": "user",
                    "name": message.peer_id,
                    "content": message.content,
                }));
            }
        }

        result
    }

    /// Convert the context to Anthropic-compatible message format.
    ///
    /// System-like messages (`peer_representation`, `peer_card`, summary) use `role: "user"`
    /// since Anthropic uses a separate `system` parameter.
    /// Assistant messages get `role: "assistant"`, others get `role: "user"` with
    /// `PEER_ID: CONTENT` format.
    #[must_use]
    pub fn to_anthropic(&self, assistant: &str) -> Vec<serde_json::Value> {
        let mut result: Vec<serde_json::Value> = Vec::new();

        if let Some(ref rep) = self.peer_representation {
            result.push(serde_json::json!({
                "role": "user",
                "content": format!("<peer_representation>{rep}</peer_representation>"),
            }));
        }

        if let Some(ref card) = self.peer_card {
            result.push(serde_json::json!({
                "role": "user",
                "content": format!("<peer_card>{}</peer_card>", serde_json::to_string(card).unwrap_or_default()),
            }));
        }

        if let Some(ref summary) = self.summary {
            result.push(serde_json::json!({
                "role": "user",
                "content": format!("<summary>{}</summary>", summary.content),
            }));
        }

        for message in &self.messages {
            if message.peer_id == assistant {
                result.push(serde_json::json!({
                    "role": "assistant",
                    "content": message.content,
                }));
            } else {
                result.push(serde_json::json!({
                    "role": "user",
                    "content": format!("{}: {}", message.peer_id, message.content),
                }));
            }
        }

        result
    }

    /// Returns the number of messages plus 1 if a summary is present.
    #[must_use]
    pub fn len(&self) -> usize {
        self.messages.len() + usize::from(self.summary.is_some())
    }

    /// Returns `true` if the context contains no messages and no summary.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.messages.is_empty() && self.summary.is_none()
    }
}
