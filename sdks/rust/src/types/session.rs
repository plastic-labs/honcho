//! Session-related types for the Honcho API.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

pub use super::common::{
    DreamConfiguration, PeerCardConfiguration, ReasoningConfiguration, SummaryConfiguration,
};
pub use super::dream::SessionQueueStatus;

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
    #[serde(default)]
    pub configuration: SessionConfiguration,
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

/// Request body for setting session metadata.
#[non_exhaustive]
#[derive(Debug, Clone, Serialize)]
pub struct SessionMetadataSet {
    /// Metadata to set.
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Request body for setting session configuration.
#[non_exhaustive]
#[derive(Debug, Clone, Serialize)]
pub struct SessionConfigurationSet {
    /// Configuration to set.
    pub configuration: HashMap<String, serde_json::Value>,
}

/// Session-level configuration overrides.
///
/// All fields are optional. Session-level configuration overrides
/// workspace-level configuration, which overrides global configuration.
#[non_exhaustive]
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
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

/// Per-peer observation settings within a session.
#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
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

const SEARCH_TOP_K_MIN: u32 = 1;
const SEARCH_TOP_K_MAX: u32 = 100;
const SEARCH_MAX_DISTANCE_MIN: f64 = 0.0;
const SEARCH_MAX_DISTANCE_MAX: f64 = 1.0;
const MAX_CONCLUSIONS_MIN: u32 = 1;
const MAX_CONCLUSIONS_MAX: u32 = 100;

impl SessionContextOptions {
    /// Validate cross-field constraints.
    pub fn validate(&self) -> std::result::Result<(), crate::error::HonchoError> {
        if self.peer_perspective.is_some() && self.peer_target.is_none() {
            return Err(crate::error::HonchoError::Validation(
                "peer_perspective requires peer_target to be set".into(),
            ));
        }
        if self.search_query.is_some() && self.peer_target.is_none() {
            return Err(crate::error::HonchoError::Validation(
                "search_query requires peer_target to be set".into(),
            ));
        }
        if let Some(k) = self.search_top_k
            && !(SEARCH_TOP_K_MIN..=SEARCH_TOP_K_MAX).contains(&k)
        {
            return Err(crate::error::HonchoError::Validation(format!(
                "search_top_k must be between {SEARCH_TOP_K_MIN} and {SEARCH_TOP_K_MAX}"
            )));
        }
        if let Some(d) = self.search_max_distance
            && !(SEARCH_MAX_DISTANCE_MIN..=SEARCH_MAX_DISTANCE_MAX).contains(&d)
        {
            return Err(crate::error::HonchoError::Validation(format!(
                "search_max_distance must be between {SEARCH_MAX_DISTANCE_MIN} and {SEARCH_MAX_DISTANCE_MAX}"
            )));
        }
        if let Some(m) = self.max_conclusions
            && !(MAX_CONCLUSIONS_MIN..=MAX_CONCLUSIONS_MAX).contains(&m)
        {
            return Err(crate::error::HonchoError::Validation(format!(
                "max_conclusions must be between {MAX_CONCLUSIONS_MIN} and {MAX_CONCLUSIONS_MAX}"
            )));
        }
        Ok(())
    }
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

/// Resolves an assistant name from various reference types.
///
/// Implemented for `&str`, `String`, and `&Peer` so that
/// [`SessionContext::to_openai`] and [`SessionContext::to_anthropic`]
/// can accept any of these without extra boilerplate.
pub trait IntoAssistantRef {
    /// Return the string name/id to use as the assistant.
    fn as_assistant_name(&self) -> &str;
}

impl IntoAssistantRef for &str {
    fn as_assistant_name(&self) -> &str {
        self
    }
}

impl IntoAssistantRef for String {
    fn as_assistant_name(&self) -> &str {
        self.as_str()
    }
}

impl IntoAssistantRef for &crate::Peer {
    fn as_assistant_name(&self) -> &str {
        self.id()
    }
}

impl SessionContext {
    /// Convert the context to OpenAI-compatible message format.
    ///
    /// System messages (`peer_representation`, `peer_card`, summary) are prepended.
    /// Assistant messages get `role: "assistant"`, all others get `role: "user"`.
    /// Each message also includes a `"name"` field set to the peer ID.
    ///
    /// `assistant` can be a `&str`, `String`, or `&Peer`.
    ///
    /// ```
    /// use honcho_ai::types::session::SessionContext;
    /// let ctx: SessionContext = serde_json::from_value(serde_json::json!({
    ///     "id": "s1",
    ///     "messages": [],
    /// })).unwrap();
    /// let messages = ctx.to_openai("assistant-1");
    /// assert!(messages.is_empty());
    /// ```
    #[must_use]
    #[allow(clippy::needless_pass_by_value)]
    pub fn to_openai(&self, assistant: impl IntoAssistantRef) -> Vec<serde_json::Value> {
        let assistant = assistant.as_assistant_name();
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
                "content": format!("<peer_card>[{}]</peer_card>", card.iter().map(|s| format!("'{}'", s.replace('\'', "\\'"))).collect::<Vec<_>>().join(", ")),
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
    ///
    /// `assistant` can be a `&str`, `String`, or `&Peer`.
    #[must_use]
    #[allow(clippy::needless_pass_by_value)]
    pub fn to_anthropic(&self, assistant: impl IntoAssistantRef) -> Vec<serde_json::Value> {
        let assistant = assistant.as_assistant_name();
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
                "content": format!("<peer_card>[{}]</peer_card>", card.iter().map(|s| format!("'{}'", s.replace('\'', "\\'"))).collect::<Vec<_>>().join(", ")),
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

    /// Returns the number of entries that `to_openai` / `to_anthropic` would produce.
    #[must_use]
    pub fn len(&self) -> usize {
        self.messages.len()
            + usize::from(self.summary.is_some())
            + usize::from(self.peer_representation.is_some())
            + usize::from(self.peer_card.is_some())
    }

    /// Returns `true` if the context contains no messages, summary,
    /// peer representation, or peer card.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.messages.is_empty()
            && self.summary.is_none()
            && self.peer_representation.is_none()
            && self.peer_card.is_none()
    }
}
