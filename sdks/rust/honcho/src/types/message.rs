//! Message-related types for the Honcho API.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

pub use super::dream::ReasoningConfiguration;

/// A message in a session.
///
/// Represents a single message created by a peer within a session.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Message {
    /// Unique message identifier.
    pub id: String,
    /// Message content text.
    pub content: String,
    /// ID of the peer that authored this message.
    pub peer_id: String,
    /// ID of the session this message belongs to.
    pub session_id: String,
    /// Arbitrary key-value metadata attached to the message.
    #[serde(default)]
    pub metadata: HashMap<String, serde_json::Value>,
    /// Timestamp when the message was created.
    pub created_at: DateTime<Utc>,
    /// ID of the workspace this message belongs to.
    pub workspace_id: String,
    /// Token count for the message content.
    pub token_count: u64,
}

/// Parameters for creating a single message.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MessageCreate {
    /// Message content text (max 25 000 characters).
    pub content: String,
    /// ID of the peer authoring the message.
    pub peer_id: String,
    /// Optional arbitrary metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, serde_json::Value>>,
    /// Optional message-level configuration.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub configuration: Option<MessageConfiguration>,
    /// Optional override for the creation timestamp.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub created_at: Option<DateTime<Utc>>,
}

/// Parameters for batch-creating messages (1–100).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MessageBatchCreate {
    /// List of messages to create.
    pub messages: Vec<MessageCreate>,
}

/// Parameters for updating a message.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct MessageUpdate {
    /// Updated metadata (replaces existing).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, serde_json::Value>>,
}

/// Parameters for listing / filtering messages.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct MessageGet {
    /// Optional filter predicates.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filters: Option<HashMap<String, serde_json::Value>>,
}

/// Configuration that can be attached to a message.
///
/// All fields optional; message-level config overrides session and workspace config.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct MessageConfiguration {
    /// Reasoning configuration for this message.
    pub reasoning: Option<ReasoningConfiguration>,
}

/// Parameters for searching messages.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MessageSearchOptions {
    /// Search query string.
    pub query: String,
    /// Optional filters to scope the search.
    pub filters: Option<HashMap<String, serde_json::Value>>,
    /// Maximum number of results (1–100, default 10).
    #[serde(default = "default_limit")]
    pub limit: u32,
}

fn default_limit() -> u32 {
    10
}

/// Multipart form body for file-based message upload.
///
/// Used by the `POST …/messages/upload` endpoint. `file` and `peer_id` are
/// required; the remaining fields are optional string-encoded payloads.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MessageUploadForm {
    /// Raw file contents as bytes.
    pub file: Vec<u8>,
    /// ID of the peer uploading the file.
    pub peer_id: String,
    /// Optional JSON-encoded metadata string.
    pub metadata: Option<String>,
    /// Optional JSON-encoded configuration string.
    pub configuration: Option<String>,
    /// Optional ISO-8601 timestamp override.
    pub created_at: Option<String>,
}

/// Paginated response of [`Message`] items.
pub type MessagePage = super::pagination::Page<Message>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_search_options_default_limit() {
        let opts: MessageSearchOptions = serde_json::from_str(r#"{"query":"hello"}"#).unwrap();
        assert_eq!(opts.limit, 10);
    }
}
