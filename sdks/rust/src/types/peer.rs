//! Peer-related types for the Honcho API.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// A peer in a Honcho workspace.
///
/// Peers represent participants (users or agents) within a workspace.
/// Each peer has a unique ID scoped to its workspace.
#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Peer {
    /// Unique identifier for the peer.
    pub id: String,
    /// ID of the workspace this peer belongs to.
    pub workspace_id: String,
    /// Timestamp when the peer was created.
    pub created_at: DateTime<Utc>,
    /// Optional metadata attached to the peer.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, serde_json::Value>,
    /// Optional configuration for the peer.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub configuration: HashMap<String, serde_json::Value>,
}

/// Parameters for creating a new peer.
#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize, bon::Builder)]
#[builder(on(String, into))]
pub struct PeerCreate {
    /// Unique identifier for the peer (alphanumeric, hyphens, underscores).
    pub id: String,
    /// Optional metadata to attach to the peer.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, serde_json::Value>>,
    /// Optional configuration for the peer.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub configuration: Option<HashMap<String, serde_json::Value>>,
}

/// Parameters for updating an existing peer.
#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize, bon::Builder)]
#[builder(on(String, into))]
pub struct PeerUpdate {
    /// Updated metadata. If provided, replaces the existing metadata entirely.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, serde_json::Value>>,
    /// Updated configuration. If provided, replaces the existing configuration entirely.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub configuration: Option<HashMap<String, serde_json::Value>>,
}

/// Filter parameters for listing peers.
#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize, bon::Builder)]
#[builder(on(String, into))]
pub struct PeerGet {
    /// Optional metadata-based filters.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub filters: Option<HashMap<String, serde_json::Value>>,
}

/// Configuration for peer card behavior.
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

/// Response from getting a peer card.
#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PeerCardResponse {
    /// The peer card content lines, or `None` if no card exists.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peer_card: Option<Vec<String>>,
}

/// Request body for setting a peer card.
#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize, bon::Builder)]
pub struct PeerCardSet {
    /// The peer card content lines to set.
    pub peer_card: Vec<String>,
}

/// Context for a peer, combining representation and peer card.
#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PeerContext {
    /// The ID of the observer peer.
    pub peer_id: String,
    /// The ID of the target peer being observed.
    pub target_id: String,
    /// Curated subset of the representation of the target from the observer's perspective.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub representation: Option<String>,
    /// The peer card for the target from the observer's perspective.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub peer_card: Option<Vec<String>>,
}

/// Parameters for getting a peer's representation.
#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize, bon::Builder)]
#[builder(on(String, into))]
pub struct PeerRepresentationGet {
    /// Optional session ID to scope the representation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
    /// Optional target peer ID to get the representation for.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target: Option<String>,
    /// Optional query to curate the representation around semantic search results.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub search_query: Option<String>,
    /// Number of semantic-search-retrieved conclusions to include.
    /// Only used if `search_query` is provided.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub search_top_k: Option<u32>,
    /// Maximum distance for semantically relevant conclusions.
    /// Only used if `search_query` is provided.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub search_max_distance: Option<f64>,
    /// Whether to include the most frequent conclusions.
    /// Only used if `search_query` is provided.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub include_most_frequent: Option<bool>,
    /// Maximum number of conclusions to include in the representation.
    /// Only used if `search_query` is provided.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_conclusions: Option<u32>,
}

/// A page of [`Peer`] results from a paginated list endpoint.
pub type PeerPage = super::pagination::Page<Peer>;
