//! Workspace domain types.

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

pub use super::common::{
    DreamConfiguration, PeerCardConfiguration, ReasoningConfiguration, SummaryConfiguration,
};

/// A workspace resource.
#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Workspace {
    /// Unique identifier for the workspace.
    pub id: String,
    /// Arbitrary metadata attached to the workspace.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub metadata: HashMap<String, serde_json::Value>,
    /// Workspace-level configuration overrides.
    #[serde(default)]
    pub configuration: WorkspaceConfiguration,
    /// When the workspace was created.
    pub created_at: DateTime<Utc>,
}

/// The set of options that can be in a workspace-level configuration dictionary.
///
/// All fields are optional. Session-level configuration overrides workspace-level
/// configuration, which overrides global configuration.
#[non_exhaustive]
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct WorkspaceConfiguration {
    /// Configuration for reasoning functionality.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning: Option<ReasoningConfiguration>,
    /// Configuration for peer card functionality.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub peer_card: Option<PeerCardConfiguration>,
    /// Configuration for summary functionality.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub summary: Option<SummaryConfiguration>,
    /// Configuration for dream functionality.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dream: Option<DreamConfiguration>,
}

/// Request body for creating a workspace.
#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, bon::Builder)]
#[builder(on(String, into))]
#[builder(finish_fn = build)]
pub struct WorkspaceCreate {
    /// Unique identifier for the new workspace (1-100 chars, `[a-zA-Z0-9_-]+`).
    pub id: String,
    /// Arbitrary metadata. Defaults to `{}`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub metadata: Option<HashMap<String, serde_json::Value>>,
    /// Workspace-level configuration overrides.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub configuration: Option<WorkspaceConfiguration>,
}

/// Request body for updating a workspace.
#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, bon::Builder)]
#[builder(on(String, into))]
#[builder(finish_fn = build)]
pub struct WorkspaceUpdate {
    /// Updated metadata. `null` means "leave unchanged".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<Option<HashMap<String, serde_json::Value>>>,
    /// Updated configuration. `null` means "leave unchanged".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub configuration: Option<Option<WorkspaceConfiguration>>,
}

/// Request body for listing/getting workspaces.
#[non_exhaustive]
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, bon::Builder)]
#[builder(on(String, into))]
#[builder(finish_fn = build)]
pub struct WorkspaceGet {
    /// Optional metadata filters.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filters: Option<HashMap<String, serde_json::Value>>,
}

/// Request body for setting workspace metadata.
#[non_exhaustive]
#[derive(Debug, Clone, Serialize)]
pub struct WorkspaceMetadataSet {
    /// Metadata to set.
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Request body for setting workspace configuration.
#[non_exhaustive]
#[derive(Debug, Clone, Serialize)]
pub struct WorkspaceConfigurationSet {
    /// Configuration to set.
    pub configuration: serde_json::Value,
}

/// Request body for workspace search.
#[non_exhaustive]
#[derive(Debug, Clone, Serialize)]
pub struct WorkspaceSearchRequest {
    /// Search query string.
    pub query: String,
    /// Maximum number of results.
    pub limit: u32,
    /// Optional metadata-based filters.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub filters: Option<HashMap<String, serde_json::Value>>,
}

/// A page of workspace results.
pub type WorkspacePage = super::pagination::Page<Workspace>;
