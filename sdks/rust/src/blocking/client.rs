use std::collections::HashMap;

use serde_json::Value;
use url::Url;

use crate::client::HonchoParams;
use crate::error::Result;
use crate::session::PeerSpec;
use crate::types::dream::QueueStatus;
use crate::types::peer::Peer as PeerResponse;
use crate::types::session::Session as SessionResponse;
use crate::types::workspace::WorkspaceConfiguration;

use super::Peer as BlockingPeer;
use super::Session as BlockingSession;
use super::iter::collect_all_pages;
use super::runtime::block_on;

/// Synchronous wrapper around [`crate::Honcho`].
#[derive(Clone)]
pub struct Honcho {
    inner: crate::Honcho,
}

impl std::fmt::Debug for Honcho {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Honcho")
            .field("workspace_id", &self.inner.workspace_id())
            .field("base_url", &self.inner.base_url().as_str())
            .finish()
    }
}

impl Honcho {
    /// Create a blocking client pointed at `base_url` for `workspace_id`.
    pub fn new(base_url: &str, workspace_id: &str) -> Result<Self> {
        let inner = crate::Honcho::new(base_url, workspace_id)?;
        Ok(Self { inner })
    }

    /// Returns a builder for [`HonchoParams`].
    pub fn builder() -> crate::client::HonchoParamsBuilder {
        crate::Honcho::builder()
    }

    /// Build from explicit params.
    pub fn from_params(params: HonchoParams) -> Result<Self> {
        let inner = crate::Honcho::from_params(params)?;
        Ok(Self { inner })
    }

    /// Eagerly ensure the workspace exists on the server.
    pub fn force_ensure(&self) -> Result<()> {
        block_on(self.inner.force_ensure())
    }

    /// Workspace ID this client is scoped to.
    #[must_use]
    pub fn workspace_id(&self) -> &str {
        self.inner.workspace_id()
    }

    /// Resolved base URL.
    #[must_use]
    pub fn base_url(&self) -> &Url {
        self.inner.base_url()
    }

    /// Get or create a peer by ID.
    pub fn peer(
        &self,
        id: impl Into<String>,
        metadata: Option<HashMap<String, Value>>,
        configuration: Option<HashMap<String, Value>>,
    ) -> Result<BlockingPeer> {
        block_on(self.inner.peer(id, metadata, configuration)).map(BlockingPeer::new)
    }

    /// Get or create a session by ID.
    pub fn session(
        &self,
        id: impl Into<String>,
        metadata: Option<HashMap<String, Value>>,
        peers: Option<Vec<PeerSpec>>,
        configuration: Option<crate::SessionConfiguration>,
    ) -> Result<BlockingSession> {
        block_on(self.inner.session(id, metadata, peers, configuration)).map(BlockingSession::new)
    }

    /// Search messages across the workspace.
    pub fn search(
        &self,
        query: &str,
        limit: Option<u32>,
        filters: Option<HashMap<String, Value>>,
    ) -> Result<Vec<crate::Message>> {
        block_on(self.inner.search(query, limit, filters))
    }

    /// Refresh workspace state.
    pub fn refresh(&self) -> Result<()> {
        block_on(self.inner.refresh())
    }

    /// Get queue processing status.
    pub fn queue_status(
        &self,
        observer_id: Option<&str>,
        sender_id: Option<&str>,
        session_id: Option<&str>,
    ) -> Result<QueueStatus> {
        block_on(self.inner.queue_status(observer_id, sender_id, session_id))
    }

    /// Schedule a dream task for memory consolidation.
    pub fn schedule_dream(
        &self,
        observer: &str,
        session_id: Option<&str>,
        observed_peer: Option<&str>,
    ) -> Result<()> {
        block_on(
            self.inner
                .schedule_dream(observer, session_id, observed_peer),
        )
    }

    /// Delete a workspace by ID.
    pub fn delete_workspace(&self, id: &str) -> Result<()> {
        block_on(self.inner.delete_workspace(id))
    }

    /// Fetch workspace metadata.
    pub fn get_metadata(&self) -> Result<HashMap<String, Value>> {
        block_on(self.inner.get_metadata())
    }

    /// Set workspace metadata.
    pub fn set_metadata(&self, metadata: HashMap<String, Value>) -> Result<()> {
        block_on(self.inner.set_metadata(metadata))
    }

    /// Fetch workspace configuration as a typed [`WorkspaceConfiguration`].
    pub fn get_configuration(&self) -> Result<WorkspaceConfiguration> {
        block_on(self.inner.get_configuration())
    }

    /// Set workspace configuration from a typed [`WorkspaceConfiguration`].
    pub fn set_configuration(&self, config: &WorkspaceConfiguration) -> Result<()> {
        block_on(self.inner.set_configuration(config))
    }

    /// Fetch workspace configuration as a raw JSON map.
    pub fn get_configuration_raw(&self) -> Result<HashMap<String, Value>> {
        block_on(self.inner.get_configuration_raw())
    }

    /// Set workspace configuration from a raw JSON map.
    pub fn set_configuration_raw(&self, configuration: HashMap<String, Value>) -> Result<()> {
        block_on(self.inner.set_configuration_raw(configuration))
    }

    /// List all peers in the workspace, collecting across pages.
    pub fn peers(&self) -> Result<Vec<PeerResponse>> {
        block_on(async {
            let page = self.inner.peers().await?;
            collect_all_pages(page).await
        })
    }

    /// List peers with filters, collecting across pages.
    pub fn peers_with_filters(
        &self,
        filters: HashMap<String, Value>,
        page: u64,
        size: u64,
        reverse: bool,
    ) -> Result<Vec<PeerResponse>> {
        block_on(async {
            let page = self
                .inner
                .peers_with_filters(filters, page, size, reverse)
                .await?;
            collect_all_pages(page).await
        })
    }

    /// List all sessions in the workspace, collecting across pages.
    pub fn sessions(&self) -> Result<Vec<SessionResponse>> {
        block_on(async {
            let page = self.inner.sessions().await?;
            collect_all_pages(page).await
        })
    }

    /// List sessions with filters, collecting across pages.
    pub fn sessions_with_filters(
        &self,
        filters: HashMap<String, Value>,
        page: u64,
        size: u64,
        reverse: bool,
    ) -> Result<Vec<SessionResponse>> {
        block_on(async {
            let page = self
                .inner
                .sessions_with_filters(filters, page, size, reverse)
                .await?;
            collect_all_pages(page).await
        })
    }

    /// List all workspace IDs, collecting across pages.
    pub fn workspaces(&self) -> Result<Vec<String>> {
        block_on(async {
            let page = self.inner.workspaces().await?;
            collect_all_pages(page).await
        })
    }
}
