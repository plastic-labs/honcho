use std::collections::HashMap;

use serde_json::Value;
use url::Url;

use crate::client::HonchoParams;
use crate::error::Result;
use crate::types::dream::QueueStatus;
use crate::types::message::MessageResponse;
use crate::types::peer::Peer as PeerResponse;
use crate::types::session::Session as SessionResponse;

use super::iter::collect_all_pages;
use super::runtime::block_on;
use super::Peer as BlockingPeer;
use super::Session as BlockingSession;

/// Synchronous wrapper around [`crate::Honcho`].
#[derive(Clone)]
pub struct Honcho {
    inner: crate::Honcho,
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
    pub fn peer(&self, id: impl Into<String>) -> Result<BlockingPeer> {
        block_on(self.inner.peer(id)).map(BlockingPeer::new)
    }

    /// Get or create a session by ID.
    pub fn session(&self, id: impl Into<String>) -> Result<BlockingSession> {
        block_on(self.inner.session(id)).map(BlockingSession::new)
    }

    /// Search messages across the workspace.
    pub fn search(&self, query: &str) -> Result<Vec<MessageResponse>> {
        block_on(self.inner.search(query))
    }

    /// Get queue processing status.
    pub fn queue_status(&self) -> Result<QueueStatus> {
        block_on(self.inner.queue_status())
    }

    /// Schedule a dream task for memory consolidation.
    pub fn schedule_dream(&self, observer: &str) -> Result<()> {
        block_on(self.inner.schedule_dream(observer))
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

    /// Fetch workspace configuration.
    pub fn get_configuration(&self) -> Result<HashMap<String, Value>> {
        block_on(self.inner.get_configuration())
    }

    /// Set workspace configuration.
    pub fn set_configuration(&self, configuration: HashMap<String, Value>) -> Result<()> {
        block_on(self.inner.set_configuration(configuration))
    }

    /// List all peers in the workspace, collecting across pages.
    pub fn peers(&self) -> Result<Vec<PeerResponse>> {
        block_on(async {
            let page = self.inner.peers().await?;
            Ok(collect_all_pages(page).await)
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
            Ok(collect_all_pages(page).await)
        })
    }

    /// List all sessions in the workspace, collecting across pages.
    pub fn sessions(&self) -> Result<Vec<SessionResponse>> {
        block_on(async {
            let page = self.inner.sessions().await?;
            Ok(collect_all_pages(page).await)
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
            Ok(collect_all_pages(page).await)
        })
    }

    /// List all workspace IDs, collecting across pages.
    #[allow(clippy::cast_possible_truncation)]
    pub fn workspaces(&self) -> Result<Vec<String>> {
        block_on(async {
            let mut page = self.inner.workspaces().await?;
            let mut all = Vec::with_capacity(page.total() as usize);
            let mut first_items = page.items();
            all.append(&mut first_items);
            while let Some(next) = page.next_page().await {
                let mut next_items = next.items();
                all.append(&mut next_items);
                page = next;
            }
            Ok(all)
        })
    }
}
