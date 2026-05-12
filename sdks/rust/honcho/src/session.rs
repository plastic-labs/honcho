//! Session wrapper — construction, metadata, peer management, per-peer config.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};

use reqwest::Method;
use serde::Deserialize;
use serde_json::Value;

use crate::error::Result;
use crate::http::client::HttpClient;
use crate::http::routes;
use crate::types::session::Session as SessionResponse;
use crate::types::session::SessionPeerConfig;

pub(crate) struct SessionInner {
    http: HttpClient,
    workspace_id: String,
    id: String,
    is_active: AtomicBool,
    metadata: RwLock<Option<HashMap<String, Value>>>,
    configuration: RwLock<Option<HashMap<String, Value>>>,
}

/// A session in a Honcho workspace.
///
/// Wraps the API response and provides methods for metadata, configuration,
/// peer management, messages, and more.
#[derive(Clone)]
pub struct Session {
    inner: Arc<SessionInner>,
}

/// Specification for adding/setting peers on a session.
///
/// Use [`PeerSpec::Id`] for a bare peer ID or [`PeerSpec::WithConfig`] to
/// include per-peer observation settings.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum PeerSpec {
    /// A peer identified by ID with no explicit config.
    Id(String),
    /// A peer identified by ID with per-session configuration.
    WithConfig(String, SessionPeerConfig),
}

impl PeerSpec {
    fn id(&self) -> &str {
        match self {
            Self::Id(id) | Self::WithConfig(id, _) => id,
        }
    }
}

impl From<&str> for PeerSpec {
    fn from(s: &str) -> Self {
        Self::Id(s.to_owned())
    }
}

impl From<String> for PeerSpec {
    fn from(s: String) -> Self {
        Self::Id(s)
    }
}

impl From<&crate::Peer> for PeerSpec {
    fn from(p: &crate::Peer) -> Self {
        Self::Id(p.id().to_owned())
    }
}

impl From<(String, SessionPeerConfig)> for PeerSpec {
    fn from((id, cfg): (String, SessionPeerConfig)) -> Self {
        Self::WithConfig(id, cfg)
    }
}

impl From<(&str, SessionPeerConfig)> for PeerSpec {
    fn from((id, cfg): (&str, SessionPeerConfig)) -> Self {
        Self::WithConfig(id.to_owned(), cfg)
    }
}

#[derive(Deserialize)]
struct PeersPageResponse {
    items: Vec<crate::types::peer::Peer>,
    #[serde(default)]
    #[allow(dead_code)]
    total: u64,
    #[serde(default)]
    #[allow(dead_code)]
    pages: u64,
}

impl Session {
    pub(crate) fn from_parts(
        http: HttpClient,
        workspace_id: String,
        resp: SessionResponse,
    ) -> Self {
        Self {
            inner: Arc::new(SessionInner {
                http,
                workspace_id,
                id: resp.id,
                is_active: AtomicBool::new(resp.is_active),
                metadata: RwLock::new(Some(resp.metadata)),
                configuration: RwLock::new(Some(resp.configuration)),
            }),
        }
    }

    pub(crate) fn from_response(honcho: &crate::Honcho, resp: SessionResponse) -> Self {
        Self::from_parts(
            honcho.http().clone(),
            honcho.workspace_id().to_owned(),
            resp,
        )
    }

    /// The session's unique identifier.
    #[must_use]
    pub fn id(&self) -> &str {
        &self.inner.id
    }

    /// Whether the session is currently active.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.inner.is_active.load(Ordering::Relaxed)
    }

    /// Cached metadata from the last API response.
    #[must_use]
    pub fn metadata(&self) -> Option<HashMap<String, Value>> {
        self.inner
            .metadata
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// Cached configuration from the last API response.
    #[must_use]
    pub fn configuration(&self) -> Option<HashMap<String, Value>> {
        self.inner
            .configuration
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    // ── F6.1: Refresh / Metadata / Configuration CRUD ──────────────────

    /// Refresh the session's cached metadata and configuration from the server.
    pub async fn refresh(&self) -> Result<()> {
        let body = serde_json::json!({"id": self.inner.id});
        let resp: SessionResponse = self
            .inner
            .http
            .post(
                &routes::sessions(&self.inner.workspace_id),
                Some(&body),
                &[],
            )
            .await?;
        self.inner
            .is_active
            .store(resp.is_active, Ordering::Relaxed);
        *self
            .inner
            .metadata
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(resp.metadata);
        *self
            .inner
            .configuration
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(resp.configuration);
        Ok(())
    }

    /// Fetch and return the session's metadata, updating the cache.
    pub async fn get_metadata(&self) -> Result<HashMap<String, Value>> {
        self.refresh().await?;
        Ok(self
            .inner
            .metadata
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
            .unwrap_or_default())
    }

    /// Set session metadata on the server and update the cache.
    pub async fn set_metadata(&self, metadata: HashMap<String, Value>) -> Result<()> {
        let body = serde_json::json!({"metadata": metadata});
        let resp: SessionResponse = self
            .inner
            .http
            .put(
                &routes::session(&self.inner.workspace_id, &self.inner.id),
                Some(&body),
                &[],
            )
            .await?;
        *self
            .inner
            .metadata
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(resp.metadata);
        Ok(())
    }

    /// Fetch and return session configuration, updating the cache.
    pub async fn get_configuration(&self) -> Result<HashMap<String, Value>> {
        self.refresh().await?;
        Ok(self
            .inner
            .configuration
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
            .unwrap_or_default())
    }

    /// Set session configuration on the server and update the cache.
    pub async fn set_configuration(&self, configuration: HashMap<String, Value>) -> Result<()> {
        let body = serde_json::json!({"configuration": configuration});
        let resp: SessionResponse = self
            .inner
            .http
            .put(
                &routes::session(&self.inner.workspace_id, &self.inner.id),
                Some(&body),
                &[],
            )
            .await?;
        *self
            .inner
            .configuration
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(resp.configuration);
        Ok(())
    }

    // ── F6.2: Peer Management ──────────────────────────────────────────

    /// Add a single peer to this session.
    pub async fn add_peer(&self, id: impl Into<String>) -> Result<()> {
        self.add_peers(std::iter::once(PeerSpec::Id(id.into())))
            .await
    }

    /// Add multiple peers to this session.
    pub async fn add_peers(
        &self,
        specs: impl IntoIterator<Item = impl Into<PeerSpec>>,
    ) -> Result<()> {
        let peers_map = normalize_peers(specs);
        let body = serde_json::json!({"peers": peers_map});
        let route = routes::session_peers(&self.inner.workspace_id, &self.inner.id);
        self.inner.http.post(&route, Some(&body), &[]).await
    }

    /// Set the complete peer list for this session (replaces existing).
    pub async fn set_peers(
        &self,
        specs: impl IntoIterator<Item = impl Into<PeerSpec>>,
    ) -> Result<()> {
        let peers_map = normalize_peers(specs);
        let body = serde_json::json!({"peers": peers_map});
        let route = routes::session_peers(&self.inner.workspace_id, &self.inner.id);
        self.inner.http.put(&route, Some(&body), &[]).await
    }

    /// Remove peers from this session.
    pub async fn remove_peers(
        &self,
        ids: impl IntoIterator<Item = impl Into<String>>,
    ) -> Result<()> {
        let id_list: Vec<String> = ids.into_iter().map(Into::into).collect();
        let route = routes::session_peers(&self.inner.workspace_id, &self.inner.id);
        self.inner
            .http
            .request::<_, ()>(Method::DELETE, &route, Some(&id_list), &[])
            .await
    }

    /// List peers in this session.
    pub async fn peers(&self) -> Result<Vec<crate::Peer>> {
        let route = routes::session_peers(&self.inner.workspace_id, &self.inner.id);
        let page: PeersPageResponse = self.inner.http.get(&route, &[]).await?;
        Ok(page
            .items
            .into_iter()
            .map(|resp| {
                crate::Peer::from_parts(
                    self.inner.http.clone(),
                    self.inner.workspace_id.clone(),
                    resp,
                )
            })
            .collect())
    }

    // ── F6.3: Per-peer configuration ───────────────────────────────────

    /// Get per-peer configuration for a specific peer in this session.
    pub async fn get_peer_configuration(&self, peer_id: &str) -> Result<SessionPeerConfig> {
        let route = routes::session_peer_config(&self.inner.workspace_id, &self.inner.id, peer_id);
        self.inner.http.get(&route, &[]).await
    }

    /// Set per-peer configuration for a specific peer in this session.
    pub async fn set_peer_configuration(
        &self,
        peer_id: &str,
        config: &SessionPeerConfig,
    ) -> Result<()> {
        let route = routes::session_peer_config(&self.inner.workspace_id, &self.inner.id, peer_id);
        self.inner.http.put(&route, Some(config), &[]).await
    }

    // ── F6.4: Messages ─────────────────────────────────────────────────

    /// Add messages to this session.
    ///
    /// If more than 100 messages are provided, they are automatically chunked
    /// into batches of 100 and sent as separate requests. On chunk failure the
    /// already-sent messages are **not** rolled back (non-atomic).
    pub async fn add_messages(
        &self,
        messages: Vec<crate::types::message::MessageCreate>,
    ) -> Result<Vec<crate::types::message::Message>> {
        use crate::types::message::Message;

        if messages.is_empty() {
            return Ok(Vec::new());
        }

        let route = routes::messages(&self.inner.workspace_id, &self.inner.id);

        if messages.len() <= 100 {
            let body = serde_json::json!({"messages": messages});
            return self
                .inner
                .http
                .post::<_, Vec<Message>>(&route, Some(&body), &[])
                .await;
        }

        let mut all = Vec::with_capacity(messages.len());
        for chunk in messages.chunks(100) {
            let body = serde_json::json!({"messages": chunk});
            let batch: Vec<Message> = self.inner.http.post(&route, Some(&body), &[]).await?;
            all.extend(batch);
        }
        Ok(all)
    }

    /// List messages in this session (paginated).
    pub async fn messages(
        &self,
    ) -> Result<crate::types::pagination::Page<crate::types::message::Message>> {
        let route = routes::messages_list(&self.inner.workspace_id, &self.inner.id);
        crate::types::pagination::paginate_post(&self.inner.http, &route, None, 1, 50, false).await
    }

    // ── F6.5: Delete, clone, get/update message ────────────────────────

    /// Delete this session.
    pub async fn delete(&self) -> Result<()> {
        self.inner
            .http
            .delete(
                &routes::session(&self.inner.workspace_id, &self.inner.id),
                &[],
            )
            .await
    }

    /// Clone this session, returning a new `Session`.
    pub async fn clone_session(&self) -> Result<Session> {
        let route = routes::session_clone(&self.inner.workspace_id, &self.inner.id);
        let resp: SessionResponse = self.inner.http.post(&route, None::<&Value>, &[]).await?;
        Ok(Self::from_parts(
            self.inner.http.clone(),
            self.inner.workspace_id.clone(),
            resp,
        ))
    }

    /// Clone this session up to (and including) the given message.
    pub async fn clone_session_with_message(&self, message_id: &str) -> Result<Session> {
        let route = routes::session_clone(&self.inner.workspace_id, &self.inner.id);
        let resp: SessionResponse = self
            .inner
            .http
            .post(&route, None::<&Value>, &[("message_id", message_id)])
            .await?;
        Ok(Self::from_parts(
            self.inner.http.clone(),
            self.inner.workspace_id.clone(),
            resp,
        ))
    }

    /// Get a single message by ID.
    pub async fn get_message(&self, id: &str) -> Result<crate::types::message::Message> {
        let route = routes::message(&self.inner.workspace_id, &self.inner.id, id);
        self.inner.http.get(&route, &[]).await
    }

    /// Update a message's metadata.
    pub async fn update_message(
        &self,
        id: &str,
        metadata: HashMap<String, Value>,
    ) -> Result<crate::types::message::Message> {
        let route = routes::message(&self.inner.workspace_id, &self.inner.id, id);
        let body = serde_json::json!({"metadata": metadata});
        self.inner.http.put(&route, Some(&body), &[]).await
    }

    // ── F6.6: Context ───────────────────────────────────────────────────

    /// Get the session context with default parameters.
    ///
    /// Fetches messages, summary, peer representation, and peer card for this session.
    pub async fn context(&self) -> Result<crate::types::session::SessionContext> {
        let route = routes::session_context(&self.inner.workspace_id, &self.inner.id);
        self.inner
            .http
            .get(
                &route,
                &[("summary", "true"), ("limit_to_session", "false")],
            )
            .await
    }

    // ── F6.8: Summaries ─────────────────────────────────────────────────

    /// Get available summaries for this session.
    ///
    /// Returns both short and long summaries if they are available.
    /// Summaries are created asynchronously as messages are added.
    pub async fn summaries(&self) -> Result<crate::types::session::SessionSummaries> {
        let route = routes::session_summaries(&self.inner.workspace_id, &self.inner.id);
        self.inner.http.get(&route, &[]).await
    }

    // ── F6.9: Search, representation, queue_status ──────────────────────

    /// Search messages within this session.
    ///
    /// Returns `Err(HonchoError::Configuration)` when `query` is empty.
    pub async fn search(&self, query: &str) -> Result<Vec<crate::types::message::Message>> {
        if query.is_empty() {
            return Err(crate::error::HonchoError::Configuration(
                "query must not be empty".to_string(),
            ));
        }
        let body = crate::types::message::MessageSearchOptions {
            query: query.to_string(),
            filters: None,
            limit: 10,
        };
        let route = routes::session_search(&self.inner.workspace_id, &self.inner.id);
        self.inner.http.post(&route, Some(&body), &[]).await
    }

    /// Get a peer's representation scoped to this session.
    ///
    /// Uses the peer representation endpoint with `session_id` filter.
    pub async fn representation(&self, peer_id: &str) -> Result<String> {
        let route = routes::peer_representation(&self.inner.workspace_id, peer_id);
        let body = serde_json::json!({
            "session_id": self.inner.id,
        });
        let resp: crate::types::dialectic::RepresentationResponse =
            self.inner.http.post(&route, Some(&body), &[]).await?;
        Ok(resp.representation)
    }

    /// Get the processing queue status for this session.
    pub async fn queue_status(&self) -> Result<crate::types::dream::QueueStatus> {
        let route = routes::workspace_queue_status(&self.inner.workspace_id);
        self.inner
            .http
            .get(&route, &[("session_id", &self.inner.id)])
            .await
    }
}

fn normalize_peers(specs: impl IntoIterator<Item = impl Into<PeerSpec>>) -> serde_json::Value {
    let map: serde_json::Map<String, Value> = specs
        .into_iter()
        .map(|s| {
            let spec = s.into();
            let val = match &spec {
                PeerSpec::Id(_) => serde_json::json!({}),
                PeerSpec::WithConfig(_, cfg) => {
                    serde_json::to_value(cfg).unwrap_or_else(|_| serde_json::json!({}))
                }
            };
            (spec.id().to_owned(), val)
        })
        .collect();
    Value::Object(map)
}
