//! Peer wrapper — construction, metadata, chat, representation, context, search, and card.

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use serde::Deserialize;
use serde_json::Value;

use crate::error::{HonchoError, Result};
use crate::http::client::HttpClient;
use crate::http::routes;
use crate::types::dialectic::DialecticOptions;
use crate::types::dialectic::RepresentationResponse;
use crate::types::message::{Message, MessageCreate, MessageSearchOptions};
use crate::types::pagination::{self, Page};
use crate::types::peer::Peer as PeerResponse;
use crate::types::peer::{PeerCardResponse, PeerCardSet, PeerContext};
use crate::types::session::Session;

pub(crate) struct PeerInner {
    http: HttpClient,
    workspace_id: String,
    id: String,
    metadata: RwLock<Option<HashMap<String, Value>>>,
    configuration: RwLock<Option<HashMap<String, Value>>>,
}

/// A peer in a Honcho workspace.
///
/// Wraps the API response and provides methods for metadata, configuration,
/// chat, representation, context, search, and cards.
#[derive(Clone)]
pub struct Peer {
    inner: Arc<PeerInner>,
}

#[derive(Deserialize)]
struct ChatResponse {
    #[serde(default)]
    content: Option<String>,
}

impl Peer {
    pub(crate) fn from_parts(http: HttpClient, workspace_id: String, resp: PeerResponse) -> Self {
        Self {
            inner: Arc::new(PeerInner {
                http,
                workspace_id,
                id: resp.id,
                metadata: RwLock::new(Some(resp.metadata)),
                configuration: RwLock::new(Some(resp.configuration)),
            }),
        }
    }

    pub(crate) fn from_response(honcho: &crate::Honcho, resp: PeerResponse) -> Self {
        Self::from_parts(
            honcho.http().clone(),
            honcho.workspace_id().to_owned(),
            resp,
        )
    }

    // ── F5.1: Construction + Metadata ──────────────────────────────────

    /// The peer's unique identifier.
    #[must_use]
    pub fn id(&self) -> &str {
        &self.inner.id
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

    /// Refresh the peer's cached metadata and configuration from the server.
    ///
    /// POSTs to the peers get-or-create endpoint with `{"id": peer_id}`.
    pub async fn refresh(&self) -> Result<()> {
        let body = serde_json::json!({"id": self.inner.id});
        let resp: PeerResponse = self
            .inner
            .http
            .post(&routes::peers(&self.inner.workspace_id), Some(&body), &[])
            .await?;
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

    /// Fetch and return the peer's metadata, updating the cache.
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

    /// Set the peer's metadata on the server and update the cache.
    pub async fn set_metadata(&self, metadata: HashMap<String, Value>) -> Result<()> {
        let body = serde_json::json!({"metadata": metadata});
        let resp: PeerResponse = self
            .inner
            .http
            .put(
                &routes::peer(&self.inner.workspace_id, &self.inner.id),
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

    /// Fetch and return the peer's configuration, updating the cache.
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

    /// Set the peer's configuration on the server and update the cache.
    pub async fn set_configuration(&self, configuration: HashMap<String, Value>) -> Result<()> {
        let body = serde_json::json!({"configuration": configuration});
        let resp: PeerResponse = self
            .inner
            .http
            .put(
                &routes::peer(&self.inner.workspace_id, &self.inner.id),
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

    /// Patch-update the peer's metadata on the server and update the cache.
    pub async fn update(&self, metadata: HashMap<String, Value>) -> Result<()> {
        let body = serde_json::json!({"metadata": metadata});
        let resp: PeerResponse = self
            .inner
            .http
            .patch(
                &routes::peer(&self.inner.workspace_id, &self.inner.id),
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

    // ── F5.2: Chat (non-streaming) ─────────────────────────────────────

    /// Send a simple non-streaming dialectic chat query.
    ///
    /// Returns `Ok(None)` when the server response has no content.
    /// Returns `Err(HonchoError::Configuration)` when `query` is empty.
    pub async fn chat(&self, query: &str) -> Result<Option<String>> {
        if query.is_empty() {
            return Err(HonchoError::Configuration(
                "query must not be empty".to_owned(),
            ));
        }
        let body = serde_json::json!({
            "query": query,
            "stream": false,
        });
        let resp: ChatResponse = self
            .inner
            .http
            .post(
                &routes::peer_chat(&self.inner.workspace_id, &self.inner.id),
                Some(&body),
                &[],
            )
            .await?;
        match resp.content {
            Some(c) if !c.is_empty() => Ok(Some(c)),
            _ => Ok(None),
        }
    }

    /// Send a dialectic chat query with full options (session, target, reasoning level).
    ///
    /// Returns `Ok(None)` when the server response has no content.
    /// Returns `Err(HonchoError::Configuration)` when the query in options is empty.
    pub async fn chat_with_options(&self, options: &DialecticOptions) -> Result<Option<String>> {
        if options.query.is_empty() {
            return Err(HonchoError::Configuration(
                "query must not be empty".to_owned(),
            ));
        }
        let resp: ChatResponse = self
            .inner
            .http
            .post(
                &routes::peer_chat(&self.inner.workspace_id, &self.inner.id),
                Some(options),
                &[],
            )
            .await?;
        match resp.content {
            Some(c) if !c.is_empty() => Ok(Some(c)),
            _ => Ok(None),
        }
    }

    // ── Representation ─────────────────────────────────────────────────

    /// Get the peer's representation (default parameters).
    pub async fn representation(&self) -> Result<String> {
        let route = routes::peer_representation(&self.inner.workspace_id, &self.inner.id);
        let body = serde_json::json!({});
        let resp: RepresentationResponse = self.inner.http.post(&route, Some(&body), &[]).await?;
        Ok(resp.representation)
    }

    /// Get a representation builder for fine-grained control over parameters.
    #[must_use]
    pub fn representation_builder(&self) -> RepresentationBuilder {
        RepresentationBuilder {
            http: self.inner.http.clone(),
            workspace_id: self.inner.workspace_id.clone(),
            peer_id: self.inner.id.clone(),
            session_id: None,
            target: None,
            search_query: None,
            search_top_k: None,
            search_max_distance: None,
            include_most_frequent: None,
            max_conclusions: None,
        }
    }

    // ── Context ────────────────────────────────────────────────────────

    /// Get the peer's context.
    pub async fn context(&self) -> Result<PeerContext> {
        let route = routes::peer_context(&self.inner.workspace_id, &self.inner.id);
        self.inner.http.get(&route, &[]).await
    }

    /// Get the peer's context scoped to a target peer.
    pub async fn context_with_target(&self, target: &str) -> Result<PeerContext> {
        let route = routes::peer_context(&self.inner.workspace_id, &self.inner.id);
        self.inner.http.get(&route, &[("target", target)]).await
    }

    // ── Sessions ───────────────────────────────────────────────────────

    /// List sessions for this peer (paginated).
    pub async fn sessions(&self) -> Result<Page<Session>> {
        let route = routes::peer_sessions_list(&self.inner.workspace_id, &self.inner.id);
        pagination::paginate_post(&self.inner.http, &route, None, 1, 50, false).await
    }

    // ── Search ─────────────────────────────────────────────────────────

    /// Search messages for this peer.
    pub async fn search(&self, query: &str) -> Result<Vec<Message>> {
        if query.is_empty() {
            return Err(HonchoError::Configuration(
                "query must not be empty".to_string(),
            ));
        }
        let body = MessageSearchOptions {
            query: query.to_string(),
            filters: None,
            limit: 10,
        };
        self.inner
            .http
            .post(
                &routes::peer_search(&self.inner.workspace_id, &self.inner.id),
                Some(&body),
                &[],
            )
            .await
    }

    // ── Card ───────────────────────────────────────────────────────────

    /// Get this peer's card.
    pub async fn get_card(&self) -> Result<Option<Vec<String>>> {
        let resp: PeerCardResponse = self
            .inner
            .http
            .get(
                &routes::peer_card(&self.inner.workspace_id, &self.inner.id),
                &[],
            )
            .await?;
        Ok(resp.peer_card)
    }

    /// Get this peer's card scoped to a target peer.
    pub async fn get_card_with_target(&self, target: &str) -> Result<Option<Vec<String>>> {
        let resp: PeerCardResponse = self
            .inner
            .http
            .get(
                &routes::peer_card(&self.inner.workspace_id, &self.inner.id),
                &[("target", target)],
            )
            .await?;
        Ok(resp.peer_card)
    }

    /// Set this peer's card.
    pub async fn set_card(&self, card: Vec<String>) -> Result<Option<Vec<String>>> {
        let body = PeerCardSet::builder().peer_card(card).build();
        let resp: PeerCardResponse = self
            .inner
            .http
            .put(
                &routes::peer_card(&self.inner.workspace_id, &self.inner.id),
                Some(&body),
                &[],
            )
            .await?;
        Ok(resp.peer_card)
    }

    /// Set this peer's card scoped to a target peer.
    pub async fn set_card_with_target(
        &self,
        card: Vec<String>,
        target: &str,
    ) -> Result<Option<Vec<String>>> {
        let body = PeerCardSet::builder().peer_card(card).build();
        let resp: PeerCardResponse = self
            .inner
            .http
            .put(
                &routes::peer_card(&self.inner.workspace_id, &self.inner.id),
                Some(&body),
                &[("target", target)],
            )
            .await?;
        Ok(resp.peer_card)
    }

    /// Get this peer's card (deprecated: use [`Peer::get_card`] instead).
    #[deprecated(since = "0.1.0", note = "use get_card instead")]
    #[allow(clippy::missing_panics_doc)]
    pub async fn card(&self) -> Result<Option<Vec<String>>> {
        self.get_card().await
    }

    // ── F5.8: Message builder (sync, no API call) ─────────────────────

    /// Create a message builder for this peer.
    ///
    /// This is purely synchronous and does NOT send any API request.
    /// Call `.build()` to get a `MessageCreate` that can be passed to
    /// `Session::add_messages()`.
    #[must_use]
    pub fn message(&self, content: impl Into<String>) -> MessageBuilder {
        MessageBuilder {
            peer_id: self.inner.id.clone(),
            content: content.into(),
            metadata: None,
            configuration: None,
            created_at: None,
        }
    }
}

/// Builder for fine-grained representation requests.
pub struct RepresentationBuilder {
    http: HttpClient,
    workspace_id: String,
    peer_id: String,
    session_id: Option<String>,
    target: Option<String>,
    search_query: Option<String>,
    search_top_k: Option<u32>,
    search_max_distance: Option<f64>,
    include_most_frequent: Option<bool>,
    max_conclusions: Option<u32>,
}

impl RepresentationBuilder {
    /// Scope the representation to a session.
    #[must_use]
    pub fn session_id(mut self, val: impl Into<String>) -> Self {
        self.session_id = Some(val.into());
        self
    }

    /// Get the representation for a specific target peer.
    #[must_use]
    pub fn target(mut self, val: impl Into<String>) -> Self {
        self.target = Some(val.into());
        self
    }

    /// Semantic search query to curate the representation.
    #[must_use]
    pub fn search_query(mut self, val: impl Into<String>) -> Self {
        self.search_query = Some(val.into());
        self
    }

    /// Number of semantic-search-retrieved conclusions (1–100).
    #[must_use]
    pub fn search_top_k(mut self, val: u32) -> Self {
        self.search_top_k = Some(val);
        self
    }

    /// Maximum distance for semantically relevant conclusions (0.0–1.0).
    #[must_use]
    pub fn search_max_distance(mut self, val: f64) -> Self {
        self.search_max_distance = Some(val);
        self
    }

    /// Whether to include the most frequent conclusions.
    #[must_use]
    pub fn include_most_frequent(mut self, val: bool) -> Self {
        self.include_most_frequent = Some(val);
        self
    }

    /// Maximum number of conclusions to include (1–100).
    #[must_use]
    pub fn max_conclusions(mut self, val: u32) -> Self {
        self.max_conclusions = Some(val);
        self
    }

    /// Send the representation request with the configured parameters.
    ///
    /// # Errors
    ///
    /// Returns `HonchoError::Configuration` if `search_top_k`, `search_max_distance`,
    /// or `max_conclusions` are out of range.
    pub async fn send(self) -> Result<String> {
        if let Some(k) = self.search_top_k {
            if !(1..=100).contains(&k) {
                return Err(HonchoError::Configuration(format!(
                    "search_top_k must be between 1 and 100, got {k}"
                )));
            }
        }
        if let Some(d) = self.search_max_distance {
            if !(0.0..=1.0).contains(&d) {
                return Err(HonchoError::Configuration(format!(
                    "search_max_distance must be between 0.0 and 1.0, got {d}"
                )));
            }
        }
        if let Some(c) = self.max_conclusions {
            if !(1..=100).contains(&c) {
                return Err(HonchoError::Configuration(format!(
                    "max_conclusions must be between 1 and 100, got {c}"
                )));
            }
        }

        let params = serde_json::json!({
            "session_id": self.session_id,
            "target": self.target,
            "search_query": self.search_query,
            "search_top_k": self.search_top_k,
            "search_max_distance": self.search_max_distance,
            "include_most_frequent": self.include_most_frequent,
            "max_conclusions": self.max_conclusions,
        });

        let route = routes::peer_representation(&self.workspace_id, &self.peer_id);
        let resp: RepresentationResponse = self.http.post(&route, Some(&params), &[]).await?;
        Ok(resp.representation)
    }
}

/// Synchronous builder for [`MessageCreate`] params.
///
/// Created via [`Peer::message()`]. Does **not** send any API request.
pub struct MessageBuilder {
    peer_id: String,
    content: String,
    metadata: Option<HashMap<String, Value>>,
    configuration: Option<crate::types::message::MessageConfiguration>,
    created_at: Option<chrono::DateTime<chrono::Utc>>,
}

impl MessageBuilder {
    /// Attach metadata to the message.
    #[must_use]
    pub fn metadata(mut self, val: HashMap<String, Value>) -> Self {
        self.metadata = Some(val);
        self
    }

    /// Attach message-level configuration.
    #[must_use]
    pub fn configuration(mut self, val: crate::types::message::MessageConfiguration) -> Self {
        self.configuration = Some(val);
        self
    }

    /// Override the creation timestamp.
    #[must_use]
    pub fn created_at(mut self, val: chrono::DateTime<chrono::Utc>) -> Self {
        self.created_at = Some(val);
        self
    }

    /// Build the message params.
    pub fn build(self) -> Result<MessageCreate> {
        Ok(MessageCreate {
            peer_id: self.peer_id,
            content: self.content,
            metadata: self.metadata,
            configuration: self.configuration,
            created_at: self.created_at,
        })
    }
}
