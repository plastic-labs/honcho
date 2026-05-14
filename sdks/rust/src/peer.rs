//! Peer wrapper — construction, metadata, chat, representation, context, search, and card.

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::{Arc, RwLock};

use futures_util::Stream;
use reqwest::Method;
use serde::Deserialize;
use serde_json::Value;

use crate::conclusion::ConclusionScope;
use crate::dialectic_stream::DialecticStream;
use crate::error::{HonchoError, Result};
use crate::http::client::HttpClient;
use crate::http::routes;
use crate::http::sse::parse_sse_stream;
use crate::types::dialectic::RepresentationResponse;
use crate::types::dialectic::{DialecticOptions, ReasoningLevel};
use crate::types::message::{MessageCreate, MessageResponse, MessageSearchOptions};
use crate::types::pagination::{self, Page};
use crate::types::peer::Peer as PeerResponse;
use crate::types::peer::{PeerCardResponse, PeerCardSet, PeerConfig, PeerContext};
use crate::types::session::{Session, SessionListOptions};

pub(crate) struct PeerInner {
    http: HttpClient,
    workspace_id: String,
    id: String,
    metadata: RwLock<Option<HashMap<String, Value>>>,
    configuration: RwLock<Option<PeerConfig>>,
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

impl std::fmt::Debug for Peer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Peer")
            .field("id", &self.inner.id)
            .field("workspace_id", &self.inner.workspace_id)
            .finish()
    }
}

impl Peer {
    pub(crate) fn from_parts(
        http: HttpClient,
        workspace_id: String,
        resp: PeerResponse,
    ) -> Result<Self> {
        let config = map_to_peer_config(&resp.configuration)?;
        Ok(Self {
            inner: Arc::new(PeerInner {
                http,
                workspace_id,
                id: resp.id,
                metadata: RwLock::new(Some(resp.metadata)),
                configuration: RwLock::new(config),
            }),
        })
    }

    pub(crate) fn from_response(honcho: &crate::Honcho, resp: PeerResponse) -> Result<Self> {
        Self::from_parts(
            honcho.http().clone(),
            honcho.workspace_id().to_owned(),
            resp,
        )
    }

    // ── F5.1: Construction + Metadata ──────────────────────────────────

    /// The peer's unique identifier.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # fn example(peer: &honcho_ai::Peer) {
    /// println!("peer id: {}", peer.id());
    /// # }
    /// ```
    #[must_use]
    pub fn id(&self) -> &str {
        &self.inner.id
    }

    /// Cached metadata from the last API response.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # fn example(peer: &honcho_ai::Peer) {
    /// if let Some(meta) = peer.metadata() {
    ///     println!("{meta:?}");
    /// }
    /// # }
    /// ```
    #[must_use]
    pub fn metadata(&self) -> Option<HashMap<String, Value>> {
        self.inner
            .metadata
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// Cached configuration from the last API response.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # fn example(peer: &honcho_ai::Peer) {
    /// if let Some(config) = peer.configuration() {
    ///     println!("observe_me: {:?}", config.observe_me);
    /// }
    /// # }
    /// ```
    #[must_use]
    pub fn configuration(&self) -> Option<PeerConfig> {
        self.inner
            .configuration
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// Refresh the peer's cached metadata and configuration from the server.
    ///
    /// POSTs to the peers get-or-create endpoint with `{"id": peer_id}`.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # async fn example(peer: &honcho_ai::Peer) -> honcho_ai::error::Result<()> {
    /// peer.refresh().await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn refresh(&self) -> Result<()> {
        let mut body_map = serde_json::Map::new();
        body_map.insert("id".into(), Value::String(self.inner.id.clone()));
        let body = Value::Object(body_map);
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
            .unwrap_or_else(std::sync::PoisonError::into_inner) =
            map_to_peer_config(&resp.configuration)?;
        Ok(())
    }

    /// Fetch and return the peer's metadata, updating the cache.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # async fn example(peer: &honcho_ai::Peer) -> honcho_ai::error::Result<()> {
    /// let meta = peer.get_metadata().await?;
    /// # Ok(())
    /// # }
    /// ```
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self), fields(peer_id = self.inner.id.as_str())))]
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
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # async fn example(peer: &honcho_ai::Peer) -> honcho_ai::error::Result<()> {
    /// let mut meta = std::collections::HashMap::new();
    /// meta.insert("role".into(), "admin".into());
    /// peer.set_metadata(meta).await?;
    /// # Ok(())
    /// # }
    /// ```
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self, metadata), fields(peer_id = self.inner.id.as_str())))]
    pub async fn set_metadata(&self, metadata: HashMap<String, Value>) -> Result<()> {
        let body = crate::types::peer::PeerMetadataSet { metadata };
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
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # async fn example(peer: &honcho_ai::Peer) -> honcho_ai::error::Result<()> {
    /// let config = peer.get_configuration().await?;
    /// # Ok(())
    /// # }
    /// ```
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self), fields(peer_id = self.inner.id.as_str())))]
    pub async fn get_configuration(&self) -> Result<PeerConfig> {
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
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # async fn example(peer: &honcho_ai::Peer) -> honcho_ai::error::Result<()> {
    /// use honcho_ai::PeerConfig;
    /// let mut config = PeerConfig::default();
    /// config.observe_me = Some(true);
    /// peer.set_configuration(&config).await?;
    /// # Ok(())
    /// # }
    /// ```
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self, config), fields(peer_id = self.inner.id.as_str())))]
    pub async fn set_configuration(&self, config: &PeerConfig) -> Result<()> {
        let body = crate::types::peer::PeerConfigurationSet {
            configuration: config.clone(),
        };
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
            .unwrap_or_else(std::sync::PoisonError::into_inner) =
            map_to_peer_config(&resp.configuration)?;
        Ok(())
    }

    /// Fetch and return the peer's configuration as a raw JSON map.
    ///
    /// As a side effect, this method updates the cached metadata
    /// ([`Self::metadata`]) and typed configuration
    /// ([`Self::configuration`]) from the server response.
    ///
    /// Prefer [`get_configuration`](Self::get_configuration) for typed access.
    /// Use this when the server returns fields not yet represented in
    /// [`PeerConfig`].
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self), fields(peer_id = self.inner.id.as_str())))]
    pub async fn get_configuration_raw(&self) -> Result<HashMap<String, Value>> {
        let mut body_map = serde_json::Map::new();
        body_map.insert("id".into(), Value::String(self.inner.id.clone()));
        let body = Value::Object(body_map);
        let resp: PeerResponse = self
            .inner
            .http
            .post(&routes::peers(&self.inner.workspace_id), Some(&body), &[])
            .await?;
        *self
            .inner
            .metadata
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(resp.metadata.clone());
        *self
            .inner
            .configuration
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner) =
            map_to_peer_config(&resp.configuration)?;
        Ok(resp.configuration)
    }

    /// Set the peer's configuration on the server from a raw JSON map.
    ///
    /// Prefer [`set_configuration`](Self::set_configuration) for typed access.
    /// Use this when you need to send fields not yet represented in
    /// [`PeerConfig`].
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self, config), fields(peer_id = self.inner.id.as_str())))]
    pub async fn set_configuration_raw(&self, config: HashMap<String, Value>) -> Result<()> {
        let body = serde_json::json!({"configuration": config});
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
            .unwrap_or_else(std::sync::PoisonError::into_inner) =
            map_to_peer_config(&resp.configuration)?;
        Ok(())
    }

    /// Patch-update the peer's metadata on the server and update the cache.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # async fn example(peer: &honcho_ai::Peer) -> honcho_ai::error::Result<()> {
    /// let mut patch = std::collections::HashMap::new();
    /// patch.insert("status".into(), "active".into());
    /// peer.update(patch).await?;
    /// # Ok(())
    /// # }
    /// ```
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self, metadata), fields(peer_id = self.inner.id.as_str())))]
    pub async fn update(&self, metadata: HashMap<String, Value>) -> Result<()> {
        let body = crate::types::peer::PeerMetadataSet { metadata };
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
    /// Returns `Err(HonchoError::Validation)` when `query` is empty.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # async fn example(peer: &honcho_ai::Peer) -> honcho_ai::error::Result<()> {
    /// if let Some(reply) = peer.chat("What does Alice like?").await? {
    ///     println!("{reply}");
    /// }
    /// # Ok(())
    /// # }
    /// ```
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self), fields(peer_id = self.inner.id.as_str())))]
    pub async fn chat(&self, query: &str) -> Result<Option<String>> {
        if query.is_empty() {
            return Err(HonchoError::Validation(
                "query must not be empty".to_owned(),
            ));
        }
        let body = crate::types::dialectic::DialecticOptions {
            query: query.to_owned(),
            session_id: None,
            target: None,
            stream: false,
            reasoning_level: crate::types::dialectic::ReasoningLevel::default(),
        };
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
    /// Returns `Err(HonchoError::Validation)` when the query in options is empty.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # async fn example(peer: &honcho_ai::Peer) -> honcho_ai::error::Result<()> {
    /// use honcho_ai::types::dialectic::DialecticOptions;
    /// let opts = DialecticOptions::builder()
    ///     .query("What does Alice like?")
    ///     .stream(false)
    ///     .build();
    /// let reply = peer.chat_with_options(&opts).await?;
    /// # Ok(())
    /// # }
    /// ```
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self, options), fields(peer_id = self.inner.id.as_str())))]
    pub async fn chat_with_options(&self, options: &DialecticOptions) -> Result<Option<String>> {
        if options.query.is_empty() {
            return Err(HonchoError::Validation(
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

    /// Create a streaming dialectic chat builder.
    ///
    /// Returns a [`ChatStreamBuilder`] that sends the request on `.send()`.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # async fn example(peer: &honcho_ai::Peer) -> honcho_ai::error::Result<()> {
    /// use futures_util::StreamExt;
    /// let mut stream = peer.chat_stream("Hello").target("bob").send().await?;
    /// while let Some(chunk) = stream.next().await {
    ///     println!("{}", chunk?);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub fn chat_stream(&self, query: impl Into<String>) -> ChatStreamBuilder {
        ChatStreamBuilder {
            http: self.inner.http.clone(),
            workspace_id: self.inner.workspace_id.clone(),
            peer_id: self.inner.id.clone(),
            query: query.into(),
            target: None,
            session_id: None,
            reasoning_level: None,
        }
    }

    // ── Representation ─────────────────────────────────────────────────

    /// Get the peer's representation (default parameters).
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # async fn example(peer: &honcho_ai::Peer) -> honcho_ai::error::Result<()> {
    /// let rep = peer.representation().await?;
    /// println!("{rep}");
    /// # Ok(())
    /// # }
    /// ```
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self), fields(peer_id = self.inner.id.as_str())))]
    pub async fn representation(&self) -> Result<String> {
        let route = routes::peer_representation(&self.inner.workspace_id, &self.inner.id);
        let body = crate::types::peer::PeerRepresentationGet {
            session_id: None,
            target: None,
            search_query: None,
            search_top_k: None,
            search_max_distance: None,
            include_most_frequent: None,
            max_conclusions: None,
        };
        let resp: RepresentationResponse = self.inner.http.post(&route, Some(&body), &[]).await?;
        Ok(resp.representation)
    }

    /// Get a representation builder for fine-grained control over parameters.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # async fn example(peer: &honcho_ai::Peer) -> honcho_ai::error::Result<()> {
    /// let rep = peer.representation_builder()
    ///     .search_query("hobbies")
    ///     .search_top_k(10)
    ///     .send()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
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
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # async fn example(peer: &honcho_ai::Peer) -> honcho_ai::error::Result<()> {
    /// let ctx = peer.context().await?;
    /// # Ok(())
    /// # }
    /// ```
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self), fields(peer_id = self.inner.id.as_str())))]
    pub async fn context(&self) -> Result<PeerContext> {
        self.context_builder().send().await
    }

    /// Get a context builder for fine-grained control over parameters.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # async fn example(peer: &honcho_ai::Peer) -> honcho_ai::error::Result<()> {
    /// let ctx = peer.context_builder()
    ///     .target("bob")
    ///     .summary(true)
    ///     .search_query("preferences")
    ///     .search_top_k(10)
    ///     .send()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn context_builder(&self) -> ContextBuilder {
        ContextBuilder {
            http: self.inner.http.clone(),
            workspace_id: self.inner.workspace_id.clone(),
            peer_id: self.inner.id.clone(),
            target: None,
            summary: None,
            limit_to_session: None,
            max_conclusions: None,
            search_query: None,
            search_top_k: None,
            search_max_distance: None,
            include_most_frequent: None,
        }
    }

    /// Get the peer's context scoped to a target peer.
    #[deprecated(since = "0.1.1", note = "use `Peer::context_builder()` instead")]
    #[allow(deprecated)]
    pub async fn context_with_target(&self, target: &str) -> Result<PeerContext> {
        let opts = crate::types::peer::PeerContextOptions::builder()
            .target(target)
            .build();
        self.context_with_options(&opts).await
    }

    /// Get the peer's context with custom options.
    #[deprecated(since = "0.1.1", note = "use `Peer::context_builder()` instead")]
    #[allow(deprecated)]
    pub async fn context_with_options(
        &self,
        options: &crate::types::peer::PeerContextOptions,
    ) -> Result<PeerContext> {
        let route = routes::peer_context(&self.inner.workspace_id, &self.inner.id);
        let mut params: Vec<(&str, String)> = Vec::new();
        if let Some(ref v) = options.target {
            params.push(("target", v.clone()));
        }
        if let Some(ref v) = options.search_query {
            params.push(("search_query", v.clone()));
        }
        if let Some(ref v) = options.search_top_k {
            params.push(("search_top_k", v.to_string()));
        }
        if let Some(ref v) = options.search_max_distance {
            params.push(("search_max_distance", v.to_string()));
        }
        if let Some(ref v) = options.include_most_frequent {
            params.push((
                "include_most_frequent",
                if *v { "true" } else { "false" }.to_string(),
            ));
        }
        if let Some(ref v) = options.max_conclusions {
            params.push(("max_conclusions", v.to_string()));
        }
        let refs: Vec<(&str, &str)> = params.iter().map(|(k, v)| (*k, v.as_str())).collect();
        self.inner.http.get(&route, &refs).await
    }

    // ── Sessions ───────────────────────────────────────────────────────

    /// List sessions for this peer (paginated).
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # async fn example(peer: &honcho_ai::Peer) -> honcho_ai::error::Result<()> {
    /// let page = peer.sessions().await?;
    /// for session in page.items() {
    ///     println!("{}", session.id);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self), fields(peer_id = self.inner.id.as_str())))]
    pub async fn sessions(&self) -> Result<Page<Session>> {
        let route = routes::peer_sessions_list(&self.inner.workspace_id, &self.inner.id);
        pagination::paginate_post(&self.inner.http, &route, None, 1, 50, false).await
    }

    /// List sessions for this peer with filters and pagination options.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # async fn example(peer: &honcho_ai::Peer) -> honcho_ai::error::Result<()> {
    /// use honcho_ai::types::session::SessionListOptions;
    /// let opts = SessionListOptions::builder().page(2).size(10).build();
    /// let page = peer.sessions_with_options(&opts).await?;
    /// for session in page.items() {
    ///     println!("{}", session.id);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self, options), fields(peer_id = self.inner.id.as_str())))]
    pub async fn sessions_with_options(
        &self,
        options: &SessionListOptions,
    ) -> Result<Page<Session>> {
        let route = routes::peer_sessions_list(&self.inner.workspace_id, &self.inner.id);
        let body = options
            .filters
            .as_ref()
            .map(|f| crate::types::session::SessionGet {
                filters: Some(f.clone()),
            });
        let body_val = body
            .as_ref()
            .map(|b| serde_json::to_value(b).map_err(|e| HonchoError::Configuration(e.to_string())))
            .transpose()?;
        pagination::paginate_post(
            &self.inner.http,
            &route,
            body_val.as_ref(),
            options.page,
            options.size,
            options.reverse,
        )
        .await
    }

    // ── Search ─────────────────────────────────────────────────────────

    /// Search messages for this peer (default limit of 10).
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # async fn example(peer: &honcho_ai::Peer) -> honcho_ai::error::Result<()> {
    /// let results = peer.search("important topic").await?;
    /// for msg in results {
    ///     println!("{}", msg.content());
    /// }
    /// # Ok(())
    /// # }
    /// ```
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self), fields(peer_id = self.inner.id.as_str())))]
    pub async fn search(&self, query: &str) -> Result<Vec<crate::Message>> {
        self.search_with_options(&MessageSearchOptions {
            query: query.to_string(),
            filters: None,
            limit: 10,
        })
        .await
    }

    /// Search messages for this peer with custom options (limit, filters).
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # async fn example(peer: &honcho_ai::Peer) -> honcho_ai::error::Result<()> {
    /// use honcho_ai::types::message::MessageSearchOptions;
    /// let opts = MessageSearchOptions {
    ///     query: "topic".into(),
    ///     filters: None,
    ///     limit: 20,
    /// };
    /// let results = peer.search_with_options(&opts).await?;
    /// # Ok(())
    /// # }
    /// ```
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self, options), fields(peer_id = self.inner.id.as_str())))]
    pub async fn search_with_options(
        &self,
        options: &MessageSearchOptions,
    ) -> Result<Vec<crate::Message>> {
        if options.query.is_empty() {
            return Err(HonchoError::Validation(
                "query must not be empty".to_string(),
            ));
        }
        let responses: Vec<MessageResponse> = self
            .inner
            .http
            .post(
                &routes::peer_search(&self.inner.workspace_id, &self.inner.id),
                Some(&options),
                &[],
            )
            .await?;
        Ok(responses
            .into_iter()
            .map(|r| crate::Message::from_raw(self.inner.workspace_id.clone(), r))
            .collect())
    }

    // ── Card ───────────────────────────────────────────────────────────

    /// Get this peer's card.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # async fn example(peer: &honcho_ai::Peer) -> honcho_ai::error::Result<()> {
    /// if let Some(card) = peer.get_card().await? {
    ///     for line in &card {
    ///         println!("{line}");
    ///     }
    /// }
    /// # Ok(())
    /// # }
    /// ```
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self), fields(peer_id = self.inner.id.as_str())))]
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
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # async fn example(peer: &honcho_ai::Peer) -> honcho_ai::error::Result<()> {
    /// let card = peer.get_card_with_target("bob").await?;
    /// # Ok(())
    /// # }
    /// ```
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self), fields(peer_id = self.inner.id.as_str())))]
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
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # async fn example(peer: &honcho_ai::Peer) -> honcho_ai::error::Result<()> {
    /// peer.set_card(vec!["friendly".into(), "likes dogs".into()]).await?;
    /// # Ok(())
    /// # }
    /// ```
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self, card), fields(peer_id = self.inner.id.as_str())))]
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
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # async fn example(peer: &honcho_ai::Peer) -> honcho_ai::error::Result<()> {
    /// peer.set_card_with_target(vec!["reliable".into()], "bob").await?;
    /// # Ok(())
    /// # }
    /// ```
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self, card), fields(peer_id = self.inner.id.as_str())))]
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

    // ── F9.8: Conclusions ──────────────────────────────────────────────

    /// Get a self-scoped conclusion handle (observer = observed = self).
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # fn example(peer: &honcho_ai::Peer) {
    /// let scope = peer.conclusions();
    /// assert_eq!(scope.observer_id(), peer.id());
    /// # }
    /// ```
    #[must_use]
    pub fn conclusions(&self) -> ConclusionScope {
        ConclusionScope::new(
            self.inner.http.clone(),
            self.inner.workspace_id.clone(),
            self.id().to_owned(),
            self.id().to_owned(),
        )
    }

    /// Get a cross-peer conclusion handle (observer = self, observed = target).
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # fn example(peer: &honcho_ai::Peer) {
    /// let scope = peer.conclusions_of("bob");
    /// # }
    /// ```
    #[must_use]
    pub fn conclusions_of(&self, target: impl Into<String>) -> ConclusionScope {
        ConclusionScope::new(
            self.inner.http.clone(),
            self.inner.workspace_id.clone(),
            self.id().to_owned(),
            target.into(),
        )
    }

    // ── F5.8: Message builder (sync, no API call) ─────────────────────

    /// Create a message builder for this peer.
    ///
    /// This is purely synchronous and does NOT send any API request.
    /// Call `.build()` to get a `MessageCreate` that can be passed to
    /// `Session::add_messages()`.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # fn example(peer: &honcho_ai::Peer) -> honcho_ai::error::Result<()> {
    /// let msg = peer.message("Hello world").build()?;
    /// # Ok(())
    /// # }
    /// ```
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

/// Builder for streaming dialectic chat requests.
///
/// Created via [`Peer::chat_stream()`]. Call `.send()` to start streaming.
pub struct ChatStreamBuilder {
    http: HttpClient,
    workspace_id: String,
    peer_id: String,
    query: String,
    target: Option<String>,
    session_id: Option<String>,
    reasoning_level: Option<ReasoningLevel>,
}

impl ChatStreamBuilder {
    /// Scope the chat to a target peer.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # fn example(peer: &honcho_ai::Peer) {
    /// let _builder = peer.chat_stream("hello").target("bob");
    /// # }
    /// ```
    #[must_use]
    pub fn target(mut self, target: impl Into<String>) -> Self {
        self.target = Some(target.into());
        self
    }

    /// Scope the chat to a session.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # fn example(peer: &honcho_ai::Peer) {
    /// let _builder = peer.chat_stream("hello").session("sess-1");
    /// # }
    /// ```
    #[must_use]
    pub fn session(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    /// Set the reasoning level (defaults to `Low` if unset).
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # fn example(peer: &honcho_ai::Peer) {
    /// use honcho_ai::types::dialectic::ReasoningLevel;
    /// let _builder = peer.chat_stream("hello").reasoning_level(ReasoningLevel::High);
    /// # }
    /// ```
    #[must_use]
    pub fn reasoning_level(mut self, level: ReasoningLevel) -> Self {
        self.reasoning_level = Some(level);
        self
    }

    /// Send the streaming chat request.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # async fn example(peer: &honcho_ai::Peer) -> honcho_ai::error::Result<()> {
    /// use futures_util::StreamExt;
    /// let mut stream = peer.chat_stream("hello").send().await?;
    /// while let Some(chunk) = stream.next().await {
    ///     println!("{}", chunk?);
    /// }
    /// println!("full: {}", stream.final_response().content());
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `HonchoError::Validation` if the query is empty.
    /// Returns transport/API errors if the request fails.
    #[allow(clippy::type_complexity)]
    pub async fn send(
        self,
    ) -> Result<DialecticStream<Pin<Box<dyn Stream<Item = Result<String>> + Send>>>> {
        if self.query.is_empty() {
            return Err(HonchoError::Validation(
                "query must not be empty".to_owned(),
            ));
        }

        let opts = DialecticOptions::builder()
            .query(self.query)
            .stream(true)
            .maybe_target(self.target)
            .maybe_session_id(self.session_id)
            .reasoning_level(self.reasoning_level.unwrap_or_default())
            .build();

        let route = routes::peer_chat(&self.workspace_id, &self.peer_id);
        let response = self
            .http
            .request_streaming(
                Method::POST,
                &route,
                Some(
                    &serde_json::to_value(&opts)
                        .map_err(|e| HonchoError::Configuration(e.to_string()))?,
                ),
                &[],
            )
            .await?;

        Ok(DialecticStream::new(Box::pin(parse_sse_stream(
            response.bytes_stream(),
        ))))
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
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # fn example(peer: &honcho_ai::Peer) {
    /// let _builder = peer.representation_builder().session_id("sess-1");
    /// # }
    /// ```
    #[must_use]
    pub fn session_id(mut self, val: impl Into<String>) -> Self {
        self.session_id = Some(val.into());
        self
    }

    /// Get the representation for a specific target peer.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # fn example(peer: &honcho_ai::Peer) {
    /// let _builder = peer.representation_builder().target("bob");
    /// # }
    /// ```
    #[must_use]
    pub fn target(mut self, val: impl Into<String>) -> Self {
        self.target = Some(val.into());
        self
    }

    /// Semantic search query to curate the representation.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # fn example(peer: &honcho_ai::Peer) {
    /// let _builder = peer.representation_builder().search_query("hobbies");
    /// # }
    /// ```
    #[must_use]
    pub fn search_query(mut self, val: impl Into<String>) -> Self {
        self.search_query = Some(val.into());
        self
    }

    /// Number of semantic-search-retrieved conclusions (1–100).
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # fn example(peer: &honcho_ai::Peer) {
    /// let _builder = peer.representation_builder().search_top_k(20);
    /// # }
    /// ```
    #[must_use]
    pub fn search_top_k(mut self, val: u32) -> Self {
        self.search_top_k = Some(val);
        self
    }

    /// Maximum distance for semantically relevant conclusions (0.0–1.0).
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # fn example(peer: &honcho_ai::Peer) {
    /// let _builder = peer.representation_builder().search_max_distance(0.5);
    /// # }
    /// ```
    #[must_use]
    pub fn search_max_distance(mut self, val: f64) -> Self {
        self.search_max_distance = Some(val);
        self
    }

    /// Whether to include the most frequent conclusions.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # fn example(peer: &honcho_ai::Peer) {
    /// let _builder = peer.representation_builder().include_most_frequent(true);
    /// # }
    /// ```
    #[must_use]
    pub fn include_most_frequent(mut self, val: bool) -> Self {
        self.include_most_frequent = Some(val);
        self
    }

    /// Maximum number of conclusions to include (1–100).
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # fn example(peer: &honcho_ai::Peer) {
    /// let _builder = peer.representation_builder().max_conclusions(25);
    /// # }
    /// ```
    #[must_use]
    pub fn max_conclusions(mut self, val: u32) -> Self {
        self.max_conclusions = Some(val);
        self
    }

    /// Send the representation request with the configured parameters.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # async fn example(peer: &honcho_ai::Peer) -> honcho_ai::error::Result<()> {
    /// let rep = peer.representation_builder()
    ///     .search_query("hobbies")
    ///     .search_top_k(10)
    ///     .send()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    ///
    /// # Errors
    ///
    /// Returns `HonchoError::Configuration` if `search_top_k`, `search_max_distance`,
    /// or `max_conclusions` are out of range.
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self), fields(peer_id = self.peer_id.as_str())))]
    pub async fn send(self) -> Result<String> {
        if let Some(k) = self.search_top_k
            && !(1..=100).contains(&k)
        {
            return Err(HonchoError::Validation(format!(
                "search_top_k must be between 1 and 100, got {k}"
            )));
        }
        if let Some(d) = self.search_max_distance
            && !(0.0..=1.0).contains(&d)
        {
            return Err(HonchoError::Validation(format!(
                "search_max_distance must be between 0.0 and 1.0, got {d}"
            )));
        }
        if let Some(c) = self.max_conclusions
            && !(1..=100).contains(&c)
        {
            return Err(HonchoError::Validation(format!(
                "max_conclusions must be between 1 and 100, got {c}"
            )));
        }

        let params = crate::types::peer::PeerRepresentationGet {
            session_id: self.session_id,
            target: self.target,
            search_query: self.search_query,
            search_top_k: self.search_top_k,
            search_max_distance: self.search_max_distance,
            include_most_frequent: self.include_most_frequent,
            max_conclusions: self.max_conclusions,
        };

        let route = routes::peer_representation(&self.workspace_id, &self.peer_id);
        let resp: RepresentationResponse = self.http.post(&route, Some(&params), &[]).await?;
        Ok(resp.representation)
    }
}

/// Builder for fine-grained context requests.
///
/// Created via [`Peer::context_builder()`].
///
/// # Examples
///
/// ```no_run
/// # async fn example(peer: &honcho_ai::Peer) -> honcho_ai::error::Result<()> {
/// use honcho_ai::types::peer::PeerContext;
/// let ctx: PeerContext = peer.context_builder()
///     .target("bob")
///     .summary(true)
///     .limit_to_session(true)
///     .search_query("preferences")
///     .search_top_k(10)
///     .send()
///     .await?;
/// # Ok(())
/// # }
/// ```
pub struct ContextBuilder {
    http: HttpClient,
    workspace_id: String,
    peer_id: String,
    target: Option<String>,
    summary: Option<bool>,
    limit_to_session: Option<bool>,
    max_conclusions: Option<u32>,
    search_query: Option<String>,
    search_top_k: Option<u32>,
    search_max_distance: Option<f64>,
    include_most_frequent: Option<bool>,
}

impl ContextBuilder {
    /// Scope the context to a specific target peer.
    #[must_use]
    pub fn target(mut self, val: impl Into<String>) -> Self {
        self.target = Some(val.into());
        self
    }

    /// Whether to include the peer card summary.
    #[must_use]
    pub fn summary(mut self, val: bool) -> Self {
        self.summary = Some(val);
        self
    }

    /// Limit the representation context to the specified session only.
    #[must_use]
    pub fn limit_to_session(mut self, val: bool) -> Self {
        self.limit_to_session = Some(val);
        self
    }

    /// Maximum number of conclusions to include (1–100).
    #[must_use]
    pub fn max_conclusions(mut self, val: u32) -> Self {
        self.max_conclusions = Some(val);
        self
    }

    /// Semantic search query to filter relevant conclusions.
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

    /// Send the context request with the configured parameters.
    ///
    /// # Errors
    ///
    /// Returns `HonchoError::Validation` if `search_top_k`, `search_max_distance`,
    /// or `max_conclusions` are out of range.
    #[cfg_attr(feature = "tracing", tracing::instrument(skip(self), fields(peer_id = self.peer_id.as_str())))]
    pub async fn send(self) -> Result<PeerContext> {
        if let Some(k) = self.search_top_k
            && !(1..=100).contains(&k)
        {
            return Err(HonchoError::Validation(format!(
                "search_top_k must be between 1 and 100, got {k}"
            )));
        }
        if let Some(d) = self.search_max_distance
            && !(0.0..=1.0).contains(&d)
        {
            return Err(HonchoError::Validation(format!(
                "search_max_distance must be between 0.0 and 1.0, got {d}"
            )));
        }
        if let Some(c) = self.max_conclusions
            && !(1..=100).contains(&c)
        {
            return Err(HonchoError::Validation(format!(
                "max_conclusions must be between 1 and 100, got {c}"
            )));
        }

        let route = routes::peer_context(&self.workspace_id, &self.peer_id);
        let mut params: Vec<(&str, String)> = Vec::new();
        if let Some(ref v) = self.target {
            params.push(("target", v.clone()));
        }
        if let Some(v) = self.summary {
            params.push(("summary", if v { "true" } else { "false" }.to_string()));
        }
        if let Some(v) = self.limit_to_session {
            params.push((
                "limit_to_session",
                if v { "true" } else { "false" }.to_string(),
            ));
        }
        if let Some(ref v) = self.search_query {
            params.push(("search_query", v.clone()));
        }
        if let Some(v) = self.search_top_k {
            params.push(("search_top_k", v.to_string()));
        }
        if let Some(v) = self.search_max_distance {
            params.push(("search_max_distance", v.to_string()));
        }
        if let Some(v) = self.include_most_frequent {
            params.push((
                "include_most_frequent",
                if v { "true" } else { "false" }.to_string(),
            ));
        }
        if let Some(v) = self.max_conclusions {
            params.push(("max_conclusions", v.to_string()));
        }
        let refs: Vec<(&str, &str)> = params.iter().map(|(k, v)| (*k, v.as_str())).collect();
        self.http.get(&route, &refs).await
    }
}

/// Synchronous builder for [`MessageCreate`] params.
///
const MAX_MESSAGE_CONTENT_LENGTH: usize = 25_000;

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
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # fn example(peer: &honcho_ai::Peer) -> honcho_ai::error::Result<()> {
    /// let mut meta = std::collections::HashMap::new();
    /// meta.insert("source".into(), "chat".into());
    /// let msg = peer.message("hello").metadata(meta).build()?;
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn metadata(mut self, val: HashMap<String, Value>) -> Self {
        self.metadata = Some(val);
        self
    }

    /// Attach message-level configuration.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # fn example(peer: &honcho_ai::Peer) -> honcho_ai::error::Result<()> {
    /// use honcho_ai::types::message::MessageConfiguration;
    /// let cfg = MessageConfiguration::default();
    /// let msg = peer.message("hello").configuration(cfg).build()?;
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn configuration(mut self, val: crate::types::message::MessageConfiguration) -> Self {
        self.configuration = Some(val);
        self
    }

    /// Override the creation timestamp.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # fn example(peer: &honcho_ai::Peer) -> honcho_ai::error::Result<()> {
    /// let msg = peer.message("hello").created_at(chrono::Utc::now()).build()?;
    /// # Ok(())
    /// # }
    /// ```
    #[must_use]
    pub fn created_at(mut self, val: chrono::DateTime<chrono::Utc>) -> Self {
        self.created_at = Some(val);
        self
    }

    /// Build the message params.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # fn example(peer: &honcho_ai::Peer) -> honcho_ai::error::Result<()> {
    /// let msg = peer.message("hello").build()?;
    /// assert_eq!(msg.content, "hello");
    /// # Ok(())
    /// # }
    /// ```
    pub fn build(self) -> Result<MessageCreate> {
        if self.content.trim().is_empty() {
            return Err(HonchoError::Validation("content must not be empty".into()));
        }
        if self.content.len() > MAX_MESSAGE_CONTENT_LENGTH {
            return Err(HonchoError::Validation(format!(
                "content must be at most {} characters, got {}",
                MAX_MESSAGE_CONTENT_LENGTH,
                self.content.len(),
            )));
        }
        Ok(MessageCreate {
            peer_id: self.peer_id,
            content: self.content,
            metadata: self.metadata,
            configuration: self.configuration,
            created_at: self.created_at,
        })
    }
}

fn map_to_peer_config(map: &HashMap<String, Value>) -> Result<Option<PeerConfig>> {
    let val = serde_json::to_value(map).map_err(|e| HonchoError::Configuration(e.to_string()))?;
    serde_json::from_value(val)
        .map(Some)
        .map_err(|e| HonchoError::Configuration(e.to_string()))
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::unnecessary_wraps,
    clippy::needless_pass_by_value,
    clippy::unused_async
)]
mod tests {
    use super::*;
    use std::pin::Pin;

    #[allow(dead_code)]
    fn assert_send_static<T: Send + 'static>(_: &T) {}

    #[test]
    fn chat_stream_return_type_is_send_static() {
        fn _assertion(
            stream: DialecticStream<
                Pin<Box<dyn futures_util::Stream<Item = Result<String>> + Send>>,
            >,
        ) {
            assert_send_static(&stream);
        }
    }
}
