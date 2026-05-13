//! Session wrapper — construction, metadata, peer management, per-peer config.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};

use chrono::{DateTime, Utc};
use reqwest::multipart::Form;
use reqwest::Method;
use serde::Deserialize;
use serde_json::Value;

use crate::error::{HonchoError, Result};
use crate::http::client::HttpClient;
use crate::http::routes;
use crate::types::message::MessageResponse;
use crate::types::session::Session as SessionResponse;
use crate::types::session::SessionPeerConfig;
use crate::upload::{self, FileSource};

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
    #[expect(dead_code)]
    total: u64,
    #[serde(default)]
    #[expect(dead_code)]
    pages: u64,
}

/// Builder for the file-upload operation returned by [`Session::upload_file`].
///
/// Call `.peer(id)` (required) and optionally chain `.metadata()`,
/// `.configuration()`, `.created_at()` before calling `.send()`.
#[must_use]
pub struct UploadFileBuilder<'a> {
    session: &'a Session,
    source: Option<FileSource>,
    peer_id: Option<String>,
    metadata: Option<Value>,
    configuration: Option<Value>,
    created_at: Option<DateTime<Utc>>,
}

impl UploadFileBuilder<'_> {
    /// Set the peer that owns the uploaded file (required).
    pub fn peer(mut self, id: impl Into<String>) -> Self {
        self.peer_id = Some(id.into());
        self
    }

    /// Attach arbitrary JSON metadata to the created message(s).
    pub fn metadata(mut self, value: Value) -> Self {
        self.metadata = Some(value);
        self
    }

    /// Attach configuration to the created message(s).
    pub fn configuration(mut self, value: Value) -> Self {
        self.configuration = Some(value);
        self
    }

    /// Override the creation timestamp (ISO 3339).
    pub fn created_at(mut self, dt: DateTime<Utc>) -> Self {
        self.created_at = Some(dt);
        self
    }

    /// Resolve the file source, build the multipart form, POST, and return
    /// the created messages.
    ///
    /// # Errors
    ///
    /// Returns [`HonchoError::Validation`] if no peer was set via `.peer()`.
    pub async fn send(self) -> Result<Vec<crate::Message>> {
        let peer_id = self
            .peer_id
            .ok_or_else(|| HonchoError::Validation("peer_id is required".into()))?;

        let source = self
            .source
            .ok_or_else(|| HonchoError::Validation("file source is required".into()))?;

        let file_part = match source {
            FileSource::Stream {
                filename,
                reader,
                content_type,
            } => {
                let stream = tokio_util::io::ReaderStream::new(reader);
                let body = reqwest::Body::wrap_stream(stream);
                reqwest::multipart::Part::stream(body)
                    .file_name(filename)
                    .mime_str(&content_type)
                    .map_err(|e| HonchoError::Configuration(format!("invalid mime type: {e}")))?
            }
            buffered => {
                let (filename, bytes, content_type) = upload::resolve_to_bytes(buffered).await?;
                reqwest::multipart::Part::bytes(bytes)
                    .file_name(filename)
                    .mime_str(&content_type)
                    .map_err(|e| HonchoError::Configuration(format!("invalid mime type: {e}")))?
            }
        };

        let mut form = Form::new().part("file", file_part).text("peer_id", peer_id);

        if let Some(ref md) = self.metadata {
            form = form.text(
                "metadata",
                serde_json::to_string(md)
                    .map_err(|e| HonchoError::Configuration(format!("metadata: {e}")))?,
            );
        }

        if let Some(ref cfg) = self.configuration {
            form = form.text(
                "configuration",
                serde_json::to_string(cfg)
                    .map_err(|e| HonchoError::Configuration(format!("configuration: {e}")))?,
            );
        }

        if let Some(dt) = self.created_at {
            form = form.text("created_at", dt.to_rfc3339());
        }

        let route =
            routes::messages_upload(&self.session.inner.workspace_id, &self.session.inner.id);

        let responses: Vec<MessageResponse> = self
            .session
            .inner
            .http
            .post_multipart(&route, form, &[])
            .await?;

        Ok(responses
            .into_iter()
            .map(|r| {
                crate::Message::from_raw(
                    self.session.inner.http.clone(),
                    self.session.inner.workspace_id.clone(),
                    r,
                )
            })
            .collect())
    }
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
        let peers_map = normalize_peers(specs)?;
        let body = serde_json::json!({"peers": peers_map});
        let route = routes::session_peers(&self.inner.workspace_id, &self.inner.id);
        self.inner.http.post(&route, Some(&body), &[]).await
    }

    /// Set the complete peer list for this session (replaces existing).
    pub async fn set_peers(
        &self,
        specs: impl IntoIterator<Item = impl Into<PeerSpec>>,
    ) -> Result<()> {
        let peers_map = normalize_peers(specs)?;
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
    ) -> Result<Vec<crate::types::message::MessageResponse>> {
        use crate::types::message::MessageResponse;

        if messages.is_empty() {
            return Ok(Vec::new());
        }

        let route = routes::messages(&self.inner.workspace_id, &self.inner.id);

        if messages.len() <= 100 {
            let body = serde_json::json!({"messages": messages});
            return self
                .inner
                .http
                .post::<_, Vec<MessageResponse>>(&route, Some(&body), &[])
                .await;
        }

        let mut all = Vec::with_capacity(messages.len());
        for chunk in messages.chunks(100) {
            let body = serde_json::json!({"messages": chunk});
            let batch: Vec<MessageResponse> =
                self.inner.http.post(&route, Some(&body), &[]).await?;
            all.extend(batch);
        }
        Ok(all)
    }

    /// List messages in this session (paginated).
    pub async fn messages(
        &self,
    ) -> Result<crate::types::pagination::Page<crate::types::message::MessageResponse>> {
        let route = routes::messages_list(&self.inner.workspace_id, &self.inner.id);
        crate::types::pagination::paginate_post(&self.inner.http, &route, None, 1, 50, false).await
    }

    // ── F7.3: File upload ───────────────────────────────────────────────

    /// Begin a file upload to this session.
    ///
    /// Returns an [`UploadFileBuilder`]. You **must** call `.peer(id)` and
    /// then `.send()` to complete the upload.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let messages = session
    ///     .upload_file(FileSource::bytes("doc.pdf", data, "application/pdf"))
    ///     .peer("alice")
    ///     .send()
    ///     .await?;
    /// ```
    pub fn upload_file(&self, source: impl Into<FileSource>) -> UploadFileBuilder<'_> {
        UploadFileBuilder {
            session: self,
            source: Some(source.into()),
            peer_id: None,
            metadata: None,
            configuration: None,
            created_at: None,
        }
    }

    /// Begin a **streaming** file upload to this session.
    ///
    /// Unlike [`Session::upload_file`], the reader is consumed lazily via
    /// [`tokio_util::io::ReaderStream`] so the entire file is never buffered
    /// in memory.
    ///
    /// Returns an [`UploadFileBuilder`]. You **must** call `.peer(id)` and
    /// then `.send()` to complete the upload.
    pub fn upload_file_streamed(
        &self,
        filename: impl Into<String>,
        reader: impl tokio::io::AsyncRead + Send + 'static,
        content_type: impl Into<String>,
    ) -> UploadFileBuilder<'_> {
        UploadFileBuilder {
            session: self,
            source: Some(FileSource::stream(filename, reader, content_type)),
            peer_id: None,
            metadata: None,
            configuration: None,
            created_at: None,
        }
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
    pub async fn get_message(&self, id: &str) -> Result<crate::types::message::MessageResponse> {
        let route = routes::message(&self.inner.workspace_id, &self.inner.id, id);
        self.inner.http.get(&route, &[]).await
    }

    /// Update a message's metadata.
    pub async fn update_message(
        &self,
        id: &str,
        metadata: HashMap<String, Value>,
    ) -> Result<crate::types::message::MessageResponse> {
        let route = routes::message(&self.inner.workspace_id, &self.inner.id, id);
        let body = serde_json::json!({"metadata": metadata});
        self.inner.http.put(&route, Some(&body), &[]).await
    }

    // ── F6.6: Context ───────────────────────────────────────────────────

    /// Get the session context with default parameters.
    ///
    /// Fetches messages, summary, peer representation, and peer card for this session.
    pub async fn context(&self) -> Result<crate::types::session::SessionContext> {
        self.context_with_options(true, false).await
    }

    /// Get the session context with custom parameters.
    ///
    /// - `summary`: whether to include the session summary.
    /// - `limit_to_session`: whether to limit representation context to this session.
    pub async fn context_with_options(
        &self,
        summary: bool,
        limit_to_session: bool,
    ) -> Result<crate::types::session::SessionContext> {
        let route = routes::session_context(&self.inner.workspace_id, &self.inner.id);
        self.inner
            .http
            .get(
                &route,
                &[
                    ("summary", if summary { "true" } else { "false" }),
                    (
                        "limit_to_session",
                        if limit_to_session { "true" } else { "false" },
                    ),
                ],
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

    /// Search messages within this session (default limit of 10).
    ///
    /// Returns `Err(HonchoError::Configuration)` when `query` is empty.
    pub async fn search(&self, query: &str) -> Result<Vec<crate::types::message::MessageResponse>> {
        self.search_with_options(crate::types::message::MessageSearchOptions {
            query: query.to_string(),
            filters: None,
            limit: 10,
        })
        .await
    }

    /// Search messages within this session with custom options (limit, filters).
    ///
    /// Returns `Err(HonchoError::Configuration)` when `query` is empty.
    pub async fn search_with_options(
        &self,
        options: crate::types::message::MessageSearchOptions,
    ) -> Result<Vec<crate::types::message::MessageResponse>> {
        if options.query.is_empty() {
            return Err(crate::error::HonchoError::Configuration(
                "query must not be empty".to_string(),
            ));
        }
        let route = routes::session_search(&self.inner.workspace_id, &self.inner.id);
        self.inner.http.post(&route, Some(&options), &[]).await
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

fn normalize_peers(
    specs: impl IntoIterator<Item = impl Into<PeerSpec>>,
) -> Result<serde_json::Value> {
    let map: serde_json::Map<String, Value> = specs
        .into_iter()
        .map(|s| {
            let spec = s.into();
            let val = match &spec {
                PeerSpec::Id(_) => serde_json::json!({}),
                PeerSpec::WithConfig(_, cfg) => serde_json::to_value(cfg).map_err(|e| {
                    HonchoError::Configuration(format!(
                        "failed to serialize peer config for {}: {e}",
                        spec.id()
                    ))
                })?,
            };
            Ok((spec.id().to_owned(), val))
        })
        .collect::<Result<_>>()?;
    Ok(Value::Object(map))
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, clippy::expect_used)]

    use static_assertions::assert_impl_all;

    use super::*;
    use crate::http::client::HttpClient;
    use crate::types::session::Session as SessionResponse;
    use chrono::TimeZone;
    use wiremock::matchers::{body_string_contains, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    assert_impl_all!(UploadFileBuilder<'_>: Send);

    fn session_json(id: &str) -> serde_json::Value {
        serde_json::json!({
            "id": id,
            "workspace_id": "ws1",
            "is_active": true,
            "metadata": {},
            "configuration": {},
            "created_at": "2025-01-15T10:30:00Z"
        })
    }

    fn message_response_json(content: &str, peer_id: &str) -> serde_json::Value {
        serde_json::json!({
            "id": "msg_1",
            "content": content,
            "peer_id": peer_id,
            "session_id": "sess1",
            "metadata": {},
            "created_at": "2025-01-15T10:30:00Z",
            "workspace_id": "ws1",
            "token_count": 5
        })
    }

    fn make_session(http: HttpClient, id: &str) -> Session {
        let resp: SessionResponse = serde_json::from_value(session_json(id)).unwrap();
        Session::from_parts(http, "ws1".to_owned(), resp)
    }

    fn upload_response_json() -> serde_json::Value {
        serde_json::json!([message_response_json("extracted text", "alice")])
    }

    #[tokio::test]
    async fn upload_file_with_bytes_sends_correct_multipart() {
        let server = MockServer::start().await;
        let http =
            HttpClient::from_params(HttpClient::builder().base_url(server.uri()).build()).unwrap();
        let session = make_session(http, "sess1");

        Mock::given(method("POST"))
            .and(path("/v3/workspaces/ws1/sessions/sess1/messages/upload"))
            .and(body_string_contains("file content here"))
            .and(body_string_contains("peer_id"))
            .and(body_string_contains("alice"))
            .respond_with(ResponseTemplate::new(200).set_body_json(upload_response_json()))
            .mount(&server)
            .await;

        let msgs = session
            .upload_file(FileSource::bytes(
                "test.txt",
                b"file content here".as_slice(),
                "text/plain",
            ))
            .peer("alice")
            .send()
            .await
            .unwrap();

        assert_eq!(msgs.len(), 1);
        assert_eq!(msgs[0].content(), "extracted text");
        assert_eq!(msgs[0].peer_id(), "alice");
    }

    #[tokio::test]
    async fn upload_file_with_metadata_sends_json_stringified_field() {
        let server = MockServer::start().await;
        let http =
            HttpClient::from_params(HttpClient::builder().base_url(server.uri()).build()).unwrap();
        let session = make_session(http, "sess1");

        let metadata = serde_json::json!({"source": "upload", "priority": 1});

        Mock::given(method("POST"))
            .and(path("/v3/workspaces/ws1/sessions/sess1/messages/upload"))
            .and(body_string_contains("\"source\":\"upload\""))
            .and(body_string_contains("\"priority\":1"))
            .respond_with(ResponseTemplate::new(200).set_body_json(upload_response_json()))
            .mount(&server)
            .await;

        let msgs = session
            .upload_file(FileSource::bytes("f.txt", b"data", "text/plain"))
            .peer("alice")
            .metadata(metadata)
            .send()
            .await
            .unwrap();

        assert_eq!(msgs.len(), 1);
    }

    #[tokio::test]
    async fn upload_file_with_configuration_sends_json_stringified() {
        let server = MockServer::start().await;
        let http =
            HttpClient::from_params(HttpClient::builder().base_url(server.uri()).build()).unwrap();
        let session = make_session(http, "sess1");

        let config = serde_json::json!({"reasoning": {"enabled": true}});

        Mock::given(method("POST"))
            .and(path("/v3/workspaces/ws1/sessions/sess1/messages/upload"))
            .and(body_string_contains("\"reasoning\""))
            .and(body_string_contains("\"enabled\":true"))
            .respond_with(ResponseTemplate::new(200).set_body_json(upload_response_json()))
            .mount(&server)
            .await;

        let msgs = session
            .upload_file(FileSource::bytes("f.txt", b"data", "text/plain"))
            .peer("bob")
            .configuration(config)
            .send()
            .await
            .unwrap();

        assert_eq!(msgs.len(), 1);
    }

    #[tokio::test]
    async fn upload_file_with_created_at_datetime_sends_iso_string() {
        let server = MockServer::start().await;
        let http =
            HttpClient::from_params(HttpClient::builder().base_url(server.uri()).build()).unwrap();
        let session = make_session(http, "sess1");

        let dt = Utc.with_ymd_and_hms(2025, 3, 14, 9, 26, 53).unwrap();

        Mock::given(method("POST"))
            .and(path("/v3/workspaces/ws1/sessions/sess1/messages/upload"))
            .and(body_string_contains("2025-03-14T09:26:53+00:00"))
            .respond_with(ResponseTemplate::new(200).set_body_json(upload_response_json()))
            .mount(&server)
            .await;

        let msgs = session
            .upload_file(FileSource::bytes("f.txt", b"data", "text/plain"))
            .peer("alice")
            .created_at(dt)
            .send()
            .await
            .unwrap();

        assert_eq!(msgs.len(), 1);
    }

    #[tokio::test]
    async fn upload_file_with_path_reads_file_and_uploads() {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join("notes.txt");
        std::fs::write(&file_path, "file from disk").unwrap();

        let server = MockServer::start().await;
        let http =
            HttpClient::from_params(HttpClient::builder().base_url(server.uri()).build()).unwrap();
        let session = make_session(http, "sess1");

        Mock::given(method("POST"))
            .and(path("/v3/workspaces/ws1/sessions/sess1/messages/upload"))
            .and(body_string_contains("file from disk"))
            .respond_with(ResponseTemplate::new(200).set_body_json(upload_response_json()))
            .mount(&server)
            .await;

        let msgs = session
            .upload_file(FileSource::path(&file_path))
            .peer("alice")
            .send()
            .await
            .unwrap();

        assert_eq!(msgs.len(), 1);
    }

    #[tokio::test]
    async fn upload_file_without_peer_returns_validation_error() {
        let server = MockServer::start().await;
        let http =
            HttpClient::from_params(HttpClient::builder().base_url(server.uri()).build()).unwrap();
        let session = make_session(http, "sess1");

        let err = session
            .upload_file(FileSource::bytes("f.txt", b"data", "text/plain"))
            .send()
            .await
            .unwrap_err();

        assert_eq!(err.code(), "validation_error");
    }

    #[tokio::test]
    async fn upload_file_streamed_uses_reader_stream() {
        let server = MockServer::start().await;
        let http =
            HttpClient::from_params(HttpClient::builder().base_url(server.uri()).build()).unwrap();
        let session = make_session(http, "sess1");

        Mock::given(method("POST"))
            .and(path("/v3/workspaces/ws1/sessions/sess1/messages/upload"))
            .and(body_string_contains("streamed payload"))
            .and(body_string_contains("peer_id"))
            .and(body_string_contains("carol"))
            .respond_with(ResponseTemplate::new(200).set_body_json(upload_response_json()))
            .mount(&server)
            .await;

        let cursor = std::io::Cursor::new(b"streamed payload".to_vec());
        let msgs = session
            .upload_file_streamed("doc.txt", cursor, "text/plain")
            .peer("carol")
            .send()
            .await
            .unwrap();

        assert_eq!(msgs.len(), 1);
    }
}
