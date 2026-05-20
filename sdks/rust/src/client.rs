//! High-level Honcho SDK client.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use reqwest::header::HeaderMap;
use serde_json::Value;
use tokio::sync::OnceCell;
use url::Url;

use crate::error::{HonchoError, Result};
use crate::http::client::HttpClient;
use crate::http::routes;
use crate::peer::Peer;
use crate::session::{PeerSpec, Session};
use crate::types::dream::QueueStatus;
use crate::types::message::MessageResponse;
use crate::types::peer::Peer as PeerResponse;
use crate::types::session::Session as SessionResponse;
use crate::types::workspace::{Workspace, WorkspaceConfiguration};

/// API environment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Environment {
    /// Local development server.
    Local,
    /// Production API at <https://api.honcho.dev>.
    #[default]
    Production,
}

impl Environment {
    fn base_url(self) -> &'static str {
        match self {
            Self::Local => "http://localhost:8000",
            Self::Production => "https://api.honcho.dev",
        }
    }
}

struct Inner {
    http: HttpClient,
    workspace_id: String,
    base_url: Url,
    ensure_workspace_once: OnceCell<()>,
}

/// Entry point for the Honcho SDK.
///
/// Construct via [`Honcho::builder()`] followed by [`Honcho::from_params()`].
#[derive(Clone)]
pub struct Honcho {
    inner: Arc<Inner>,
}

/// Parameters for constructing a [`Honcho`] client.
///
/// Resolution order: explicit argument -> environment variable -> default.
#[derive(bon::Builder)]
#[builder(on(String, into))]
#[builder(finish_fn = build)]
pub struct HonchoParams {
    /// API key. Falls back to `HONCHO_API_KEY` env var.
    api_key: Option<String>,
    /// Base URL. Falls back to `HONCHO_URL` env var, then `HONCHO_API_URL`, then [`Environment::base_url`].
    base_url: Option<String>,
    /// API environment. Defaults to [`Environment::Production`].
    #[builder(default)]
    environment: Environment,
    /// Workspace ID. Falls back to `HONCHO_WORKSPACE_ID` env var, then "default".
    workspace_id: Option<String>,
    /// Custom `reqwest::Client`.
    http_client: Option<reqwest::Client>,
    /// Request timeout. Falls back to `HttpClient` default (60s).
    timeout: Option<Duration>,
    /// Max retries for transient errors. Falls back to `HttpClient` default (2).
    max_retries: Option<u32>,
    /// Extra default headers sent with every request.
    default_headers: Option<HeaderMap>,
    /// Extra default query parameters appended to every request.
    default_query: Option<Vec<(String, String)>>,
}

impl Honcho {
    /// Quick constructor pointing at `base_url` for `workspace_id`.
    ///
    /// # Examples
    ///
    /// ```
    /// let client = honcho_ai::Honcho::new("http://localhost:8000", "my-workspace")?;
    /// # Ok::<(), honcho_ai::error::HonchoError>(())
    /// ```
    pub fn new(base_url: &str, workspace_id: &str) -> Result<Self> {
        if workspace_id.is_empty() {
            return Err(HonchoError::Configuration(
                "workspace_id must not be empty".into(),
            ));
        }
        let http =
            HttpClient::from_params(HttpClient::builder().base_url(base_url.to_string()).build())?;
        let url = Url::parse(base_url)
            .map_err(|e| HonchoError::Configuration(format!("invalid base_url: {e}")))?;
        Ok(Self {
            inner: Arc::new(Inner {
                http,
                workspace_id: workspace_id.to_string(),
                base_url: url,
                ensure_workspace_once: OnceCell::new(),
            }),
        })
    }

    /// Returns a builder for [`HonchoParams`].
    ///
    /// # Examples
    ///
    /// ```
    /// let params = honcho_ai::Honcho::builder()
    ///     .base_url("http://localhost:8000".to_owned())
    ///     .workspace_id("my-workspace".to_owned())
    ///     .build();
    /// let client = honcho_ai::Honcho::from_params(params)?;
    /// # Ok::<(), honcho_ai::error::HonchoError>(())
    /// ```
    pub fn builder() -> HonchoParamsBuilder {
        HonchoParams::builder()
    }

    /// Constructs a [`Honcho`] from params.
    ///
    /// # Examples
    ///
    /// ```
    /// let params = honcho_ai::Honcho::builder()
    ///     .base_url("http://localhost:8000".to_owned())
    ///     .build();
    /// let client = honcho_ai::Honcho::from_params(params)?;
    /// # Ok::<(), honcho_ai::error::HonchoError>(())
    /// ```
    pub fn from_params(params: HonchoParams) -> Result<Self> {
        let resolved_base_url = params
            .base_url
            .or_else(|| std::env::var("HONCHO_URL").ok())
            .or_else(|| std::env::var("HONCHO_API_URL").ok())
            .unwrap_or_else(|| params.environment.base_url().to_owned());

        let resolved_api_key = params
            .api_key
            .or_else(|| std::env::var("HONCHO_API_KEY").ok());

        let resolved_workspace_id = params
            .workspace_id
            .or_else(|| std::env::var("HONCHO_WORKSPACE_ID").ok())
            .unwrap_or_else(|| "default".to_owned());

        if resolved_workspace_id.is_empty() {
            return Err(HonchoError::Configuration(
                "workspace_id must not be empty".into(),
            ));
        }

        let base_url = Url::parse(&resolved_base_url)
            .map_err(|e| HonchoError::Configuration(format!("invalid base_url: {e}")))?;

        let http = HttpClient::from_params(
            HttpClient::builder()
                .base_url(resolved_base_url)
                .maybe_api_key(resolved_api_key)
                .maybe_http_client(params.http_client)
                .timeout(params.timeout.unwrap_or(Duration::from_secs(60)))
                .max_retries(params.max_retries.unwrap_or(2))
                .default_headers(params.default_headers.unwrap_or_default())
                .default_query(params.default_query.unwrap_or_default())
                .build(),
        )?;

        Ok(Self {
            inner: Arc::new(Inner {
                http,
                workspace_id: resolved_workspace_id,
                base_url,
                ensure_workspace_once: OnceCell::new(),
            }),
        })
    }

    /// Ensure the workspace exists on the server (`POST /v3/workspaces`).
    pub(crate) async fn ensure_workspace(&self) -> Result<()> {
        self.inner
            .ensure_workspace_once
            .get_or_try_init(|| async {
                let body = crate::types::workspace::WorkspaceCreate {
                    id: self.inner.workspace_id.clone(),
                    metadata: None,
                    configuration: None,
                };
                match self
                    .inner
                    .http
                    .post::<_, Workspace>(&routes::workspaces(), Some(&body), &[])
                    .await
                {
                    Ok(_) => Ok(()),
                    Err(e) if e.status_code() == Some(409) => Ok(()),
                    Err(e) => Err(e),
                }
            })
            .await
            .map(drop)
    }

    /// Eagerly ensure the workspace exists.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # async fn example() -> honcho_ai::error::Result<()> {
    /// let client = honcho_ai::Honcho::new("http://localhost:8000", "ws-1")?;
    /// client.force_ensure().await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn force_ensure(&self) -> Result<()> {
        self.ensure_workspace().await
    }

    /// Returns the workspace ID this client is scoped to.
    ///
    /// # Examples
    ///
    /// ```
    /// let client = honcho_ai::Honcho::new("http://localhost:8000", "ws-1")?;
    /// assert_eq!(client.workspace_id(), "ws-1");
    /// # Ok::<(), honcho_ai::error::HonchoError>(())
    /// ```
    #[must_use]
    pub fn workspace_id(&self) -> &str {
        &self.inner.workspace_id
    }

    /// Returns the resolved base URL.
    #[must_use]
    pub fn base_url(&self) -> &Url {
        &self.inner.base_url
    }

    /// Returns the underlying HTTP client.
    pub(crate) fn http(&self) -> &HttpClient {
        &self.inner.http
    }

    /// Fetch workspace metadata from the server.
    pub async fn get_metadata(&self) -> Result<HashMap<String, Value>> {
        let ws: Workspace = self
            .inner
            .http
            .get(&routes::workspace(self.workspace_id())?, &[])
            .await?;
        Ok(ws.metadata)
    }

    /// Set workspace metadata on the server.
    pub async fn set_metadata(&self, metadata: HashMap<String, Value>) -> Result<()> {
        let body = crate::types::workspace::WorkspaceMetadataSet { metadata };
        let _: Workspace = self
            .inner
            .http
            .put(&routes::workspace(self.workspace_id())?, Some(&body), &[])
            .await?;
        Ok(())
    }

    /// Fetch workspace configuration as a typed [`WorkspaceConfiguration`].
    ///
    /// # Example
    ///
    /// ```ignore
    /// let config = client.get_configuration().await?;
    /// if let Some(reasoning) = &config.reasoning {
    ///     println!("reasoning enabled: {:?}", reasoning.enabled);
    /// }
    /// ```
    pub async fn get_configuration(&self) -> Result<WorkspaceConfiguration> {
        let ws: Workspace = self
            .inner
            .http
            .get(&routes::workspace(self.workspace_id())?, &[])
            .await?;
        Ok(ws.configuration)
    }

    /// Set workspace configuration from a typed [`WorkspaceConfiguration`].
    ///
    /// # Example
    ///
    /// ```ignore
    /// let config = WorkspaceConfiguration {
    ///     reasoning: Some(ReasoningConfiguration { enabled: Some(true), custom_instructions: None, ..Default::default() }),
    ///     ..Default::default()
    /// };
    /// client.set_configuration(&config).await?;
    /// ```
    pub async fn set_configuration(&self, config: &WorkspaceConfiguration) -> Result<()> {
        let body = crate::types::workspace::WorkspaceConfigurationSet {
            configuration: serde_json::to_value(config)
                .map_err(|e| HonchoError::Configuration(e.to_string()))?,
        };
        let _: Workspace = self
            .inner
            .http
            .put(&routes::workspace(self.workspace_id())?, Some(&body), &[])
            .await?;
        Ok(())
    }

    /// Fetch workspace configuration as a raw JSON map.
    ///
    /// Prefer [`get_configuration`](Self::get_configuration) for typed access.
    /// Use this when the server returns fields not yet represented in
    /// [`WorkspaceConfiguration`].
    pub async fn get_configuration_raw(&self) -> Result<HashMap<String, Value>> {
        let raw: serde_json::Value = self
            .inner
            .http
            .get(&routes::workspace(self.workspace_id())?, &[])
            .await?;
        match raw.get("configuration") {
            Some(serde_json::Value::Object(map)) => {
                Ok(map.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
            }
            _ => Ok(HashMap::new()),
        }
    }

    /// Set workspace configuration from a raw JSON map.
    ///
    /// Prefer [`set_configuration`](Self::set_configuration) for typed access.
    /// Use this when you need to send fields not yet represented in
    /// [`WorkspaceConfiguration`].
    pub async fn set_configuration_raw(&self, configuration: HashMap<String, Value>) -> Result<()> {
        let body = crate::types::workspace::WorkspaceConfigurationSet {
            configuration: serde_json::Value::Object(configuration.into_iter().collect()),
        };
        let _: Workspace = self
            .inner
            .http
            .put(&routes::workspace(self.workspace_id())?, Some(&body), &[])
            .await?;
        Ok(())
    }

    /// Get or create a peer by ID.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # async fn example() -> honcho_ai::error::Result<()> {
    /// let client = honcho_ai::Honcho::new("http://localhost:8000", "ws-1")?;
    ///     let peer = client.peer("alice", None, None).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn peer(
        &self,
        id: impl Into<String>,
        metadata: Option<HashMap<String, Value>>,
        configuration: Option<HashMap<String, Value>>,
    ) -> Result<Peer> {
        let peer_id: String = id.into();
        if peer_id.is_empty() {
            return Err(HonchoError::Configuration(
                "peer_id must not be empty".into(),
            ));
        }
        self.ensure_workspace().await?;
        let body = crate::types::peer::PeerCreate {
            id: peer_id,
            metadata,
            configuration,
        };
        let resp: PeerResponse = self
            .inner
            .http
            .post(&routes::peers(&self.inner.workspace_id)?, Some(&body), &[])
            .await?;
        Peer::from_response(self, resp)
    }

    /// Get or create a session by ID.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # async fn example() -> honcho_ai::error::Result<()> {
    /// let client = honcho_ai::Honcho::new("http://localhost:8000", "ws-1")?;
    ///     let session = client.session("s-42", None, None, None).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn session(
        &self,
        id: impl Into<String>,
        metadata: Option<HashMap<String, Value>>,
        peers: Option<Vec<PeerSpec>>,
        configuration: Option<crate::SessionConfiguration>,
    ) -> Result<Session> {
        let session_id: String = id.into();
        if session_id.is_empty() {
            return Err(HonchoError::Configuration(
                "session_id must not be empty".into(),
            ));
        }
        self.ensure_workspace().await?;
        let peers_map = peers.map(|specs| {
            specs
                .into_iter()
                .map(|s| {
                    let (sid, cfg) = match s {
                        PeerSpec::Id(id) => (
                            id,
                            crate::SessionPeerConfig {
                                observe_me: None,
                                observe_others: None,
                            },
                        ),
                        PeerSpec::WithConfig(id, cfg) => (id, cfg),
                    };
                    (sid, cfg)
                })
                .collect()
        });
        let body = crate::types::session::SessionCreate {
            id: session_id,
            metadata,
            peers: peers_map,
            configuration,
        };
        let resp: SessionResponse = self
            .inner
            .http
            .post(
                &routes::sessions(&self.inner.workspace_id)?,
                Some(&body),
                &[],
            )
            .await?;
        Ok(Session::from_response(self, resp))
    }

    /// Refresh workspace state by re-fetching metadata and configuration.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # async fn example() -> honcho_ai::error::Result<()> {
    /// let client = honcho_ai::Honcho::new("http://localhost:8000", "ws-1")?;
    /// client.refresh().await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn refresh(&self) -> Result<()> {
        let _ = self.get_metadata().await?;
        let _ = self.get_configuration().await?;
        Ok(())
    }

    /// Search messages across the workspace.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # async fn example() -> honcho_ai::error::Result<()> {
    /// let client = honcho_ai::Honcho::new("http://localhost:8000", "ws-1")?;
    /// let results = client.search("important topic", None, None).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn search(
        &self,
        query: &str,
        limit: Option<u32>,
        filters: Option<HashMap<String, Value>>,
    ) -> Result<Vec<crate::Message>> {
        self.ensure_workspace().await?;
        let body = crate::types::workspace::WorkspaceSearchRequest {
            query: query.to_owned(),
            limit: limit.unwrap_or(10),
            filters,
        };
        let responses: Vec<MessageResponse> = self
            .inner
            .http
            .post(
                &routes::workspace_search(&self.inner.workspace_id)?,
                Some(&body),
                &[],
            )
            .await?;
        Ok(responses
            .into_iter()
            .map(|r| crate::Message::from_raw(self.inner.workspace_id.clone(), r))
            .collect())
    }

    /// Get queue processing status.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # async fn example() -> honcho_ai::error::Result<()> {
    /// let client = honcho_ai::Honcho::new("http://localhost:8000", "ws-1")?;
    /// let status = client.queue_status(None, None, None).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn queue_status(
        &self,
        observer_id: Option<&str>,
        sender_id: Option<&str>,
        session_id: Option<&str>,
    ) -> Result<QueueStatus> {
        self.ensure_workspace().await?;
        let mut query: Vec<(&str, &str)> = Vec::new();
        if let Some(v) = observer_id {
            query.push(("observer_id", v));
        }
        if let Some(v) = sender_id {
            query.push(("sender_id", v));
        }
        if let Some(v) = session_id {
            query.push(("session_id", v));
        }
        self.inner
            .http
            .get(
                &routes::workspace_queue_status(&self.inner.workspace_id)?,
                &query,
            )
            .await
    }

    /// Schedule a dream task for memory consolidation.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # async fn example() -> honcho_ai::error::Result<()> {
    /// let client = honcho_ai::Honcho::new("http://localhost:8000", "ws-1")?;
    /// client.schedule_dream("alice", None, None).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn schedule_dream(
        &self,
        observer: &str,
        session_id: Option<&str>,
        observed_peer: Option<&str>,
    ) -> Result<()> {
        self.ensure_workspace().await?;
        let observed_peer = observed_peer.unwrap_or(observer);
        let body = crate::types::dream::ScheduleDreamRequest {
            observer: observer.to_owned(),
            dream_type: crate::types::dream::DreamType::Omni,
            observed: Some(observed_peer.to_owned()),
            session_id: session_id.map(std::borrow::ToOwned::to_owned),
        };
        self.inner
            .http
            .post(
                &routes::workspace_schedule_dream(&self.inner.workspace_id)?,
                Some(&body),
                &[],
            )
            .await
    }

    /// Delete a workspace by ID.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # async fn example() -> honcho_ai::error::Result<()> {
    /// let client = honcho_ai::Honcho::new("http://localhost:8000", "ws-1")?;
    /// client.delete_workspace("old-ws").await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn delete_workspace(&self, id: &str) -> Result<()> {
        self.inner.http.delete(&routes::workspace(id)?, &[]).await
    }

    // ── Paginated list methods (F4.5) ──────────────────────────────────

    /// List peers in the workspace. Returns a paginated result.
    ///
    /// Defaults: page=1, size=50, reverse=false, no filters.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # async fn example() -> honcho_ai::error::Result<()> {
    /// let client = honcho_ai::Honcho::new("http://localhost:8000", "ws-1")?;
    /// let page = client.peers().await?;
    /// for peer in page.items() {
    ///     println!("{}", peer.id);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn peers(&self) -> Result<crate::types::pagination::Page<crate::types::peer::Peer>> {
        self.ensure_workspace().await?;
        crate::types::pagination::paginate_post(
            &self.inner.http,
            &routes::peers_list(&self.inner.workspace_id)?,
            None,
            1,
            50,
            false,
        )
        .await
    }

    /// List peers with filters.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # async fn example() -> honcho_ai::error::Result<()> {
    /// let client = honcho_ai::Honcho::new("http://localhost:8000", "ws-1")?;
    /// let mut filters = std::collections::HashMap::new();
    /// filters.insert("role".into(), "admin".into());
    /// let page = client.peers_with_filters(filters, 1, 10, false).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn peers_with_filters(
        &self,
        filters: HashMap<String, Value>,
        page: u64,
        size: u64,
        reverse: bool,
    ) -> Result<crate::types::pagination::Page<crate::types::peer::Peer>> {
        self.ensure_workspace().await?;
        let body = crate::types::peer::PeerGet {
            filters: Some(filters),
        };
        let body_val =
            serde_json::to_value(&body).map_err(|e| HonchoError::Configuration(e.to_string()))?;
        crate::types::pagination::paginate_post(
            &self.inner.http,
            &routes::peers_list(&self.inner.workspace_id)?,
            Some(&body_val),
            page,
            size,
            reverse,
        )
        .await
    }

    /// List sessions in the workspace. Returns a paginated result.
    ///
    /// Defaults: page=1, size=50, reverse=false, no filters.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # async fn example() -> honcho_ai::error::Result<()> {
    /// let client = honcho_ai::Honcho::new("http://localhost:8000", "ws-1")?;
    /// let page = client.sessions().await?;
    /// for session in page.items() {
    ///     println!("{}", session.id);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn sessions(
        &self,
    ) -> Result<crate::types::pagination::Page<crate::types::session::Session>> {
        self.ensure_workspace().await?;
        crate::types::pagination::paginate_post(
            &self.inner.http,
            &routes::sessions_list(&self.inner.workspace_id)?,
            None,
            1,
            50,
            false,
        )
        .await
    }

    /// List sessions with filters.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # async fn example() -> honcho_ai::error::Result<()> {
    /// let client = honcho_ai::Honcho::new("http://localhost:8000", "ws-1")?;
    /// let mut filters = std::collections::HashMap::new();
    /// filters.insert("is_active".into(), true.into());
    /// let page = client.sessions_with_filters(filters, 1, 10, false).await?;
    /// # Ok(())
    /// # }
    /// ```
    pub async fn sessions_with_filters(
        &self,
        filters: HashMap<String, Value>,
        page: u64,
        size: u64,
        reverse: bool,
    ) -> Result<crate::types::pagination::Page<crate::types::session::Session>> {
        self.ensure_workspace().await?;
        let body = crate::types::session::SessionGet {
            filters: Some(filters),
        };
        let body_val =
            serde_json::to_value(&body).map_err(|e| HonchoError::Configuration(e.to_string()))?;
        crate::types::pagination::paginate_post(
            &self.inner.http,
            &routes::sessions_list(&self.inner.workspace_id)?,
            Some(&body_val),
            page,
            size,
            reverse,
        )
        .await
    }

    /// List workspace IDs. Returns a paginated result of ID strings.
    ///
    /// Defaults: page=1, size=50, reverse=false, no filters.
    /// No workspace scope required — queries all workspaces.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// # async fn example() -> honcho_ai::error::Result<()> {
    /// let client = honcho_ai::Honcho::new("http://localhost:8000", "ws-1")?;
    /// let page = client.workspaces().await?;
    /// for id in page.items() {
    ///     println!("{id}");
    /// }
    /// # Ok(())
    /// # }
    /// ```
    pub async fn workspaces(&self) -> Result<crate::types::pagination::Page<Workspace, String>> {
        let page = crate::types::pagination::paginate_post::<Workspace>(
            &self.inner.http,
            &routes::workspaces_list(),
            None,
            1,
            50,
            false,
        )
        .await?;
        Ok(page.map(|ws| ws.id))
    }
}
