//! High-level Honcho SDK client.

use std::collections::HashMap;
use std::sync::Arc;

use serde_json::Value;
use tokio::sync::OnceCell;
use url::Url;

use crate::error::{HonchoError, Result};
use crate::http::client::HttpClient;
use crate::http::routes;
use crate::peer::Peer;
use crate::session::Session;
use crate::types::dream::QueueStatus;
use crate::types::message::Message;
use crate::types::peer::Peer as PeerResponse;
use crate::types::session::Session as SessionResponse;
use crate::types::workspace::Workspace;

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
    /// Base URL. Falls back to `HONCHO_URL` env var, then [`Environment::base_url`].
    base_url: Option<String>,
    /// API environment. Defaults to [`Environment::Production`].
    #[builder(default)]
    environment: Environment,
    /// Workspace ID. Falls back to `HONCHO_WORKSPACE_ID` env var, then "default".
    workspace_id: Option<String>,
    /// Custom `reqwest::Client`.
    http_client: Option<reqwest::Client>,
}

impl Honcho {
    /// Quick constructor pointing at `base_url` for `workspace_id`.
    pub fn new(base_url: &str, workspace_id: &str) -> Result<Self> {
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
    pub fn builder() -> HonchoParamsBuilder {
        HonchoParams::builder()
    }

    /// Constructs a [`Honcho`] from params.
    pub fn from_params(params: HonchoParams) -> Result<Self> {
        let resolved_base_url = params
            .base_url
            .or_else(|| std::env::var("HONCHO_URL").ok())
            .unwrap_or_else(|| params.environment.base_url().to_owned());

        let resolved_api_key = params
            .api_key
            .or_else(|| std::env::var("HONCHO_API_KEY").ok());

        let resolved_workspace_id = params
            .workspace_id
            .or_else(|| std::env::var("HONCHO_WORKSPACE_ID").ok())
            .unwrap_or_else(|| "default".to_owned());

        let base_url = Url::parse(&resolved_base_url)
            .map_err(|e| HonchoError::Configuration(format!("invalid base_url: {e}")))?;

        let http = HttpClient::from_params(
            HttpClient::builder()
                .base_url(resolved_base_url)
                .maybe_api_key(resolved_api_key)
                .maybe_http_client(params.http_client)
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
                let body = serde_json::json!({"id": self.inner.workspace_id});
                self.inner
                    .http
                    .post::<_, Workspace>(&routes::workspaces(), Some(&body), &[])
                    .await
                    .map(drop)
            })
            .await
            .map(drop)
    }

    /// Eagerly ensure the workspace exists.
    pub async fn force_ensure(&self) -> Result<()> {
        self.ensure_workspace().await
    }

    /// Returns the workspace ID this client is scoped to.
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
        let body = serde_json::json!({"id": self.workspace_id()});
        let ws: Workspace = self
            .inner
            .http
            .post(&routes::workspaces(), Some(&body), &[])
            .await?;
        Ok(ws.metadata)
    }

    /// Set workspace metadata on the server.
    pub async fn set_metadata(&self, metadata: HashMap<String, Value>) -> Result<()> {
        let body = serde_json::json!({"metadata": metadata});
        let _: Workspace = self
            .inner
            .http
            .put(&routes::workspace(self.workspace_id()), Some(&body), &[])
            .await?;
        Ok(())
    }

    /// Fetch workspace configuration from the server.
    pub async fn get_configuration(&self) -> Result<HashMap<String, Value>> {
        let body = serde_json::json!({"id": self.workspace_id()});
        let ws: Workspace = self
            .inner
            .http
            .post(&routes::workspaces(), Some(&body), &[])
            .await?;
        Ok(ws.configuration)
    }

    /// Set workspace configuration on the server.
    pub async fn set_configuration(&self, configuration: HashMap<String, Value>) -> Result<()> {
        let body = serde_json::json!({"configuration": configuration});
        let _: Workspace = self
            .inner
            .http
            .put(&routes::workspace(self.workspace_id()), Some(&body), &[])
            .await?;
        Ok(())
    }

    /// Get or create a peer by ID.
    pub async fn peer(&self, id: impl Into<String>) -> Result<Peer> {
        self.ensure_workspace().await?;
        let body = serde_json::json!({"id": id.into()});
        let resp: PeerResponse = self
            .inner
            .http
            .post(&routes::peers(&self.inner.workspace_id), Some(&body), &[])
            .await?;
        Ok(Peer::from_response(self, resp))
    }

    /// Get or create a session by ID.
    pub async fn session(&self, id: impl Into<String>) -> Result<Session> {
        self.ensure_workspace().await?;
        let body = serde_json::json!({"id": id.into()});
        let resp: SessionResponse = self
            .inner
            .http
            .post(
                &routes::sessions(&self.inner.workspace_id),
                Some(&body),
                &[],
            )
            .await?;
        Ok(Session::from_response(self, resp))
    }

    /// Search messages across the workspace.
    pub async fn search(&self, query: &str) -> Result<Vec<Message>> {
        self.ensure_workspace().await?;
        let body = serde_json::json!({"query": query, "filters": null, "limit": 10});
        self.inner
            .http
            .post(
                &routes::workspace_search(&self.inner.workspace_id),
                Some(&body),
                &[],
            )
            .await
    }

    /// Get queue processing status.
    pub async fn queue_status(&self) -> Result<QueueStatus> {
        self.ensure_workspace().await?;
        self.inner
            .http
            .get(
                &routes::workspace_queue_status(&self.inner.workspace_id),
                &[],
            )
            .await
    }

    /// Schedule a dream task for memory consolidation.
    pub async fn schedule_dream(&self, observer: &str) -> Result<()> {
        self.ensure_workspace().await?;
        let body = serde_json::json!({"observer": observer, "observed": observer, "session_id": null, "dream_type": "omni"});
        self.inner
            .http
            .post(
                &routes::workspace_schedule_dream(&self.inner.workspace_id),
                Some(&body),
                &[],
            )
            .await
    }

    /// Delete a workspace by ID.
    pub async fn delete_workspace(&self, id: &str) -> Result<()> {
        self.inner.http.delete(&routes::workspace(id), &[]).await
    }

    // ── Paginated list methods (F4.5) ──────────────────────────────────

    /// List peers in the workspace. Returns a paginated result.
    ///
    /// Defaults: page=1, size=50, reverse=false, no filters.
    pub async fn peers(&self) -> Result<crate::types::pagination::Page<crate::types::peer::Peer>> {
        self.ensure_workspace().await?;
        crate::types::pagination::paginate_post(
            &self.inner.http,
            &routes::peers_list(&self.inner.workspace_id),
            None,
            1,
            50,
            false,
        )
        .await
    }

    /// List peers with filters.
    pub async fn peers_with_filters(
        &self,
        filters: HashMap<String, Value>,
        page: u64,
        size: u64,
        reverse: bool,
    ) -> Result<crate::types::pagination::Page<crate::types::peer::Peer>> {
        self.ensure_workspace().await?;
        let body = serde_json::json!({"filters": filters});
        crate::types::pagination::paginate_post(
            &self.inner.http,
            &routes::peers_list(&self.inner.workspace_id),
            Some(&body),
            page,
            size,
            reverse,
        )
        .await
    }

    /// List sessions in the workspace. Returns a paginated result.
    ///
    /// Defaults: page=1, size=50, reverse=false, no filters.
    pub async fn sessions(
        &self,
    ) -> Result<crate::types::pagination::Page<crate::types::session::Session>> {
        self.ensure_workspace().await?;
        crate::types::pagination::paginate_post(
            &self.inner.http,
            &routes::sessions_list(&self.inner.workspace_id),
            None,
            1,
            50,
            false,
        )
        .await
    }

    /// List sessions with filters.
    pub async fn sessions_with_filters(
        &self,
        filters: HashMap<String, Value>,
        page: u64,
        size: u64,
        reverse: bool,
    ) -> Result<crate::types::pagination::Page<crate::types::session::Session>> {
        self.ensure_workspace().await?;
        let body = serde_json::json!({"filters": filters});
        crate::types::pagination::paginate_post(
            &self.inner.http,
            &routes::sessions_list(&self.inner.workspace_id),
            Some(&body),
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
