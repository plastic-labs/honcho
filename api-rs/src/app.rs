use axum::extract::{Path, Query, State};
use axum::http::HeaderMap;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post, put};
use axum::{Json, Router};
use chrono::{DateTime, Utc};
use serde::Deserialize;
use serde_json::{Value, json};
use sqlx::PgPool;
use std::collections::BTreeMap;
use std::net::IpAddr;

use crate::auth::{AuthConfig, AuthError, JwtParams, authorize, create_scoped_key};
use crate::cache::PeerCache;
use crate::config::EmbeddingConfig;
use crate::db;
use crate::embedding::{self, EmbedError};
use crate::error::ApiError;
use crate::filters::{FilterTarget, build_filter_clause};
use crate::llm::http::{Credentials, ReqwestHttp};
use crate::pagination::Pagination;
use crate::search::{self, HybridSearchParams};

#[derive(Debug, Clone)]
pub struct AppState {
    pub pool: Option<PgPool>,
    pub auth: AuthConfig,
    pub db_schema: String,
    pub write_enabled: bool,
    pub peer_cache: PeerCache,
    pub embed_messages: bool,
    pub embedding_max_tokens: usize,
    pub embedding: EmbeddingConfig,
    pub dream_enabled: bool,
}

fn default_test_embedding_config() -> EmbeddingConfig {
    EmbeddingConfig {
        model: "text-embedding-3-small".to_string(),
        vector_dimensions: 1536,
        send_dimensions: false,
        max_tokens: 8192,
        api_key: None,
        base_url: None,
    }
}

impl AppState {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        pool: PgPool,
        auth: AuthConfig,
        db_schema: String,
        write_enabled: bool,
        peer_cache: PeerCache,
        embed_messages: bool,
        embedding_max_tokens: usize,
        embedding: EmbeddingConfig,
        dream_enabled: bool,
    ) -> Self {
        Self {
            pool: Some(pool),
            auth,
            db_schema,
            write_enabled,
            peer_cache,
            embed_messages,
            embedding_max_tokens,
            embedding,
            dream_enabled,
        }
    }

    pub fn for_test(auth: AuthConfig) -> Self {
        Self {
            pool: None,
            auth,
            db_schema: "public".to_string(),
            write_enabled: false,
            peer_cache: PeerCache::disabled(),
            embed_messages: false,
            embedding_max_tokens: 8192,
            embedding: default_test_embedding_config(),
            dream_enabled: true,
        }
    }

    pub fn for_test_with_writes(auth: AuthConfig) -> Self {
        Self {
            pool: None,
            auth,
            db_schema: "public".to_string(),
            write_enabled: true,
            peer_cache: PeerCache::disabled(),
            embed_messages: false,
            embedding_max_tokens: 8192,
            embedding: default_test_embedding_config(),
            dream_enabled: true,
        }
    }

    pub fn pool(&self) -> Result<&PgPool, ApiError> {
        self.pool.as_ref().ok_or(ApiError::MissingPool)
    }
}

#[derive(Debug, Deserialize)]
pub struct ListRequest {
    pub filters: Option<Value>,
}

#[derive(Debug, Deserialize)]
pub struct ListQuery {
    pub page: Option<u64>,
    pub size: Option<u64>,
    #[serde(default)]
    pub reverse: bool,
}

#[derive(Debug, Deserialize)]
pub struct QueueStatusQuery {
    pub observer_id: Option<String>,
    pub sender_id: Option<String>,
    pub session_id: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct PeerCardQuery {
    pub target: Option<String>,
}

/// Query parameters for `GET .../sessions/{id}/context`. Numeric/boolean params
/// are captured as raw strings so the handler can reproduce FastAPI's
/// `int_parsing` / `less_than_equal` / `bool_parsing` error shapes byte-for-byte.
/// Only the base (no-`peer_target`) path is ported; the perspective params are
/// captured solely to branch onto a 501.
#[derive(Debug, Deserialize)]
pub struct GetContextQuery {
    pub tokens: Option<String>,
    pub summary: Option<String>,
    pub peer_target: Option<String>,
    pub peer_perspective: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct CloneSessionQuery {
    pub message_id: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct CreateKeyQuery {
    pub workspace_id: Option<String>,
    pub peer_id: Option<String>,
    pub session_id: Option<String>,
    pub expires_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Deserialize)]
pub struct WorkspaceCreateRequest {
    pub name: String,
    name_field: &'static str,
    pub metadata: Value,
    pub configuration: Value,
}

#[derive(Debug)]
pub struct WorkspaceUpdateRequest {
    pub metadata: Option<Value>,
    pub configuration: Option<Value>,
}

#[derive(Debug)]
pub struct PeerCreateRequest {
    pub name: String,
    name_field: &'static str,
    pub metadata: Option<Value>,
    pub configuration: Option<Value>,
}

#[derive(Debug)]
pub struct PeerUpdateRequest {
    pub metadata: Option<Value>,
    pub configuration: Option<Value>,
}

#[derive(Debug)]
pub struct SessionCreateRequest {
    pub name: String,
    name_field: &'static str,
    pub metadata: Option<Value>,
    pub configuration: Option<Value>,
}

#[derive(Debug)]
pub struct SessionUpdateRequest {
    pub metadata: Option<Value>,
    pub configuration: Option<Value>,
}

#[derive(Debug)]
pub struct SessionPeerConfig {
    pub observe_me: Option<bool>,
    pub observe_others: Option<bool>,
}

#[derive(Debug)]
pub struct SessionPeersRequest {
    pub peers: BTreeMap<String, SessionPeerConfig>,
}

pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v3/keys", post(create_key))
        .route("/v3/workspaces", post(get_or_create_workspace))
        .route(
            "/v3/workspaces/{workspace_id}",
            put(update_workspace).delete(delete_workspace),
        )
        .route(
            "/v3/workspaces/{workspace_id}/peers",
            post(get_or_create_peer),
        )
        .route("/v3/workspaces/{workspace_id}/search", post(search_workspace))
        .route(
            "/v3/workspaces/{workspace_id}/peers/{peer_id}/search",
            post(search_peer),
        )
        .route(
            "/v3/workspaces/{workspace_id}/sessions/{session_id}/search",
            post(search_session),
        )
        .route("/v3/workspaces/list", post(list_workspaces))
        .route("/v3/workspaces/{workspace_id}/peers/list", post(list_peers))
        .route(
            "/v3/workspaces/{workspace_id}/peers/{peer_id}",
            put(update_peer),
        )
        .route(
            "/v3/workspaces/{workspace_id}/peers/{peer_id}/card",
            get(get_peer_card).put(set_peer_card),
        )
        .route(
            "/v3/workspaces/{workspace_id}/sessions/list",
            post(list_sessions),
        )
        .route(
            "/v3/workspaces/{workspace_id}/sessions",
            post(get_or_create_session),
        )
        .route(
            "/v3/workspaces/{workspace_id}/sessions/{session_id}",
            put(update_session).delete(delete_session),
        )
        .route(
            "/v3/workspaces/{workspace_id}/sessions/{session_id}/clone",
            post(clone_session),
        )
        .route(
            "/v3/workspaces/{workspace_id}/peers/{peer_id}/sessions",
            post(list_peer_sessions),
        )
        .route(
            "/v3/workspaces/{workspace_id}/sessions/{session_id}/peers",
            get(get_session_peers)
                .post(add_peers_to_session)
                .put(set_session_peers)
                .delete(remove_peers_from_session),
        )
        .route(
            "/v3/workspaces/{workspace_id}/sessions/{session_id}/peers/{peer_id}/config",
            get(get_session_peer_config).put(set_session_peer_config),
        )
        .route(
            "/v3/workspaces/{workspace_id}/sessions/{session_id}/summaries",
            get(get_session_summaries),
        )
        .route(
            "/v3/workspaces/{workspace_id}/sessions/{session_id}/context",
            get(get_session_context),
        )
        .route(
            "/v3/workspaces/{workspace_id}/queue/status",
            get(queue_status),
        )
        .route(
            "/v3/workspaces/{workspace_id}/schedule_dream",
            post(schedule_dream),
        )
        .route(
            "/v3/workspaces/{workspace_id}/webhooks",
            get(list_webhook_endpoints).post(get_or_create_webhook_endpoint),
        )
        .route(
            "/v3/workspaces/{workspace_id}/webhooks/{endpoint_id}",
            axum::routing::delete(delete_webhook_endpoint),
        )
        .route(
            "/v3/workspaces/{workspace_id}/sessions/{session_id}/messages",
            post(create_messages_for_session),
        )
        .route(
            "/v3/workspaces/{workspace_id}/sessions/{session_id}/messages/",
            post(create_messages_for_session),
        )
        .route(
            "/v3/workspaces/{workspace_id}/sessions/{session_id}/messages/list",
            post(list_messages),
        )
        .route(
            "/v3/workspaces/{workspace_id}/conclusions",
            post(create_conclusions),
        )
        .route(
            "/v3/workspaces/{workspace_id}/conclusions/list",
            post(list_conclusions),
        )
        .route(
            "/v3/workspaces/{workspace_id}/conclusions/query",
            post(query_conclusions),
        )
        .route(
            "/v3/workspaces/{workspace_id}/conclusions/{conclusion_id}",
            axum::routing::delete(delete_conclusion),
        )
        .route(
            "/v3/workspaces/{workspace_id}/sessions/{session_id}/messages/{message_id}",
            get(get_message).put(update_message),
        )
        .with_state(state)
}

async fn health() -> Json<Value> {
    Json(json!({ "status": "ok" }))
}

async fn create_key(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(query): Query<CreateKeyQuery>,
) -> Result<Json<Value>, ApiError> {
    authorize(
        &state.auth,
        authorization_header(&headers),
        true,
        None,
        None,
        None,
    )?;
    if !state.auth.use_auth {
        return Err(ApiError::Disabled);
    }
    if !has_non_empty_scope(&query) {
        return Err(ApiError::Validation(
            "At least one of workspace_id, peer_id, or session_id must be provided".to_string(),
        ));
    }
    let token = create_scoped_key(
        JwtParams {
            exp: query.expires_at.map(|expires_at| expires_at.to_rfc3339()),
            workspace: query.workspace_id,
            peer: query.peer_id,
            session: query.session_id,
            ..JwtParams::default()
        },
        state
            .auth
            .jwt_secret
            .as_deref()
            .ok_or(AuthError::InvalidJwt)?,
    )?;
    Ok(Json(json!({ "key": token })))
}

async fn get_or_create_workspace(
    State(state): State<AppState>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Result<Response, ApiError> {
    let params = authorize(
        &state.auth,
        authorization_header(&headers),
        false,
        None,
        None,
        None,
    )?;
    ensure_writes_enabled(&state)?;

    let WorkspaceCreateRequest {
        name: workspace_name,
        name_field,
        metadata,
        configuration,
    } = parse_workspace_create(body)?;
    if !params.admin.unwrap_or(false)
        && params.workspace.as_deref() != Some(workspace_name.as_str())
    {
        return Err(ApiError::Authentication(
            "Unauthorized access to resource".to_string(),
        ));
    }
    validate_resource_name(&workspace_name, name_field)?;

    let (value, created) =
        db::get_or_create_workspace(state.pool()?, &workspace_name, metadata, configuration)
            .await?;
    let status = if created {
        StatusCode::CREATED
    } else {
        StatusCode::OK
    };
    Ok((status, Json(value)).into_response())
}

async fn update_workspace(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(workspace_id): Path<String>,
    Json(body): Json<Value>,
) -> Result<Json<Value>, ApiError> {
    authorize(
        &state.auth,
        authorization_header(&headers),
        false,
        Some(&workspace_id),
        None,
        None,
    )?;
    ensure_writes_enabled(&state)?;
    validate_resource_name(&workspace_id, "workspace_id")?;

    let body = parse_workspace_update(body)?;
    let value = db::update_workspace(
        state.pool()?,
        &workspace_id,
        body.metadata,
        body.configuration,
    )
    .await?;
    Ok(Json(value))
}

async fn delete_workspace(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(workspace_id): Path<String>,
) -> Result<Response, ApiError> {
    authorize(
        &state.auth,
        authorization_header(&headers),
        false,
        Some(&workspace_id),
        None,
        None,
    )?;
    ensure_writes_enabled(&state)?;

    match db::enqueue_workspace_deletion(state.pool()?, &workspace_id).await {
        Ok(()) => Ok((
            StatusCode::ACCEPTED,
            Json(json!({ "message": "Workspace deletion accepted" })),
        )
            .into_response()),
        Err(db::WorkspaceDeleteError::NotFound) => {
            Err(ApiError::NotFound(format!("Workspace {workspace_id} not found")))
        }
        Err(db::WorkspaceDeleteError::ActiveSessions) => Err(ApiError::Conflict(format!(
            "Cannot delete workspace '{workspace_id}': active session(s) remain. \
             Delete all sessions first."
        ))),
        Err(db::WorkspaceDeleteError::Database(error)) => Err(ApiError::Database(error)),
    }
}

async fn get_or_create_peer(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(workspace_id): Path<String>,
    Json(body): Json<Value>,
) -> Result<Response, ApiError> {
    let params = authorize(
        &state.auth,
        authorization_header(&headers),
        false,
        None,
        None,
        None,
    )?;
    ensure_writes_enabled(&state)?;
    validate_resource_name(&workspace_id, "workspace_id")?;

    let body = parse_peer_create(body)?;
    let peer_name = body.name;
    validate_resource_name(&peer_name, body.name_field)?;
    authorize_peer_create(&params, &workspace_id, &peer_name)?;

    let (value, created) = db::get_or_create_peer(
        state.pool()?,
        &workspace_id,
        &peer_name,
        body.metadata,
        body.configuration,
    )
    .await?;
    state
        .peer_cache
        .invalidate_peer(&workspace_id, &peer_name)
        .await;
    let status = if created {
        StatusCode::CREATED
    } else {
        StatusCode::OK
    };
    Ok((status, Json(value)).into_response())
}

async fn update_peer(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path((workspace_id, peer_id)): Path<(String, String)>,
    Json(body): Json<Value>,
) -> Result<Json<Value>, ApiError> {
    authorize(
        &state.auth,
        authorization_header(&headers),
        false,
        Some(&workspace_id),
        Some(&peer_id),
        None,
    )?;
    ensure_writes_enabled(&state)?;
    validate_resource_name(&workspace_id, "workspace_id")?;
    validate_resource_name(&peer_id, "peer_id")?;

    let body = parse_peer_update(body)?;
    let value = db::update_peer(
        state.pool()?,
        &workspace_id,
        &peer_id,
        body.metadata,
        body.configuration,
    )
    .await?;
    state
        .peer_cache
        .invalidate_peer(&workspace_id, &peer_id)
        .await;
    Ok(Json(value))
}

async fn get_or_create_session(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(workspace_id): Path<String>,
    Json(body): Json<Value>,
) -> Result<Response, ApiError> {
    let params = authorize(
        &state.auth,
        authorization_header(&headers),
        false,
        None,
        None,
        None,
    )?;
    ensure_writes_enabled(&state)?;
    validate_resource_name(&workspace_id, "workspace_id")?;

    let body = parse_session_create(body)?;
    let session_name = body.name;
    validate_resource_name(&session_name, body.name_field)?;
    authorize_session_create(&params, &workspace_id, &session_name)?;

    let (value, created) = db::get_or_create_session(
        state.pool()?,
        &workspace_id,
        &session_name,
        body.metadata,
        body.configuration,
    )
    .await
    .map_err(|error| session_write_error(error, &workspace_id, &session_name))?;
    state
        .peer_cache
        .invalidate_session(&workspace_id, &session_name)
        .await;
    let status = if created {
        StatusCode::CREATED
    } else {
        StatusCode::OK
    };
    Ok((status, Json(value)).into_response())
}

async fn update_session(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path((workspace_id, session_id)): Path<(String, String)>,
    Json(body): Json<Value>,
) -> Result<Json<Value>, ApiError> {
    authorize(
        &state.auth,
        authorization_header(&headers),
        false,
        Some(&workspace_id),
        None,
        Some(&session_id),
    )?;
    ensure_writes_enabled(&state)?;
    validate_resource_name(&workspace_id, "workspace_id")?;
    validate_resource_name(&session_id, "session_id")?;

    let body = parse_session_update(body)?;
    let value = db::update_session(
        state.pool()?,
        &workspace_id,
        &session_id,
        body.metadata,
        body.configuration,
    )
    .await
    .map_err(|error| session_write_error(error, &workspace_id, &session_id))?;
    state
        .peer_cache
        .invalidate_session(&workspace_id, &session_id)
        .await;
    Ok(Json(value))
}

async fn delete_session(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path((workspace_id, session_id)): Path<(String, String)>,
) -> Result<(StatusCode, Json<Value>), ApiError> {
    authorize(
        &state.auth,
        authorization_header(&headers),
        false,
        Some(&workspace_id),
        None,
        Some(&session_id),
    )?;
    ensure_writes_enabled(&state)?;
    validate_resource_name(&workspace_id, "workspace_id")?;
    validate_resource_name(&session_id, "session_id")?;

    db::delete_session(state.pool()?, &workspace_id, &session_id)
        .await
        .map_err(|error| session_write_error(error, &workspace_id, &session_id))?;
    state
        .peer_cache
        .invalidate_session(&workspace_id, &session_id)
        .await;
    Ok((
        StatusCode::ACCEPTED,
        Json(json!({"message": "Session deleted successfully"})),
    ))
}

async fn clone_session(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path((workspace_id, session_id)): Path<(String, String)>,
    Query(query): Query<CloneSessionQuery>,
) -> Result<(StatusCode, Json<Value>), ApiError> {
    authorize(
        &state.auth,
        authorization_header(&headers),
        false,
        Some(&workspace_id),
        None,
        Some(&session_id),
    )?;
    ensure_writes_enabled(&state)?;
    validate_resource_name(&workspace_id, "workspace_id")?;
    validate_resource_name(&session_id, "session_id")?;

    let value = db::clone_session(
        state.pool()?,
        &workspace_id,
        &session_id,
        query.message_id.as_deref(),
    )
    .await
    .map_err(clone_session_error)?;
    Ok((StatusCode::CREATED, Json(value)))
}

fn clone_session_error(error: db::CloneSessionError) -> ApiError {
    match error {
        db::CloneSessionError::OriginalNotFound => {
            ApiError::NotFound("Original session not found".to_string())
        }
        db::CloneSessionError::CutoffMessageNotFound => {
            ApiError::NotFound("Session not found".to_string())
        }
        db::CloneSessionError::Database(error) => ApiError::Database(error),
    }
}

async fn add_peers_to_session(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path((workspace_id, session_id)): Path<(String, String)>,
    Json(body): Json<Value>,
) -> Result<Json<Value>, ApiError> {
    authorize(
        &state.auth,
        authorization_header(&headers),
        false,
        Some(&workspace_id),
        None,
        Some(&session_id),
    )?;
    ensure_writes_enabled(&state)?;
    validate_resource_name(&workspace_id, "workspace_id")?;
    validate_resource_name(&session_id, "session_id")?;

    let body = parse_session_peers(body)?;
    for peer_name in body.peers.keys() {
        validate_resource_name(peer_name, "peer_id")?;
    }
    ensure_session_observer_limit(state.pool()?, &workspace_id, &session_id, &body).await?;

    let peer_configs = body
        .peers
        .iter()
        .map(|(peer_name, config)| (peer_name.clone(), session_peer_config_json(config)))
        .collect::<BTreeMap<_, _>>();
    let value = db::add_peers_to_session(state.pool()?, &workspace_id, &session_id, &peer_configs)
        .await
        .map_err(|error| session_write_error(error, &workspace_id, &session_id))?;
    state
        .peer_cache
        .invalidate_session(&workspace_id, &session_id)
        .await;
    for peer_name in peer_configs.keys() {
        state
            .peer_cache
            .invalidate_peer(&workspace_id, peer_name)
            .await;
    }
    Ok(Json(value))
}

async fn set_session_peers(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path((workspace_id, session_id)): Path<(String, String)>,
    Json(body): Json<Value>,
) -> Result<Json<Value>, ApiError> {
    authorize(
        &state.auth,
        authorization_header(&headers),
        false,
        Some(&workspace_id),
        None,
        Some(&session_id),
    )?;
    ensure_writes_enabled(&state)?;
    validate_resource_name(&workspace_id, "workspace_id")?;
    validate_resource_name(&session_id, "session_id")?;

    let body = parse_session_peers(body)?;
    for peer_name in body.peers.keys() {
        validate_resource_name(peer_name, "peer_id")?;
    }
    ensure_session_set_observer_limit(&session_id, &body)?;

    let peer_configs = body
        .peers
        .iter()
        .map(|(peer_name, config)| (peer_name.clone(), session_peer_config_json(config)))
        .collect::<BTreeMap<_, _>>();
    let value = db::set_peers_for_session(state.pool()?, &workspace_id, &session_id, &peer_configs)
        .await
        .map_err(|error| session_write_error(error, &workspace_id, &session_id))?;
    state
        .peer_cache
        .invalidate_session(&workspace_id, &session_id)
        .await;
    for peer_name in peer_configs.keys() {
        state
            .peer_cache
            .invalidate_peer(&workspace_id, peer_name)
            .await;
    }
    Ok(Json(value))
}

async fn remove_peers_from_session(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path((workspace_id, session_id)): Path<(String, String)>,
    Json(body): Json<Value>,
) -> Result<Json<Value>, ApiError> {
    authorize(
        &state.auth,
        authorization_header(&headers),
        false,
        Some(&workspace_id),
        None,
        Some(&session_id),
    )?;
    ensure_writes_enabled(&state)?;
    validate_resource_name(&workspace_id, "workspace_id")?;
    validate_resource_name(&session_id, "session_id")?;

    let peer_names = parse_peer_name_list(body)?;
    for peer_name in &peer_names {
        validate_resource_name(peer_name, "peer_id")?;
    }

    let value =
        db::remove_peers_from_session(state.pool()?, &workspace_id, &session_id, &peer_names)
            .await
            .map_err(|error| session_write_error(error, &workspace_id, &session_id))?;
    state
        .peer_cache
        .invalidate_session(&workspace_id, &session_id)
        .await;
    for peer_name in &peer_names {
        state
            .peer_cache
            .invalidate_peer(&workspace_id, peer_name)
            .await;
    }
    Ok(Json(value))
}

async fn set_session_peer_config(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path((workspace_id, session_id, peer_id)): Path<(String, String, String)>,
    Json(body): Json<Value>,
) -> Result<StatusCode, ApiError> {
    authorize(
        &state.auth,
        authorization_header(&headers),
        false,
        Some(&workspace_id),
        None,
        Some(&session_id),
    )?;
    ensure_writes_enabled(&state)?;
    validate_resource_name(&workspace_id, "workspace_id")?;
    validate_resource_name(&session_id, "session_id")?;
    validate_resource_name(&peer_id, "peer_id")?;

    let config = parse_session_peer_config(body, &[])?;
    db::ensure_active_session_exists(state.pool()?, &workspace_id, &session_id)
        .await
        .map_err(|error| session_write_error(error, &workspace_id, &session_id))?;
    if !db::peer_exists(state.pool()?, &workspace_id, &peer_id).await? {
        return Err(ApiError::NotFound(format!(
            "Peer {peer_id} not found in workspace {workspace_id}"
        )));
    }
    ensure_session_peer_config_observer_limit(
        state.pool()?,
        &workspace_id,
        &session_id,
        &peer_id,
        &config,
    )
    .await?;
    db::set_session_peer_config(
        state.pool()?,
        &workspace_id,
        &session_id,
        &peer_id,
        session_peer_config_patch_json(&config),
    )
    .await?;
    state
        .peer_cache
        .invalidate_session(&workspace_id, &session_id)
        .await;
    state
        .peer_cache
        .invalidate_peer(&workspace_id, &peer_id)
        .await;
    Ok(StatusCode::NO_CONTENT)
}

async fn list_workspaces(
    State(state): State<AppState>,
    headers: HeaderMap,
    Query(query): Query<ListQuery>,
    body: Option<Json<ListRequest>>,
) -> Result<Json<Value>, ApiError> {
    authorize(
        &state.auth,
        authorization_header(&headers),
        true,
        None,
        None,
        None,
    )?;
    let filter = build_filter_clause(
        FilterTarget::Workspace,
        body.as_ref().and_then(|Json(body)| body.filters.as_ref()),
    )?;
    let page = Pagination::new(query.page, query.size);
    let value = db::list_workspaces(state.pool()?, &filter, page, query.reverse).await?;
    Ok(Json(value))
}

async fn list_peers(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(workspace_id): Path<String>,
    Query(query): Query<ListQuery>,
    body: Option<Json<ListRequest>>,
) -> Result<Json<Value>, ApiError> {
    authorize(
        &state.auth,
        authorization_header(&headers),
        false,
        Some(&workspace_id),
        None,
        None,
    )?;
    let filter = build_filter_clause(
        FilterTarget::Peer,
        body.as_ref().and_then(|Json(body)| body.filters.as_ref()),
    )?;
    let page = Pagination::new(query.page, query.size);
    let value = db::list_peers(state.pool()?, &workspace_id, &filter, page, query.reverse).await?;
    Ok(Json(value))
}

async fn list_sessions(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(workspace_id): Path<String>,
    Query(query): Query<ListQuery>,
    body: Option<Json<ListRequest>>,
) -> Result<Json<Value>, ApiError> {
    authorize(
        &state.auth,
        authorization_header(&headers),
        false,
        Some(&workspace_id),
        None,
        None,
    )?;
    let filter = build_filter_clause(
        FilterTarget::Session,
        body.as_ref().and_then(|Json(body)| body.filters.as_ref()),
    )?;
    let page = Pagination::new(query.page, query.size);
    let value =
        db::list_sessions(state.pool()?, &workspace_id, &filter, page, query.reverse).await?;
    Ok(Json(value))
}

async fn list_peer_sessions(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path((workspace_id, peer_id)): Path<(String, String)>,
    Query(query): Query<ListQuery>,
    body: Option<Json<ListRequest>>,
) -> Result<Json<Value>, ApiError> {
    authorize(
        &state.auth,
        authorization_header(&headers),
        false,
        Some(&workspace_id),
        Some(&peer_id),
        None,
    )?;
    let filter = build_filter_clause(
        FilterTarget::Session,
        body.as_ref().and_then(|Json(body)| body.filters.as_ref()),
    )?;
    let page = Pagination::new(query.page, query.size);
    let value = db::list_peer_sessions(
        state.pool()?,
        &workspace_id,
        &peer_id,
        &filter,
        page,
        query.reverse,
    )
    .await?;
    Ok(Json(value))
}

async fn get_peer_card(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path((workspace_id, peer_id)): Path<(String, String)>,
    Query(query): Query<PeerCardQuery>,
) -> Result<Response, ApiError> {
    authorize(
        &state.auth,
        authorization_header(&headers),
        false,
        Some(&workspace_id),
        Some(&peer_id),
        None,
    )?;
    let observed = query.target.as_deref().unwrap_or(&peer_id);
    match db::get_peer_card(state.pool()?, &workspace_id, &peer_id, observed).await? {
        Some(value) => Ok(Json(value).into_response()),
        None => Ok((
            StatusCode::NOT_FOUND,
            Json(json!({
                "detail": format!("Peer {peer_id} not found in workspace {workspace_id}")
            })),
        )
            .into_response()),
    }
}

async fn set_peer_card(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path((workspace_id, peer_id)): Path<(String, String)>,
    Query(query): Query<PeerCardQuery>,
    Json(body): Json<Value>,
) -> Result<Json<Value>, ApiError> {
    authorize(
        &state.auth,
        authorization_header(&headers),
        false,
        Some(&workspace_id),
        Some(&peer_id),
        None,
    )?;
    ensure_writes_enabled(&state)?;

    let peer_card = parse_peer_card_set(body)?;
    // No target -> the observer sets its own card (observer == observed).
    let observed = query.target.as_deref().unwrap_or(&peer_id);
    db::set_peer_card(state.pool()?, &workspace_id, &peer_id, observed, &peer_card).await?;
    state
        .peer_cache
        .invalidate_peer(&workspace_id, &peer_id)
        .await;
    Ok(Json(json!({ "peer_card": peer_card })))
}

async fn get_session_peers(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path((workspace_id, session_id)): Path<(String, String)>,
    Query(query): Query<ListQuery>,
) -> Result<Json<Value>, ApiError> {
    authorize(
        &state.auth,
        authorization_header(&headers),
        false,
        Some(&workspace_id),
        None,
        Some(&session_id),
    )?;
    let page = Pagination::new(query.page, query.size);
    let value = db::list_session_peers(state.pool()?, &workspace_id, &session_id, page).await?;
    Ok(Json(value))
}

async fn get_session_peer_config(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path((workspace_id, session_id, peer_id)): Path<(String, String, String)>,
) -> Result<Response, ApiError> {
    authorize(
        &state.auth,
        authorization_header(&headers),
        false,
        Some(&workspace_id),
        None,
        Some(&session_id),
    )?;
    match db::get_session_peer_config(state.pool()?, &workspace_id, &session_id, &peer_id).await? {
        Some(value) => Ok(Json(value).into_response()),
        None => Ok((
            StatusCode::NOT_FOUND,
            Json(json!({
                "detail": format!(
                    "Session peer {peer_id} not found in session {session_id} in workspace {workspace_id}"
                )
            })),
        )
            .into_response()),
    }
}

async fn queue_status(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(workspace_id): Path<String>,
    Query(query): Query<QueueStatusQuery>,
) -> Result<Json<Value>, ApiError> {
    authorize(
        &state.auth,
        authorization_header(&headers),
        false,
        Some(&workspace_id),
        None,
        None,
    )?;
    let value = db::queue_status(
        state.pool()?,
        &workspace_id,
        query.session_id.as_deref(),
        query.observer_id.as_deref(),
        query.sender_id.as_deref(),
    )
    .await?;
    Ok(Json(value))
}

async fn schedule_dream(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(workspace_id): Path<String>,
    Json(body): Json<Value>,
) -> Result<Response, ApiError> {
    authorize(
        &state.auth,
        authorization_header(&headers),
        false,
        Some(&workspace_id),
        None,
        None,
    )?;
    ensure_writes_enabled(&state)?;

    let request = parse_schedule_dream(body)?;

    // FastAPI validates the body before the handler, so a 422 precedes this 400.
    if !state.dream_enabled {
        return Err(ApiError::BadRequest(
            "Dreams are not enabled in the system configuration".to_string(),
        ));
    }

    // observed defaults to observer.
    let observed = request.observed.as_deref().unwrap_or(&request.observer);
    db::enqueue_dream(
        state.pool()?,
        &workspace_id,
        &request.observer,
        observed,
        request.dream_type.as_str(),
        request.session_id.as_deref(),
    )
    .await?;

    Ok(StatusCode::NO_CONTENT.into_response())
}

struct ScheduleDreamRequest {
    observer: String,
    observed: Option<String>,
    dream_type: String,
    session_id: Option<String>,
}

/// Port of `ScheduleDreamRequest` validation: `observer` (required string),
/// `observed` (optional string), `dream_type` (required `DreamType` enum — only
/// `"omni"`), `session_id` (optional string). First error wins in field order.
fn parse_schedule_dream(body: Value) -> Result<ScheduleDreamRequest, ApiError> {
    let object = body.as_object();

    let observer = match object.and_then(|map| map.get("observer")) {
        Some(Value::String(value)) => value.clone(),
        Some(other) => {
            return Err(validation_error(
                "string_type",
                &["observer"],
                "Input should be a valid string",
                other.clone(),
                None,
            ));
        }
        None => {
            return Err(validation_error(
                "missing",
                &["observer"],
                "Field required",
                body.clone(),
                None,
            ));
        }
    };

    let observed = match object.and_then(|map| map.get("observed")) {
        None | Some(Value::Null) => None,
        Some(Value::String(value)) => Some(value.clone()),
        Some(other) => {
            return Err(validation_error(
                "string_type",
                &["observed"],
                "Input should be a valid string",
                other.clone(),
                None,
            ));
        }
    };

    let dream_type = match object.and_then(|map| map.get("dream_type")) {
        Some(Value::String(value)) if value == "omni" => value.clone(),
        Some(value @ Value::String(_)) => {
            return Err(validation_error(
                "enum",
                &["dream_type"],
                "Input should be 'omni'",
                value.clone(),
                Some(json!({ "expected": "'omni'" })),
            ));
        }
        Some(other) => {
            return Err(validation_error(
                "enum",
                &["dream_type"],
                "Input should be 'omni'",
                other.clone(),
                Some(json!({ "expected": "'omni'" })),
            ));
        }
        None => {
            return Err(validation_error(
                "missing",
                &["dream_type"],
                "Field required",
                body.clone(),
                None,
            ));
        }
    };

    let session_id = match object.and_then(|map| map.get("session_id")) {
        None | Some(Value::Null) => None,
        Some(Value::String(value)) => Some(value.clone()),
        Some(other) => {
            return Err(validation_error(
                "string_type",
                &["session_id"],
                "Input should be a valid string",
                other.clone(),
                None,
            ));
        }
    };

    Ok(ScheduleDreamRequest {
        observer,
        observed,
        dream_type,
        session_id,
    })
}

async fn list_webhook_endpoints(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(workspace_id): Path<String>,
    Query(query): Query<ListQuery>,
) -> Result<Json<Value>, ApiError> {
    let params = authorize(
        &state.auth,
        authorization_header(&headers),
        false,
        None,
        None,
        None,
    )?;
    authorize_webhook_workspace(&params, &workspace_id)?;
    let page = Pagination::new(query.page, query.size);
    let value = db::list_webhook_endpoints(state.pool()?, &workspace_id, page).await?;
    Ok(Json(value))
}

async fn get_or_create_webhook_endpoint(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(workspace_id): Path<String>,
    Json(body): Json<Value>,
) -> Result<Response, ApiError> {
    let params = authorize(
        &state.auth,
        authorization_header(&headers),
        false,
        None,
        None,
        None,
    )?;
    authorize_webhook_workspace(&params, &workspace_id)?;
    ensure_writes_enabled(&state)?;

    let url = parse_webhook_create(body)?;
    match db::get_or_create_webhook_endpoint(state.pool()?, &workspace_id, &url).await {
        Ok((value, created)) => {
            let status = if created {
                StatusCode::CREATED
            } else {
                StatusCode::OK
            };
            Ok((status, Json(value)).into_response())
        }
        Err(db::WebhookCreateError::WorkspaceNotFound) => Err(ApiError::NotFound(format!(
            "Workspace {workspace_id} not found"
        ))),
        Err(db::WebhookCreateError::LimitReached) => Err(ApiError::Conflict(format!(
            "Maximum number of webhook endpoints ({}) reached for this workspace.",
            db::WEBHOOK_MAX_WORKSPACE_LIMIT
        ))),
        Err(db::WebhookCreateError::Database(error)) => Err(ApiError::Database(error)),
    }
}

async fn delete_webhook_endpoint(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path((workspace_id, endpoint_id)): Path<(String, String)>,
) -> Result<StatusCode, ApiError> {
    let params = authorize(
        &state.auth,
        authorization_header(&headers),
        false,
        None,
        None,
        None,
    )?;
    authorize_webhook_workspace(&params, &workspace_id)?;
    ensure_writes_enabled(&state)?;

    if db::delete_webhook_endpoint(state.pool()?, &workspace_id, &endpoint_id).await? {
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err(ApiError::NotFound(format!(
            "Webhook endpoint {endpoint_id} not found for workspace {workspace_id}"
        )))
    }
}

async fn get_session_summaries(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path((workspace_id, session_id)): Path<(String, String)>,
) -> Result<Json<Value>, ApiError> {
    authorize(
        &state.auth,
        authorization_header(&headers),
        false,
        Some(&workspace_id),
        None,
        Some(&session_id),
    )?;
    let value = db::get_session_summaries(state.pool()?, &workspace_id, &session_id).await?;
    Ok(Json(value))
}

async fn get_session_context(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path((workspace_id, session_id)): Path<(String, String)>,
    Query(query): Query<GetContextQuery>,
) -> Result<Json<Value>, ApiError> {
    authorize(
        &state.auth,
        authorization_header(&headers),
        false,
        Some(&workspace_id),
        None,
        Some(&session_id),
    )?;

    let token_limit = match query.tokens.as_deref() {
        Some(raw) => parse_context_tokens(raw)?,
        None => db::GET_CONTEXT_MAX_TOKENS,
    };
    let include_summary = match query.summary.as_deref() {
        Some(raw) => parse_query_bool(raw, "summary")?,
        None => true,
    };

    if query.peer_perspective.is_some() && query.peer_target.is_none() {
        return Err(ApiError::Validation(
            "peer_target must be provided if peer_perspective is provided".to_string(),
        ));
    }

    // The perspective path needs the representation/embedding subsystem, which
    // is not ported. Surface a 501 rather than silently dropping the requested
    // representation + peer card.
    if query.peer_target.is_some() {
        return Err(ApiError::NotImplemented(
            "Perspective-scoped session context (peer_target) is not yet supported by the Rust API"
                .to_string(),
        ));
    }

    let value = db::get_session_context(
        state.pool()?,
        &workspace_id,
        &session_id,
        token_limit,
        include_summary,
    )
    .await?;
    Ok(Json(value))
}

async fn search_workspace(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(workspace_id): Path<String>,
    Json(body): Json<Value>,
) -> Result<Json<Value>, ApiError> {
    authorize(
        &state.auth,
        authorization_header(&headers),
        false,
        Some(&workspace_id),
        None,
        None,
    )?;
    run_search(&state, &workspace_id, &[], body).await
}

async fn search_peer(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path((workspace_id, peer_id)): Path<(String, String)>,
    Json(body): Json<Value>,
) -> Result<Json<Value>, ApiError> {
    authorize(
        &state.auth,
        authorization_header(&headers),
        false,
        Some(&workspace_id),
        Some(&peer_id),
        None,
    )?;
    run_search(&state, &workspace_id, &[("peer_id", &peer_id)], body).await
}

async fn search_session(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path((workspace_id, session_id)): Path<(String, String)>,
    Json(body): Json<Value>,
) -> Result<Json<Value>, ApiError> {
    authorize(
        &state.auth,
        authorization_header(&headers),
        false,
        Some(&workspace_id),
        None,
        Some(&session_id),
    )?;
    run_search(&state, &workspace_id, &[("session_id", &session_id)], body).await
}

/// Shared core of the three search routes, porting `utils.search.search`. Builds
/// the effective filter (`body.filters` + the path-derived ids, with
/// `workspace_id` forced in like the routes do), extracts the `peer_perspective`
/// special filter, embeds the query when `EMBED_MESSAGES` is on, and fuses the
/// hybrid-search legs. Returns a plain `list[Message]` (no pagination).
async fn run_search(
    state: &AppState,
    workspace_id: &str,
    extra_filters: &[(&str, &str)],
    body: Value,
) -> Result<Json<Value>, ApiError> {
    let options = parse_search_options(body)?;

    // Build the effective filter object: user filters + path ids (workspace
    // always; peer/session when present), matching the per-route Python code.
    let mut filters = match options.filters {
        Some(Value::Object(map)) => map,
        _ => serde_json::Map::new(),
    };
    filters.insert("workspace_id".to_string(), json!(workspace_id));
    for (key, value) in extra_filters {
        filters.insert((*key).to_string(), json!(value));
    }

    // Pull out the `peer_perspective` special filter before building the SQL
    // filter clause (Python removes it so apply_filter never sees it). Workspace
    // scope is guaranteed (we just inserted workspace_id), so the scope-validation
    // raise can never fire.
    let peer_perspective = match filters.remove("peer_perspective") {
        Some(Value::String(name)) => Some(name),
        _ => None,
    };

    let filters_value = Value::Object(filters);
    let filter = build_filter_clause(FilterTarget::Message, Some(&filters_value))?;

    // Pre-compute the query embedding (best-effort semantic leg). Mirrors
    // `search()`: a token-limit / dimension `ValueError` becomes a 422
    // ValidationException; other failures propagate as 500.
    let mut embedding_vec: Option<Vec<f32>> = None;
    if state.embed_messages {
        let api_key = state.embedding.api_key.as_ref().ok_or_else(|| {
            ApiError::Embedding(
                "EMBED_MESSAGES is enabled but no embedding API key is configured".to_string(),
            )
        })?;
        let credentials =
            Credentials::with_base_url(api_key.clone(), state.embedding.base_url.clone());
        let http = ReqwestHttp::default();
        match embedding::embed_openai(
            &http,
            &credentials,
            &state.embedding.model,
            &options.query,
            state.embedding.vector_dimensions,
            state.embedding.send_dimensions,
            state.embedding.max_tokens,
        )
        .await
        {
            Ok(vector) => embedding_vec = Some(vector),
            Err(EmbedError::TokenLimit { .. } | EmbedError::DimensionMismatch { .. }) => {
                return Err(ApiError::Validation(format!(
                    "Query exceeds maximum token limit of {}.",
                    state.embedding.max_tokens
                )));
            }
            Err(other) => return Err(ApiError::Embedding(other.to_string())),
        }
    }

    let params = HybridSearchParams {
        workspace_name: workspace_id,
        query: &options.query,
        filter: &filter,
        query_embedding: embedding_vec.as_deref(),
        peer_perspective: peer_perspective.as_deref(),
        limit: options.limit as usize,
    };
    let results = search::hybrid_search(state.pool()?, &params).await?;
    Ok(Json(Value::Array(results)))
}

struct SearchOptions {
    query: String,
    filters: Option<Value>,
    limit: i64,
}

/// Port of `MessageSearchOptions` validation: `query` (required string,
/// NUL-stripped), optional `filters` dict, `limit` (int, default 10, 1..=100).
/// Reproduces Pydantic's per-field error shapes; returns the first error
/// encountered (query → filters → limit), the house single-error convention.
fn parse_search_options(body: Value) -> Result<SearchOptions, ApiError> {
    let object = body.as_object();

    let query = match object.and_then(|map| map.get("query")) {
        Some(Value::String(value)) => value.replace('\u{0}', ""),
        Some(other) => {
            return Err(validation_error(
                "string_type",
                &["query"],
                "Input should be a valid string",
                other.clone(),
                None,
            ));
        }
        None => {
            return Err(validation_error(
                "missing",
                &["query"],
                "Field required",
                body.clone(),
                None,
            ));
        }
    };

    let filters = match object.and_then(|map| map.get("filters")) {
        None | Some(Value::Null) => None,
        Some(value @ Value::Object(_)) => Some(value.clone()),
        Some(other) => {
            return Err(validation_error(
                "dict_type",
                &["filters"],
                "Input should be a valid dictionary",
                other.clone(),
                None,
            ));
        }
    };

    let limit = match object.and_then(|map| map.get("limit")) {
        None | Some(Value::Null) => 10,
        Some(value) => parse_search_limit(value)?,
    };

    Ok(SearchOptions {
        query,
        filters,
        limit,
    })
}

fn parse_search_limit(value: &Value) -> Result<i64, ApiError> {
    parse_bounded_int(value, "limit", 1, 100)
}

/// Coerce + bound-check an integer field like Pydantic's `int = Field(ge=min,
/// le=max)` in lax (JSON-body) mode, emitting `int_parsing`/`int_from_float`/
/// `int_type`/`greater_than_equal`/`less_than_equal` with the field's `loc`.
fn parse_bounded_int(value: &Value, field: &'static str, min: i64, max: i64) -> Result<i64, ApiError> {
    let parsed = match value {
        Value::Number(number) => {
            if let Some(integer) = number.as_i64() {
                integer
            } else if let Some(float) = number.as_f64() {
                if float.fract() == 0.0 {
                    float as i64
                } else {
                    return Err(validation_error(
                        "int_from_float",
                        &[field],
                        "Input should be a valid integer, got a number with a fractional part",
                        value.clone(),
                        None,
                    ));
                }
            } else {
                return Err(validation_error(
                    "int_parsing",
                    &[field],
                    "Input should be a valid integer, unable to parse string as an integer",
                    value.clone(),
                    None,
                ));
            }
        }
        Value::String(text) => text.trim().parse::<i64>().map_err(|_| {
            validation_error(
                "int_parsing",
                &[field],
                "Input should be a valid integer, unable to parse string as an integer",
                value.clone(),
                None,
            )
        })?,
        other => {
            return Err(validation_error(
                "int_type",
                &[field],
                "Input should be a valid integer",
                other.clone(),
                None,
            ));
        }
    };

    if parsed < min {
        return Err(validation_error(
            "greater_than_equal",
            &[field],
            &format!("Input should be greater than or equal to {min}"),
            json!(parsed),
            Some(json!({ "ge": min })),
        ));
    }
    if parsed > max {
        return Err(validation_error(
            "less_than_equal",
            &[field],
            &format!("Input should be less than or equal to {max}"),
            json!(parsed),
            Some(json!({ "le": max })),
        ));
    }
    Ok(parsed)
}

async fn create_conclusions(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(workspace_id): Path<String>,
    Json(body): Json<Value>,
) -> Result<Response, ApiError> {
    authorize(
        &state.auth,
        authorization_header(&headers),
        false,
        Some(&workspace_id),
        None,
        None,
    )?;
    ensure_writes_enabled(&state)?;

    let conclusions = parse_conclusion_batch(body)?;

    // Validate sessions/peers and get-or-create collections before embedding,
    // matching Python's error precedence (a missing peer 404s before any embed).
    match db::prepare_conclusions(state.pool()?, &workspace_id, &conclusions).await {
        Ok(()) => {}
        Err(db::ConclusionWriteError::SessionNotFound(name)) => {
            return Err(ApiError::NotFound(format!(
                "Session {name} not found in workspace {workspace_id}"
            )));
        }
        Err(db::ConclusionWriteError::PeerNotFound(name)) => {
            return Err(ApiError::NotFound(format!(
                "Peer {name} not found in workspace {workspace_id}"
            )));
        }
        Err(db::ConclusionWriteError::Database(error)) => return Err(ApiError::Database(error)),
    }

    // `simple_batch_embed` validates every per-input token count up front and
    // raises on the first over-limit, so check all counts before any API call.
    for (index, conclusion) in conclusions.iter().enumerate() {
        let tokens = embedding::embedding_token_count(&conclusion.content);
        if tokens > state.embedding.max_tokens {
            return Err(ApiError::Validation(format!(
                "Text at index {index} exceeds maximum token limit of {} tokens (got {tokens} tokens)",
                state.embedding.max_tokens
            )));
        }
    }

    let api_key = state.embedding.api_key.as_ref().ok_or_else(|| {
        ApiError::Embedding("no embedding API key is configured".to_string())
    })?;
    let credentials = Credentials::with_base_url(api_key.clone(), state.embedding.base_url.clone());
    let http = ReqwestHttp::default();
    let mut embeddings = Vec::with_capacity(conclusions.len());
    for (index, conclusion) in conclusions.iter().enumerate() {
        let vector = embedding::embed_openai(
            &http,
            &credentials,
            &state.embedding.model,
            &conclusion.content,
            state.embedding.vector_dimensions,
            state.embedding.send_dimensions,
            state.embedding.max_tokens,
        )
        .await
        .map_err(|error| match error {
            // Unreachable in practice — token counts are pre-validated above —
            // but kept faithful to the per-index message just in case.
            EmbedError::TokenLimit { limit, got } => ApiError::Validation(format!(
                "Text at index {index} exceeds maximum token limit of {limit} tokens (got {got} tokens)"
            )),
            other => ApiError::Embedding(other.to_string()),
        })?;
        embeddings.push(vector);
    }

    let created =
        db::insert_conclusions(state.pool()?, &workspace_id, &conclusions, &embeddings).await?;
    Ok((StatusCode::CREATED, Json(Value::Array(created))).into_response())
}

/// Port of `ConclusionBatchCreate` + `ConclusionCreate` validation. Accepts the
/// `conclusions` or `observations` alias; reproduces Pydantic's list-length and
/// per-item error shapes (with integer list-index `loc` parts). First error wins.
fn parse_conclusion_batch(body: Value) -> Result<Vec<db::NewConclusion>, ApiError> {
    let object = body.as_object();
    // AliasChoices("conclusions", "observations"): prefer the field name; the
    // `loc` reflects whichever key the caller supplied (missing → "conclusions").
    let (key, raw) = match object.and_then(|map| map.get("conclusions")) {
        Some(value) => ("conclusions", Some(value)),
        None => match object.and_then(|map| map.get("observations")) {
            Some(value) => ("observations", Some(value)),
            None => ("conclusions", None),
        },
    };

    let Some(raw) = raw else {
        return Err(validation_error(
            "missing",
            &["conclusions"],
            "Field required",
            body.clone(),
            None,
        ));
    };
    let Some(items) = raw.as_array() else {
        return Err(validation_error(
            "list_type",
            &[key],
            "Input should be a valid list",
            raw.clone(),
            None,
        ));
    };

    let mut parsed = Vec::with_capacity(items.len());
    for (index, item) in items.iter().enumerate() {
        parsed.push(parse_conclusion_item(key, index, item)?);
    }

    if parsed.is_empty() {
        return Err(validation_error_parts(
            "too_short",
            &[Value::String(key.to_string())],
            "List should have at least 1 item after validation, not 0",
            raw.clone(),
            Some(json!({ "field_type": "List", "min_length": 1, "actual_length": 0 })),
        ));
    }
    if parsed.len() > 100 {
        return Err(validation_error_parts(
            "too_long",
            &[Value::String(key.to_string())],
            "List should have at most 100 items after validation, not 100",
            raw.clone(),
            Some(json!({ "field_type": "List", "max_length": 100, "actual_length": parsed.len() })),
        ));
    }

    Ok(parsed)
}

fn parse_conclusion_item(
    key: &str,
    index: usize,
    item: &Value,
) -> Result<db::NewConclusion, ApiError> {
    let item_loc = |field: &str| {
        vec![
            Value::String(key.to_string()),
            Value::Number(index.into()),
            Value::String(field.to_string()),
        ]
    };

    let Some(object) = item.as_object() else {
        return Err(validation_error_parts(
            "model_type",
            &[Value::String(key.to_string()), Value::Number(index.into())],
            "Input should be a valid dictionary or instance of ConclusionCreate",
            item.clone(),
            Some(json!({ "class_name": "ConclusionCreate" })),
        ));
    };

    let content = match object.get("content") {
        Some(Value::String(value)) => {
            if value.is_empty() {
                return Err(validation_error_parts(
                    "string_too_short",
                    &item_loc("content"),
                    "String should have at least 1 character",
                    Value::String(value.clone()),
                    Some(json!({ "min_length": 1 })),
                ));
            }
            if value.chars().count() > 65535 {
                return Err(validation_error_parts(
                    "string_too_long",
                    &item_loc("content"),
                    "String should have at most 65535 characters",
                    Value::String(value.clone()),
                    Some(json!({ "max_length": 65535 })),
                ));
            }
            value.clone()
        }
        Some(other) => {
            return Err(validation_error_parts(
                "string_type",
                &item_loc("content"),
                "Input should be a valid string",
                other.clone(),
                None,
            ));
        }
        None => {
            return Err(validation_error_parts(
                "missing",
                &item_loc("content"),
                "Field required",
                item.clone(),
                None,
            ));
        }
    };

    let observer_id = require_conclusion_string(object, item, key, index, "observer_id")?;
    let observed_id = require_conclusion_string(object, item, key, index, "observed_id")?;

    let session_id = match object.get("session_id") {
        None | Some(Value::Null) => None,
        Some(Value::String(value)) => Some(value.clone()),
        Some(other) => {
            return Err(validation_error_parts(
                "string_type",
                &item_loc("session_id"),
                "Input should be a valid string",
                other.clone(),
                None,
            ));
        }
    };

    Ok(db::NewConclusion {
        content,
        observer_id,
        observed_id,
        session_id,
    })
}

fn require_conclusion_string(
    object: &serde_json::Map<String, Value>,
    item: &Value,
    key: &str,
    index: usize,
    field: &str,
) -> Result<String, ApiError> {
    let loc = vec![
        Value::String(key.to_string()),
        Value::Number(index.into()),
        Value::String(field.to_string()),
    ];
    match object.get(field) {
        Some(Value::String(value)) => Ok(value.clone()),
        Some(other) => Err(validation_error_parts(
            "string_type",
            &loc,
            "Input should be a valid string",
            other.clone(),
            None,
        )),
        None => Err(validation_error_parts(
            "missing",
            &loc,
            "Field required",
            item.clone(),
            None,
        )),
    }
}

async fn query_conclusions(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(workspace_id): Path<String>,
    Json(body): Json<Value>,
) -> Result<Json<Value>, ApiError> {
    authorize(
        &state.auth,
        authorization_header(&headers),
        false,
        Some(&workspace_id),
        None,
        None,
    )?;

    let query = parse_conclusion_query(body)?;

    // observer/observed come from the filters dict (`x or x_id`), matching the
    // route; both must be truthy or it's a 422 ValidationException.
    let (observer, observed) = match query.filters.as_ref().and_then(Value::as_object) {
        Some(map) => (
            truthy_string(map, "observer").or_else(|| truthy_string(map, "observer_id")),
            truthy_string(map, "observed").or_else(|| truthy_string(map, "observed_id")),
        ),
        None => (None, None),
    };
    let (Some(observer), Some(observed)) = (observer, observed) else {
        return Err(ApiError::Validation(
            "observer and observed must be specified for semantic search".to_string(),
        ));
    };

    // The conclusions query always embeds (it is not gated on EMBED_MESSAGES).
    let api_key = state.embedding.api_key.as_ref().ok_or_else(|| {
        ApiError::Embedding("no embedding API key is configured".to_string())
    })?;
    let credentials = Credentials::with_base_url(api_key.clone(), state.embedding.base_url.clone());
    let http = ReqwestHttp::default();
    let embedding_vec = match embedding::embed_openai(
        &http,
        &credentials,
        &state.embedding.model,
        &query.query,
        state.embedding.vector_dimensions,
        state.embedding.send_dimensions,
        state.embedding.max_tokens,
    )
    .await
    {
        Ok(vector) => vector,
        Err(EmbedError::TokenLimit { .. } | EmbedError::DimensionMismatch { .. }) => {
            return Err(ApiError::Validation(format!(
                "Query exceeds maximum token limit of {}.",
                state.embedding.max_tokens
            )));
        }
        Err(other) => return Err(ApiError::Embedding(other.to_string())),
    };

    let filter = build_filter_clause(FilterTarget::Conclusion, query.filters.as_ref())?;
    let results = db::query_documents_pgvector(
        state.pool()?,
        &workspace_id,
        &observer,
        &observed,
        &embedding_vec,
        &filter,
        query.distance,
        query.top_k,
    )
    .await?;
    Ok(Json(Value::Array(results)))
}

struct ConclusionQuery {
    query: String,
    top_k: i64,
    distance: Option<f64>,
    filters: Option<Value>,
}

/// Port of `ConclusionQuery` validation: `query` (required string), `top_k`
/// (int, default 10, 1..=100), `distance` (optional float, 0.0..=1.0), `filters`
/// (optional dict). First error wins (query → top_k → distance → filters).
fn parse_conclusion_query(body: Value) -> Result<ConclusionQuery, ApiError> {
    let object = body.as_object();

    let query = match object.and_then(|map| map.get("query")) {
        Some(Value::String(value)) => value.clone(),
        Some(other) => {
            return Err(validation_error(
                "string_type",
                &["query"],
                "Input should be a valid string",
                other.clone(),
                None,
            ));
        }
        None => {
            return Err(validation_error(
                "missing",
                &["query"],
                "Field required",
                body.clone(),
                None,
            ));
        }
    };

    let top_k = match object.and_then(|map| map.get("top_k")) {
        None | Some(Value::Null) => 10,
        Some(value) => parse_bounded_int(value, "top_k", 1, 100)?,
    };

    let distance = match object.and_then(|map| map.get("distance")) {
        None | Some(Value::Null) => None,
        Some(value) => Some(parse_distance(value)?),
    };

    let filters = match object.and_then(|map| map.get("filters")) {
        None | Some(Value::Null) => None,
        Some(value @ Value::Object(_)) => Some(value.clone()),
        Some(other) => {
            return Err(validation_error(
                "dict_type",
                &["filters"],
                "Input should be a valid dictionary",
                other.clone(),
                None,
            ));
        }
    };

    Ok(ConclusionQuery {
        query,
        top_k,
        distance,
        filters,
    })
}

/// Coerce + bound-check the optional `distance` field like Pydantic's `float =
/// Field(ge=0.0, le=1.0)`.
fn parse_distance(value: &Value) -> Result<f64, ApiError> {
    let parsed = match value {
        Value::Number(number) => number.as_f64().ok_or_else(|| {
            validation_error(
                "float_parsing",
                &["distance"],
                "Input should be a valid number, unable to parse string as a number",
                value.clone(),
                None,
            )
        })?,
        Value::String(text) => text.trim().parse::<f64>().map_err(|_| {
            validation_error(
                "float_parsing",
                &["distance"],
                "Input should be a valid number, unable to parse string as a number",
                value.clone(),
                None,
            )
        })?,
        other => {
            return Err(validation_error(
                "float_type",
                &["distance"],
                "Input should be a valid number",
                other.clone(),
                None,
            ));
        }
    };

    if parsed < 0.0 {
        return Err(validation_error(
            "greater_than_equal",
            &["distance"],
            "Input should be greater than or equal to 0",
            value.clone(),
            Some(json!({ "ge": 0.0 })),
        ));
    }
    if parsed > 1.0 {
        return Err(validation_error(
            "less_than_equal",
            &["distance"],
            "Input should be less than or equal to 1",
            value.clone(),
            Some(json!({ "le": 1.0 })),
        ));
    }
    Ok(parsed)
}

/// Return the value at `key` only when it is a non-empty string (Python's
/// truthiness for the `observer or observer_id` chain).
fn truthy_string(map: &serde_json::Map<String, Value>, key: &str) -> Option<String> {
    match map.get(key) {
        Some(Value::String(value)) if !value.is_empty() => Some(value.clone()),
        _ => None,
    }
}

async fn list_messages(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path((workspace_id, session_id)): Path<(String, String)>,
    Query(query): Query<ListQuery>,
    body: Option<Json<ListRequest>>,
) -> Result<Json<Value>, ApiError> {
    authorize(
        &state.auth,
        authorization_header(&headers),
        false,
        Some(&workspace_id),
        None,
        Some(&session_id),
    )?;
    let filter = build_filter_clause(
        FilterTarget::Message,
        body.as_ref().and_then(|Json(body)| body.filters.as_ref()),
    )?;
    let page = Pagination::new(query.page, query.size);
    let value = db::list_messages(
        state.pool()?,
        &workspace_id,
        &session_id,
        &filter,
        page,
        query.reverse,
    )
    .await?;
    Ok(Json(value))
}

async fn list_conclusions(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path(workspace_id): Path<String>,
    Query(query): Query<ListQuery>,
    body: Option<Json<ListRequest>>,
) -> Result<Json<Value>, ApiError> {
    authorize(
        &state.auth,
        authorization_header(&headers),
        false,
        Some(&workspace_id),
        None,
        None,
    )?;
    let filter = build_filter_clause(
        FilterTarget::Conclusion,
        body.as_ref().and_then(|Json(body)| body.filters.as_ref()),
    )?;
    let page = Pagination::new(query.page, query.size);
    let value =
        db::list_conclusions(state.pool()?, &workspace_id, &filter, page, query.reverse).await?;
    Ok(Json(value))
}

async fn delete_conclusion(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path((workspace_id, conclusion_id)): Path<(String, String)>,
) -> Result<StatusCode, ApiError> {
    authorize(
        &state.auth,
        authorization_header(&headers),
        false,
        Some(&workspace_id),
        None,
        None,
    )?;
    ensure_writes_enabled(&state)?;

    if db::delete_conclusion(state.pool()?, &workspace_id, &conclusion_id).await? {
        Ok(StatusCode::NO_CONTENT)
    } else {
        Err(ApiError::NotFound(format!(
            "Document {conclusion_id} not found or does not belong to workspace {workspace_id}"
        )))
    }
}

async fn get_message(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path((workspace_id, session_id, message_id)): Path<(String, String, String)>,
) -> Result<Response, ApiError> {
    authorize(
        &state.auth,
        authorization_header(&headers),
        false,
        Some(&workspace_id),
        None,
        Some(&session_id),
    )?;
    match db::get_message(state.pool()?, &workspace_id, &session_id, &message_id).await? {
        Some(value) => Ok(Json(value).into_response()),
        None => Ok((
            StatusCode::NOT_FOUND,
            Json(json!({ "detail": format!("Message with ID {message_id} not found") })),
        )
            .into_response()),
    }
}

async fn create_messages_for_session(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path((workspace_id, session_id)): Path<(String, String)>,
    Json(body): Json<Value>,
) -> Result<Response, ApiError> {
    authorize(
        &state.auth,
        authorization_header(&headers),
        false,
        Some(&workspace_id),
        None,
        Some(&session_id),
    )?;
    ensure_writes_enabled(&state)?;
    validate_resource_name(&workspace_id, "workspace_id")?;
    validate_resource_name(&session_id, "session_id")?;

    let parsed = parse_message_batch(body)?;
    let inserts = parsed
        .iter()
        .map(|message| db::MessageInsert {
            peer_name: message.peer_name.clone(),
            content: message.content.clone(),
            metadata: message.metadata.clone(),
            created_at: message.created_at,
            token_count: message.token_count,
        })
        .collect::<Vec<_>>();

    let created = db::create_messages(
        state.pool()?,
        &workspace_id,
        &session_id,
        &inserts,
        state.embed_messages,
        state.embedding_max_tokens,
    )
    .await?;
    let response = Value::Array(created.iter().map(db::CreatedMessage::to_json).collect());

    // Enqueue is fire-and-forget in Python (a BackgroundTask whose failures are
    // swallowed), so a queue-write failure must never fail message creation.
    if let Err(error) =
        enqueue_created_messages(&state, &workspace_id, &session_id, &created, &parsed).await
    {
        tracing::warn!(?error, "failed to enqueue created messages");
    }

    Ok((StatusCode::CREATED, Json(response)).into_response())
}

/// Build and insert the queue records for a freshly-created batch, porting the
/// `enqueue` → `handle_session` → `generate_queue_records` path. Best-effort:
/// the caller swallows errors to match Python's background-task semantics.
async fn enqueue_created_messages(
    state: &AppState,
    workspace_id: &str,
    session_id: &str,
    created: &[db::CreatedMessage],
    parsed: &[ParsedMessage],
) -> Result<(), ApiError> {
    let pool = state.pool()?;
    let Some((session_internal_id, session_config)) =
        db::fetch_session_for_enqueue(pool, workspace_id, session_id).await?
    else {
        return Ok(());
    };
    let workspace_config = db::get_workspace_configuration(pool, workspace_id).await?;
    let peers = db::get_session_peer_configuration(pool, workspace_id, session_id).await?;

    let mut records = Vec::new();
    for (created_message, parsed_message) in created.iter().zip(parsed) {
        let conf = crate::producer::resolve_configuration(
            Some(&workspace_config),
            Some(&session_config),
            parsed_message.configuration.as_ref(),
        );
        let message = crate::producer::MessageForEnqueue {
            workspace_name: workspace_id,
            session_name: session_id,
            message_id: created_message.id,
            message_public_id: &created_message.public_id,
            content: &created_message.content,
            peer_name: &created_message.peer_name,
            created_at: created_message.created_at,
            seq_in_session: created_message.seq_in_session,
        };
        records.extend(crate::producer::generate_queue_records(
            &message,
            &peers,
            &session_internal_id,
            &conf,
        )?);
    }

    db::insert_queue_records(pool, &records).await?;
    Ok(())
}

async fn update_message(
    State(state): State<AppState>,
    headers: HeaderMap,
    Path((workspace_id, session_id, message_id)): Path<(String, String, String)>,
    Json(body): Json<Value>,
) -> Result<Response, ApiError> {
    authorize(
        &state.auth,
        authorization_header(&headers),
        false,
        Some(&workspace_id),
        None,
        Some(&session_id),
    )?;
    ensure_writes_enabled(&state)?;
    validate_resource_name(&workspace_id, "workspace_id")?;
    validate_resource_name(&session_id, "session_id")?;

    let metadata = parse_message_update(body)?;
    match db::update_message(
        state.pool()?,
        &workspace_id,
        &session_id,
        &message_id,
        metadata,
    )
    .await?
    {
        Some(value) => Ok(Json(value).into_response()),
        None => Err(ApiError::NotFound("Message not found".to_string())),
    }
}

fn authorization_header(headers: &HeaderMap) -> Option<&str> {
    headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|value| value.to_str().ok())
}

fn has_non_empty_scope(query: &CreateKeyQuery) -> bool {
    [
        query.workspace_id.as_deref(),
        query.peer_id.as_deref(),
        query.session_id.as_deref(),
    ]
    .into_iter()
    .flatten()
    .any(|value| !value.is_empty())
}

fn ensure_writes_enabled(state: &AppState) -> Result<(), ApiError> {
    if state.write_enabled {
        Ok(())
    } else {
        Err(ApiError::WriteDisabled)
    }
}

fn parse_workspace_create(input: Value) -> Result<WorkspaceCreateRequest, ApiError> {
    let object = input.as_object().ok_or_else(|| {
        validation_error(
            "model_attributes_type",
            &[],
            "Input should be a valid dictionary or object to extract fields from",
            input.clone(),
            None,
        )
    })?;
    let name_field = if object.contains_key("id") {
        "id"
    } else {
        "name"
    };
    let name = object.get("id").or_else(|| object.get("name"));
    let name = match name {
        Some(Value::String(name)) => name.clone(),
        Some(value) => {
            return Err(validation_error(
                "string_type",
                &[name_field],
                "Input should be a valid string",
                value.clone(),
                None,
            ));
        }
        None => {
            return Err(validation_error(
                "missing",
                &["id"],
                "Field required",
                input,
                None,
            ));
        }
    };

    let metadata = object.get("metadata").cloned().unwrap_or_else(empty_object);
    let metadata = validate_metadata(metadata)?;
    let configuration = object
        .get("configuration")
        .cloned()
        .unwrap_or_else(empty_object);
    let configuration = validate_workspace_configuration(configuration)?;

    Ok(WorkspaceCreateRequest {
        name,
        name_field,
        metadata,
        configuration,
    })
}

fn parse_workspace_update(input: Value) -> Result<WorkspaceUpdateRequest, ApiError> {
    let object = input.as_object().ok_or_else(|| {
        validation_error(
            "model_attributes_type",
            &[],
            "Input should be a valid dictionary or object to extract fields from",
            input.clone(),
            None,
        )
    })?;
    let metadata = match object.get("metadata") {
        Some(value) if !value.is_null() => Some(validate_metadata(value.clone())?),
        _ => None,
    };
    let configuration = match object.get("configuration") {
        Some(value) if !value.is_null() => Some(validate_workspace_configuration(value.clone())?),
        _ => None,
    };
    Ok(WorkspaceUpdateRequest {
        metadata,
        configuration,
    })
}

fn parse_peer_create(input: Value) -> Result<PeerCreateRequest, ApiError> {
    let object = input.as_object().ok_or_else(|| {
        validation_error(
            "model_attributes_type",
            &[],
            "Input should be a valid dictionary or object to extract fields from",
            input.clone(),
            None,
        )
    })?;
    let name_field = if object.contains_key("id") {
        "id"
    } else {
        "name"
    };
    let name = object.get("id").or_else(|| object.get("name"));
    let name = match name {
        Some(Value::String(name)) => name.clone(),
        Some(value) => {
            return Err(validation_error(
                "string_type",
                &[name_field],
                "Input should be a valid string",
                value.clone(),
                None,
            ));
        }
        None => {
            return Err(validation_error(
                "missing",
                &["id"],
                "Field required",
                input,
                None,
            ));
        }
    };

    let metadata = match object.get("metadata") {
        Some(value) if !value.is_null() => Some(validate_metadata(value.clone())?),
        _ => None,
    };
    let configuration = match object.get("configuration") {
        Some(value) if !value.is_null() => {
            Some(validate_plain_object(value.clone(), &["configuration"])?)
        }
        _ => None,
    };

    Ok(PeerCreateRequest {
        name,
        name_field,
        metadata,
        configuration,
    })
}

fn parse_peer_update(input: Value) -> Result<PeerUpdateRequest, ApiError> {
    let object = input.as_object().ok_or_else(|| {
        validation_error(
            "model_attributes_type",
            &[],
            "Input should be a valid dictionary or object to extract fields from",
            input.clone(),
            None,
        )
    })?;
    let metadata = match object.get("metadata") {
        Some(value) if !value.is_null() => Some(validate_metadata(value.clone())?),
        _ => None,
    };
    let configuration = match object.get("configuration") {
        Some(value) if !value.is_null() => {
            Some(validate_plain_object(value.clone(), &["configuration"])?)
        }
        _ => None,
    };
    Ok(PeerUpdateRequest {
        metadata,
        configuration,
    })
}

fn parse_session_create(input: Value) -> Result<SessionCreateRequest, ApiError> {
    let object = input.as_object().ok_or_else(|| {
        validation_error(
            "model_attributes_type",
            &[],
            "Input should be a valid dictionary or object to extract fields from",
            input.clone(),
            None,
        )
    })?;
    reject_session_peers(object)?;
    let name_field = if object.contains_key("id") {
        "id"
    } else {
        "name"
    };
    let name = object.get("id").or_else(|| object.get("name"));
    let name = match name {
        Some(Value::String(name)) => name.clone(),
        Some(value) => {
            return Err(validation_error(
                "string_type",
                &[name_field],
                "Input should be a valid string",
                value.clone(),
                None,
            ));
        }
        None => {
            return Err(validation_error(
                "missing",
                &["id"],
                "Field required",
                input,
                None,
            ));
        }
    };

    let metadata = match object.get("metadata") {
        Some(value) if !value.is_null() => Some(validate_metadata(value.clone())?),
        _ => None,
    };
    let configuration = match object.get("configuration") {
        Some(value) if !value.is_null() => Some(validate_workspace_configuration(value.clone())?),
        _ => None,
    };

    Ok(SessionCreateRequest {
        name,
        name_field,
        metadata,
        configuration,
    })
}

fn parse_session_update(input: Value) -> Result<SessionUpdateRequest, ApiError> {
    let object = input.as_object().ok_or_else(|| {
        validation_error(
            "model_attributes_type",
            &[],
            "Input should be a valid dictionary or object to extract fields from",
            input.clone(),
            None,
        )
    })?;
    reject_session_peers(object)?;
    let metadata = match object.get("metadata") {
        Some(value) if !value.is_null() => Some(validate_metadata(value.clone())?),
        _ => None,
    };
    let configuration = match object.get("configuration") {
        Some(value) if !value.is_null() => Some(validate_workspace_configuration(value.clone())?),
        _ => None,
    };
    Ok(SessionUpdateRequest {
        metadata,
        configuration,
    })
}

fn parse_session_peers(input: Value) -> Result<SessionPeersRequest, ApiError> {
    let object = input.as_object().ok_or_else(|| {
        validation_error(
            "dict_type",
            &[],
            "Input should be a valid dictionary",
            input.clone(),
            None,
        )
    })?;
    let mut peers = BTreeMap::new();
    for (peer_name, config) in object {
        peers.insert(
            peer_name.clone(),
            parse_session_peer_config(config.clone(), &[peer_name])?,
        );
    }
    Ok(SessionPeersRequest { peers })
}

fn parse_session_peer_config(input: Value, loc: &[&str]) -> Result<SessionPeerConfig, ApiError> {
    let config_object = input.as_object().ok_or_else(|| {
        validation_error(
            "model_attributes_type",
            loc,
            "Input should be a valid dictionary or object to extract fields from",
            input.clone(),
            None,
        )
    })?;
    let mut observe_me_loc = loc.to_vec();
    observe_me_loc.push("observe_me");
    let observe_me = validate_optional_bool(config_object.get("observe_me"), &observe_me_loc)?;
    let mut observe_others_loc = loc.to_vec();
    observe_others_loc.push("observe_others");
    let observe_others =
        validate_optional_bool(config_object.get("observe_others"), &observe_others_loc)?;
    Ok(SessionPeerConfig {
        observe_me,
        observe_others,
    })
}

fn parse_peer_name_list(input: Value) -> Result<Vec<String>, ApiError> {
    let values = input.as_array().ok_or_else(|| {
        validation_error(
            "list_type",
            &[],
            "Input should be a valid list",
            input.clone(),
            None,
        )
    })?;
    let mut peer_names = Vec::with_capacity(values.len());
    for (index, value) in values.iter().enumerate() {
        let peer_name = match value {
            Value::String(peer_name) => peer_name.clone(),
            _ => {
                return Err(validation_error(
                    "string_type",
                    &[&index.to_string()],
                    "Input should be a valid string",
                    value.clone(),
                    None,
                ));
            }
        };
        if !peer_names.iter().any(|existing| existing == &peer_name) {
            peer_names.push(peer_name);
        }
    }
    Ok(peer_names)
}

/// Mirror of Python's `settings.MAX_MESSAGE_SIZE` default (25_000), the
/// `max_length` on `MessageCreate.content`. Making it config-driven is a
/// follow-up like the custom-instructions token budget.
const MAX_MESSAGE_SIZE: usize = 25_000;

/// A single validated message ready for insertion plus enqueueing. `metadata` is
/// the sanitized object (`{}` when absent); `configuration` is the normalized
/// message-level override (reasoning-only, matching `MessageConfiguration`) or
/// `None`; `token_count` is the o200k_base length of the sanitized content.
#[derive(Debug, Clone)]
pub struct ParsedMessage {
    pub peer_name: String,
    pub content: String,
    pub metadata: Value,
    pub configuration: Option<Value>,
    pub created_at: Option<DateTime<Utc>>,
    pub token_count: i32,
}

fn parse_message_batch(input: Value) -> Result<Vec<ParsedMessage>, ApiError> {
    let object = input.as_object().ok_or_else(|| {
        validation_error(
            "model_attributes_type",
            &[],
            "Input should be a valid dictionary or object to extract fields from",
            input.clone(),
            None,
        )
    })?;
    let messages = match object.get("messages") {
        None | Some(Value::Null) => {
            return Err(validation_error(
                "missing",
                &["messages"],
                "Field required",
                input.clone(),
                None,
            ));
        }
        Some(value) => value,
    };
    let array = messages.as_array().ok_or_else(|| {
        validation_error(
            "list_type",
            &["messages"],
            "Input should be a valid list",
            messages.clone(),
            None,
        )
    })?;
    if array.is_empty() {
        return Err(validation_error(
            "too_short",
            &["messages"],
            "List should have at least 1 item after validation, not 0",
            messages.clone(),
            Some(json!({"field_type": "List", "min_length": 1, "actual_length": 0})),
        ));
    }
    if array.len() > 100 {
        return Err(validation_error(
            "too_long",
            &["messages"],
            &format!(
                "List should have at most 100 items after validation, not {}",
                array.len()
            ),
            messages.clone(),
            Some(json!({"field_type": "List", "max_length": 100, "actual_length": array.len()})),
        ));
    }
    array
        .iter()
        .enumerate()
        .map(|(index, item)| parse_message_create(item, index))
        .collect()
}

/// Parse a `PeerCardSet` body: a required `peer_card` list of strings, with NUL
/// bytes stripped from each item (`sanitize_peer_card`).
fn parse_peer_card_set(input: Value) -> Result<Vec<String>, ApiError> {
    let object = input.as_object().ok_or_else(|| {
        validation_error(
            "model_attributes_type",
            &[],
            "Input should be a valid dictionary or object to extract fields from",
            input.clone(),
            None,
        )
    })?;
    let items = match object.get("peer_card") {
        None | Some(Value::Null) => {
            return Err(validation_error(
                "missing",
                &["peer_card"],
                "Field required",
                input.clone(),
                None,
            ));
        }
        Some(Value::Array(items)) => items,
        Some(value) => {
            return Err(validation_error(
                "list_type",
                &["peer_card"],
                "Input should be a valid list",
                value.clone(),
                None,
            ));
        }
    };
    let mut peer_card = Vec::with_capacity(items.len());
    for (index, item) in items.iter().enumerate() {
        match item {
            Value::String(value) => peer_card.push(value.replace('\0', "")),
            other => {
                return Err(validation_error_parts(
                    "string_type",
                    &[
                        Value::String("peer_card".to_string()),
                        Value::Number(index.into()),
                    ],
                    "Input should be a valid string",
                    other.clone(),
                    None,
                ));
            }
        }
    }
    Ok(peer_card)
}

/// Parse a `MessageUpdate` body, returning the sanitized metadata override (or
/// `None` when absent/null, leaving the message unchanged).
fn parse_message_update(input: Value) -> Result<Option<Value>, ApiError> {
    let object = input.as_object().ok_or_else(|| {
        validation_error(
            "model_attributes_type",
            &[],
            "Input should be a valid dictionary or object to extract fields from",
            input.clone(),
            None,
        )
    })?;
    match object.get("metadata") {
        Some(value) if !value.is_null() => Ok(Some(validate_metadata(value.clone())?)),
        _ => Ok(None),
    }
}

fn parse_message_create(input: &Value, index: usize) -> Result<ParsedMessage, ApiError> {
    let object = input.as_object().ok_or_else(|| {
        message_error(
            index,
            &[],
            "model_attributes_type",
            "Input should be a valid dictionary or object to extract fields from",
            input.clone(),
            None,
        )
    })?;

    // content: required, str, 0..=MAX_MESSAGE_SIZE, then NUL-sanitized.
    let content = match object.get("content") {
        None | Some(Value::Null) => {
            return Err(message_error(
                index,
                &["content"],
                "missing",
                "Field required",
                input.clone(),
                None,
            ));
        }
        Some(Value::String(content)) => content.clone(),
        Some(value) => {
            return Err(message_error(
                index,
                &["content"],
                "string_type",
                "Input should be a valid string",
                value.clone(),
                None,
            ));
        }
    };
    if content.chars().count() > MAX_MESSAGE_SIZE {
        return Err(message_error(
            index,
            &["content"],
            "string_too_long",
            &format!("String should have at most {MAX_MESSAGE_SIZE} characters"),
            Value::String(content),
            Some(json!({"max_length": MAX_MESSAGE_SIZE})),
        ));
    }
    let content = content.replace('\0', "");

    // peer_id: required str (alias only — MessageCreate has no populate_by_name).
    let peer_name = match object.get("peer_id") {
        None | Some(Value::Null) => {
            return Err(message_error(
                index,
                &["peer_id"],
                "missing",
                "Field required",
                input.clone(),
                None,
            ));
        }
        Some(Value::String(peer_name)) => peer_name.clone(),
        Some(value) => {
            return Err(message_error(
                index,
                &["peer_id"],
                "string_type",
                "Input should be a valid string",
                value.clone(),
                None,
            ));
        }
    };

    let metadata = match object.get("metadata") {
        Some(value) if !value.is_null() => validate_metadata_at(
            value.clone(),
            &[
                Value::String("messages".to_string()),
                Value::Number(index.into()),
                Value::String("metadata".to_string()),
            ],
        )?,
        _ => json!({}),
    };

    let configuration = match object.get("configuration") {
        Some(value) if !value.is_null() => parse_message_configuration(value, index)?,
        _ => None,
    };

    let created_at = match object.get("created_at") {
        Some(value) if !value.is_null() => Some(parse_message_created_at(value, index)?),
        _ => None,
    };

    let token_count = crate::tokens::estimate_tokens(&content) as i32;

    Ok(ParsedMessage {
        peer_name,
        content,
        metadata,
        configuration,
        created_at,
        token_count,
    })
}

/// Normalize a message-level `configuration` to the reasoning-only shape Pydantic
/// keeps for `MessageConfiguration` (extras ignored). Returns `None` when no
/// recognized override remains, so the producer applies no message-level change.
fn parse_message_configuration(value: &Value, index: usize) -> Result<Option<Value>, ApiError> {
    let object = value.as_object().ok_or_else(|| {
        message_error(
            index,
            &["configuration"],
            "model_attributes_type",
            "Input should be a valid dictionary or object to extract fields from",
            value.clone(),
            None,
        )
    })?;
    let mut normalized = serde_json::Map::new();
    if let Some(reasoning) = object.get("reasoning").filter(|value| !value.is_null()) {
        let reasoning = validate_message_reasoning(reasoning, index)?;
        if !reasoning.as_object().is_none_or(serde_json::Map::is_empty) {
            normalized.insert("reasoning".to_string(), reasoning);
        }
    }
    if normalized.is_empty() {
        Ok(None)
    } else {
        Ok(Some(Value::Object(normalized)))
    }
}

fn validate_message_reasoning(value: &Value, index: usize) -> Result<Value, ApiError> {
    let object = value.as_object().ok_or_else(|| {
        message_error(
            index,
            &["configuration", "reasoning"],
            "model_attributes_type",
            "Input should be a valid dictionary or object to extract fields from",
            value.clone(),
            None,
        )
    })?;
    let mut out = serde_json::Map::new();
    match object.get("enabled") {
        None | Some(Value::Null) => {}
        Some(value) => {
            let parsed = coerce_bool(value).ok_or_else(|| {
                message_error(
                    index,
                    &["configuration", "reasoning", "enabled"],
                    "bool_parsing",
                    "Input should be a valid boolean, unable to interpret input",
                    value.clone(),
                    None,
                )
            })?;
            out.insert("enabled".to_string(), Value::Bool(parsed));
        }
    }
    match object.get("custom_instructions") {
        None | Some(Value::Null) => {}
        Some(Value::String(instructions)) => {
            validate_custom_instructions_msg(instructions, index)?;
            out.insert(
                "custom_instructions".to_string(),
                Value::String(instructions.clone()),
            );
        }
        Some(value) => {
            return Err(message_error(
                index,
                &["configuration", "reasoning", "custom_instructions"],
                "string_type",
                "Input should be a valid string",
                value.clone(),
                None,
            ));
        }
    }
    Ok(Value::Object(out))
}

fn validate_custom_instructions_msg(value: &str, index: usize) -> Result<(), ApiError> {
    if value.trim().is_empty() {
        return Ok(());
    }
    if crate::tokens::estimate_tokens(value) > 2000 {
        return Err(message_error(
            index,
            &["configuration", "reasoning", "custom_instructions"],
            "value_error",
            "Value error, custom_instructions exceeds DERIVER.MAX_CUSTOM_INSTRUCTIONS_TOKENS (2000 tokens)",
            Value::String(value.to_string()),
            Some(json!({"error": {}})),
        ));
    }
    Ok(())
}

fn parse_message_created_at(value: &Value, index: usize) -> Result<DateTime<Utc>, ApiError> {
    if let Value::String(text) = value {
        if let Ok(parsed) = DateTime::parse_from_rfc3339(text) {
            return Ok(parsed.with_timezone(&Utc));
        }
        // Accept a naive ISO timestamp (no offset) as UTC, matching Pydantic's
        // lenient datetime parsing for offset-less inputs.
        if let Ok(parsed) = chrono::NaiveDateTime::parse_from_str(text, "%Y-%m-%dT%H:%M:%S%.f") {
            return Ok(DateTime::from_naive_utc_and_offset(parsed, Utc));
        }
        return Err(message_error(
            index,
            &["created_at"],
            "datetime_from_date_parsing",
            "Input should be a valid datetime or date, input is too short",
            value.clone(),
            Some(json!({"error": "input is too short"})),
        ));
    }
    Err(message_error(
        index,
        &["created_at"],
        "datetime_type",
        "Input should be a valid datetime",
        value.clone(),
        None,
    ))
}

/// Parse a `WebhookEndpointCreate` body and validate its `url`, porting the
/// `validate_webhook_url` field validator (scheme + host required, http/https
/// only, private/internal IP literals rejected).
fn parse_webhook_create(input: Value) -> Result<String, ApiError> {
    let object = input.as_object().ok_or_else(|| {
        validation_error(
            "model_attributes_type",
            &[],
            "Input should be a valid dictionary or object to extract fields from",
            input.clone(),
            None,
        )
    })?;
    let url = match object.get("url") {
        None | Some(Value::Null) => {
            return Err(validation_error(
                "missing",
                &["url"],
                "Field required",
                input.clone(),
                None,
            ));
        }
        Some(Value::String(url)) => url.clone(),
        Some(value) => {
            return Err(validation_error(
                "string_type",
                &["url"],
                "Input should be a valid string",
                value.clone(),
                None,
            ));
        }
    };
    validate_webhook_url(&url)?;
    Ok(url)
}

fn validate_webhook_url(url: &str) -> Result<(), ApiError> {
    let Some((scheme, netloc)) = split_scheme_netloc(url) else {
        return Err(webhook_url_error(url, "Invalid URL format"));
    };
    if !scheme.eq_ignore_ascii_case("http") && !scheme.eq_ignore_ascii_case("https") {
        return Err(webhook_url_error(
            url,
            "Only HTTP and HTTPS URLs are allowed",
        ));
    }
    if hostname_from_netloc(&netloc)
        .and_then(|hostname| hostname.parse::<IpAddr>().ok())
        .is_some_and(|ip| is_blocked_ip(&ip))
    {
        return Err(webhook_url_error(
            url,
            "Private/internal IP addresses are not allowed",
        ));
    }
    Ok(())
}

/// Approximate `urlparse`'s scheme/netloc split: requires a non-empty scheme
/// before `://` and a non-empty network location before the first `/`, `?`, or
/// `#`. Returns `None` when either is missing (Python's "Invalid URL format").
fn split_scheme_netloc(url: &str) -> Option<(String, String)> {
    let (scheme, rest) = url.split_once("://")?;
    if scheme.is_empty() {
        return None;
    }
    let netloc_end = rest.find(['/', '?', '#']).unwrap_or(rest.len());
    let netloc = &rest[..netloc_end];
    if netloc.is_empty() {
        return None;
    }
    Some((scheme.to_string(), netloc.to_string()))
}

/// Extract the host from a netloc, stripping any `user:pass@` userinfo and the
/// `:port` suffix, and unwrapping a `[..]` IPv6 literal.
fn hostname_from_netloc(netloc: &str) -> Option<String> {
    let host_port = netloc.rsplit_once('@').map_or(netloc, |(_, host)| host);
    if let Some(after_bracket) = host_port.strip_prefix('[') {
        let end = after_bracket.find(']')?;
        return Some(after_bracket[..end].to_string());
    }
    let host = host_port.split(':').next().unwrap_or(host_port);
    if host.is_empty() {
        None
    } else {
        Some(host.to_string())
    }
}

/// Block the same IP-literal categories as Python's `ipaddress` checks. IPv4 is
/// covered by std helpers; for IPv6 the std private/link-local helpers are
/// unstable, so unique-local (`fc00::/7`) and link-local (`fe80::/10`) are
/// matched on the leading segment.
fn is_blocked_ip(ip: &IpAddr) -> bool {
    if ip.is_loopback() || ip.is_multicast() || ip.is_unspecified() {
        return true;
    }
    match ip {
        IpAddr::V4(addr) => {
            // `Ipv4Addr::is_reserved` is unstable; match Python's reserved range
            // (240.0.0.0/4) on the leading octet instead.
            addr.is_private() || addr.is_link_local() || addr.octets()[0] >= 240
        }
        IpAddr::V6(addr) => {
            let leading = addr.segments()[0];
            (leading & 0xfe00) == 0xfc00 || (leading & 0xffc0) == 0xfe80
        }
    }
}

fn webhook_url_error(input: &str, message: &str) -> ApiError {
    validation_error(
        "value_error",
        &["url"],
        &format!("Value error, {message}"),
        Value::String(input.to_string()),
        Some(json!({"error": {}})),
    )
}

fn message_error(
    index: usize,
    tail: &[&str],
    error_type: &str,
    msg: &str,
    input: Value,
    ctx: Option<Value>,
) -> ApiError {
    let mut loc = vec![
        Value::String("messages".to_string()),
        Value::Number(index.into()),
    ];
    loc.extend(tail.iter().map(|part| Value::String((*part).to_string())));
    validation_error_parts(error_type, &loc, msg, input, ctx)
}

fn session_peer_config_json(config: &SessionPeerConfig) -> Value {
    json!({
        "observe_me": config.observe_me,
        "observe_others": config.observe_others
    })
}

fn session_peer_config_patch_json(config: &SessionPeerConfig) -> Value {
    let mut object = serde_json::Map::new();
    if let Some(observe_me) = config.observe_me {
        object.insert("observe_me".to_string(), Value::Bool(observe_me));
    }
    if let Some(observe_others) = config.observe_others {
        object.insert("observe_others".to_string(), Value::Bool(observe_others));
    }
    Value::Object(object)
}

fn reject_session_peers(object: &serde_json::Map<String, Value>) -> Result<(), ApiError> {
    for field in ["peers", "peer_names"] {
        match object.get(field) {
            Some(Value::Null) | None => {}
            Some(Value::Object(peers)) if peers.is_empty() => {}
            Some(value) => {
                return Err(validation_error(
                    "value_error",
                    &[field],
                    "Value error, session peer membership is not implemented in the Rust write shadow yet",
                    value.clone(),
                    Some(json!({"error": {}})),
                ));
            }
        }
    }
    Ok(())
}

fn validate_resource_name(value: &str, field: &str) -> Result<(), ApiError> {
    if value.is_empty() {
        return Err(validation_error(
            "string_too_short",
            &[field],
            "String should have at least 1 character",
            Value::String(value.to_string()),
            Some(json!({"min_length": 1})),
        ));
    }
    if value.len() > 512 {
        return Err(validation_error(
            "string_too_long",
            &[field],
            "String should have at most 512 characters",
            Value::String(value.to_string()),
            Some(json!({"max_length": 512})),
        ));
    }
    if !value
        .bytes()
        .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-')
    {
        return Err(validation_error(
            "string_pattern_mismatch",
            &[field],
            "String should match pattern '^[a-zA-Z0-9_-]+$'",
            Value::String(value.to_string()),
            Some(json!({"pattern": "^[a-zA-Z0-9_-]+$"})),
        ));
    }
    Ok(())
}

fn validate_metadata(value: Value) -> Result<Value, ApiError> {
    validate_metadata_at(value, &[Value::String("metadata".to_string())])
}

/// Metadata validation (`_validate_metadata` BeforeValidator) at an arbitrary
/// field location, so nested message metadata reports the right `loc`.
fn validate_metadata_at(value: Value, loc: &[Value]) -> Result<Value, ApiError> {
    if !value.is_object() {
        return Err(validation_error_parts(
            "dict_type",
            loc,
            "Input should be a valid dictionary",
            value,
            None,
        ));
    }
    check_metadata_limits(&value, 1, &value, loc)?;
    Ok(sanitize_value(value))
}

fn validate_plain_object(value: Value, loc: &[&str]) -> Result<Value, ApiError> {
    if value.is_object() {
        Ok(value)
    } else {
        Err(validation_error(
            "dict_type",
            loc,
            "Input should be a valid dictionary",
            value,
            None,
        ))
    }
}

fn empty_object() -> Value {
    json!({})
}

fn check_metadata_limits(
    value: &Value,
    depth: usize,
    root: &Value,
    loc: &[Value],
) -> Result<(), ApiError> {
    let Some(object) = value.as_object() else {
        return Ok(());
    };
    if depth > 5 {
        return Err(validation_error_parts(
            "value_error",
            loc,
            "Value error, Metadata nesting exceeds maximum depth of 5",
            root.clone(),
            Some(json!({"error": {}})),
        ));
    }
    if depth == 1 && object.len() > 100 {
        return Err(validation_error_parts(
            "value_error",
            loc,
            "Value error, Metadata exceeds maximum of 100 top-level keys",
            value.clone(),
            Some(json!({"error": {}})),
        ));
    }
    for nested in object.values() {
        if nested.is_object() {
            check_metadata_limits(nested, depth + 1, root, loc)?;
        }
    }
    Ok(())
}

fn sanitize_value(value: Value) -> Value {
    match value {
        Value::String(value) => Value::String(value.replace('\0', "")),
        Value::Array(values) => Value::Array(values.into_iter().map(sanitize_value).collect()),
        Value::Object(values) => Value::Object(
            values
                .into_iter()
                .map(|(key, value)| (key.replace('\0', ""), sanitize_value(value)))
                .collect(),
        ),
        other => other,
    }
}

fn validate_workspace_configuration(value: Value) -> Result<Value, ApiError> {
    if !value.is_object() {
        return Err(validation_error(
            "model_attributes_type",
            &["configuration"],
            "Input should be a valid dictionary or object to extract fields from",
            value,
            None,
        ));
    }
    let mut value = strip_nulls(value);
    validate_reasoning_configuration(&mut value)?;
    validate_bool_field(
        &mut value,
        &["configuration", "peer_card"],
        "use",
        &["use", "create"],
    )?;
    validate_bool_field(
        &mut value,
        &["configuration", "peer_card"],
        "create",
        &["use", "create"],
    )?;
    validate_summary_configuration(&mut value)?;
    validate_bool_field(
        &mut value,
        &["configuration", "dream"],
        "enabled",
        &["enabled"],
    )?;
    Ok(strip_nulls(value))
}

fn validate_reasoning_configuration(configuration: &mut Value) -> Result<(), ApiError> {
    let Some(reasoning_value) = configuration.get_mut("reasoning") else {
        return Ok(());
    };
    let Some(reasoning) = reasoning_value.as_object_mut() else {
        return Err(validation_error(
            "model_attributes_type",
            &["configuration", "reasoning"],
            "Input should be a valid dictionary or object to extract fields from",
            reasoning_value.clone(),
            None,
        ));
    };
    if let Some(enabled) = validate_optional_bool(
        reasoning.get("enabled"),
        &["configuration", "reasoning", "enabled"],
    )? {
        reasoning.insert("enabled".to_string(), Value::Bool(enabled));
    }
    if let Some(custom_instructions) = reasoning.get("custom_instructions") {
        if custom_instructions.is_null() {
            return Ok(());
        }
        let Some(custom_instructions) = custom_instructions.as_str() else {
            return Err(validation_error(
                "string_type",
                &["configuration", "reasoning", "custom_instructions"],
                "Input should be a valid string",
                custom_instructions.clone(),
                None,
            ));
        };
        validate_custom_instructions(custom_instructions)?;
    }
    retain_keys(reasoning, &["enabled", "custom_instructions"]);
    Ok(())
}

fn validate_summary_configuration(configuration: &mut Value) -> Result<(), ApiError> {
    let Some(summary_value) = configuration.get_mut("summary") else {
        return Ok(());
    };
    let Some(summary) = summary_value.as_object_mut() else {
        return Err(validation_error(
            "model_attributes_type",
            &["configuration", "summary"],
            "Input should be a valid dictionary or object to extract fields from",
            summary_value.clone(),
            None,
        ));
    };

    if let Some(enabled) = validate_optional_bool(
        summary.get("enabled"),
        &["configuration", "summary", "enabled"],
    )? {
        summary.insert("enabled".to_string(), Value::Bool(enabled));
    }
    let short = validate_optional_i64(
        summary.get("messages_per_short_summary"),
        &["configuration", "summary", "messages_per_short_summary"],
        10,
    )?;
    if let Some(short) = short {
        summary.insert("messages_per_short_summary".to_string(), json!(short));
    }
    let long = validate_optional_i64(
        summary.get("messages_per_long_summary"),
        &["configuration", "summary", "messages_per_long_summary"],
        20,
    )?;
    if let Some(long) = long {
        summary.insert("messages_per_long_summary".to_string(), json!(long));
    }
    if let (Some(short), Some(long)) = (short, long) {
        if short >= long {
            return Err(validation_error(
                "value_error",
                &["configuration", "summary"],
                "Value error, messages_per_short_summary must be less than messages_per_long_summary",
                Value::Object(summary.clone()),
                Some(json!({"error": {}})),
            ));
        }
    }
    retain_keys(
        summary,
        &[
            "enabled",
            "messages_per_short_summary",
            "messages_per_long_summary",
        ],
    );
    Ok(())
}

fn validate_bool_field(
    configuration: &mut Value,
    parent_loc: &[&str],
    field: &str,
    allowed_fields: &[&str],
) -> Result<(), ApiError> {
    let section_name = parent_loc
        .last()
        .expect("parent loc should include section name");
    let Some(section_value) = configuration.get_mut(*section_name) else {
        return Ok(());
    };
    let Some(section) = section_value.as_object_mut() else {
        return Err(validation_error(
            "model_attributes_type",
            parent_loc,
            "Input should be a valid dictionary or object to extract fields from",
            section_value.clone(),
            None,
        ));
    };
    let mut loc = parent_loc.to_vec();
    loc.push(field);
    if let Some(value) = validate_optional_bool(section.get(field), &loc)? {
        section.insert(field.to_string(), Value::Bool(value));
    }
    retain_keys(section, allowed_fields);
    Ok(())
}

fn retain_keys(object: &mut serde_json::Map<String, Value>, allowed: &[&str]) {
    object.retain(|key, _| allowed.iter().any(|allowed| key == allowed));
}

fn validate_optional_bool(value: Option<&Value>, loc: &[&str]) -> Result<Option<bool>, ApiError> {
    let Some(value) = value else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    match coerce_bool(value) {
        Some(parsed) => Ok(Some(parsed)),
        None => Err(bool_parsing_error(loc, value.clone())),
    }
}

/// Pydantic-lax boolean coercion: native bools, integers `0`/`1`, and the string
/// spellings Pydantic accepts. Returns `None` when the value cannot be coerced.
fn coerce_bool(value: &Value) -> Option<bool> {
    if let Some(parsed) = value.as_bool() {
        return Some(parsed);
    }
    if let Some(number) = value.as_i64() {
        return match number {
            0 => Some(false),
            1 => Some(true),
            _ => None,
        };
    }
    if let Some(value_str) = value.as_str() {
        return match value_str.to_ascii_lowercase().as_str() {
            "0" | "false" | "f" | "n" | "no" | "off" => Some(false),
            "1" | "true" | "t" | "y" | "yes" | "on" => Some(true),
            _ => None,
        };
    }
    None
}

fn bool_parsing_error(loc: &[&str], input: Value) -> ApiError {
    validation_error(
        "bool_parsing",
        loc,
        "Input should be a valid boolean, unable to interpret input",
        input,
        None,
    )
}

fn validate_custom_instructions(value: &str) -> Result<(), ApiError> {
    if value.trim().is_empty() {
        return Ok(());
    }
    // Exact o200k_base parity with Python's `estimate_tokens`. The token budget
    // itself is still hardcoded to the Python default
    // (DERIVER.MAX_CUSTOM_INSTRUCTIONS_TOKENS = 2000); making it config-driven is
    // a separate change.
    if crate::tokens::estimate_tokens(value) > 2000 {
        return Err(validation_error(
            "value_error",
            &["configuration", "reasoning", "custom_instructions"],
            "Value error, custom_instructions exceeds DERIVER.MAX_CUSTOM_INSTRUCTIONS_TOKENS (2000 tokens)",
            Value::String(value.to_string()),
            Some(json!({"error": {}})),
        ));
    }
    Ok(())
}

fn validate_optional_i64(
    value: Option<&Value>,
    loc: &[&str],
    minimum: i64,
) -> Result<Option<i64>, ApiError> {
    let Some(value) = value else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    let number = if let Some(number) = value.as_i64() {
        number
    } else if let Some(number) = value.as_f64() {
        if number.fract() == 0.0 && number.is_finite() {
            number as i64
        } else {
            return Err(int_parsing_error(loc, value.clone()));
        }
    } else if let Some(value_str) = value.as_str() {
        value_str
            .parse::<i64>()
            .map_err(|_| int_parsing_error(loc, value.clone()))?
    } else {
        return Err(int_type_error(loc, value.clone()));
    };
    if number < minimum {
        return Err(validation_error(
            "greater_than_equal",
            loc,
            &format!("Input should be greater than or equal to {minimum}"),
            value.clone(),
            Some(json!({"ge": minimum})),
        ));
    }
    Ok(Some(number))
}

fn int_type_error(loc: &[&str], input: Value) -> ApiError {
    validation_error(
        "int_type",
        loc,
        "Input should be a valid integer",
        input,
        None,
    )
}

fn int_parsing_error(loc: &[&str], input: Value) -> ApiError {
    validation_error(
        "int_parsing",
        loc,
        "Input should be a valid integer, unable to parse string as an integer",
        input,
        None,
    )
}

fn strip_nulls(value: Value) -> Value {
    match value {
        Value::Array(values) => Value::Array(values.into_iter().map(strip_nulls).collect()),
        Value::Object(values) => Value::Object(
            values
                .into_iter()
                .filter_map(|(key, value)| {
                    if value.is_null() {
                        None
                    } else {
                        Some((key, strip_nulls(value)))
                    }
                })
                .collect(),
        ),
        other => other,
    }
}

/// Build a query-parameter validation error (`loc: ["query", name]`, no leading
/// `"body"`), matching FastAPI's coercion errors. `input` is the raw query
/// string, exactly as FastAPI reports it.
fn query_validation_error(
    error_type: &str,
    name: &str,
    msg: &str,
    raw: &str,
    ctx: Option<Value>,
) -> ApiError {
    let mut error = serde_json::Map::from_iter([
        ("type".to_string(), Value::String(error_type.to_string())),
        (
            "loc".to_string(),
            Value::Array(vec![
                Value::String("query".to_string()),
                Value::String(name.to_string()),
            ]),
        ),
        ("msg".to_string(), Value::String(msg.to_string())),
        ("input".to_string(), Value::String(raw.to_string())),
    ]);
    if let Some(ctx) = ctx {
        error.insert("ctx".to_string(), ctx);
    }
    ApiError::RequestValidation(Value::Array(vec![Value::Object(error)]))
}

/// Parse the `tokens` query param the way FastAPI's `int | None = Query(None,
/// le=GET_CONTEXT_MAX_TOKENS)` does: integer coercion (whitespace-trimmed) then
/// the `le` upper bound. No lower bound — negative/zero values are accepted and
/// short-circuit to an empty context downstream.
fn parse_context_tokens(raw: &str) -> Result<i64, ApiError> {
    let value: i64 = raw.trim().parse().map_err(|_| {
        query_validation_error(
            "int_parsing",
            "tokens",
            "Input should be a valid integer, unable to parse string as an integer",
            raw,
            None,
        )
    })?;
    if value > db::GET_CONTEXT_MAX_TOKENS {
        return Err(query_validation_error(
            "less_than_equal",
            "tokens",
            &format!("Input should be less than or equal to {}", db::GET_CONTEXT_MAX_TOKENS),
            raw,
            Some(json!({ "le": db::GET_CONTEXT_MAX_TOKENS })),
        ));
    }
    Ok(value)
}

/// Parse a boolean query param using Pydantic v2's accepted token set
/// (case-insensitive, whitespace-trimmed); anything else yields a `bool_parsing`
/// error.
fn parse_query_bool(raw: &str, name: &str) -> Result<bool, ApiError> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "1" | "on" | "t" | "true" | "y" | "yes" => Ok(true),
        "0" | "off" | "f" | "false" | "n" | "no" => Ok(false),
        _ => Err(query_validation_error(
            "bool_parsing",
            name,
            "Input should be a valid boolean, unable to interpret input",
            raw,
            None,
        )),
    }
}

fn validation_error(
    error_type: &str,
    loc: &[&str],
    msg: &str,
    input: Value,
    ctx: Option<Value>,
) -> ApiError {
    let parts = loc
        .iter()
        .map(|value| Value::String((*value).to_string()))
        .collect::<Vec<_>>();
    validation_error_parts(error_type, &parts, msg, input, ctx)
}

/// Like `validation_error`, but accepts pre-built `loc` parts so list indices can
/// be emitted as JSON numbers (matching Pydantic/FastAPI, which use integer
/// indices in the `loc` of list-item errors).
fn validation_error_parts(
    error_type: &str,
    loc: &[Value],
    msg: &str,
    input: Value,
    ctx: Option<Value>,
) -> ApiError {
    let mut location = vec![Value::String("body".to_string())];
    location.extend(loc.iter().cloned());
    let mut error = serde_json::Map::from_iter([
        ("type".to_string(), Value::String(error_type.to_string())),
        ("loc".to_string(), Value::Array(location)),
        ("msg".to_string(), Value::String(msg.to_string())),
        ("input".to_string(), input),
    ]);
    if let Some(ctx) = ctx {
        error.insert("ctx".to_string(), ctx);
    }
    ApiError::RequestValidation(Value::Array(vec![Value::Object(error)]))
}

fn authorize_peer_create(
    params: &JwtParams,
    workspace_id: &str,
    peer_name: &str,
) -> Result<(), ApiError> {
    if params.admin.unwrap_or(false) {
        return Ok(());
    }
    if params
        .workspace
        .as_deref()
        .is_some_and(|workspace| workspace != workspace_id)
    {
        return Err(ApiError::Authentication(
            "Unauthorized access to resource".to_string(),
        ));
    }
    if params.peer.as_deref().is_some_and(|peer| peer != peer_name) {
        return Err(ApiError::Authentication(
            "Unauthorized access to resource".to_string(),
        ));
    }
    Ok(())
}

fn authorize_session_create(
    params: &JwtParams,
    workspace_id: &str,
    session_name: &str,
) -> Result<(), ApiError> {
    if params.admin.unwrap_or(false) {
        return Ok(());
    }
    if params
        .workspace
        .as_deref()
        .is_some_and(|workspace| workspace != workspace_id)
    {
        return Err(ApiError::Authentication(
            "Unauthorized access to resource".to_string(),
        ));
    }
    if params
        .session
        .as_deref()
        .is_some_and(|session| session != session_name)
    {
        return Err(ApiError::Authentication(
            "Unauthorized access to resource".to_string(),
        ));
    }
    Ok(())
}

async fn ensure_session_observer_limit(
    pool: &PgPool,
    workspace_id: &str,
    session_id: &str,
    body: &SessionPeersRequest,
) -> Result<(), ApiError> {
    let incoming_observers = body
        .peers
        .values()
        .filter(|config| config.observe_others == Some(true))
        .count() as i64;
    if incoming_observers == 0 {
        return Ok(());
    }
    let limit = db::SESSION_OBSERVERS_LIMIT;
    if incoming_observers > limit {
        return Err(session_observer_limit_error(session_id, incoming_observers));
    }
    let peer_names = body.peers.keys().cloned().collect::<Vec<_>>();
    let existing_observers =
        db::count_active_session_observers_excluding(pool, workspace_id, session_id, &peer_names)
            .await?;
    let total_observers = existing_observers + incoming_observers;
    if total_observers > limit {
        return Err(session_observer_limit_error(session_id, total_observers));
    }
    Ok(())
}

fn session_observer_limit_error(session_id: &str, observer_count: i64) -> ApiError {
    ApiError::BadRequest(format!(
        "Cannot create session {session_id} with {observer_count} observers. Maximum allowed is {} observers per session. Observers are peers with 'observe_others' set to true.",
        db::SESSION_OBSERVERS_LIMIT
    ))
}

fn ensure_session_set_observer_limit(
    session_id: &str,
    body: &SessionPeersRequest,
) -> Result<(), ApiError> {
    let incoming_observers = body
        .peers
        .values()
        .filter(|config| config.observe_others == Some(true))
        .count() as i64;
    if incoming_observers > db::SESSION_OBSERVERS_LIMIT {
        return Err(session_observer_limit_error(session_id, incoming_observers));
    }
    Ok(())
}

async fn ensure_session_peer_config_observer_limit(
    pool: &PgPool,
    workspace_id: &str,
    session_id: &str,
    peer_id: &str,
    config: &SessionPeerConfig,
) -> Result<(), ApiError> {
    if config.observe_others != Some(true) {
        return Ok(());
    }
    let current_config =
        db::get_session_peer_configuration_value(pool, workspace_id, session_id, peer_id).await?;
    let already_observer = current_config
        .as_ref()
        .and_then(|value| value.get("observe_others"))
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if already_observer {
        return Ok(());
    }
    let existing_observers = db::count_active_session_observers_excluding(
        pool,
        workspace_id,
        session_id,
        &[peer_id.to_string()],
    )
    .await?;
    let observer_count = existing_observers + 1;
    if observer_count > db::SESSION_OBSERVERS_LIMIT {
        return Err(session_observer_limit_error(session_id, observer_count));
    }
    Ok(())
}

fn authorize_webhook_workspace(params: &JwtParams, workspace_id: &str) -> Result<(), AuthError> {
    if params.admin.unwrap_or(false) {
        return Ok(());
    }
    if params
        .workspace
        .as_deref()
        .is_some_and(|workspace| workspace != workspace_id)
    {
        return Err(AuthError::PermissionDenied);
    }
    Ok(())
}

fn session_write_error(error: sqlx::Error, workspace_id: &str, session_id: &str) -> ApiError {
    match error {
        sqlx::Error::RowNotFound => ApiError::NotFound(format!(
            "Session {session_id} not found in workspace {workspace_id}"
        )),
        error => ApiError::Database(error),
    }
}

#[cfg(test)]
mod tests {
    use crate::app::{authorize_webhook_workspace, validate_webhook_url};
    use crate::auth::{AuthError, JwtParams};

    #[test]
    fn webhook_url_accepts_public_http_urls() {
        assert!(validate_webhook_url("https://example.com/hook").is_ok());
        assert!(validate_webhook_url("http://example.com:8080/hook?x=1").is_ok());
        // A public IP literal is allowed.
        assert!(validate_webhook_url("https://8.8.8.8/hook").is_ok());
    }

    #[test]
    fn webhook_url_rejects_missing_scheme_or_host() {
        for url in ["example.com/hook", "https://", "/just/a/path", "ftp://"] {
            assert!(
                validate_webhook_url(url).is_err(),
                "expected {url:?} to be rejected"
            );
        }
    }

    #[test]
    fn webhook_url_rejects_non_http_schemes() {
        assert!(validate_webhook_url("ftp://example.com/x").is_err());
        assert!(validate_webhook_url("ws://example.com/x").is_err());
    }

    #[test]
    fn webhook_url_rejects_private_and_internal_ips() {
        for url in [
            "http://127.0.0.1/hook",   // loopback
            "http://10.0.0.5/hook",    // private
            "http://192.168.1.1/hook", // private
            "http://169.254.1.1/hook", // link-local
            "http://0.0.0.0/hook",     // unspecified
            "http://240.0.0.1/hook",   // reserved
            "http://[::1]/hook",       // ipv6 loopback
            "http://[fc00::1]/hook",   // ipv6 unique-local
            "http://[fe80::1]/hook",   // ipv6 link-local
        ] {
            assert!(
                validate_webhook_url(url).is_err(),
                "expected {url:?} to be rejected"
            );
        }
    }

    #[test]
    fn webhook_workspace_auth_matches_unscoped_python_route() {
        assert!(
            authorize_webhook_workspace(
                &JwtParams {
                    workspace: None,
                    ..JwtParams::default()
                },
                "workspace-a",
            )
            .is_ok()
        );
        assert!(
            authorize_webhook_workspace(
                &JwtParams {
                    workspace: Some("workspace-a".to_string()),
                    ..JwtParams::default()
                },
                "workspace-a",
            )
            .is_ok()
        );
        assert!(
            authorize_webhook_workspace(
                &JwtParams {
                    admin: Some(true),
                    workspace: Some("workspace-b".to_string()),
                    ..JwtParams::default()
                },
                "workspace-a",
            )
            .is_ok()
        );

        let error = authorize_webhook_workspace(
            &JwtParams {
                workspace: Some("workspace-b".to_string()),
                ..JwtParams::default()
            },
            "workspace-a",
        )
        .expect_err("mismatched workspace scope should fail");

        assert_eq!(error, AuthError::PermissionDenied);
    }
}
