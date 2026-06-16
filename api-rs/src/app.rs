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

use crate::auth::{AuthConfig, AuthError, JwtParams, authorize, create_scoped_key};
use crate::cache::PeerCache;
use crate::db;
use crate::error::ApiError;
use crate::filters::{FilterTarget, build_filter_clause};
use crate::pagination::Pagination;

#[derive(Debug, Clone)]
pub struct AppState {
    pub pool: Option<PgPool>,
    pub auth: AuthConfig,
    pub db_schema: String,
    pub write_enabled: bool,
    pub peer_cache: PeerCache,
}

impl AppState {
    pub fn new(
        pool: PgPool,
        auth: AuthConfig,
        db_schema: String,
        write_enabled: bool,
        peer_cache: PeerCache,
    ) -> Self {
        Self {
            pool: Some(pool),
            auth,
            db_schema,
            write_enabled,
            peer_cache,
        }
    }

    pub fn for_test(auth: AuthConfig) -> Self {
        Self {
            pool: None,
            auth,
            db_schema: "public".to_string(),
            write_enabled: false,
            peer_cache: PeerCache::disabled(),
        }
    }

    pub fn for_test_with_writes(auth: AuthConfig) -> Self {
        Self {
            pool: None,
            auth,
            db_schema: "public".to_string(),
            write_enabled: true,
            peer_cache: PeerCache::disabled(),
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
        .route("/v3/workspaces/{workspace_id}", put(update_workspace))
        .route(
            "/v3/workspaces/{workspace_id}/peers",
            post(get_or_create_peer),
        )
        .route("/v3/workspaces/list", post(list_workspaces))
        .route("/v3/workspaces/{workspace_id}/peers/list", post(list_peers))
        .route(
            "/v3/workspaces/{workspace_id}/peers/{peer_id}",
            put(update_peer),
        )
        .route(
            "/v3/workspaces/{workspace_id}/peers/{peer_id}/card",
            get(get_peer_card),
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
            "/v3/workspaces/{workspace_id}/queue/status",
            get(queue_status),
        )
        .route(
            "/v3/workspaces/{workspace_id}/webhooks",
            get(list_webhook_endpoints),
        )
        .route(
            "/v3/workspaces/{workspace_id}/sessions/{session_id}/messages/list",
            post(list_messages),
        )
        .route(
            "/v3/workspaces/{workspace_id}/conclusions/list",
            post(list_conclusions),
        )
        .route(
            "/v3/workspaces/{workspace_id}/sessions/{session_id}/messages/{message_id}",
            get(get_message),
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
    if !value.is_object() {
        return Err(validation_error(
            "dict_type",
            &["metadata"],
            "Input should be a valid dictionary",
            value,
            None,
        ));
    }
    check_metadata_limits(&value, 1, &value)?;
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

fn check_metadata_limits(value: &Value, depth: usize, root: &Value) -> Result<(), ApiError> {
    let Some(object) = value.as_object() else {
        return Ok(());
    };
    if depth > 5 {
        return Err(validation_error(
            "value_error",
            &["metadata"],
            "Value error, Metadata nesting exceeds maximum depth of 5",
            root.clone(),
            Some(json!({"error": {}})),
        ));
    }
    if depth == 1 && object.len() > 100 {
        return Err(validation_error(
            "value_error",
            &["metadata"],
            "Value error, Metadata exceeds maximum of 100 top-level keys",
            value.clone(),
            Some(json!({"error": {}})),
        ));
    }
    for nested in object.values() {
        if nested.is_object() {
            check_metadata_limits(nested, depth + 1, root)?;
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
    if let Some(value) = value.as_bool() {
        return Ok(Some(value));
    }
    if let Some(number) = value.as_i64() {
        return match number {
            0 => Ok(Some(false)),
            1 => Ok(Some(true)),
            _ => Err(bool_parsing_error(loc, value.clone())),
        };
    }
    if let Some(value_str) = value.as_str() {
        return match value_str.to_ascii_lowercase().as_str() {
            "0" | "false" | "f" | "n" | "no" | "off" => Ok(Some(false)),
            "1" | "true" | "t" | "y" | "yes" | "on" => Ok(Some(true)),
            _ => Err(bool_parsing_error(loc, value.clone())),
        };
    }
    Err(bool_parsing_error(loc, value.clone()))
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
    if approximate_token_count(value) > 2000 {
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

fn approximate_token_count(value: &str) -> usize {
    value
        .split_whitespace()
        .count()
        .max(value.chars().count() / 4)
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

fn validation_error(
    error_type: &str,
    loc: &[&str],
    msg: &str,
    input: Value,
    ctx: Option<Value>,
) -> ApiError {
    let mut location = vec![Value::String("body".to_string())];
    location.extend(loc.iter().map(|value| Value::String((*value).to_string())));
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
    use crate::app::authorize_webhook_workspace;
    use crate::auth::{AuthError, JwtParams};

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
