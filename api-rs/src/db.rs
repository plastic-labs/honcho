use chrono::{DateTime, Utc};
use rand::Rng;
use serde_json::{Value, json};
use sqlx::postgres::{PgArguments, PgRow};
use sqlx::query::{Query, QueryAs};
use sqlx::{FromRow, PgPool, Postgres, Row};
use std::collections::{BTreeMap, HashMap};
use std::io;

use crate::filters::FilterClause;
use crate::pagination::{Pagination, page_response};
use crate::producer::{
    PeerConfigEntry, QueueRecord, ResolvedConfiguration, resolve_batch_configuration_prefix,
};
use crate::queue_status::{QueueStatusCounts, build_queue_status};

const NANOID_ALPHABET: &[u8] = b"_-0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
const NANOID_LENGTH: usize = 21;
pub const SESSION_OBSERVERS_LIMIT: i64 = 10;

pub fn quote_identifier(value: &str) -> String {
    format!("\"{}\"", value.replace('"', "\"\""))
}

pub(crate) fn generate_nanoid() -> String {
    let mut rng = rand::thread_rng();
    (0..NANOID_LENGTH)
        .map(|_| {
            let index = rng.gen_range(0..NANOID_ALPHABET.len());
            NANOID_ALPHABET[index] as char
        })
        .collect()
}

#[derive(Debug, FromRow)]
struct WorkspaceRow {
    name: String,
    metadata: Value,
    configuration: Value,
    created_at: DateTime<Utc>,
}

#[derive(Debug, FromRow)]
struct PeerRow {
    name: String,
    workspace_name: String,
    metadata: Value,
    configuration: Value,
    created_at: DateTime<Utc>,
}

#[derive(Debug, FromRow)]
struct SessionRow {
    name: String,
    workspace_name: String,
    is_active: bool,
    metadata: Value,
    configuration: Value,
    created_at: DateTime<Utc>,
}

#[derive(Debug, FromRow)]
struct MessageRow {
    public_id: String,
    content: String,
    peer_name: String,
    session_name: String,
    metadata: Value,
    created_at: DateTime<Utc>,
    workspace_name: String,
    token_count: i32,
}

/// A message row as returned from `create_messages`, carrying both the public
/// response fields and the internal identifiers (`id`, `seq_in_session`) the
/// enqueue step needs.
#[derive(Debug, FromRow)]
pub struct CreatedMessage {
    pub id: i64,
    pub public_id: String,
    pub content: String,
    pub peer_name: String,
    pub session_name: String,
    pub metadata: Value,
    pub created_at: DateTime<Utc>,
    pub workspace_name: String,
    pub token_count: i32,
    pub seq_in_session: i64,
}

impl CreatedMessage {
    pub fn to_json(&self) -> Value {
        json!({
            "id": self.public_id,
            "content": self.content,
            "peer_id": self.peer_name,
            "session_id": self.session_name,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "workspace_id": self.workspace_name,
            "token_count": self.token_count
        })
    }
}

#[derive(Debug, FromRow)]
struct ConclusionRow {
    id: String,
    content: String,
    observer: String,
    observed: String,
    session_name: Option<String>,
    created_at: DateTime<Utc>,
}

/// The full document shape consumed by [`crate::representation::Representation::from_documents`].
/// `source_ids` is nullable JSONB (an array of ids or `NULL`); `internal_metadata`
/// is the non-null JSONB metadata object.
#[derive(Debug, FromRow)]
struct DocumentRow {
    id: String,
    content: String,
    level: String,
    created_at: DateTime<Utc>,
    session_name: Option<String>,
    source_ids: Option<Value>,
    internal_metadata: Value,
}

/// Convert a [`DocumentRow`] into a [`crate::representation::Document`], decoding
/// the `source_ids` JSONB array into a string vec (NULL / non-array → empty).
fn document_from_row(row: DocumentRow) -> crate::representation::Document {
    let source_ids = row
        .source_ids
        .as_ref()
        .and_then(Value::as_array)
        .map(|array| {
            array
                .iter()
                .filter_map(|value| value.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();
    crate::representation::Document {
        id: row.id,
        content: row.content,
        level: row.level,
        created_at: row.created_at,
        session_name: row.session_name,
        source_ids,
        internal_metadata: row.internal_metadata,
    }
}

#[derive(Debug, FromRow)]
struct WebhookEndpointRow {
    id: String,
    workspace_name: String,
    url: String,
    created_at: DateTime<Utc>,
}

pub async fn list_workspaces(
    pool: &PgPool,
    filter: &FilterClause,
    page: Pagination,
    reverse: bool,
) -> Result<Value, sqlx::Error> {
    let where_sql = format!("WHERE true{}", filter.sql);
    let count_sql = format!("SELECT count(*) AS count FROM workspaces {where_sql}");
    let total = fetch_count(pool, &count_sql, &filter.bindings).await?;

    let direction = direction(reverse);
    let limit_idx = filter.bindings.len() + 1;
    let offset_idx = filter.bindings.len() + 2;
    let sql = format!(
        "SELECT name, metadata, configuration, created_at \
         FROM workspaces {where_sql} \
         ORDER BY created_at {direction}, id {direction} \
         LIMIT ${limit_idx} OFFSET ${offset_idx}"
    );
    let mut query = bind_values_as(sqlx::query_as::<_, WorkspaceRow>(&sql), &filter.bindings);
    query = query.bind(page.limit()).bind(page.offset());
    let items = query
        .fetch_all(pool)
        .await?
        .into_iter()
        .map(workspace_json)
        .collect::<Vec<_>>();

    Ok(page_response(items, total as u64, page))
}

pub async fn get_or_create_workspace(
    pool: &PgPool,
    workspace_name: &str,
    metadata: Value,
    configuration: Value,
) -> Result<(Value, bool), sqlx::Error> {
    let row = sqlx::query_as::<_, WorkspaceRow>(
        "INSERT INTO workspaces (id, name, metadata, configuration) \
         VALUES ($1, $2, $3, $4) \
         ON CONFLICT (name) DO NOTHING \
         RETURNING name, metadata, configuration, created_at",
    )
    .bind(generate_nanoid())
    .bind(workspace_name)
    .bind(metadata)
    .bind(configuration)
    .fetch_optional(pool)
    .await?;

    if let Some(row) = row {
        return Ok((workspace_json(row), true));
    }

    let row = sqlx::query_as::<_, WorkspaceRow>(
        "SELECT name, metadata, configuration, created_at \
         FROM workspaces \
         WHERE name = $1",
    )
    .bind(workspace_name)
    .fetch_one(pool)
    .await?;
    Ok((workspace_json(row), false))
}

async fn ensure_workspace_exists(pool: &PgPool, workspace_name: &str) -> Result<(), sqlx::Error> {
    sqlx::query(
        "INSERT INTO workspaces (id, name, metadata, configuration) \
         VALUES ($1, $2, $3, $4) \
         ON CONFLICT (name) DO NOTHING",
    )
    .bind(generate_nanoid())
    .bind(workspace_name)
    .bind(json!({}))
    .bind(json!({}))
    .execute(pool)
    .await?;
    Ok(())
}

pub async fn update_workspace(
    pool: &PgPool,
    workspace_name: &str,
    metadata: Option<Value>,
    configuration: Option<Value>,
) -> Result<Value, sqlx::Error> {
    ensure_workspace_exists(pool, workspace_name).await?;

    let row = sqlx::query_as::<_, WorkspaceRow>(
        "UPDATE workspaces \
         SET metadata = COALESCE($2::jsonb, metadata), \
             configuration = CASE \
                 WHEN $3::jsonb IS NULL THEN configuration \
                 ELSE configuration || $3::jsonb \
             END \
         WHERE name = $1 \
         RETURNING name, metadata, configuration, created_at",
    )
    .bind(workspace_name)
    .bind(metadata)
    .bind(configuration)
    .fetch_one(pool)
    .await?;
    Ok(workspace_json(row))
}

pub async fn get_or_create_peer(
    pool: &PgPool,
    workspace_name: &str,
    peer_name: &str,
    metadata: Option<Value>,
    configuration: Option<Value>,
) -> Result<(Value, bool), sqlx::Error> {
    ensure_workspace_exists(pool, workspace_name).await?;

    let row = sqlx::query_as::<_, PeerRow>(
        "INSERT INTO peers (id, name, workspace_name, metadata, configuration) \
         VALUES ($1, $2, $3, $4, $5) \
         ON CONFLICT (name, workspace_name) DO NOTHING \
         RETURNING name, workspace_name, metadata, configuration, created_at",
    )
    .bind(generate_nanoid())
    .bind(peer_name)
    .bind(workspace_name)
    .bind(metadata.clone().unwrap_or_else(|| json!({})))
    .bind(configuration.clone().unwrap_or_else(|| json!({})))
    .fetch_optional(pool)
    .await?;

    if let Some(row) = row {
        return Ok((peer_json(row), true));
    }

    let row = sqlx::query_as::<_, PeerRow>(
        "UPDATE peers \
         SET metadata = COALESCE($3::jsonb, metadata), \
             configuration = COALESCE($4::jsonb, configuration) \
         WHERE workspace_name = $1 AND name = $2 \
         RETURNING name, workspace_name, metadata, configuration, created_at",
    )
    .bind(workspace_name)
    .bind(peer_name)
    .bind(metadata)
    .bind(configuration)
    .fetch_one(pool)
    .await?;

    Ok((peer_json(row), false))
}

pub async fn update_peer(
    pool: &PgPool,
    workspace_name: &str,
    peer_name: &str,
    metadata: Option<Value>,
    configuration: Option<Value>,
) -> Result<Value, sqlx::Error> {
    let (value, _) =
        get_or_create_peer(pool, workspace_name, peer_name, metadata, configuration).await?;
    Ok(value)
}

pub async fn get_or_create_session(
    pool: &PgPool,
    workspace_name: &str,
    session_name: &str,
    metadata: Option<Value>,
    configuration: Option<Value>,
) -> Result<(Value, bool), sqlx::Error> {
    ensure_workspace_exists(pool, workspace_name).await?;

    let row = sqlx::query_as::<_, SessionRow>(
        "INSERT INTO sessions (id, name, workspace_name, is_active, metadata, configuration) \
         VALUES ($1, $2, $3, true, $4, $5) \
         ON CONFLICT (name, workspace_name) DO NOTHING \
         RETURNING name, workspace_name, is_active, metadata, configuration, created_at",
    )
    .bind(generate_nanoid())
    .bind(session_name)
    .bind(workspace_name)
    .bind(metadata.clone().unwrap_or_else(|| json!({})))
    .bind(configuration.clone().unwrap_or_else(|| json!({})))
    .fetch_optional(pool)
    .await?;

    if let Some(row) = row {
        return Ok((session_json(row), true));
    }

    let row = sqlx::query_as::<_, SessionRow>(
        "UPDATE sessions \
         SET metadata = COALESCE($3::jsonb, metadata), \
             configuration = CASE \
                 WHEN $4::jsonb IS NULL THEN configuration \
                 ELSE configuration || $4::jsonb \
             END \
         WHERE workspace_name = $1 AND name = $2 AND is_active = true \
         RETURNING name, workspace_name, is_active, metadata, configuration, created_at",
    )
    .bind(workspace_name)
    .bind(session_name)
    .bind(metadata)
    .bind(configuration)
    .fetch_one(pool)
    .await?;

    Ok((session_json(row), false))
}

pub async fn update_session(
    pool: &PgPool,
    workspace_name: &str,
    session_name: &str,
    metadata: Option<Value>,
    configuration: Option<Value>,
) -> Result<Value, sqlx::Error> {
    let (value, _) =
        get_or_create_session(pool, workspace_name, session_name, metadata, configuration).await?;
    Ok(value)
}

pub async fn delete_session(
    pool: &PgPool,
    workspace_name: &str,
    session_name: &str,
) -> Result<(), sqlx::Error> {
    let mut transaction = pool.begin().await?;
    sqlx::query(
        "UPDATE sessions \
         SET is_active = false \
         WHERE workspace_name = $1 \
         AND name = $2 \
         AND is_active = true \
         RETURNING id",
    )
    .bind(workspace_name)
    .bind(session_name)
    .fetch_one(&mut *transaction)
    .await?;

    let work_unit_key = format!("deletion:{workspace_name}:session:{session_name}");
    sqlx::query(
        "INSERT INTO queue \
         (work_unit_key, payload, session_id, task_type, workspace_name, message_id) \
         VALUES ($1, $2, NULL, 'deletion', $3, NULL)",
    )
    .bind(work_unit_key)
    .bind(json!({
        "task_type": "deletion",
        "deletion_type": "session",
        "resource_id": session_name,
    }))
    .bind(workspace_name)
    .execute(&mut *transaction)
    .await?;

    transaction.commit().await?;
    Ok(())
}

#[derive(Debug)]
pub enum CloneSessionError {
    OriginalNotFound,
    CutoffMessageNotFound,
    Database(sqlx::Error),
}

impl From<sqlx::Error> for CloneSessionError {
    fn from(error: sqlx::Error) -> Self {
        CloneSessionError::Database(error)
    }
}

#[derive(Debug, FromRow)]
struct ClonedMessageRow {
    content: String,
    metadata: Value,
    peer_name: String,
    seq_in_session: i64,
}

#[derive(Debug, FromRow)]
struct ClonedPeerRow {
    peer_name: String,
    configuration: Value,
}

/// Clone an active session, mirroring Python's `crud.clone_session` semantics.
///
/// The new session gets a fresh nanoid name and copies the original session's
/// metadata and configuration. Messages (optionally truncated at and including
/// `cutoff_message_id`) are copied with fresh public ids and default
/// `token_count` / `internal_metadata`. Session-peer memberships are copied with
/// their configuration only when at least one message is cloned, matching
/// Python's early return for empty source sessions.
pub async fn clone_session(
    pool: &PgPool,
    workspace_name: &str,
    session_name: &str,
    cutoff_message_id: Option<&str>,
) -> Result<Value, CloneSessionError> {
    let mut transaction = pool.begin().await?;

    let original = sqlx::query_as::<_, SessionRow>(
        "SELECT name, workspace_name, is_active, metadata, configuration, created_at \
         FROM sessions \
         WHERE workspace_name = $1 AND name = $2 AND is_active = true",
    )
    .bind(workspace_name)
    .bind(session_name)
    .fetch_optional(&mut *transaction)
    .await?
    .ok_or(CloneSessionError::OriginalNotFound)?;

    let cutoff_id: Option<i64> = match cutoff_message_id {
        Some(message_id) => {
            let row = sqlx::query(
                "SELECT id FROM messages \
                 WHERE public_id = $1 AND session_name = $2 AND workspace_name = $3",
            )
            .bind(message_id)
            .bind(session_name)
            .bind(workspace_name)
            .fetch_optional(&mut *transaction)
            .await?
            .ok_or(CloneSessionError::CutoffMessageNotFound)?;
            Some(row.try_get::<i64, _>("id")?)
        }
        None => None,
    };

    let new_session_name = generate_nanoid();
    let new_session = sqlx::query_as::<_, SessionRow>(
        "INSERT INTO sessions (id, name, workspace_name, is_active, metadata, configuration) \
         VALUES ($1, $2, $3, true, $4, $5) \
         RETURNING name, workspace_name, is_active, metadata, configuration, created_at",
    )
    .bind(generate_nanoid())
    .bind(&new_session_name)
    .bind(workspace_name)
    .bind(original.metadata)
    .bind(original.configuration)
    .fetch_one(&mut *transaction)
    .await?;

    let messages = sqlx::query_as::<_, ClonedMessageRow>(
        "SELECT content, metadata, peer_name, seq_in_session \
         FROM messages \
         WHERE session_name = $1 AND workspace_name = $2 \
         AND ($3::bigint IS NULL OR id <= $3) \
         ORDER BY id",
    )
    .bind(session_name)
    .bind(workspace_name)
    .bind(cutoff_id)
    .fetch_all(&mut *transaction)
    .await?;

    // Python returns early (skipping peer copy) when the source has no messages.
    if messages.is_empty() {
        transaction.commit().await?;
        return Ok(session_json(new_session));
    }

    for message in &messages {
        // token_count has only an application-side default in Python; the column
        // is NOT NULL with no DB default, so we set the cloned default (0)
        // explicitly to match Python's omitted-value insert behavior.
        sqlx::query(
            "INSERT INTO messages \
             (public_id, session_name, content, metadata, workspace_name, peer_name, \
              seq_in_session, token_count) \
             VALUES ($1, $2, $3, $4, $5, $6, $7, 0)",
        )
        .bind(generate_nanoid())
        .bind(&new_session_name)
        .bind(&message.content)
        .bind(&message.metadata)
        .bind(workspace_name)
        .bind(&message.peer_name)
        .bind(message.seq_in_session)
        .execute(&mut *transaction)
        .await?;
    }

    let peers = sqlx::query_as::<_, ClonedPeerRow>(
        "SELECT peer_name, configuration FROM session_peers \
         WHERE workspace_name = $1 AND session_name = $2",
    )
    .bind(workspace_name)
    .bind(session_name)
    .fetch_all(&mut *transaction)
    .await?;

    for peer in &peers {
        sqlx::query(
            "INSERT INTO session_peers \
             (workspace_name, session_name, peer_name, configuration) \
             VALUES ($1, $2, $3, $4)",
        )
        .bind(workspace_name)
        .bind(&new_session_name)
        .bind(&peer.peer_name)
        .bind(&peer.configuration)
        .execute(&mut *transaction)
        .await?;
    }

    transaction.commit().await?;
    Ok(session_json(new_session))
}

pub async fn count_active_session_observers_excluding(
    pool: &PgPool,
    workspace_name: &str,
    session_name: &str,
    peer_names: &[String],
) -> Result<i64, sqlx::Error> {
    let row = sqlx::query(
        "SELECT count(*) AS count \
         FROM session_peers \
         WHERE workspace_name = $1 \
         AND session_name = $2 \
         AND left_at IS NULL \
         AND NOT (peer_name = ANY($3::text[])) \
         AND COALESCE((configuration->>'observe_others')::boolean, false)",
    )
    .bind(workspace_name)
    .bind(session_name)
    .bind(peer_names)
    .fetch_one(pool)
    .await?;
    row.try_get("count")
}

pub async fn add_peers_to_session(
    pool: &PgPool,
    workspace_name: &str,
    session_name: &str,
    peer_configs: &BTreeMap<String, Value>,
) -> Result<Value, sqlx::Error> {
    let (session, _) =
        get_or_create_session(pool, workspace_name, session_name, None, None).await?;

    for peer_name in peer_configs.keys() {
        get_or_create_peer(pool, workspace_name, peer_name, None, None).await?;
    }

    upsert_session_peers(pool, workspace_name, session_name, peer_configs).await?;

    Ok(session)
}

pub async fn set_peers_for_session(
    pool: &PgPool,
    workspace_name: &str,
    session_name: &str,
    peer_configs: &BTreeMap<String, Value>,
) -> Result<Value, sqlx::Error> {
    let session = fetch_active_session(pool, workspace_name, session_name).await?;

    sqlx::query(
        "UPDATE session_peers \
         SET left_at = now() \
         WHERE workspace_name = $1 \
         AND session_name = $2 \
         AND left_at IS NULL",
    )
    .bind(workspace_name)
    .bind(session_name)
    .execute(pool)
    .await?;

    for peer_name in peer_configs.keys() {
        get_or_create_peer(pool, workspace_name, peer_name, None, None).await?;
    }

    upsert_session_peers(pool, workspace_name, session_name, peer_configs).await?;

    Ok(session_json(session))
}

pub async fn remove_peers_from_session(
    pool: &PgPool,
    workspace_name: &str,
    session_name: &str,
    peer_names: &[String],
) -> Result<Value, sqlx::Error> {
    let session = fetch_active_session(pool, workspace_name, session_name).await?;

    sqlx::query(
        "UPDATE session_peers \
         SET left_at = now() \
         WHERE workspace_name = $1 \
         AND session_name = $2 \
         AND peer_name = ANY($3::text[]) \
         AND left_at IS NULL",
    )
    .bind(workspace_name)
    .bind(session_name)
    .bind(peer_names)
    .execute(pool)
    .await?;

    Ok(session_json(session))
}

async fn fetch_active_session(
    pool: &PgPool,
    workspace_name: &str,
    session_name: &str,
) -> Result<SessionRow, sqlx::Error> {
    sqlx::query_as::<_, SessionRow>(
        "SELECT name, workspace_name, is_active, metadata, configuration, created_at \
         FROM sessions \
         WHERE workspace_name = $1 AND name = $2 AND is_active = true",
    )
    .bind(workspace_name)
    .bind(session_name)
    .fetch_one(pool)
    .await
}

pub async fn ensure_active_session_exists(
    pool: &PgPool,
    workspace_name: &str,
    session_name: &str,
) -> Result<(), sqlx::Error> {
    fetch_active_session(pool, workspace_name, session_name)
        .await
        .map(|_| ())
}

pub async fn peer_exists(
    pool: &PgPool,
    workspace_name: &str,
    peer_name: &str,
) -> Result<bool, sqlx::Error> {
    let row = sqlx::query(
        "SELECT EXISTS( \
             SELECT 1 \
             FROM peers \
             WHERE workspace_name = $1 \
             AND name = $2 \
         ) AS exists",
    )
    .bind(workspace_name)
    .bind(peer_name)
    .fetch_one(pool)
    .await?;

    row.try_get("exists")
}

pub async fn get_session_peer_configuration_value(
    pool: &PgPool,
    workspace_name: &str,
    session_name: &str,
    peer_name: &str,
) -> Result<Option<Value>, sqlx::Error> {
    let row = sqlx::query(
        "SELECT configuration \
         FROM session_peers \
         WHERE workspace_name = $1 \
         AND session_name = $2 \
         AND peer_name = $3",
    )
    .bind(workspace_name)
    .bind(session_name)
    .bind(peer_name)
    .fetch_optional(pool)
    .await?;

    match row {
        Some(row) => row.try_get("configuration").map(Some),
        None => Ok(None),
    }
}

pub async fn set_session_peer_config(
    pool: &PgPool,
    workspace_name: &str,
    session_name: &str,
    peer_name: &str,
    configuration_patch: Value,
) -> Result<(), sqlx::Error> {
    sqlx::query(
        "INSERT INTO session_peers \
         (workspace_name, session_name, peer_name, configuration) \
         VALUES ($1, $2, $3, $4) \
         ON CONFLICT (session_name, peer_name, workspace_name) DO UPDATE \
         SET configuration = session_peers.configuration || EXCLUDED.configuration",
    )
    .bind(workspace_name)
    .bind(session_name)
    .bind(peer_name)
    .bind(configuration_patch)
    .execute(pool)
    .await?;

    Ok(())
}

async fn upsert_session_peers(
    pool: &PgPool,
    workspace_name: &str,
    session_name: &str,
    peer_configs: &BTreeMap<String, Value>,
) -> Result<(), sqlx::Error> {
    for (peer_name, configuration) in peer_configs {
        sqlx::query(
            "INSERT INTO session_peers \
             (workspace_name, session_name, peer_name, joined_at, left_at, configuration) \
             VALUES ($1, $2, $3, now(), NULL, $4) \
             ON CONFLICT (session_name, peer_name, workspace_name) DO UPDATE \
             SET joined_at = now(), \
                 left_at = NULL, \
                 configuration = CASE \
                     WHEN session_peers.left_at IS NOT NULL THEN EXCLUDED.configuration \
                     ELSE session_peers.configuration \
                 END",
        )
        .bind(workspace_name)
        .bind(session_name)
        .bind(peer_name)
        .bind(configuration.clone())
        .execute(pool)
        .await?;
    }

    Ok(())
}

pub async fn list_peers(
    pool: &PgPool,
    workspace_name: &str,
    filter: &FilterClause,
    page: Pagination,
    reverse: bool,
) -> Result<Value, sqlx::Error> {
    let workspace_idx = filter.bindings.len() + 1;
    let where_sql = format!(
        "WHERE true{} AND workspace_name = ${workspace_idx}",
        filter.sql
    );
    let count_sql = format!("SELECT count(*) AS count FROM peers {where_sql}");
    let total =
        fetch_count_with_tail(pool, &count_sql, &filter.bindings, &[workspace_name]).await?;

    let direction = direction(reverse);
    let limit_idx = filter.bindings.len() + 2;
    let offset_idx = filter.bindings.len() + 3;
    let sql = format!(
        "SELECT name, workspace_name, metadata, configuration, created_at \
         FROM peers {where_sql} \
         ORDER BY created_at {direction}, id {direction} \
         LIMIT ${limit_idx} OFFSET ${offset_idx}"
    );
    let mut query = bind_values_as(sqlx::query_as::<_, PeerRow>(&sql), &filter.bindings);
    query = query
        .bind(workspace_name)
        .bind(page.limit())
        .bind(page.offset());
    let items = query
        .fetch_all(pool)
        .await?
        .into_iter()
        .map(peer_json)
        .collect::<Vec<_>>();

    Ok(page_response(items, total as u64, page))
}

pub async fn list_sessions(
    pool: &PgPool,
    workspace_name: &str,
    filter: &FilterClause,
    page: Pagination,
    reverse: bool,
) -> Result<Value, sqlx::Error> {
    let workspace_idx = filter.bindings.len() + 1;
    let where_sql = format!(
        "WHERE true{} AND workspace_name = ${workspace_idx} AND is_active = true",
        filter.sql
    );
    list_sessions_with_where(pool, filter, page, reverse, &where_sql, &[workspace_name]).await
}

pub async fn list_peer_sessions(
    pool: &PgPool,
    workspace_name: &str,
    peer_name: &str,
    filter: &FilterClause,
    page: Pagination,
    reverse: bool,
) -> Result<Value, sqlx::Error> {
    let workspace_idx = filter.bindings.len() + 1;
    let peer_idx = filter.bindings.len() + 2;
    let where_sql = format!(
        "JOIN session_peers sp \
         ON sessions.name = sp.session_name \
         AND sessions.workspace_name = sp.workspace_name \
         WHERE true{} \
         AND sessions.workspace_name = ${workspace_idx} \
         AND sp.peer_name = ${peer_idx}",
        filter.sql
    );
    list_sessions_with_where(
        pool,
        filter,
        page,
        reverse,
        &where_sql,
        &[workspace_name, peer_name],
    )
    .await
}

pub async fn list_session_peers(
    pool: &PgPool,
    workspace_name: &str,
    session_name: &str,
    page: Pagination,
) -> Result<Value, sqlx::Error> {
    let count_sql = "SELECT count(*) AS count \
         FROM peers \
         JOIN session_peers sp \
         ON peers.name = sp.peer_name \
         AND peers.workspace_name = sp.workspace_name \
         WHERE peers.workspace_name = $1 \
         AND sp.session_name = $2 \
         AND sp.left_at IS NULL";
    let row = sqlx::query(count_sql)
        .bind(workspace_name)
        .bind(session_name)
        .fetch_one(pool)
        .await?;
    let total: i64 = row.try_get("count")?;

    let rows = sqlx::query_as::<_, PeerRow>(
        "SELECT peers.name, peers.workspace_name, peers.metadata, \
         peers.configuration, peers.created_at \
         FROM peers \
         JOIN session_peers sp \
         ON peers.name = sp.peer_name \
         AND peers.workspace_name = sp.workspace_name \
         WHERE peers.workspace_name = $1 \
         AND sp.session_name = $2 \
         AND sp.left_at IS NULL \
         LIMIT $3 OFFSET $4",
    )
    .bind(workspace_name)
    .bind(session_name)
    .bind(page.limit())
    .bind(page.offset())
    .fetch_all(pool)
    .await?;

    let items = rows.into_iter().map(peer_json).collect::<Vec<_>>();
    Ok(page_response(items, total as u64, page))
}

pub async fn get_peer_card(
    pool: &PgPool,
    workspace_name: &str,
    observer: &str,
    observed: &str,
) -> Result<Option<Value>, sqlx::Error> {
    let row = sqlx::query(
        "SELECT internal_metadata \
         FROM peers \
         WHERE workspace_name = $1 \
         AND name = $2",
    )
    .bind(workspace_name)
    .bind(observer)
    .fetch_optional(pool)
    .await?;

    let Some(row) = row else {
        return Ok(None);
    };
    let internal_metadata: Value = row.try_get("internal_metadata")?;

    Ok(Some(peer_card_json(observer, observed, &internal_metadata)))
}

pub async fn get_session_peer_config(
    pool: &PgPool,
    workspace_name: &str,
    session_name: &str,
    peer_name: &str,
) -> Result<Option<Value>, sqlx::Error> {
    let row = sqlx::query(
        "SELECT configuration \
         FROM session_peers \
         WHERE workspace_name = $1 \
         AND session_name = $2 \
         AND peer_name = $3",
    )
    .bind(workspace_name)
    .bind(session_name)
    .bind(peer_name)
    .fetch_optional(pool)
    .await?;

    let Some(row) = row else {
        return Ok(None);
    };
    let configuration: Value = row.try_get("configuration")?;

    Ok(Some(json!({
        "observe_me": configuration
            .get("observe_me")
            .cloned()
            .unwrap_or(Value::Null),
        "observe_others": configuration
            .get("observe_others")
            .cloned()
            .unwrap_or(Value::Null)
    })))
}

pub async fn get_session_summaries(
    pool: &PgPool,
    workspace_name: &str,
    session_name: &str,
) -> Result<Value, sqlx::Error> {
    let row = sqlx::query(
        "SELECT internal_metadata \
         FROM sessions \
         WHERE workspace_name = $1 \
         AND name = $2",
    )
    .bind(workspace_name)
    .bind(session_name)
    .fetch_optional(pool)
    .await?;

    let internal_metadata = match row {
        Some(row) => row.try_get("internal_metadata")?,
        None => Value::Null,
    };

    session_summaries_json(session_name, &internal_metadata)
}

/// Port of `summarizer.get_summary`: fetch a single stored summary from
/// `internal_metadata["summaries"][summary_type]`. Returns `None` when the
/// session is missing or carries no summary of that type.
pub async fn get_summary(
    pool: &PgPool,
    workspace_name: &str,
    session_name: &str,
    summary_type: &str,
) -> Result<Option<Value>, sqlx::Error> {
    let value: Option<Value> = sqlx::query_scalar(
        "SELECT internal_metadata -> 'summaries' -> $3 \
         FROM sessions WHERE workspace_name = $1 AND name = $2",
    )
    .bind(workspace_name)
    .bind(session_name)
    .bind(summary_type)
    .fetch_optional(pool)
    .await?
    .flatten();
    // `null` JSONB decodes to `Value::Null`; normalize it to `None`.
    Ok(value.filter(|v| !v.is_null()))
}

/// Port of `crud.get_messages_by_seq_range`: fetch a session's messages whose
/// `seq_in_session` falls in `[start_seq, end_seq]` (inclusive; open-ended when
/// `end_seq` is `None`), ordered ascending by sequence. Returns empty for an
/// invalid range (`start_seq < 1` or `start_seq > end_seq`), matching Python.
pub async fn get_messages_by_seq_range(
    pool: &PgPool,
    workspace_name: &str,
    session_name: &str,
    start_seq: i64,
    end_seq: Option<i64>,
) -> Result<Vec<BatchMessage>, sqlx::Error> {
    if start_seq < 1 || end_seq.is_some_and(|end| start_seq > end) {
        return Ok(Vec::new());
    }

    let rows = if let Some(end) = end_seq {
        sqlx::query(
            "SELECT id, public_id, content, created_at, peer_name, token_count, \
             session_name, workspace_name FROM messages \
             WHERE workspace_name = $1 AND session_name = $2 \
             AND seq_in_session >= $3 AND seq_in_session <= $4 \
             ORDER BY seq_in_session ASC",
        )
        .bind(workspace_name)
        .bind(session_name)
        .bind(start_seq)
        .bind(end)
        .fetch_all(pool)
        .await?
    } else {
        sqlx::query(
            "SELECT id, public_id, content, created_at, peer_name, token_count, \
             session_name, workspace_name FROM messages \
             WHERE workspace_name = $1 AND session_name = $2 \
             AND seq_in_session >= $3 \
             ORDER BY seq_in_session ASC",
        )
        .bind(workspace_name)
        .bind(session_name)
        .bind(start_seq)
        .fetch_all(pool)
        .await?
    };

    Ok(rows
        .into_iter()
        .map(|row| BatchMessage {
            id: row.get("id"),
            public_id: row.get("public_id"),
            content: row.get("content"),
            created_at: row.get("created_at"),
            peer_name: row.get("peer_name"),
            token_count: row.get("token_count"),
            session_name: row.get("session_name"),
            workspace_name: row.get("workspace_name"),
        })
        .collect())
}

/// Fetch a message's `public_id` by its primary-key `id`, scoped to
/// workspace + session. Backs the summary task's `message_public_id` fallback
/// (Python `consumer.py` `summary_fallback`): when the queue payload omits the
/// public id, the worker looks it up before summarizing. Returns `None` when no
/// such message exists (Python logs and returns without summarizing).
pub async fn get_message_public_id(
    pool: &PgPool,
    workspace_name: &str,
    session_name: &str,
    message_id: i64,
) -> Result<Option<String>, sqlx::Error> {
    let row = sqlx::query(
        "SELECT public_id FROM messages \
         WHERE workspace_name = $1 AND session_name = $2 AND id = $3",
    )
    .bind(workspace_name)
    .bind(session_name)
    .bind(message_id)
    .fetch_optional(pool)
    .await?;

    Ok(row.map(|row| row.get("public_id")))
}

/// Port of `summarizer._save_summary`: shallow-merge `summary` under
/// `internal_metadata["summaries"][summary_type]` for the session, atomically via
/// nested JSONB `||` (existing summaries of other types are preserved; the same
/// type is overwritten). A no-op when the session does not exist (Python logs and
/// returns). Returns whether a row was updated.
pub async fn save_summary(
    pool: &PgPool,
    workspace_name: &str,
    session_name: &str,
    summary_type: &str,
    summary: &Value,
) -> Result<bool, sqlx::Error> {
    let type_patch = json!({ summary_type: summary });
    let affected = sqlx::query(
        "UPDATE sessions \
         SET internal_metadata = internal_metadata || jsonb_build_object( \
             'summaries', \
             COALESCE(internal_metadata -> 'summaries', '{}'::jsonb) || $3::jsonb \
         ) \
         WHERE workspace_name = $1 AND name = $2",
    )
    .bind(workspace_name)
    .bind(session_name)
    .bind(type_patch)
    .execute(pool)
    .await?
    .rows_affected();
    Ok(affected > 0)
}

async fn list_sessions_with_where(
    pool: &PgPool,
    filter: &FilterClause,
    page: Pagination,
    reverse: bool,
    where_sql: &str,
    tail_bindings: &[&str],
) -> Result<Value, sqlx::Error> {
    let count_sql = format!("SELECT count(*) AS count FROM sessions {where_sql}");
    let total = fetch_count_with_tail(pool, &count_sql, &filter.bindings, tail_bindings).await?;

    let direction = direction(reverse);
    let limit_idx = filter.bindings.len() + tail_bindings.len() + 1;
    let offset_idx = filter.bindings.len() + tail_bindings.len() + 2;
    let sql = format!(
        "SELECT sessions.name, sessions.workspace_name, sessions.is_active, \
         sessions.metadata, sessions.configuration, sessions.created_at \
         FROM sessions {where_sql} \
         ORDER BY sessions.created_at {direction}, sessions.id {direction} \
         LIMIT ${limit_idx} OFFSET ${offset_idx}"
    );
    let mut query = bind_values_as(sqlx::query_as::<_, SessionRow>(&sql), &filter.bindings);
    for value in tail_bindings {
        query = query.bind(*value);
    }
    query = query.bind(page.limit()).bind(page.offset());
    let items = query
        .fetch_all(pool)
        .await?
        .into_iter()
        .map(session_json)
        .collect::<Vec<_>>();

    Ok(page_response(items, total as u64, page))
}

pub async fn queue_status(
    pool: &PgPool,
    workspace_name: &str,
    session_name: Option<&str>,
    observer: Option<&str>,
    observed: Option<&str>,
) -> Result<Value, sqlx::Error> {
    let mut sql = String::from(
        "SELECT queue.session_id, queue.processed, active_queue_sessions.id IS NOT NULL AS active \
         FROM queue \
         LEFT JOIN active_queue_sessions \
         ON queue.work_unit_key = active_queue_sessions.work_unit_key",
    );
    let mut next_idx = 1;

    if session_name.is_some() {
        sql.push_str(" JOIN sessions ON queue.session_id = sessions.id");
    }

    sql.push_str(&format!(
        " WHERE queue.workspace_name = ${next_idx} \
         AND queue.task_type IN ('representation', 'summary', 'dream')"
    ));
    next_idx += 1;

    if session_name.is_some() {
        sql.push_str(&format!(" AND sessions.name = ${next_idx}"));
        next_idx += 1;
    }
    match (observer.is_some(), observed.is_some()) {
        (true, true) => {
            sql.push_str(&format!(
                " AND (queue.payload->>'observer' = ${next_idx} \
                 OR queue.payload->>'observed' = ${})",
                next_idx + 1
            ));
        }
        (true, false) => {
            sql.push_str(&format!(" AND queue.payload->>'observer' = ${next_idx}"));
        }
        (false, true) => {
            sql.push_str(&format!(" AND queue.payload->>'observed' = ${next_idx}"));
        }
        (false, false) => {}
    }

    let mut query = sqlx::query(&sql).bind(workspace_name);
    if let Some(session_name) = session_name {
        query = query.bind(session_name);
    }
    if let Some(observer) = observer {
        query = query.bind(observer);
    }
    if let Some(observed) = observed {
        query = query.bind(observed);
    }

    let rows = query.fetch_all(pool).await?;
    let mut total = 0;
    let mut completed = 0;
    let mut in_progress = 0;
    let mut pending = 0;
    let mut session_counts = std::collections::BTreeMap::<String, (i64, i64, i64)>::new();

    for row in rows {
        total += 1;
        let processed: bool = row.try_get("processed")?;
        let active: bool = row.try_get("active")?;
        let session_id: Option<String> = row.try_get("session_id")?;
        let bucket = if processed {
            completed += 1;
            0
        } else if active {
            in_progress += 1;
            1
        } else {
            pending += 1;
            2
        };
        if let Some(session_id) = session_id {
            let entry = session_counts.entry(session_id).or_insert((0, 0, 0));
            match bucket {
                0 => entry.0 += 1,
                1 => entry.1 += 1,
                _ => entry.2 += 1,
            }
        }
    }

    Ok(build_queue_status(
        session_name,
        QueueStatusCounts {
            total,
            completed,
            in_progress,
            pending,
            sessions: session_counts
                .into_iter()
                .map(|(session_id, (completed, in_progress, pending))| {
                    (session_id, completed, in_progress, pending)
                })
                .collect(),
        },
    ))
}

pub async fn list_webhook_endpoints(
    pool: &PgPool,
    workspace_name: &str,
    page: Pagination,
) -> Result<Value, sqlx::Error> {
    let row = sqlx::query(
        "SELECT count(*) AS count \
         FROM webhook_endpoints \
         WHERE workspace_name = $1",
    )
    .bind(workspace_name)
    .fetch_one(pool)
    .await?;
    let total: i64 = row.try_get("count")?;

    let rows = sqlx::query_as::<_, WebhookEndpointRow>(
        "SELECT id, workspace_name, url, created_at \
         FROM webhook_endpoints \
         WHERE workspace_name = $1 \
         LIMIT $2 OFFSET $3",
    )
    .bind(workspace_name)
    .bind(page.limit())
    .bind(page.offset())
    .fetch_all(pool)
    .await?;

    let items = rows
        .into_iter()
        .map(|row| webhook_endpoint_json(row.id, row.workspace_name, row.url, row.created_at))
        .collect::<Vec<_>>();
    Ok(page_response(items, total as u64, page))
}

/// All webhook endpoint URLs for a workspace (unpaginated), for the delivery
/// path. Port of `webhook_delivery._get_webhook_urls` / `crud.list_webhook_endpoints`
/// used in the worker context (which fetches every endpoint, not a page).
pub async fn get_webhook_endpoint_urls(
    pool: &PgPool,
    workspace_name: &str,
) -> Result<Vec<String>, sqlx::Error> {
    let urls: Vec<String> = sqlx::query_scalar(
        "SELECT url FROM webhook_endpoints WHERE workspace_name = $1",
    )
    .bind(workspace_name)
    .fetch_all(pool)
    .await?;
    Ok(urls)
}

pub async fn list_messages(
    pool: &PgPool,
    workspace_name: &str,
    session_name: &str,
    filter: &FilterClause,
    page: Pagination,
    reverse: bool,
) -> Result<Value, sqlx::Error> {
    let workspace_idx = filter.bindings.len() + 1;
    let session_idx = filter.bindings.len() + 2;
    let where_sql = format!(
        "WHERE true{} \
         AND workspace_name = ${workspace_idx} \
         AND session_name = ${session_idx}",
        filter.sql
    );
    let count_sql = format!("SELECT count(*) AS count FROM messages {where_sql}");
    let total = fetch_count_with_tail(
        pool,
        &count_sql,
        &filter.bindings,
        &[workspace_name, session_name],
    )
    .await?;

    let direction = direction(reverse);
    let limit_idx = filter.bindings.len() + 3;
    let offset_idx = filter.bindings.len() + 4;
    let sql = format!(
        "SELECT public_id, content, peer_name, session_name, metadata, created_at, \
         workspace_name, token_count \
         FROM messages {where_sql} \
         ORDER BY id {direction} \
         LIMIT ${limit_idx} OFFSET ${offset_idx}"
    );
    let mut query = bind_values_as(sqlx::query_as::<_, MessageRow>(&sql), &filter.bindings);
    query = query
        .bind(workspace_name)
        .bind(session_name)
        .bind(page.limit())
        .bind(page.offset());
    let items = query
        .fetch_all(pool)
        .await?
        .into_iter()
        .map(message_json)
        .collect::<Vec<_>>();

    Ok(page_response(items, total as u64, page))
}

/// Fetch messages by public id, preserving the input ordering and dropping ids
/// that don't resolve, porting `fetch_messages_by_ids`. Used by the search route
/// to hydrate the RRF-fused id list back into message JSON.
pub async fn fetch_messages_by_ids(
    pool: &PgPool,
    message_ids: &[String],
) -> Result<Vec<Value>, sqlx::Error> {
    if message_ids.is_empty() {
        return Ok(Vec::new());
    }

    let rows = sqlx::query_as::<_, MessageRow>(
        "SELECT public_id, content, peer_name, session_name, metadata, created_at, \
         workspace_name, token_count \
         FROM messages WHERE public_id = ANY($1)",
    )
    .bind(message_ids)
    .fetch_all(pool)
    .await?;

    let mut by_id: BTreeMap<String, Value> = rows
        .into_iter()
        .map(|row| (row.public_id.clone(), message_json(row)))
        .collect();

    Ok(message_ids
        .iter()
        .filter_map(|id| by_id.remove(id))
        .collect())
}

/// Mirror of `settings.GET_CONTEXT_MAX_TOKENS` (default 100_000): the default
/// and upper bound for the `get_context` `tokens` query parameter.
pub const GET_CONTEXT_MAX_TOKENS: i64 = 100_000;

/// Mirror of `settings.DERIVER.WORKING_REPRESENTATION_MAX_OBSERVATIONS` (default
/// 100): the observation budget for a working representation when the caller does
/// not pass `max_conclusions`.
pub const WORKING_REPRESENTATION_MAX_OBSERVATIONS: i64 = 100;

/// Mirror of `settings.MAX_FILE_SIZE` (default 5 MiB): the upper bound for an
/// uploaded file before `FileTooLargeError`.
pub const MAX_FILE_SIZE: i64 = 5_242_880;

/// Mirror of `settings.MAX_MESSAGE_SIZE` (default 25_000): the per-message
/// character cap, also used as the file-upload chunk size.
pub const MAX_MESSAGE_SIZE: usize = 25_000;

/// Shallow-merge `patch` into a message's `internal_metadata` (JSONB `||`),
/// porting the file-upload post-create update (`message.internal_metadata.update(
/// file_metadata); flag_modified`). Scoped by workspace + session + id. Returns
/// whether a row was updated.
pub async fn merge_message_internal_metadata(
    pool: &PgPool,
    workspace_name: &str,
    session_name: &str,
    message_id: i64,
    patch: &Value,
) -> Result<bool, sqlx::Error> {
    let result = sqlx::query(
        "UPDATE messages \
         SET internal_metadata = COALESCE(internal_metadata, '{}'::jsonb) || $4::jsonb \
         WHERE workspace_name = $1 AND session_name = $2 AND id = $3",
    )
    .bind(workspace_name)
    .bind(session_name)
    .bind(message_id)
    .bind(patch)
    .execute(pool)
    .await?;
    Ok(result.rows_affected() > 0)
}

/// Port of `crud.get_messages_id_range`. Returns message rows by internal PK id
/// range, optionally constrained by a descending running-token-sum window.
///
/// Faithful to the Python guard (`start_id < 0`, or a degenerate `end_id` range,
/// → empty), the `if token_limit:` truthiness (`None`/`Some(0)` take the
/// unordered "all rows" branch; any non-zero limit takes the windowed branch),
/// and the ascending-id ordering of the windowed branch.
async fn get_messages_id_range(
    pool: &PgPool,
    workspace_name: &str,
    session_name: &str,
    start_id: i64,
    end_id: Option<i64>,
    token_limit: Option<i64>,
) -> Result<Vec<MessageRow>, sqlx::Error> {
    if start_id < 0 {
        return Ok(Vec::new());
    }
    if let Some(end) = end_id
        && (start_id >= end || end <= 0)
    {
        return Ok(Vec::new());
    }

    // `if token_limit:` in Python — 0 (and None) is falsy.
    let token_window = matches!(token_limit, Some(limit) if limit != 0);
    let limit_value = token_limit.unwrap_or(0);

    let rows = match (end_id, token_window) {
        (None, true) => {
            sqlx::query_as::<_, MessageRow>(
                "SELECT m.public_id, m.content, m.peer_name, m.session_name, m.metadata, \
                 m.created_at, m.workspace_name, m.token_count \
                 FROM messages m \
                 JOIN (SELECT id, SUM(token_count) OVER (ORDER BY id DESC) AS running_token_sum \
                       FROM messages \
                       WHERE workspace_name = $1 AND session_name = $2 AND id >= $3) sub \
                 ON m.id = sub.id \
                 WHERE sub.running_token_sum <= $4 \
                 ORDER BY m.id",
            )
            .bind(workspace_name)
            .bind(session_name)
            .bind(start_id)
            .bind(limit_value)
            .fetch_all(pool)
            .await?
        }
        (Some(end), true) => {
            sqlx::query_as::<_, MessageRow>(
                "SELECT m.public_id, m.content, m.peer_name, m.session_name, m.metadata, \
                 m.created_at, m.workspace_name, m.token_count \
                 FROM messages m \
                 JOIN (SELECT id, SUM(token_count) OVER (ORDER BY id DESC) AS running_token_sum \
                       FROM messages \
                       WHERE workspace_name = $1 AND session_name = $2 AND id >= $3 AND id < $4) sub \
                 ON m.id = sub.id \
                 WHERE sub.running_token_sum <= $5 \
                 ORDER BY m.id",
            )
            .bind(workspace_name)
            .bind(session_name)
            .bind(start_id)
            .bind(end)
            .bind(limit_value)
            .fetch_all(pool)
            .await?
        }
        (None, false) => {
            sqlx::query_as::<_, MessageRow>(
                "SELECT public_id, content, peer_name, session_name, metadata, created_at, \
                 workspace_name, token_count \
                 FROM messages \
                 WHERE workspace_name = $1 AND session_name = $2 AND id >= $3",
            )
            .bind(workspace_name)
            .bind(session_name)
            .bind(start_id)
            .fetch_all(pool)
            .await?
        }
        (Some(end), false) => {
            sqlx::query_as::<_, MessageRow>(
                "SELECT public_id, content, peer_name, session_name, metadata, created_at, \
                 workspace_name, token_count \
                 FROM messages \
                 WHERE workspace_name = $1 AND session_name = $2 AND id >= $3 AND id < $4",
            )
            .bind(workspace_name)
            .bind(session_name)
            .bind(start_id)
            .bind(end)
            .fetch_all(pool)
            .await?
        }
    };

    Ok(rows)
}

/// Port of the base (no-perspective) path of `get_session_context` —
/// `summarizer.get_session_context` + the router's `SessionContext`
/// serialization. Returns the JSON body: `{id, messages, summary,
/// peer_representation, peer_card}` with the latter two always null on this path.
///
/// Allocates 40% of the token budget to a summary (longest that fits, long
/// preferred over short when strictly larger) and the remainder to the most
/// recent messages after the summary's coverage point. `token_limit <= 0`
/// short-circuits to an empty context with no DB access, matching Python.
pub async fn get_session_context(
    pool: &PgPool,
    workspace_name: &str,
    session_name: &str,
    token_limit: i64,
    include_summary: bool,
) -> Result<Value, sqlx::Error> {
    let (summary_json, messages) =
        session_summary_and_messages(pool, workspace_name, session_name, token_limit, include_summary)
            .await?;
    Ok(session_context_json(session_name, summary_json, messages))
}

/// Perspective-scoped session context, porting the `peer_target` branch of
/// Python `get_session_context`: the same 40%-summary / token-budgeted-messages
/// selection as [`get_session_context`], but run against the budget already
/// reduced by the target peer's working-representation + peer-card token cost,
/// with both injected into the response. The caller computes `adjusted_token_limit`
/// (and the markdown) because the representation + card token estimates depend on
/// the LLM-facing string forms, which live outside the DB layer.
pub async fn get_perspective_session_context(
    pool: &PgPool,
    workspace_name: &str,
    session_name: &str,
    adjusted_token_limit: i64,
    include_summary: bool,
    peer_representation_markdown: String,
    peer_card: Option<Vec<String>>,
) -> Result<Value, sqlx::Error> {
    let (summary_json, messages) = session_summary_and_messages(
        pool,
        workspace_name,
        session_name,
        adjusted_token_limit,
        include_summary,
    )
    .await?;
    Ok(json!({
        "id": session_name,
        "messages": Value::Array(messages.into_iter().map(message_json).collect()),
        "summary": summary_json,
        "peer_representation": peer_representation_markdown,
        "peer_card": match peer_card {
            Some(card) => Value::Array(card.into_iter().map(Value::String).collect()),
            None => Value::Null,
        },
    }))
}

/// Shared summary-selection + token-budgeted message fetch behind
/// [`get_session_context`] and [`get_perspective_session_context`]. Ports
/// `_select_summary_for_context` (40% summary budget, long-preferred when it both
/// fits and beats the short summary) followed by the `get_messages_id_range`
/// fetch. Returns `(summary JSON or Null, messages)`.
async fn session_summary_and_messages(
    pool: &PgPool,
    workspace_name: &str,
    session_name: &str,
    token_limit: i64,
    include_summary: bool,
) -> Result<(Value, Vec<MessageRow>), sqlx::Error> {
    if token_limit <= 0 {
        return Ok((Value::Null, Vec::new()));
    }

    let mut messages_tokens = token_limit;
    let mut messages_start_id: i64 = 0;
    let mut summary_json = Value::Null;

    if include_summary {
        let summary_tokens_limit = (token_limit as f64 * 0.4) as i64;
        let internal_metadata =
            fetch_session_internal_metadata(pool, workspace_name, session_name).await?;
        let summaries = internal_metadata.get("summaries").and_then(Value::as_object);
        let short = summaries.and_then(|items| items.get("honcho_chat_summary_short"));
        let long = summaries.and_then(|items| items.get("honcho_chat_summary_long"));

        let long_len = match long {
            Some(summary) => summary_field_i64(summary, "token_count")?,
            None => 0,
        };
        let short_len = match short {
            Some(summary) => summary_field_i64(summary, "token_count")?,
            None => 0,
        };

        if let Some(long_summary) = long
            && long_len <= summary_tokens_limit
            && long_len > short_len
        {
            summary_json = schema_summary_json(long_summary)?;
            messages_tokens = token_limit - long_len;
            messages_start_id = summary_field_i64(long_summary, "message_id")?;
        }

        if summary_json.is_null()
            && let Some(short_summary) = short
            && short_len <= summary_tokens_limit
            && short_len > 0
        {
            summary_json = schema_summary_json(short_summary)?;
            messages_tokens = token_limit - short_len;
            messages_start_id = summary_field_i64(short_summary, "message_id")?;
        }
    }

    let messages = get_messages_id_range(
        pool,
        workspace_name,
        session_name,
        messages_start_id,
        None,
        Some(messages_tokens),
    )
    .await?;

    Ok((summary_json, messages))
}

fn session_context_json(session_name: &str, summary: Value, messages: Vec<MessageRow>) -> Value {
    json!({
        "id": session_name,
        "messages": Value::Array(messages.into_iter().map(message_json).collect()),
        "summary": summary,
        "peer_representation": Value::Null,
        "peer_card": Value::Null
    })
}

async fn fetch_session_internal_metadata(
    pool: &PgPool,
    workspace_name: &str,
    session_name: &str,
) -> Result<Value, sqlx::Error> {
    let row = sqlx::query(
        "SELECT internal_metadata FROM sessions WHERE workspace_name = $1 AND name = $2",
    )
    .bind(workspace_name)
    .bind(session_name)
    .fetch_optional(pool)
    .await?;

    Ok(match row {
        Some(row) => row.try_get("internal_metadata")?,
        None => Value::Null,
    })
}

fn summary_field_i64(summary: &Value, key: &str) -> Result<i64, sqlx::Error> {
    let object = summary
        .as_object()
        .ok_or_else(|| invalid_summary_error("summary entry must be an object"))?;
    require_i64(object, key)
}

pub async fn list_conclusions(
    pool: &PgPool,
    workspace_name: &str,
    filter: &FilterClause,
    page: Pagination,
    reverse: bool,
) -> Result<Value, sqlx::Error> {
    let workspace_idx = filter.bindings.len() + 1;
    let where_sql = format!(
        "WHERE true{} \
         AND workspace_name = ${workspace_idx} \
         AND deleted_at IS NULL",
        filter.sql
    );
    let count_sql = format!("SELECT count(*) AS count FROM documents {where_sql}");
    let total =
        fetch_count_with_tail(pool, &count_sql, &filter.bindings, &[workspace_name]).await?;

    let direction = if reverse { "ASC" } else { "DESC" };
    let limit_idx = filter.bindings.len() + 2;
    let offset_idx = filter.bindings.len() + 3;
    let sql = format!(
        "SELECT id, content, observer, observed, session_name, created_at \
         FROM documents {where_sql} \
         ORDER BY created_at {direction} \
         LIMIT ${limit_idx} OFFSET ${offset_idx}"
    );
    let mut query = bind_values_as(sqlx::query_as::<_, ConclusionRow>(&sql), &filter.bindings);
    query = query
        .bind(workspace_name)
        .bind(page.limit())
        .bind(page.offset());
    let items = query
        .fetch_all(pool)
        .await?
        .into_iter()
        .map(conclusion_json)
        .collect::<Vec<_>>();

    Ok(page_response(items, total as u64, page))
}

/// Distinct session names where `peer_name` has any membership record, porting
/// `crud.get_peer_session_names` (any record grants visibility, ignoring the
/// joined/left window).
async fn fetch_peer_session_names(
    pool: &PgPool,
    workspace_name: &str,
    peer_name: &str,
) -> Result<Vec<String>, sqlx::Error> {
    let rows: Vec<(String,)> = sqlx::query_as(
        "SELECT DISTINCT session_name FROM session_peers \
         WHERE workspace_name = $1 AND peer_name = $2",
    )
    .bind(workspace_name)
    .bind(peer_name)
    .fetch_all(pool)
    .await?;
    Ok(rows.into_iter().map(|(name,)| name).collect())
}

/// Port of `crud.get_messages_by_date_range` (the dialectic
/// `get_messages_by_date_range` tool's data layer). Returns messages in the
/// workspace optionally scoped to a session — or, when `observer` is given
/// without a session, to the sessions that peer belongs to (no memberships →
/// empty) — bounded by an inclusive `[after, before]` `created_at` window,
/// ordered by `created_at` (desc unless `order_desc` is false), limited.
#[allow(clippy::too_many_arguments)]
pub async fn get_messages_by_date_range(
    pool: &PgPool,
    workspace_name: &str,
    session_name: Option<&str>,
    observer: Option<&str>,
    after: Option<DateTime<Utc>>,
    before: Option<DateTime<Utc>>,
    limit: i64,
    order_desc: bool,
) -> Result<Vec<Value>, sqlx::Error> {
    let allowed_sessions: Option<Vec<String>> = match (observer, session_name) {
        (Some(observer), None) => {
            let names = fetch_peer_session_names(pool, workspace_name, observer).await?;
            if names.is_empty() {
                return Ok(Vec::new());
            }
            Some(names)
        }
        _ => None,
    };

    let mut sql = String::from(
        "SELECT public_id, content, peer_name, session_name, metadata, created_at, \
         workspace_name, token_count FROM messages WHERE workspace_name = $1",
    );
    let mut placeholder = 1;
    if session_name.is_some() {
        placeholder += 1;
        sql.push_str(&format!(" AND session_name = ${placeholder}"));
    } else if allowed_sessions.is_some() {
        placeholder += 1;
        sql.push_str(&format!(" AND session_name = ANY(${placeholder})"));
    }
    if after.is_some() {
        placeholder += 1;
        sql.push_str(&format!(" AND created_at >= ${placeholder}"));
    }
    if before.is_some() {
        placeholder += 1;
        sql.push_str(&format!(" AND created_at <= ${placeholder}"));
    }
    sql.push_str(if order_desc {
        " ORDER BY created_at DESC"
    } else {
        " ORDER BY created_at ASC"
    });
    placeholder += 1;
    sql.push_str(&format!(" LIMIT ${placeholder}"));

    let mut query = sqlx::query_as::<_, MessageRow>(&sql).bind(workspace_name);
    if let Some(session) = session_name {
        query = query.bind(session);
    } else if let Some(names) = allowed_sessions.as_ref() {
        query = query.bind(names);
    }
    if let Some(after) = after {
        query = query.bind(after);
    }
    if let Some(before) = before {
        query = query.bind(before);
    }
    query = query.bind(limit);

    let rows = query.fetch_all(pool).await?;
    Ok(rows.into_iter().map(message_json).collect())
}

/// A message row carrying `seq_in_session`, needed by the snippet-merging logic
/// behind `grep_messages` / `search_messages` (the date-range/list paths don't
/// need the sequence number, so the leaner [`MessageRow`] is used there).
#[derive(Debug, FromRow)]
struct SnippetMessageRow {
    public_id: String,
    content: String,
    peer_name: String,
    session_name: String,
    metadata: Value,
    created_at: DateTime<Utc>,
    workspace_name: String,
    token_count: i32,
    seq_in_session: i64,
}

fn snippet_message_json(row: &SnippetMessageRow) -> Value {
    json!({
        "id": row.public_id,
        "content": row.content,
        "peer_id": row.peer_name,
        "session_id": row.session_name,
        "metadata": row.metadata,
        "created_at": row.created_at,
        "workspace_id": row.workspace_name,
        "token_count": row.token_count
    })
}

/// One conversation snippet: the matched messages merged into this range, plus
/// the full context window around them (which includes the matched messages),
/// ordered chronologically by `seq_in_session`. Ports the
/// `(matched, context)` tuple the dialectic `grep_messages` / `search_messages`
/// tools return. Messages are serialized in the public message JSON shape.
#[derive(Debug, Clone, PartialEq)]
pub struct MessageSnippet {
    pub matched: Vec<Value>,
    pub context: Vec<Value>,
}

/// Port of `crud.grep_messages` (the dialectic `grep_messages` tool's data
/// layer): case-insensitive substring match (`ILIKE %text%` with `\` escape,
/// newest-first, limited), optionally scoped to a session — or, when `observer`
/// is given without a session, to that peer's sessions (no memberships → empty)
/// — then grouped into merged context-window snippets via
/// [`build_merged_snippets`].
#[allow(clippy::too_many_arguments)]
pub async fn grep_messages(
    pool: &PgPool,
    workspace_name: &str,
    session_name: Option<&str>,
    text: &str,
    limit: i64,
    context_window: i64,
    observer: Option<&str>,
) -> Result<Vec<MessageSnippet>, sqlx::Error> {
    let allowed_sessions: Option<Vec<String>> = match (observer, session_name) {
        (Some(observer), None) => {
            let names = fetch_peer_session_names(pool, workspace_name, observer).await?;
            if names.is_empty() {
                return Ok(Vec::new());
            }
            Some(names)
        }
        _ => None,
    };

    let pattern = format!("%{}%", crate::search::escape_ilike_pattern(text));
    let mut sql = String::from(
        "SELECT public_id, content, peer_name, session_name, metadata, created_at, \
         workspace_name, token_count, seq_in_session FROM messages \
         WHERE workspace_name = $1 AND content ILIKE $2 ESCAPE '\\'",
    );
    let mut placeholder = 2;
    if session_name.is_some() {
        placeholder += 1;
        sql.push_str(&format!(" AND session_name = ${placeholder}"));
    } else if allowed_sessions.is_some() {
        placeholder += 1;
        sql.push_str(&format!(" AND session_name = ANY(${placeholder})"));
    }
    placeholder += 1;
    sql.push_str(&format!(" ORDER BY created_at DESC LIMIT ${placeholder}"));

    let mut query = sqlx::query_as::<_, SnippetMessageRow>(&sql)
        .bind(workspace_name)
        .bind(pattern);
    if let Some(session) = session_name {
        query = query.bind(session);
    } else if let Some(names) = allowed_sessions.as_ref() {
        query = query.bind(names);
    }
    query = query.bind(limit);
    let matched = query.fetch_all(pool).await?;

    build_merged_snippets(pool, workspace_name, matched, context_window).await
}

/// Port of `_build_merged_snippets`: group matches by session (preserving the
/// first-seen session order, like Python's insertion-ordered dict), sort each
/// session's matches by `seq_in_session`, merge overlapping/adjacent
/// context windows (`start <= prev_end + 1`), then fetch the full message range
/// for each merged window. Returns one snippet per merged range.
async fn build_merged_snippets(
    pool: &PgPool,
    workspace_name: &str,
    matched_messages: Vec<SnippetMessageRow>,
    context_window: i64,
) -> Result<Vec<MessageSnippet>, sqlx::Error> {
    if matched_messages.is_empty() {
        return Ok(Vec::new());
    }

    // Group by session, preserving first-seen order (Python dict semantics).
    let mut session_matches: Vec<(String, Vec<SnippetMessageRow>)> = Vec::new();
    for msg in matched_messages {
        match session_matches
            .iter_mut()
            .find(|(name, _)| name == &msg.session_name)
        {
            Some((_, matches)) => matches.push(msg),
            None => session_matches.push((msg.session_name.clone(), vec![msg])),
        }
    }

    let mut snippets: Vec<MessageSnippet> = Vec::new();
    for (sess_name, mut matches) in session_matches {
        // Stable sort by sequence number (Python list.sort is stable).
        matches.sort_by_key(|m| m.seq_in_session);

        // Merge ranges: (start, end, matched-rows-in-range).
        let mut merged: Vec<(i64, i64, Vec<SnippetMessageRow>)> = Vec::new();
        for m in matches {
            let start = m.seq_in_session - context_window;
            let end = m.seq_in_session + context_window;
            if let Some(last) = merged.last_mut()
                && start <= last.1 + 1
            {
                last.1 = last.1.max(end);
                last.2.push(m);
                continue;
            }
            merged.push((start, end, vec![m]));
        }

        for (start, end, range_matches) in merged {
            let context_rows: Vec<SnippetMessageRow> = sqlx::query_as(
                "SELECT public_id, content, peer_name, session_name, metadata, created_at, \
                 workspace_name, token_count, seq_in_session FROM messages \
                 WHERE workspace_name = $1 AND session_name = $2 \
                 AND seq_in_session BETWEEN $3 AND $4 \
                 ORDER BY seq_in_session ASC",
            )
            .bind(workspace_name)
            .bind(&sess_name)
            .bind(start)
            .bind(end)
            .fetch_all(pool)
            .await?;

            snippets.push(MessageSnippet {
                matched: range_matches.iter().map(snippet_message_json).collect(),
                context: context_rows.iter().map(snippet_message_json).collect(),
            });
        }
    }

    Ok(snippets)
}

/// Port of `_search_messages_pgvector` + `_deduplicate_messages` (the data layer
/// behind the dialectic `search_messages` / `search_messages_temporal` tools).
/// Cosine-distance (`<=>`) match over `message_embeddings`, oversampling
/// `limit * 2` (one message may have several chunks), deduped by `public_id` to
/// `limit` preserving rank order, then grouped into merged context-window
/// snippets. The workspace and session scope live on the embedding side
/// (mirroring Python's `MessageEmbedding.*` predicates); the inclusive
/// `[after, before]` `created_at` window applies to the messages. When
/// `observer` is given without a session, scopes to that peer's sessions (no
/// memberships → empty). Takes a pre-computed `query_embedding`, so the call is
/// DB-only (the embed leg is the caller's concern, like `hybrid_search`).
#[allow(clippy::too_many_arguments)]
pub async fn search_messages_semantic(
    pool: &PgPool,
    workspace_name: &str,
    session_name: Option<&str>,
    observer: Option<&str>,
    query_embedding: &[f32],
    after: Option<DateTime<Utc>>,
    before: Option<DateTime<Utc>>,
    limit: i64,
    context_window: i64,
) -> Result<Vec<MessageSnippet>, sqlx::Error> {
    let allowed_sessions: Option<Vec<String>> = match (observer, session_name) {
        (Some(observer), None) => {
            let names = fetch_peer_session_names(pool, workspace_name, observer).await?;
            if names.is_empty() {
                return Ok(Vec::new());
            }
            Some(names)
        }
        _ => None,
    };

    let mut sql = String::from(
        "SELECT m.public_id, m.content, m.peer_name, m.session_name, m.metadata, \
         m.created_at, m.workspace_name, m.token_count, m.seq_in_session \
         FROM messages m \
         JOIN message_embeddings me ON m.public_id = me.message_id \
         WHERE me.embedding IS NOT NULL AND me.workspace_name = $1",
    );
    let mut placeholder = 1;
    if session_name.is_some() {
        placeholder += 1;
        sql.push_str(&format!(" AND me.session_name = ${placeholder}"));
    } else if allowed_sessions.is_some() {
        placeholder += 1;
        sql.push_str(&format!(" AND me.session_name = ANY(${placeholder})"));
    }
    if after.is_some() {
        placeholder += 1;
        sql.push_str(&format!(" AND m.created_at >= ${placeholder}"));
    }
    if before.is_some() {
        placeholder += 1;
        sql.push_str(&format!(" AND m.created_at <= ${placeholder}"));
    }
    let vec_idx = placeholder + 1;
    let limit_idx = placeholder + 2;
    sql.push_str(&format!(
        " ORDER BY me.embedding <=> ${vec_idx}::vector LIMIT ${limit_idx}"
    ));

    let mut query = sqlx::query_as::<_, SnippetMessageRow>(&sql).bind(workspace_name);
    if let Some(session) = session_name {
        query = query.bind(session);
    } else if let Some(names) = allowed_sessions.as_ref() {
        query = query.bind(names);
    }
    if let Some(after) = after {
        query = query.bind(after);
    }
    if let Some(before) = before {
        query = query.bind(before);
    }
    query = query
        .bind(crate::search::vector_literal(query_embedding))
        .bind(limit * 2);
    let rows = query.fetch_all(pool).await?;

    // _deduplicate_messages: dedup by public_id preserving order, cap at limit.
    let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut matched: Vec<SnippetMessageRow> = Vec::new();
    for row in rows {
        if seen.insert(row.public_id.clone()) {
            matched.push(row);
            if matched.len() as i64 >= limit {
                break;
            }
        }
    }

    build_merged_snippets(pool, workspace_name, matched, context_window).await
}

/// Port of `agent_tools.get_observation_context` (the dialectic data layer that
/// expands an observation's `message_ids` into surrounding conversation
/// context). Collects the target messages' `seq_in_session` values, then returns
/// every message whose sequence is within ±1 of any target, ordered ascending.
/// Optional session scope, or observer-session scoping when no session is given
/// (no memberships → empty). Empty `message_ids` → empty, no query.
///
/// Faithful quirk: the ±1 match is purely arithmetic on `seq_in_session` and is
/// *not* correlated to the target's session, so under observer/allowed-session
/// scope (multiple sessions) a candidate can match a target sequence from a
/// different session — exactly as the Python EXISTS-over-CTE does.
pub async fn get_observation_context(
    pool: &PgPool,
    workspace_name: &str,
    session_name: Option<&str>,
    message_ids: &[String],
    observer: Option<&str>,
) -> Result<Vec<Value>, sqlx::Error> {
    if message_ids.is_empty() {
        return Ok(Vec::new());
    }

    let allowed_sessions: Option<Vec<String>> = match (observer, session_name) {
        (Some(observer), None) => {
            let names = fetch_peer_session_names(pool, workspace_name, observer).await?;
            if names.is_empty() {
                return Ok(Vec::new());
            }
            Some(names)
        }
        _ => None,
    };

    // $1 = workspace (referenced in the CTE and the outer query), $2 = ids,
    // $3 = session scope when present (referenced in both clauses). Reusing one
    // placeholder across both keeps the bind list to three.
    let (cte_session_clause, outer_session_clause) = if session_name.is_some() {
        (" AND session_name = $3", " AND m.session_name = $3")
    } else if allowed_sessions.is_some() {
        (
            " AND session_name = ANY($3)",
            " AND m.session_name = ANY($3)",
        )
    } else {
        ("", "")
    };

    let sql = format!(
        "WITH target_seqs AS (\
           SELECT seq_in_session FROM messages \
           WHERE workspace_name = $1 AND public_id = ANY($2){cte_session_clause}\
         ) \
         SELECT m.public_id, m.content, m.peer_name, m.session_name, m.metadata, \
         m.created_at, m.workspace_name, m.token_count, m.seq_in_session \
         FROM messages m \
         WHERE m.workspace_name = $1 \
         AND EXISTS (SELECT 1 FROM target_seqs t \
                     WHERE (t.seq_in_session - m.seq_in_session) BETWEEN -1 AND 1)\
         {outer_session_clause} \
         ORDER BY m.seq_in_session ASC"
    );

    let mut query = sqlx::query_as::<_, SnippetMessageRow>(&sql)
        .bind(workspace_name)
        .bind(message_ids);
    if let Some(session) = session_name {
        query = query.bind(session);
    } else if let Some(names) = allowed_sessions.as_ref() {
        query = query.bind(names);
    }
    let rows = query.fetch_all(pool).await?;
    Ok(rows.iter().map(snippet_message_json).collect())
}

/// A reasoning-tree document as consumed by the dialectic `get_reasoning_chain`
/// tool: just the fields the chain traversal needs (`id`, `content`, `level`,
/// and the premise/source `source_ids`). `level` is `NOT NULL` in the schema
/// (defaults to `'explicit'`); `source_ids` is a nullable JSONB array, flattened
/// to a `Vec<String>` (absent/empty → empty).
#[derive(Debug, Clone, PartialEq)]
pub struct ReasoningDocument {
    pub id: String,
    pub content: String,
    pub level: String,
    pub source_ids: Vec<String>,
}

#[derive(Debug, FromRow)]
struct ReasoningDocumentRow {
    id: String,
    content: String,
    level: String,
    source_ids: Option<Value>,
}

fn reasoning_document(row: ReasoningDocumentRow) -> ReasoningDocument {
    let source_ids = row
        .source_ids
        .as_ref()
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item.as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    ReasoningDocument {
        id: row.id,
        content: row.content,
        level: row.level,
        source_ids,
    }
}

/// Port of `crud.get_documents_by_ids`: live documents (`deleted_at IS NULL`) in
/// the workspace matching any of `document_ids`. May return fewer than requested
/// (missing/deleted ids are dropped). Empty input → empty, no query. No
/// ordering, matching Python's plain `IN` query.
pub async fn get_documents_by_ids(
    pool: &PgPool,
    workspace_name: &str,
    document_ids: &[String],
) -> Result<Vec<ReasoningDocument>, sqlx::Error> {
    if document_ids.is_empty() {
        return Ok(Vec::new());
    }
    let rows: Vec<ReasoningDocumentRow> = sqlx::query_as(
        "SELECT id, content, level, source_ids FROM documents \
         WHERE workspace_name = $1 AND id = ANY($2) AND deleted_at IS NULL",
    )
    .bind(workspace_name)
    .bind(document_ids)
    .fetch_all(pool)
    .await?;
    Ok(rows.into_iter().map(reasoning_document).collect())
}

/// Port of `crud.get_child_observations`: live documents whose `source_ids`
/// JSONB array contains `parent_id` (`@>`, GIN-indexed), i.e. observations
/// derived from this one (reasoning-tree traversal downward). Optionally
/// filtered by observer/observed.
pub async fn get_child_observations(
    pool: &PgPool,
    workspace_name: &str,
    parent_id: &str,
    observer: Option<&str>,
    observed: Option<&str>,
) -> Result<Vec<ReasoningDocument>, sqlx::Error> {
    let mut sql = String::from(
        "SELECT id, content, level, source_ids FROM documents \
         WHERE workspace_name = $1 AND source_ids @> $2::jsonb AND deleted_at IS NULL",
    );
    let mut placeholder = 2;
    if observer.is_some() {
        placeholder += 1;
        sql.push_str(&format!(" AND observer = ${placeholder}"));
    }
    if observed.is_some() {
        placeholder += 1;
        sql.push_str(&format!(" AND observed = ${placeholder}"));
    }

    let mut query = sqlx::query_as::<_, ReasoningDocumentRow>(&sql)
        .bind(workspace_name)
        .bind(json!([parent_id]));
    if let Some(observer) = observer {
        query = query.bind(observer);
    }
    if let Some(observed) = observed {
        query = query.bind(observed);
    }
    let rows = query.fetch_all(pool).await?;
    Ok(rows.into_iter().map(reasoning_document).collect())
}

/// A validated conclusion to create, plus its pre-computed embedding (computed
/// in the handler, outside any DB session).
#[derive(Debug, Clone)]
pub struct NewConclusion {
    pub content: String,
    pub observer_id: String,
    pub observed_id: String,
    pub session_id: Option<String>,
}

/// Failure modes of `prepare_conclusions`, mapped to Python's statuses.
#[derive(Debug)]
pub enum ConclusionWriteError {
    /// `get_session` raised — 404 `Session {name} not found in workspace {ws}`.
    SessionNotFound(String),
    /// `get_peer` raised — 404 `Peer {name} not found in workspace {ws}`.
    PeerNotFound(String),
    Database(sqlx::Error),
}

impl From<sqlx::Error> for ConclusionWriteError {
    fn from(error: sqlx::Error) -> Self {
        ConclusionWriteError::Database(error)
    }
}

/// Port of the pre-embedding half of `create_observations`: validate every
/// referenced session and peer exists (else 404), then get-or-create the
/// `(observer, observed)` collections. Runs before the embedding call so error
/// precedence matches Python (a missing peer 404s before any embed 422).
pub async fn prepare_conclusions(
    pool: &PgPool,
    workspace_name: &str,
    conclusions: &[NewConclusion],
) -> Result<(), ConclusionWriteError> {
    // Distinct sessions / peers / collection pairs, first-seen order.
    let mut sessions: Vec<&str> = Vec::new();
    let mut peers: Vec<&str> = Vec::new();
    let mut pairs: Vec<(&str, &str)> = Vec::new();
    for conclusion in conclusions {
        if let Some(session) = conclusion.session_id.as_deref()
            && !sessions.contains(&session)
        {
            sessions.push(session);
        }
        for peer in [conclusion.observer_id.as_str(), conclusion.observed_id.as_str()] {
            if !peers.contains(&peer) {
                peers.push(peer);
            }
        }
        let pair = (conclusion.observer_id.as_str(), conclusion.observed_id.as_str());
        if !pairs.contains(&pair) {
            pairs.push(pair);
        }
    }

    for session in sessions {
        let exists: bool = sqlx::query_scalar(
            "SELECT EXISTS(SELECT 1 FROM sessions WHERE workspace_name = $1 AND name = $2)",
        )
        .bind(workspace_name)
        .bind(session)
        .fetch_one(pool)
        .await?;
        if !exists {
            return Err(ConclusionWriteError::SessionNotFound(session.to_string()));
        }
    }

    for peer in peers {
        let exists: bool = sqlx::query_scalar(
            "SELECT EXISTS(SELECT 1 FROM peers WHERE workspace_name = $1 AND name = $2)",
        )
        .bind(workspace_name)
        .bind(peer)
        .fetch_one(pool)
        .await?;
        if !exists {
            return Err(ConclusionWriteError::PeerNotFound(peer.to_string()));
        }
    }

    for (observer, observed) in pairs {
        sqlx::query(
            "INSERT INTO collections (id, workspace_name, observer, observed) \
             VALUES ($1, $2, $3, $4) \
             ON CONFLICT (observer, observed, workspace_name) DO NOTHING",
        )
        .bind(generate_nanoid())
        .bind(workspace_name)
        .bind(observer)
        .bind(observed)
        .execute(pool)
        .await?;
    }

    Ok(())
}

/// Port of the insert half of `create_observations`: insert one `explicit`
/// document per conclusion (with its pre-computed embedding, `sync_state` =
/// `synced` since pgvector stores the vector inline), returning the created rows
/// as conclusion JSON. `embeddings` is parallel to `conclusions`.
pub async fn insert_conclusions(
    pool: &PgPool,
    workspace_name: &str,
    conclusions: &[NewConclusion],
    embeddings: &[Vec<f32>],
) -> Result<Vec<Value>, sqlx::Error> {
    let mut created = Vec::with_capacity(conclusions.len());
    let mut transaction = pool.begin().await?;
    for (conclusion, embedding) in conclusions.iter().zip(embeddings.iter()) {
        let row = sqlx::query_as::<_, ConclusionRow>(
            "INSERT INTO documents \
             (id, workspace_name, observer, observed, content, level, times_derived, \
              internal_metadata, session_name, embedding, sync_state) \
             VALUES ($1, $2, $3, $4, $5, 'explicit', 1, '{}'::jsonb, $6, $7::vector, 'synced') \
             RETURNING id, content, observer, observed, session_name, created_at",
        )
        .bind(generate_nanoid())
        .bind(workspace_name)
        .bind(&conclusion.observer_id)
        .bind(&conclusion.observed_id)
        .bind(&conclusion.content)
        .bind(&conclusion.session_id)
        .bind(crate::search::vector_literal(embedding))
        .fetch_one(&mut *transaction)
        .await?;
        created.push(conclusion_json(row));
    }
    transaction.commit().await?;
    Ok(created)
}

/// Most-recent conclusions for `(observer, observed)`, porting
/// `crud.query_documents_recent`: live documents ordered by `created_at` desc,
/// optionally narrowed to a session, limited. Returns conclusion JSON.
pub async fn query_documents_recent(
    pool: &PgPool,
    workspace_name: &str,
    observer: &str,
    observed: &str,
    session_name: Option<&str>,
    limit: i64,
) -> Result<Vec<Value>, sqlx::Error> {
    let session_clause = if session_name.is_some() {
        " AND session_name = $4"
    } else {
        ""
    };
    let limit_idx = if session_name.is_some() { 5 } else { 4 };
    let sql = format!(
        "SELECT id, content, observer, observed, session_name, created_at \
         FROM documents \
         WHERE workspace_name = $1 AND observer = $2 AND observed = $3 \
           AND deleted_at IS NULL{session_clause} \
         ORDER BY created_at DESC \
         LIMIT ${limit_idx}"
    );
    let mut query = sqlx::query_as::<_, ConclusionRow>(&sql)
        .bind(workspace_name)
        .bind(observer)
        .bind(observed);
    if let Some(session) = session_name {
        query = query.bind(session);
    }
    query = query.bind(limit);
    Ok(query
        .fetch_all(pool)
        .await?
        .into_iter()
        .map(conclusion_json)
        .collect())
}

/// Like [`query_documents_recent`] but returns the full
/// [`crate::representation::Document`] shape (level/source_ids/internal_metadata)
/// so callers can rebuild a `Representation`. The dreamer's
/// `get_recent_observations` tool needs this rather than the conclusion-JSON
/// projection.
pub async fn query_documents_recent_full(
    pool: &PgPool,
    workspace_name: &str,
    observer: &str,
    observed: &str,
    session_name: Option<&str>,
    limit: i64,
) -> Result<Vec<crate::representation::Document>, sqlx::Error> {
    let session_clause = if session_name.is_some() {
        " AND session_name = $4"
    } else {
        ""
    };
    let limit_idx = if session_name.is_some() { 5 } else { 4 };
    let sql = format!(
        "SELECT id, content, level, created_at, session_name, source_ids, internal_metadata \
         FROM documents \
         WHERE workspace_name = $1 AND observer = $2 AND observed = $3 \
           AND deleted_at IS NULL{session_clause} \
         ORDER BY created_at DESC \
         LIMIT ${limit_idx}"
    );
    let mut query = sqlx::query_as::<_, DocumentRow>(&sql)
        .bind(workspace_name)
        .bind(observer)
        .bind(observed);
    if let Some(session) = session_name {
        query = query.bind(session);
    }
    query = query.bind(limit);
    Ok(query
        .fetch_all(pool)
        .await?
        .into_iter()
        .map(document_from_row)
        .collect())
}

/// Most-reinforced conclusions for `(observer, observed)`, porting
/// `crud.query_documents_most_derived`: live documents ordered by
/// `times_derived` desc, then `created_at` desc, then `id` (stable tiebreak),
/// limited. Returns conclusion JSON.
pub async fn query_documents_most_derived(
    pool: &PgPool,
    workspace_name: &str,
    observer: &str,
    observed: &str,
    limit: i64,
) -> Result<Vec<Value>, sqlx::Error> {
    let rows = sqlx::query_as::<_, ConclusionRow>(
        "SELECT id, content, observer, observed, session_name, created_at \
         FROM documents \
         WHERE workspace_name = $1 AND observer = $2 AND observed = $3 \
           AND deleted_at IS NULL \
         ORDER BY times_derived DESC, created_at DESC, id \
         LIMIT $4",
    )
    .bind(workspace_name)
    .bind(observer)
    .bind(observed)
    .bind(limit)
    .fetch_all(pool)
    .await?;
    Ok(rows.into_iter().map(conclusion_json).collect())
}

/// Most-reinforced conclusions for `(observer, observed)` as full
/// [`representation::Document`]s (the [`Representation::from_documents`] shape),
/// porting `crud._query_documents_most_derived`: live documents ordered by
/// `times_derived` desc, then `created_at` desc, then `id`, limited. The
/// `Value`-returning [`query_documents_most_derived`] backs the conclusions list
/// endpoint; this variant feeds the working representation.
pub async fn query_documents_most_derived_full(
    pool: &PgPool,
    workspace_name: &str,
    observer: &str,
    observed: &str,
    limit: i64,
) -> Result<Vec<crate::representation::Document>, sqlx::Error> {
    let rows = sqlx::query_as::<_, DocumentRow>(
        "SELECT id, content, level, created_at, session_name, source_ids, internal_metadata \
         FROM documents \
         WHERE workspace_name = $1 AND observer = $2 AND observed = $3 \
           AND deleted_at IS NULL \
         ORDER BY times_derived DESC, created_at DESC, id \
         LIMIT $4",
    )
    .bind(workspace_name)
    .bind(observer)
    .bind(observed)
    .bind(limit)
    .fetch_all(pool)
    .await?;
    Ok(rows.into_iter().map(document_from_row).collect())
}

/// Assemble a working representation for `(observer, observed)`, porting
/// `RepresentationManager._get_working_representation_internal`: split the
/// `max_observations` budget across semantic, most-derived, and recent
/// conclusions, fetch each source, and merge (dedup + sort by `created_at`).
///
/// `embedding` is the query vector pre-computed by the caller (the route embeds
/// outside the DB session). When a semantic query was requested but no embedding
/// is available (the embed failed), the semantic slots stay reserved-but-unfilled
/// — the same outcome Python reaches when its fallback embed fails — so the recent
/// allocation is unchanged.
#[allow(clippy::too_many_arguments)]
pub async fn query_working_representation(
    pool: &PgPool,
    workspace_name: &str,
    observer: &str,
    observed: &str,
    session_name: Option<&str>,
    embedding: Option<&[f32]>,
    include_semantic_query: bool,
    semantic_search_top_k: Option<i64>,
    semantic_search_max_distance: Option<f64>,
    include_most_derived: bool,
    max_observations: i64,
) -> Result<crate::representation::Representation, sqlx::Error> {
    use crate::representation::Representation;

    let total = max_observations.max(0);

    let semantic = if include_semantic_query {
        semantic_search_top_k.unwrap_or(total / 3).max(0).min(total)
    } else {
        0
    };
    let top = if include_semantic_query && include_most_derived {
        (total / 3).max(0).min(total - semantic)
    } else if include_most_derived {
        (total / 2).max(0).min(total - semantic)
    } else {
        0
    };
    let recent = total - semantic - top;

    let mut representation = Representation::default();

    if include_semantic_query
        && let Some(embedding) = embedding
    {
        let filter = FilterClause {
            sql: String::new(),
            bindings: Vec::new(),
        };
        let docs = query_documents_full(
            pool,
            workspace_name,
            observer,
            observed,
            embedding,
            &filter,
            semantic_search_max_distance,
            semantic,
        )
        .await?;
        representation.merge(Representation::from_documents(&docs));
    }

    if include_most_derived {
        let docs =
            query_documents_most_derived_full(pool, workspace_name, observer, observed, top).await?;
        representation.merge(Representation::from_documents(&docs));
    }

    let recent_docs = query_documents_recent_full(
        pool,
        workspace_name,
        observer,
        observed,
        session_name,
        recent,
    )
    .await?;
    representation.merge(Representation::from_documents(&recent_docs));

    Ok(representation)
}

/// Semantic search over conclusions (documents) by `(observer, observed)`,
/// porting `crud._query_documents_pgvector`. Orders by pgvector cosine distance
/// to the pre-computed `embedding`, optionally bounded by `max_distance`, with
/// the request's `filter` (a `FilterTarget::Conclusion` clause) threaded in.
/// Returns conclusion JSON in the same shape as the list endpoint.
#[allow(clippy::too_many_arguments)]
pub async fn query_documents_pgvector(
    pool: &PgPool,
    workspace_name: &str,
    observer: &str,
    observed: &str,
    embedding: &[f32],
    filter: &FilterClause,
    max_distance: Option<f64>,
    top_k: i64,
) -> Result<Vec<Value>, sqlx::Error> {
    // filter.sql hardcodes $1..$k for its own bindings, so those bind first and
    // the fixed params follow at $(k+1)+.
    let base = filter.bindings.len();
    let ws_idx = base + 1;
    let observer_idx = base + 2;
    let observed_idx = base + 3;
    let vec_idx = base + 4;
    let (distance_clause, limit_idx) = match max_distance {
        Some(_) => (
            format!(" AND embedding <=> ${vec_idx}::vector <= ${}", base + 5),
            base + 6,
        ),
        None => (String::new(), base + 5),
    };

    let sql = format!(
        "SELECT id, content, observer, observed, session_name, created_at \
         FROM documents \
         WHERE workspace_name = ${ws_idx} AND observer = ${observer_idx} \
           AND observed = ${observed_idx} \
           AND embedding IS NOT NULL AND deleted_at IS NULL{distance_clause}{} \
         ORDER BY embedding <=> ${vec_idx}::vector \
         LIMIT ${limit_idx}",
        filter.sql
    );

    let mut query = bind_values_as(sqlx::query_as::<_, ConclusionRow>(&sql), &filter.bindings);
    query = query
        .bind(workspace_name)
        .bind(observer)
        .bind(observed)
        .bind(crate::search::vector_literal(embedding));
    if let Some(distance) = max_distance {
        query = query.bind(distance);
    }
    query = query.bind(top_k);

    let items = query
        .fetch_all(pool)
        .await?
        .into_iter()
        .map(conclusion_json)
        .collect();
    Ok(items)
}

/// Semantic search returning full document rows for
/// [`crate::representation::Representation::from_documents`], porting
/// `crud._query_documents_pgvector`. Identical ordering, filter, and
/// `max_distance` handling to [`query_documents_pgvector`], but selects the
/// columns `from_documents` reads (`level`, `source_ids`, `internal_metadata`)
/// instead of the trimmed conclusion JSON. This is the data layer the dialectic
/// `search_memory` tool (and the prefetch path) builds its `Representation` from.
#[allow(clippy::too_many_arguments)]
pub async fn query_documents_full(
    pool: &PgPool,
    workspace_name: &str,
    observer: &str,
    observed: &str,
    embedding: &[f32],
    filter: &FilterClause,
    max_distance: Option<f64>,
    top_k: i64,
) -> Result<Vec<crate::representation::Document>, sqlx::Error> {
    // filter.sql hardcodes $1..$k for its own bindings, so those bind first and
    // the fixed params follow at $(k+1)+.
    let base = filter.bindings.len();
    let ws_idx = base + 1;
    let observer_idx = base + 2;
    let observed_idx = base + 3;
    let vec_idx = base + 4;
    let (distance_clause, limit_idx) = match max_distance {
        Some(_) => (
            format!(" AND embedding <=> ${vec_idx}::vector <= ${}", base + 5),
            base + 6,
        ),
        None => (String::new(), base + 5),
    };

    let sql = format!(
        "SELECT id, content, level, created_at, session_name, source_ids, internal_metadata \
         FROM documents \
         WHERE workspace_name = ${ws_idx} AND observer = ${observer_idx} \
           AND observed = ${observed_idx} \
           AND embedding IS NOT NULL AND deleted_at IS NULL{distance_clause}{} \
         ORDER BY embedding <=> ${vec_idx}::vector \
         LIMIT ${limit_idx}",
        filter.sql
    );

    let mut query = bind_values_as(sqlx::query_as::<_, DocumentRow>(&sql), &filter.bindings);
    query = query
        .bind(workspace_name)
        .bind(observer)
        .bind(observed)
        .bind(crate::search::vector_literal(embedding));
    if let Some(distance) = max_distance {
        query = query.bind(distance);
    }
    query = query.bind(top_k);

    let documents = query
        .fetch_all(pool)
        .await?
        .into_iter()
        .map(document_from_row)
        .collect();
    Ok(documents)
}

/// Semantic document search restricted to the given reasoning `levels`, for the
/// dialectic prefetch path (Python `search_memory` with
/// `filters={"level": {"in": levels}}`). The Conclusion `FilterTarget` does not
/// map the `level` column, so this binds a Postgres `text[]` for the `ANY` test
/// directly rather than going through the generic filter builder. Returns the
/// full document shape ordered by pgvector cosine distance.
pub async fn query_documents_by_levels(
    pool: &PgPool,
    workspace_name: &str,
    observer: &str,
    observed: &str,
    embedding: &[f32],
    levels: &[String],
    top_k: i64,
) -> Result<Vec<crate::representation::Document>, sqlx::Error> {
    let documents = sqlx::query_as::<_, DocumentRow>(
        "SELECT id, content, level, created_at, session_name, source_ids, internal_metadata \
         FROM documents \
         WHERE workspace_name = $1 AND observer = $2 AND observed = $3 \
           AND embedding IS NOT NULL AND deleted_at IS NULL \
           AND level = ANY($4) \
         ORDER BY embedding <=> $5::vector \
         LIMIT $6",
    )
    .bind(workspace_name)
    .bind(observer)
    .bind(observed)
    .bind(levels)
    .bind(crate::search::vector_literal(embedding))
    .bind(top_k)
    .fetch_all(pool)
    .await?
    .into_iter()
    .map(document_from_row)
    .collect();
    Ok(documents)
}

/// The most-recent session messages whose running token sum (newest-first) stays
/// within `token_limit`, returned chronologically (ascending id). Ports the
/// `token_limit` + `reverse=False` path of `crud.get_messages` (via its
/// `_apply_token_limit` window function), which the dialectic uses to seed
/// session history into the system prompt. Returns message JSON.
pub async fn get_session_messages_within_token_limit(
    pool: &PgPool,
    workspace_name: &str,
    session_name: &str,
    token_limit: i64,
) -> Result<Vec<Value>, sqlx::Error> {
    let rows = sqlx::query_as::<_, MessageRow>(
        "SELECT m.public_id, m.content, m.peer_name, m.session_name, m.metadata, \
                m.created_at, m.workspace_name, m.token_count \
         FROM messages m \
         JOIN ( \
             SELECT id, SUM(token_count) OVER (ORDER BY id DESC) AS running_token_sum \
             FROM messages \
             WHERE workspace_name = $1 AND session_name = $2 \
         ) t ON m.id = t.id \
         WHERE t.running_token_sum <= $3 \
         ORDER BY m.id ASC",
    )
    .bind(workspace_name)
    .bind(session_name)
    .bind(token_limit)
    .fetch_all(pool)
    .await?;
    Ok(rows.into_iter().map(message_json).collect())
}

pub async fn get_message(
    pool: &PgPool,
    workspace_name: &str,
    session_name: &str,
    message_id: &str,
) -> Result<Option<Value>, sqlx::Error> {
    let row = sqlx::query_as::<_, MessageRow>(
        "SELECT public_id, content, peer_name, session_name, metadata, created_at, \
         workspace_name, token_count \
         FROM messages \
         WHERE workspace_name = $1 AND session_name = $2 AND public_id = $3",
    )
    .bind(workspace_name)
    .bind(session_name)
    .bind(message_id)
    .fetch_optional(pool)
    .await?;

    Ok(row.map(message_json))
}

/// Soft-delete a conclusion (document) by id, porting
/// `crud.delete_document_by_id`. Sets `deleted_at`; the reconciler handles
/// vector-store cleanup and hard deletion. Returns `false` when no live document
/// with that id exists in the workspace.
pub async fn delete_conclusion(
    pool: &PgPool,
    workspace_name: &str,
    conclusion_id: &str,
) -> Result<bool, sqlx::Error> {
    let result = sqlx::query(
        "UPDATE documents SET deleted_at = now() \
         WHERE id = $1 AND workspace_name = $2 AND deleted_at IS NULL",
    )
    .bind(conclusion_id)
    .bind(workspace_name)
    .execute(pool)
    .await?;
    Ok(result.rows_affected() > 0)
}

/// Overwrite a message's metadata, porting `crud.update_message`. When
/// `metadata` is `None` the row is returned unchanged (Python skips the
/// assignment but still commits and returns the message). Returns `None` when no
/// such message exists in the session.
pub async fn update_message(
    pool: &PgPool,
    workspace_name: &str,
    session_name: &str,
    message_id: &str,
    metadata: Option<Value>,
) -> Result<Option<Value>, sqlx::Error> {
    let row = match metadata {
        Some(metadata) => {
            sqlx::query_as::<_, MessageRow>(
                "UPDATE messages SET metadata = $4 \
                 WHERE workspace_name = $1 AND session_name = $2 AND public_id = $3 \
                 RETURNING public_id, content, peer_name, session_name, metadata, \
                           created_at, workspace_name, token_count",
            )
            .bind(workspace_name)
            .bind(session_name)
            .bind(message_id)
            .bind(metadata)
            .fetch_optional(pool)
            .await?
        }
        None => {
            sqlx::query_as::<_, MessageRow>(
                "SELECT public_id, content, peer_name, session_name, metadata, created_at, \
                 workspace_name, token_count \
                 FROM messages \
                 WHERE workspace_name = $1 AND session_name = $2 AND public_id = $3",
            )
            .bind(workspace_name)
            .bind(session_name)
            .bind(message_id)
            .fetch_optional(pool)
            .await?
        }
    };

    Ok(row.map(message_json))
}

/// One message to insert, after request validation. `metadata` is the sanitized
/// object (`{}` when absent), `created_at` is the optional caller-supplied
/// timestamp, and `token_count` is the precomputed o200k_base length.
#[derive(Debug, Clone)]
pub struct MessageInsert {
    pub peer_name: String,
    pub content: String,
    pub metadata: Value,
    pub created_at: Option<DateTime<Utc>>,
    pub token_count: i32,
}

/// Bulk-create messages for a session, porting `crud.create_messages`.
///
/// Mirrors the Python ordering: ensure the session exists with its sender peers
/// joined, take a transaction-scoped advisory lock keyed on
/// `(workspace, session)`, read the current max `seq_in_session`, then insert
/// each message with a monotonically increasing sequence number. The advisory
/// lock serializes concurrent writers so sequence numbers stay gap-free.
///
/// When `embed_messages` is set, one pending `MessageEmbedding` row is inserted
/// per content chunk (within the same transaction), mirroring Python's
/// `EMBED_MESSAGES` block — only the chunk text and `sync_state='pending'` are
/// written; the reconciler generates vectors later.
pub async fn create_messages(
    pool: &PgPool,
    workspace_name: &str,
    session_name: &str,
    messages: &[MessageInsert],
    embed_messages: bool,
    embedding_max_tokens: usize,
) -> Result<Vec<CreatedMessage>, sqlx::Error> {
    // Ensure the session exists and every sender is joined with default config,
    // matching Python's `get_or_create_session(peers={sender: SessionPeerConfig()})`.
    get_or_create_session(pool, workspace_name, session_name, None, None).await?;
    let mut sender_configs: BTreeMap<String, Value> = BTreeMap::new();
    for message in messages {
        sender_configs
            .entry(message.peer_name.clone())
            .or_insert_with(|| json!({"observe_me": null, "observe_others": null}));
    }
    for peer_name in sender_configs.keys() {
        get_or_create_peer(pool, workspace_name, peer_name, None, None).await?;
    }
    upsert_session_peers(pool, workspace_name, session_name, &sender_configs).await?;

    let mut transaction = pool.begin().await?;

    sqlx::query("SET LOCAL lock_timeout = '5s'")
        .execute(&mut *transaction)
        .await?;
    sqlx::query("SELECT pg_advisory_xact_lock(hashtext($1), hashtext($2))")
        .bind(workspace_name)
        .bind(session_name)
        .execute(&mut *transaction)
        .await?;

    let last_seq: i64 = sqlx::query_scalar(
        "SELECT seq_in_session FROM messages \
         WHERE workspace_name = $1 AND session_name = $2 \
         ORDER BY seq_in_session DESC LIMIT 1",
    )
    .bind(workspace_name)
    .bind(session_name)
    .fetch_optional(&mut *transaction)
    .await?
    .unwrap_or(0);

    let mut created = Vec::with_capacity(messages.len());
    for (offset, message) in messages.iter().enumerate() {
        let seq_in_session = last_seq + (offset as i64) + 1;
        let row = sqlx::query_as::<_, CreatedMessage>(
            "INSERT INTO messages \
             (public_id, session_name, content, metadata, workspace_name, peer_name, \
              seq_in_session, token_count, created_at) \
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, COALESCE($9::timestamptz, now())) \
             RETURNING id, public_id, content, peer_name, session_name, metadata, \
                       created_at, workspace_name, token_count, seq_in_session",
        )
        .bind(generate_nanoid())
        .bind(session_name)
        .bind(&message.content)
        .bind(&message.metadata)
        .bind(workspace_name)
        .bind(&message.peer_name)
        .bind(seq_in_session)
        .bind(message.token_count)
        .bind(message.created_at)
        .fetch_one(&mut *transaction)
        .await?;
        created.push(row);
    }

    // EMBED_MESSAGES: persist one pending embedding row per content chunk, for
    // messages whose content is non-empty after trimming (matching Python's
    // `if content and content.strip()` filter).
    if embed_messages {
        let id_resource_dict = created
            .iter()
            .filter(|message| !message.content.trim().is_empty())
            .map(|message| (message.public_id.clone(), message.content.clone()))
            .collect::<BTreeMap<_, _>>();
        let chunks_by_id =
            crate::embedding::prepare_chunks(&id_resource_dict, embedding_max_tokens);
        let peer_by_id = created
            .iter()
            .map(|message| (message.public_id.as_str(), message.peer_name.as_str()))
            .collect::<BTreeMap<_, _>>();
        for message in &created {
            let Some(chunks) = chunks_by_id.get(&message.public_id) else {
                continue;
            };
            let peer_name = peer_by_id[message.public_id.as_str()];
            for chunk in chunks {
                sqlx::query(
                    "INSERT INTO message_embeddings \
                     (content, embedding, message_id, workspace_name, session_name, \
                      peer_name, sync_state) \
                     VALUES ($1, NULL, $2, $3, $4, $5, 'pending')",
                )
                .bind(chunk)
                .bind(&message.public_id)
                .bind(workspace_name)
                .bind(session_name)
                .bind(peer_name)
                .execute(&mut *transaction)
                .await?;
            }
        }
    }

    transaction.commit().await?;
    Ok(created)
}

/// The active session's internal id plus its stored configuration, used as the
/// `session_id` queue column and one rung of the configuration hierarchy.
pub async fn fetch_session_for_enqueue(
    pool: &PgPool,
    workspace_name: &str,
    session_name: &str,
) -> Result<Option<(String, Value)>, sqlx::Error> {
    let row = sqlx::query(
        "SELECT id, configuration FROM sessions \
         WHERE workspace_name = $1 AND name = $2 AND is_active = true",
    )
    .bind(workspace_name)
    .bind(session_name)
    .fetch_optional(pool)
    .await?;

    match row {
        Some(row) => Ok(Some((row.try_get("id")?, row.try_get("configuration")?))),
        None => Ok(None),
    }
}

/// The workspace's stored configuration object, the top rung of the hierarchy.
pub async fn get_workspace_configuration(
    pool: &PgPool,
    workspace_name: &str,
) -> Result<Value, sqlx::Error> {
    let row = sqlx::query("SELECT configuration FROM workspaces WHERE name = $1")
        .bind(workspace_name)
        .fetch_optional(pool)
        .await?;
    match row {
        Some(row) => row.try_get("configuration"),
        None => Ok(json!({})),
    }
}

/// Observer-selection query, porting `crud.get_session_peer_configuration`.
///
/// Returns every peer ever in the session (including those who left), keyed by
/// peer name and ordered for deterministic observer ordering. `is_active`
/// distinguishes current members (`left_at IS NULL`) from peers who left.
pub async fn get_session_peer_configuration(
    pool: &PgPool,
    workspace_name: &str,
    session_name: &str,
) -> Result<BTreeMap<String, PeerConfigEntry>, sqlx::Error> {
    let rows = sqlx::query(
        "SELECT peer.name AS peer_name, \
                peer.configuration AS peer_configuration, \
                session_peer.configuration AS session_peer_configuration, \
                (session_peer.left_at IS NULL) AS is_active \
         FROM peers peer \
         JOIN session_peers session_peer \
           ON peer.name = session_peer.peer_name \
          AND peer.workspace_name = session_peer.workspace_name \
         WHERE session_peer.session_name = $1 \
           AND peer.workspace_name = $2 \
           AND session_peer.workspace_name = $2 \
         ORDER BY peer.name",
    )
    .bind(session_name)
    .bind(workspace_name)
    .fetch_all(pool)
    .await?;

    let mut peers = BTreeMap::new();
    for row in rows {
        let peer_name: String = row.try_get("peer_name")?;
        peers.insert(
            peer_name,
            PeerConfigEntry {
                peer_configuration: row.try_get("peer_configuration")?,
                session_peer_configuration: row.try_get("session_peer_configuration")?,
                is_active: row.try_get("is_active")?,
            },
        );
    }
    Ok(peers)
}

/// Insert the generated queue records in one transaction, mirroring the single
/// batched `INSERT INTO queue ... ` Python performs after building the records.
pub async fn insert_queue_records(
    pool: &PgPool,
    records: &[QueueRecord],
) -> Result<(), sqlx::Error> {
    if records.is_empty() {
        return Ok(());
    }
    let mut transaction = pool.begin().await?;
    for record in records {
        sqlx::query(
            "INSERT INTO queue \
             (work_unit_key, payload, session_id, task_type, workspace_name, message_id) \
             VALUES ($1, $2, $3, $4, $5, $6)",
        )
        .bind(&record.work_unit_key)
        .bind(&record.payload)
        .bind(&record.session_id)
        .bind(&record.task_type)
        .bind(&record.workspace_name)
        .bind(record.message_id)
        .execute(&mut *transaction)
        .await?;
    }
    transaction.commit().await?;
    Ok(())
}

/// A `queue` row, mirroring the `models.QueueItem` columns the worker reads.
#[derive(Debug, Clone)]
pub struct QueueItem {
    pub id: i64,
    pub work_unit_key: String,
    pub payload: Value,
    pub session_id: Option<String>,
    pub task_type: String,
    pub workspace_name: Option<String>,
    pub message_id: Option<i64>,
    pub processed: bool,
    pub error: Option<String>,
    pub created_at: DateTime<Utc>,
}

impl QueueItem {
    fn from_row(row: &PgRow) -> Self {
        Self {
            id: row.get("id"),
            work_unit_key: row.get("work_unit_key"),
            payload: row.get("payload"),
            session_id: row.get("session_id"),
            task_type: row.get("task_type"),
            workspace_name: row.get("workspace_name"),
            message_id: row.get("message_id"),
            processed: row.get("processed"),
            error: row.get("error"),
            created_at: row.get("created_at"),
        }
    }
}

/// Port of `QueueManager.get_next_queue_item`: the earliest unprocessed queue
/// item for a work unit still owned by this worker (verified by joining
/// `active_queue_sessions` on the key and matching `aqs_id`). Returns `None` when
/// the work unit is drained or ownership was lost. Representation work units use
/// [`get_queue_item_batch`] instead (the Python guard raising for them is an
/// internal invariant the caller already upholds).
pub async fn get_next_queue_item(
    pool: &PgPool,
    work_unit_key: &str,
    aqs_id: &str,
) -> Result<Option<QueueItem>, sqlx::Error> {
    let row = sqlx::query(
        "SELECT q.id, q.work_unit_key, q.payload, q.session_id, q.task_type, \
                q.workspace_name, q.message_id, q.processed, q.error, q.created_at \
         FROM queue q \
         JOIN active_queue_sessions aqs ON q.work_unit_key = aqs.work_unit_key \
         WHERE q.work_unit_key = $1 AND NOT q.processed AND aqs.id = $2 \
         ORDER BY q.id LIMIT 1",
    )
    .bind(work_unit_key)
    .bind(aqs_id)
    .fetch_optional(pool)
    .await?;
    Ok(row.as_ref().map(QueueItem::from_row))
}

/// A `collections` row keyed by `(observer, observed, workspace_name)`.
#[derive(Debug, Clone)]
pub struct Collection {
    pub id: String,
    pub workspace_name: String,
    pub observer: String,
    pub observed: String,
    pub metadata: Value,
    pub internal_metadata: Value,
    pub created_at: DateTime<Utc>,
}

impl Collection {
    fn from_row(row: &PgRow) -> Self {
        Self {
            id: row.get("id"),
            workspace_name: row.get("workspace_name"),
            observer: row.get("observer"),
            observed: row.get("observed"),
            metadata: row.get("metadata"),
            internal_metadata: row.get("internal_metadata"),
            created_at: row.get("created_at"),
        }
    }
}

/// Port of `crud.get_or_create_collection`: return the `(observer, observed,
/// workspace)` collection, creating it (with a fresh nanoid id) if absent.
/// Implemented as an `INSERT ... ON CONFLICT DO NOTHING RETURNING` followed by a
/// `SELECT` on conflict — concurrency-safe via the
/// `uq_collections_observer_observed_workspace_name` constraint, matching the
/// Python get-then-create-with-retry semantics. (The Python best-effort cache
/// write is omitted; the Rust read path does not cache collections.)
pub async fn get_or_create_collection(
    pool: &PgPool,
    workspace_name: &str,
    observer: &str,
    observed: &str,
) -> Result<Collection, sqlx::Error> {
    const COLUMNS: &str =
        "id, workspace_name, observer, observed, metadata, internal_metadata, created_at";

    let insert_sql = format!(
        "INSERT INTO collections (id, workspace_name, observer, observed) \
         VALUES ($1, $2, $3, $4) \
         ON CONFLICT (observer, observed, workspace_name) DO NOTHING \
         RETURNING {COLUMNS}"
    );
    let inserted = sqlx::query(&insert_sql)
        .bind(generate_nanoid())
        .bind(workspace_name)
        .bind(observer)
        .bind(observed)
        .fetch_optional(pool)
        .await?;
    if let Some(row) = inserted {
        return Ok(Collection::from_row(&row));
    }

    let select_sql = format!(
        "SELECT {COLUMNS} FROM collections \
         WHERE observer = $1 AND observed = $2 AND workspace_name = $3"
    );
    let row = sqlx::query(&select_sql)
        .bind(observer)
        .bind(observed)
        .bind(workspace_name)
        .fetch_one(pool)
        .await?;
    Ok(Collection::from_row(&row))
}

/// Port of `process_dream`'s guard-pair write: in one transaction, lock the
/// `(observer, observed)` collection `FOR UPDATE`, count its live `explicit`
/// documents, and advance the collection's `internal_metadata.dream`
/// `{last_dream_at, last_dream_document_count}` together. Existing `dream`
/// sub-keys are preserved (the two guard fields are merged over them). No-op when
/// the collection does not exist. Returns the explicit count written.
pub async fn record_dream_guard(
    pool: &PgPool,
    workspace_name: &str,
    observer: &str,
    observed: &str,
    now_iso: &str,
) -> Result<Option<i64>, sqlx::Error> {
    let mut tx = pool.begin().await?;

    let existing: Option<Value> = sqlx::query_scalar(
        "SELECT internal_metadata FROM collections \
         WHERE observer = $1 AND observed = $2 AND workspace_name = $3 \
         FOR UPDATE",
    )
    .bind(observer)
    .bind(observed)
    .bind(workspace_name)
    .fetch_optional(&mut *tx)
    .await?;

    let Some(internal_metadata) = existing else {
        return Ok(None);
    };

    let explicit_count: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM documents \
         WHERE workspace_name = $1 AND observer = $2 AND observed = $3 \
           AND level = 'explicit'",
    )
    .bind(workspace_name)
    .bind(observer)
    .bind(observed)
    .fetch_one(&mut *tx)
    .await?;

    // Build the new dream sub-object from the existing one (preserve other keys).
    let mut dream_meta = internal_metadata
        .get("dream")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    dream_meta.insert("last_dream_at".to_string(), Value::String(now_iso.to_string()));
    dream_meta.insert(
        "last_dream_document_count".to_string(),
        Value::from(explicit_count),
    );
    let patch = json!({ "dream": Value::Object(dream_meta) });

    sqlx::query(
        "UPDATE collections SET internal_metadata = internal_metadata || $1::jsonb \
         WHERE observer = $2 AND observed = $3 AND workspace_name = $4",
    )
    .bind(patch)
    .bind(observer)
    .bind(observed)
    .bind(workspace_name)
    .execute(&mut *tx)
    .await?;

    tx.commit().await?;
    Ok(Some(explicit_count))
}

/// An observation document to persist, mirroring the fields of
/// `schemas.DocumentCreate` the deriver populates. `embedding` is computed by
/// the caller (`save_representation`); `internal_metadata` is the already-built
/// metadata object (message_ids / premises / message_created_at).
#[derive(Debug, Clone)]
pub struct DocumentToCreate {
    pub content: String,
    pub session_name: Option<String>,
    pub level: String,
    pub internal_metadata: Value,
    pub embedding: Vec<f32>,
    pub times_derived: i32,
    pub source_ids: Option<Value>,
}

/// cl100k_base token-id set, matching the deriver dedup's
/// `set(embedding_client.encoding.encode(text))` (the embedding model
/// text-embedding-3-small resolves to cl100k_base, distinct from the o200k
/// encoding `tokens::estimate_tokens` uses). Uses ordinary encoding so it
/// mirrors Python's default `encode` for non-special content.
fn cl100k_token_set(text: &str) -> std::collections::HashSet<u32> {
    use std::sync::OnceLock;
    static ENCODER: OnceLock<tiktoken_rs::CoreBPE> = OnceLock::new();
    let encoder =
        ENCODER.get_or_init(|| tiktoken_rs::cl100k_base().expect("load cl100k_base tokenizer"));
    encoder.encode_ordinary(text).into_iter().collect()
}

struct DuplicateCandidate {
    id: String,
    content: String,
    times_derived: i32,
}

/// The nearest non-deleted document within `max_distance` cosine distance of
/// `embedding` for `(observer, observed)` — `query_documents(..., top_k=1)` as
/// used by `is_rejected_duplicate`. Runs on the caller's transaction so it sees
/// in-flight soft-deletes.
async fn find_duplicate_candidate(
    conn: &mut sqlx::PgConnection,
    workspace_name: &str,
    observer: &str,
    observed: &str,
    embedding: &[f32],
    max_distance: f64,
) -> Result<Option<DuplicateCandidate>, sqlx::Error> {
    let row = sqlx::query(
        "SELECT id, content, times_derived FROM documents \
         WHERE workspace_name = $1 AND observer = $2 AND observed = $3 \
           AND embedding IS NOT NULL AND deleted_at IS NULL \
           AND embedding <=> $4::vector <= $5 \
         ORDER BY embedding <=> $4::vector LIMIT 1",
    )
    .bind(workspace_name)
    .bind(observer)
    .bind(observed)
    .bind(crate::search::vector_literal(embedding))
    .bind(max_distance)
    .fetch_optional(&mut *conn)
    .await?;
    Ok(row.map(|row| DuplicateCandidate {
        id: row.get("id"),
        content: row.get("content"),
        times_derived: row.get("times_derived"),
    }))
}

/// Port of `is_rejected_duplicate`: cosine-similarity dedup (>= 0.95, i.e.
/// distance <= 0.05) with token-set retention scoring. When a near-duplicate
/// exists, the more-informative document wins: ties and a superior new doc
/// soft-delete the existing one (carrying its reinforcement count forward into
/// `doc.times_derived`) and the new doc is kept (`false`); otherwise the
/// existing doc's `times_derived` is atomically incremented and the new doc is
/// rejected (`true`). Score = token count + 10 * tokens-unique-to-that-doc.
async fn is_rejected_duplicate(
    conn: &mut sqlx::PgConnection,
    workspace_name: &str,
    observer: &str,
    observed: &str,
    doc: &mut DocumentToCreate,
) -> Result<bool, sqlx::Error> {
    let Some(existing) =
        find_duplicate_candidate(conn, workspace_name, observer, observed, &doc.embedding, 0.05)
            .await?
    else {
        return Ok(false);
    };

    let tokens_new = cl100k_token_set(&doc.content);
    let tokens_existing = cl100k_token_set(&existing.content);
    let unique_new = tokens_new.difference(&tokens_existing).count() as i64;
    let unique_existing = tokens_existing.difference(&tokens_new).count() as i64;
    let score_new = tokens_new.len() as i64 + unique_new * 10;
    let score_existing = tokens_existing.len() as i64 + unique_existing * 10;

    if score_new >= score_existing {
        // Keep the new doc; soft-delete the existing and carry its reinforcement
        // count forward (max so a higher new count is preserved).
        doc.times_derived = doc.times_derived.max(existing.times_derived + 1);
        sqlx::query("UPDATE documents SET deleted_at = now() WHERE id = $1")
            .bind(&existing.id)
            .execute(&mut *conn)
            .await?;
        Ok(false)
    } else {
        // Existing wins; record the reinforcement atomically and reject the new.
        sqlx::query("UPDATE documents SET times_derived = times_derived + 1 WHERE id = $1")
            .bind(&existing.id)
            .execute(&mut *conn)
            .await?;
        Ok(true)
    }
}

/// Port of `crud.create_documents` for the default pgvector path (no external
/// vector store): optionally dedup each document, bulk-insert the survivors with
/// their embeddings, and mark them `synced` immediately (the
/// `external_vector_store is None` branch). Returns the number of documents
/// actually inserted. Dedup writes and inserts share one transaction, matching
/// the Python session. External-vector-store modes are not ported.
pub async fn create_documents(
    pool: &PgPool,
    documents: Vec<DocumentToCreate>,
    workspace_name: &str,
    observer: &str,
    observed: &str,
    deduplicate: bool,
) -> Result<usize, sqlx::Error> {
    Ok(
        create_documents_returning_levels(pool, documents, workspace_name, observer, observed, deduplicate)
            .await?
            .len(),
    )
}

/// Like [`create_documents`] but returns the `level` of each accepted (inserted)
/// document, in insertion order — the dreamer's `create_observations` needs the
/// per-level breakdown of what survived dedup (Python `create_documents` returns
/// the accepted `DocumentCreate` list and the caller reads `[doc.level ...]`).
pub async fn create_documents_returning_levels(
    pool: &PgPool,
    documents: Vec<DocumentToCreate>,
    workspace_name: &str,
    observer: &str,
    observed: &str,
    deduplicate: bool,
) -> Result<Vec<String>, sqlx::Error> {
    let mut transaction = pool.begin().await?;
    let mut inserted_ids: Vec<String> = Vec::new();
    let mut inserted_levels: Vec<String> = Vec::new();

    for mut doc in documents {
        if deduplicate
            && is_rejected_duplicate(
                &mut transaction,
                workspace_name,
                observer,
                observed,
                &mut doc,
            )
            .await?
        {
            continue;
        }

        let id = generate_nanoid();
        sqlx::query(
            "INSERT INTO documents \
             (id, workspace_name, observer, observed, content, level, times_derived, \
              internal_metadata, session_name, embedding, source_ids, sync_state) \
             VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10::vector, $11, 'pending')",
        )
        .bind(&id)
        .bind(workspace_name)
        .bind(observer)
        .bind(observed)
        .bind(&doc.content)
        .bind(&doc.level)
        .bind(doc.times_derived)
        .bind(&doc.internal_metadata)
        .bind(&doc.session_name)
        .bind(crate::search::vector_literal(&doc.embedding))
        .bind(&doc.source_ids)
        .execute(&mut *transaction)
        .await?;
        inserted_ids.push(id);
        inserted_levels.push(doc.level);
    }

    // pgvector mode: no external store, so mark the freshly-inserted docs synced.
    if !inserted_ids.is_empty() {
        sqlx::query(
            "UPDATE documents SET sync_state = 'synced', last_sync_at = now(), sync_attempts = 0 \
             WHERE id = ANY($1)",
        )
        .bind(&inserted_ids)
        .execute(&mut *transaction)
        .await?;
    }

    transaction.commit().await?;
    Ok(inserted_levels)
}

/// A conversation message row forming the representation-batch context window
/// (`models.Message` columns the deriver batch processor reads).
#[derive(Debug, Clone)]
pub struct BatchMessage {
    pub id: i64,
    pub public_id: String,
    pub content: String,
    pub created_at: DateTime<Utc>,
    pub peer_name: String,
    pub token_count: i32,
    pub session_name: String,
    pub workspace_name: String,
}

/// Result of [`get_queue_item_batch`], mirroring Python `QueueBatchResult`. The
/// cap flags feed the deriver's `RepresentationCompletedEvent` telemetry.
#[derive(Debug, Clone, Default)]
pub struct QueueBatchResult {
    pub messages_context: Vec<BatchMessage>,
    pub items_to_process: Vec<QueueItem>,
    pub configuration: Option<ResolvedConfiguration>,
    pub hit_batch_token_cap: bool,
    pub was_flush_enabled: bool,
    pub batch_max_tokens: i64,
}

/// Failure modes of [`get_queue_item_batch`].
#[derive(Debug)]
pub enum QueueBatchError {
    Database(sqlx::Error),
    /// A queue item carried a malformed `configuration` payload (the Python
    /// `ResolvedConfiguration.model_validate` would raise).
    Config(String),
}

impl From<sqlx::Error> for QueueBatchError {
    fn from(error: sqlx::Error) -> Self {
        QueueBatchError::Database(error)
    }
}

/// Port of `QueueManager.get_queue_item_batch`: assemble a representation work
/// unit's context window and the queue items to process within it.
///
/// After verifying ownership, a single windowed query finds the earliest
/// unprocessed message for the work unit, optionally prepends the immediately
/// preceding message when it is from a different peer (conversational context),
/// then walks forward accumulating `token_count` until `batch_max_tokens` is
/// exceeded — always including at least the first unprocessed message. The CTE's
/// `bool_or(cumulative > cap) OVER ()` reports whether the cap excluded anything
/// in one round-trip. The returned batch is then trimmed to the leading
/// homogeneous-`configuration` run (`resolve_batch_configuration_prefix`), the
/// context window is clipped to the last surviving queued message, and
/// `hit_batch_token_cap` is set only when the cap (not the config filter) bounded
/// the batch.
pub async fn get_queue_item_batch(
    pool: &PgPool,
    work_unit_key: &str,
    aqs_id: &str,
    batch_max_tokens: i64,
    flush_enabled: bool,
) -> Result<QueueBatchResult, QueueBatchError> {
    // Parsed key gives session_name / workspace_name / observed for the query.
    let parsed = crate::producer::parse_work_unit_key(work_unit_key)
        .map_err(|error| QueueBatchError::Config(error.to_string()))?;
    let session_name = parsed.session_name.unwrap_or_default();
    let workspace_name = parsed.workspace_name.unwrap_or_default();
    let observed = parsed.observed.unwrap_or_default();

    let empty = || QueueBatchResult {
        was_flush_enabled: flush_enabled,
        batch_max_tokens,
        ..QueueBatchResult::default()
    };

    // Step 1: confirm this worker still owns the work unit.
    let owned: bool = sqlx::query_scalar(
        "SELECT EXISTS(SELECT 1 FROM active_queue_sessions WHERE work_unit_key = $1 AND id = $2)",
    )
    .bind(work_unit_key)
    .bind(aqs_id)
    .fetch_one(pool)
    .await?;
    if !owned {
        return Ok(empty());
    }

    // Step 2: the windowed batch query. $1 session, $2 workspace, $3 key,
    // $4 observed, $5 batch_max_tokens.
    let rows = sqlx::query(
        "WITH inner_cte AS ( \
             SELECT m.id AS message_id, m.token_count AS token_count, m.peer_name AS peer_name, \
                    SUM(m.token_count) OVER (ORDER BY m.id) AS cumulative_token_count \
             FROM messages m \
             WHERE m.session_name = $1 AND m.workspace_name = $2 AND m.id >= COALESCE( \
                 (SELECT mp.id FROM messages mp \
                  WHERE mp.id = (SELECT MAX(m2.id) FROM messages m2 \
                                 WHERE m2.session_name = $1 AND m2.workspace_name = $2 \
                                   AND m2.id < (SELECT MIN(m3.id) FROM messages m3 \
                                                JOIN queue q3 ON q3.message_id = m3.id \
                                                WHERE NOT q3.processed AND m3.session_name = $1 \
                                                  AND m3.workspace_name = $2 AND q3.work_unit_key = $3)) \
                    AND mp.peer_name <> $4), \
                 (SELECT MIN(m4.id) FROM messages m4 \
                  JOIN queue q4 ON q4.message_id = m4.id \
                  WHERE NOT q4.processed AND m4.session_name = $1 \
                    AND m4.workspace_name = $2 AND q4.work_unit_key = $3) \
             ) \
         ), \
         cte AS ( \
             SELECT message_id, token_count, peer_name, cumulative_token_count, \
                    bool_or(cumulative_token_count > $5) OVER () AS cap_exceeded \
             FROM inner_cte \
         ) \
         SELECT m.id, m.public_id, m.content, m.created_at, m.peer_name, m.token_count, \
                m.session_name, m.workspace_name, \
                q.id AS qid, q.work_unit_key AS qwuk, q.payload AS qpayload, \
                q.session_id AS qsession_id, q.task_type AS qtask_type, \
                q.workspace_name AS qworkspace_name, q.message_id AS qmessage_id, \
                q.processed AS qprocessed, q.error AS qerror, q.created_at AS qcreated_at, \
                cte.cap_exceeded AS cap_exceeded \
         FROM cte \
         JOIN messages m ON m.id = cte.message_id \
         LEFT JOIN queue q ON q.work_unit_key = $3 AND NOT q.processed AND q.message_id = m.id \
         WHERE cte.cumulative_token_count <= $5 OR cte.message_id = ( \
             SELECT MIN(m5.id) FROM messages m5 \
             JOIN queue q5 ON q5.message_id = m5.id \
             WHERE NOT q5.processed AND m5.session_name = $1 \
               AND m5.workspace_name = $2 AND q5.work_unit_key = $3) \
         ORDER BY m.id, q.id",
    )
    .bind(&session_name)
    .bind(&workspace_name)
    .bind(work_unit_key)
    .bind(&observed)
    .bind(batch_max_tokens)
    .fetch_all(pool)
    .await?;

    if rows.is_empty() {
        return Ok(empty());
    }

    // cap_exceeded is window-aggregated (identical on every row); read once.
    let cap_exceeded_from_query = if batch_max_tokens > 0 {
        rows[0]
            .try_get::<Option<bool>, _>("cap_exceeded")
            .ok()
            .flatten()
            .unwrap_or(false)
    } else {
        false
    };

    let mut messages_context: Vec<BatchMessage> = Vec::new();
    let mut seen: std::collections::HashSet<i64> = std::collections::HashSet::new();
    let mut items_to_process: Vec<QueueItem> = Vec::new();
    for row in &rows {
        let message_id: i64 = row.get("id");
        if seen.insert(message_id) {
            messages_context.push(BatchMessage {
                id: message_id,
                public_id: row.get("public_id"),
                content: row.get("content"),
                created_at: row.get("created_at"),
                peer_name: row.get("peer_name"),
                token_count: row.get("token_count"),
                session_name: row.get("session_name"),
                workspace_name: row.get("workspace_name"),
            });
        }
        if let Ok(Some(qid)) = row.try_get::<Option<i64>, _>("qid") {
            items_to_process.push(QueueItem {
                id: qid,
                work_unit_key: row.get("qwuk"),
                payload: row.get("qpayload"),
                session_id: row.get("qsession_id"),
                task_type: row.get("qtask_type"),
                workspace_name: row.get("qworkspace_name"),
                message_id: row.get("qmessage_id"),
                processed: row.get("qprocessed"),
                error: row.get("qerror"),
                created_at: row.get("qcreated_at"),
            });
        }
    }

    // The queue-item boundary (not the context tail) drives cap detection.
    let last_queued_id_before: Option<i64> =
        items_to_process.iter().filter_map(|q| q.message_id).max();

    // Trim to the leading homogeneous-configuration run.
    let (prefix_len, resolved_config) =
        resolve_batch_configuration_prefix(items_to_process.iter().map(|q| &q.payload))
            .map_err(QueueBatchError::Config)?;
    items_to_process.truncate(prefix_len);

    if !items_to_process.is_empty() {
        if let Some(max_queue_item_message_id) =
            items_to_process.iter().filter_map(|q| q.message_id).max()
        {
            messages_context.retain(|m| m.id <= max_queue_item_message_id);
        }
    }

    let last_queued_id_after: Option<i64> =
        items_to_process.iter().filter_map(|q| q.message_id).max();

    // The cap was binding on the returned batch only when the config filter did
    // not move the queue-item boundary and the CTE saw a message past the cap.
    let hit_batch_token_cap = if batch_max_tokens > 0
        && last_queued_id_before.is_some()
        && last_queued_id_before == last_queued_id_after
    {
        cap_exceeded_from_query
    } else {
        false
    };

    Ok(QueueBatchResult {
        messages_context,
        items_to_process,
        configuration: resolved_config,
        hit_batch_token_cap,
        was_flush_enabled: flush_enabled,
        batch_max_tokens,
    })
}

/// Port of `QueueManager.claim_work_units`: insert one `active_queue_sessions`
/// row per work-unit key (each with a freshly generated nanoid id) using
/// `ON CONFLICT DO NOTHING`, returning the `work_unit_key -> aqs_id` map for the
/// rows actually claimed (a key already owned by another worker conflicts on the
/// `work_unit_key` unique constraint and is skipped). Callers pass a deduped key
/// list, so there are no intra-statement duplicates. The executor may be a pool
/// or an in-flight transaction.
pub async fn claim_work_units<'e, E>(
    executor: E,
    work_unit_keys: &[String],
) -> Result<HashMap<String, String>, sqlx::Error>
where
    E: sqlx::PgExecutor<'e>,
{
    if work_unit_keys.is_empty() {
        return Ok(HashMap::new());
    }

    let ids: Vec<String> = work_unit_keys.iter().map(|_| generate_nanoid()).collect();
    let mut sql = String::from("INSERT INTO active_queue_sessions (id, work_unit_key) VALUES ");
    for index in 0..work_unit_keys.len() {
        if index > 0 {
            sql.push(',');
        }
        let base = index * 2;
        sql.push_str(&format!("(${}, ${})", base + 1, base + 2));
    }
    sql.push_str(" ON CONFLICT DO NOTHING RETURNING work_unit_key, id");

    let mut query = sqlx::query(&sql);
    for (id, key) in ids.iter().zip(work_unit_keys) {
        query = query.bind(id).bind(key);
    }
    let rows = query.fetch_all(executor).await?;

    let mut claimed = HashMap::with_capacity(rows.len());
    for row in rows {
        let key: String = row.get("work_unit_key");
        let id: String = row.get("id");
        claimed.insert(key, id);
    }
    Ok(claimed)
}

/// Port of `QueueManager.get_and_claim_work_units`: find unprocessed work-unit
/// keys not currently owned (no `active_queue_sessions` row), optionally
/// requiring representation batches to have reached the cumulative-token
/// threshold, then claim up to `workers - owned_count` of them in one
/// transaction. `representation:` keys must meet `batch_max_tokens` (summed over
/// their unprocessed messages) unless flushing is enabled or the cap is 0; other
/// task types are always eligible. Returns the `work_unit_key -> aqs_id` map.
pub async fn get_and_claim_work_units(
    pool: &PgPool,
    workers: i64,
    owned_count: i64,
    batch_max_tokens: i64,
    flush_enabled: bool,
) -> Result<HashMap<String, String>, sqlx::Error> {
    let limit = (workers - owned_count).max(0);
    if limit == 0 {
        return Ok(HashMap::new());
    }

    let apply_threshold = !flush_enabled && batch_max_tokens > 0;

    // No ORDER BY — matches the Python query, whose LIMIT caps the count without
    // a defined ordering. The threshold filter and the cap parameter are bound
    // positionally ($1 = limit, $2 = batch_max_tokens).
    let mut sql = String::from(
        "SELECT wu.work_unit_key \
         FROM (SELECT work_unit_key FROM queue WHERE NOT processed GROUP BY work_unit_key) wu \
         LEFT JOIN ( \
             SELECT q.work_unit_key, SUM(m.token_count) AS total_tokens \
             FROM queue q JOIN messages m ON q.message_id = m.id \
             WHERE NOT q.processed AND q.work_unit_key LIKE 'representation:%' \
             GROUP BY q.work_unit_key \
         ) ts ON wu.work_unit_key = ts.work_unit_key \
         WHERE NOT EXISTS ( \
             SELECT 1 FROM active_queue_sessions aqs \
             WHERE aqs.work_unit_key = wu.work_unit_key \
         )",
    );
    if apply_threshold {
        sql.push_str(
            " AND (wu.work_unit_key NOT LIKE 'representation:%' \
             OR COALESCE(ts.total_tokens, 0) >= $2)",
        );
    }
    sql.push_str(" LIMIT $1");

    let mut transaction = pool.begin().await?;

    let mut query = sqlx::query_scalar::<_, String>(&sql).bind(limit);
    if apply_threshold {
        query = query.bind(batch_max_tokens);
    }
    let available: Vec<String> = query.fetch_all(&mut *transaction).await?;
    if available.is_empty() {
        transaction.commit().await?;
        return Ok(HashMap::new());
    }

    let claimed = claim_work_units(&mut *transaction, &available).await?;
    transaction.commit().await?;
    Ok(claimed)
}

/// Port of `QueueManager.mark_queue_items_as_processed`: flag the given queue
/// items processed and bump the owning `active_queue_sessions.last_updated`,
/// in one transaction. The `work_unit_key` guard mirrors the Python `where`.
pub async fn mark_queue_items_as_processed(
    pool: &PgPool,
    item_ids: &[i64],
    work_unit_key: &str,
) -> Result<(), sqlx::Error> {
    if item_ids.is_empty() {
        return Ok(());
    }
    let mut transaction = pool.begin().await?;
    sqlx::query("UPDATE queue SET processed = true WHERE id = ANY($1) AND work_unit_key = $2")
        .bind(item_ids)
        .bind(work_unit_key)
        .execute(&mut *transaction)
        .await?;
    sqlx::query("UPDATE active_queue_sessions SET last_updated = now() WHERE work_unit_key = $1")
        .bind(work_unit_key)
        .execute(&mut *transaction)
        .await?;
    transaction.commit().await?;
    Ok(())
}

/// Port of `QueueManager.mark_queue_item_as_errored`: mark a single queue item
/// processed with its error text (truncated to the TEXT-limit 65535 chars, by
/// code point as in Python) and bump `last_updated`, in one transaction.
pub async fn mark_queue_item_as_errored(
    pool: &PgPool,
    item_id: i64,
    work_unit_key: &str,
    error: &str,
) -> Result<(), sqlx::Error> {
    let truncated: String = error.chars().take(65535).collect();
    let mut transaction = pool.begin().await?;
    sqlx::query("UPDATE queue SET processed = true, error = $3 WHERE id = $1 AND work_unit_key = $2")
        .bind(item_id)
        .bind(work_unit_key)
        .bind(&truncated)
        .execute(&mut *transaction)
        .await?;
    sqlx::query("UPDATE active_queue_sessions SET last_updated = now() WHERE work_unit_key = $1")
        .bind(work_unit_key)
        .execute(&mut *transaction)
        .await?;
    transaction.commit().await?;
    Ok(())
}

/// Port of `QueueManager._cleanup_work_unit`: delete the `active_queue_sessions`
/// row matched by both id and key, returning whether a row was removed (the
/// Python `rowcount > 0` used to gate the `queue.empty` webhook).
pub async fn cleanup_work_unit(
    pool: &PgPool,
    aqs_id: &str,
    work_unit_key: &str,
) -> Result<bool, sqlx::Error> {
    let result =
        sqlx::query("DELETE FROM active_queue_sessions WHERE id = $1 AND work_unit_key = $2")
            .bind(aqs_id)
            .bind(work_unit_key)
            .execute(pool)
            .await?;
    Ok(result.rows_affected() > 0)
}

/// Port of `QueueManager.cleanup_stale_work_units`: within one transaction,
/// lock stale `active_queue_sessions` rows (`last_updated` older than
/// `stale_timeout_minutes`) with `FOR UPDATE SKIP LOCKED` so concurrent cleaners
/// on other instances don't collide, then delete exactly the rows locked.
/// Returns the number of rows removed. The cutoff is computed from the
/// process clock (`Utc::now()`), matching Python's `datetime.now(utc)`.
pub async fn cleanup_stale_work_units(
    pool: &PgPool,
    stale_timeout_minutes: i64,
) -> Result<u64, sqlx::Error> {
    let cutoff = Utc::now() - chrono::Duration::minutes(stale_timeout_minutes);
    let mut transaction = pool.begin().await?;
    let stale_ids: Vec<String> = sqlx::query_scalar(
        "SELECT id FROM active_queue_sessions \
         WHERE last_updated < $1 ORDER BY last_updated FOR UPDATE SKIP LOCKED",
    )
    .bind(cutoff)
    .fetch_all(&mut *transaction)
    .await?;

    let mut deleted = 0u64;
    if !stale_ids.is_empty() {
        let result = sqlx::query("DELETE FROM active_queue_sessions WHERE id = ANY($1)")
            .bind(&stale_ids)
            .execute(&mut *transaction)
            .await?;
        deleted = result.rows_affected();
    }
    transaction.commit().await?;
    Ok(deleted)
}

/// Port of `reconciler.queue_cleanup.cleanup_queue_items`: delete processed queue
/// items. Successfully processed items (`error IS NULL`) are deleted immediately;
/// errored items are kept until `created_at` is older than the error-retention
/// window (`now() - error_retention_seconds`). Returns the number deleted.
pub async fn cleanup_queue_items(
    pool: &PgPool,
    error_retention_seconds: i64,
) -> Result<u64, sqlx::Error> {
    let error_cutoff = Utc::now() - chrono::Duration::seconds(error_retention_seconds);
    let result = sqlx::query(
        "DELETE FROM queue \
         WHERE processed \
           AND (error IS NULL \
                OR (error IS NOT NULL AND created_at < $1))",
    )
    .bind(error_cutoff)
    .execute(pool)
    .await?;
    Ok(result.rows_affected())
}

/// Port of `reconciler.sync_vectors._cleanup_soft_deleted_documents_pgvector`:
/// hard-delete up to `batch_size` soft-deleted documents whose `deleted_at` is
/// older than `older_than_minutes`. Rows are claimed with `FOR UPDATE SKIP
/// LOCKED` so concurrent reconcilers don't contend, then deleted in the same
/// transaction. Returns the number deleted. (pgvector mode: no external vector
/// store cleanup is needed.)
pub async fn cleanup_soft_deleted_documents_pgvector(
    pool: &PgPool,
    batch_size: i64,
    older_than_minutes: i64,
) -> Result<u64, sqlx::Error> {
    let cutoff = Utc::now() - chrono::Duration::minutes(older_than_minutes);
    let mut transaction = pool.begin().await?;
    let doc_ids: Vec<String> = sqlx::query_scalar(
        "SELECT id FROM documents \
         WHERE deleted_at IS NOT NULL AND deleted_at < $1 \
         LIMIT $2 FOR UPDATE SKIP LOCKED",
    )
    .bind(cutoff)
    .bind(batch_size)
    .fetch_all(&mut *transaction)
    .await?;

    if doc_ids.is_empty() {
        transaction.commit().await?;
        return Ok(0);
    }

    let result = sqlx::query("DELETE FROM documents WHERE id = ANY($1)")
        .bind(&doc_ids)
        .execute(&mut *transaction)
        .await?;
    let deleted = result.rows_affected();
    transaction.commit().await?;
    Ok(deleted)
}

/// A pending `message_embeddings` row claimed by the reconciler. `has_embedding`
/// is `embedding IS NOT NULL` — the vector value itself is never read back (a row
/// that already has a vector just needs marking `synced`; a row without one is
/// re-embedded from `content`).
#[derive(Debug, Clone)]
pub struct PendingMessageEmbedding {
    pub id: i64,
    pub content: String,
    pub has_embedding: bool,
    pub message_id: String,
    pub workspace_name: String,
    pub sync_attempts: i32,
}

/// Port of `reconciler.sync_vectors._get_message_embeddings_needing_sync`: claim
/// pending message-embedding rows on the given connection (so the caller can hold
/// the `FOR UPDATE SKIP LOCKED` rows through the embed call, matching Python's
/// single-session cycle). Step 1 picks up to `batch_size` distinct `message_id`s
/// that have at least one backoff-eligible pending row, oldest `last_sync_at`
/// first; step 2 claims ALL eligible pending rows for those messages so a
/// message's chunks are always processed together. `backoff_minutes` is the flat
/// retry wait (Python `SYNC_BACKOFF`).
pub async fn claim_pending_message_embeddings(
    conn: &mut sqlx::PgConnection,
    batch_size: i64,
    backoff_minutes: i64,
) -> Result<Vec<PendingMessageEmbedding>, sqlx::Error> {
    let backoff_cutoff = Utc::now() - chrono::Duration::minutes(backoff_minutes);

    // Step 1: distinct message_ids with an eligible pending row, oldest first.
    let message_ids: Vec<String> = sqlx::query_scalar(
        "SELECT message_id \
         FROM message_embeddings \
         WHERE sync_state = 'pending' \
           AND (last_sync_at IS NULL OR last_sync_at < $1) \
         GROUP BY message_id \
         ORDER BY MIN(last_sync_at) ASC NULLS FIRST \
         LIMIT $2",
    )
    .bind(backoff_cutoff)
    .bind(batch_size)
    .fetch_all(&mut *conn)
    .await?;

    if message_ids.is_empty() {
        return Ok(Vec::new());
    }

    // Step 2: claim all eligible pending rows for those messages.
    let rows = sqlx::query(
        "SELECT id, content, (embedding IS NOT NULL) AS has_embedding, \
                message_id, workspace_name, sync_attempts \
         FROM message_embeddings \
         WHERE message_id = ANY($1) \
           AND sync_state = 'pending' \
           AND (last_sync_at IS NULL OR last_sync_at < $2) \
         ORDER BY message_id, id \
         FOR UPDATE SKIP LOCKED",
    )
    .bind(&message_ids)
    .bind(backoff_cutoff)
    .fetch_all(&mut *conn)
    .await?;

    Ok(rows
        .into_iter()
        .map(|row| PendingMessageEmbedding {
            id: row.get("id"),
            content: row.get("content"),
            has_embedding: row.get("has_embedding"),
            message_id: row.get("message_id"),
            workspace_name: row.get("workspace_name"),
            sync_attempts: row.get("sync_attempts"),
        })
        .collect())
}

/// Mark a message-embedding row `synced` (Python's per-row UPDATE in
/// `_sync_message_embeddings`, pgvector branch): set `sync_state='synced'`,
/// `last_sync_at=now()`, `sync_attempts=0`, and write `fresh_embedding` when the
/// row was just re-embedded (a pre-existing vector is left untouched).
pub async fn mark_message_embedding_synced(
    conn: &mut sqlx::PgConnection,
    id: i64,
    fresh_embedding: Option<&[f32]>,
) -> Result<(), sqlx::Error> {
    match fresh_embedding {
        Some(embedding) => {
            sqlx::query(
                "UPDATE message_embeddings \
                 SET sync_state = 'synced', last_sync_at = now(), sync_attempts = 0, \
                     embedding = $2::vector \
                 WHERE id = $1",
            )
            .bind(id)
            .bind(crate::search::vector_literal(embedding))
            .execute(&mut *conn)
            .await?;
        }
        None => {
            sqlx::query(
                "UPDATE message_embeddings \
                 SET sync_state = 'synced', last_sync_at = now(), sync_attempts = 0 \
                 WHERE id = $1",
            )
            .bind(id)
            .execute(&mut *conn)
            .await?;
        }
    }
    Ok(())
}

/// Port of `reconciler.sync_vectors._bump_message_embedding_sync_attempts` for one
/// row: increment `sync_attempts`, set `last_sync_at=now()`, and flip to `failed`
/// once attempts reach `max_attempts` (else stays `pending` for a later retry).
pub async fn bump_message_embedding_sync_attempts(
    conn: &mut sqlx::PgConnection,
    id: i64,
    current_attempts: i32,
    max_attempts: i32,
) -> Result<(), sqlx::Error> {
    let new_attempts = current_attempts + 1;
    let new_state = if new_attempts >= max_attempts {
        "failed"
    } else {
        "pending"
    };
    sqlx::query(
        "UPDATE message_embeddings \
         SET sync_state = $2, sync_attempts = $3, last_sync_at = now() \
         WHERE id = $1",
    )
    .bind(id)
    .bind(new_state)
    .bind(new_attempts)
    .execute(&mut *conn)
    .await?;
    Ok(())
}

/// Port of `enqueue_dream` (the `schedule_dream` route's manual path): enqueue
/// a `dream` queue item for `(observer, observed)`, deduplicated against an
/// in-progress `active_queue_sessions` row or a pending (`processed = false`)
/// queue item with the same work_unit_key. A skipped enqueue is a no-op (the
/// route still returns 204). Payload mirrors `DreamPayload` model_dump with the
/// manual `trigger_reason`/`delay_reason` sentinels.
pub async fn enqueue_dream(
    pool: &PgPool,
    workspace_name: &str,
    observer: &str,
    observed: &str,
    dream_type: &str,
    session_name: Option<&str>,
) -> Result<(), sqlx::Error> {
    let work_unit_key = format!("dream:{dream_type}:{workspace_name}:{observer}:{observed}");

    let mut payload = serde_json::Map::from_iter([
        ("task_type".to_string(), Value::String("dream".to_string())),
        ("dream_type".to_string(), Value::String(dream_type.to_string())),
        ("observer".to_string(), Value::String(observer.to_string())),
        ("observed".to_string(), Value::String(observed.to_string())),
    ]);
    if let Some(session) = session_name {
        payload.insert("session_name".to_string(), Value::String(session.to_string()));
    }
    payload.insert("trigger_reason".to_string(), Value::String("manual".to_string()));
    payload.insert("delay_reason".to_string(), Value::String("immediate".to_string()));

    insert_dream_if_absent(pool, workspace_name, &work_unit_key, Value::Object(payload)).await?;
    Ok(())
}

/// Port of `check_and_schedule_dream`'s threshold-triggered enqueue (the body of
/// `DreamScheduler.execute_dream`). Like [`enqueue_dream`] but carries the
/// scheduling-attribution payload fields (`trigger_reason="document_threshold"`,
/// the resolved `delay_reason`, and the count snapshots) so `DreamRunEvent` can
/// attribute the dream back to its trigger. `session_name` is required here (the
/// caller knows it from the save context). Returns whether a row was inserted.
#[allow(clippy::too_many_arguments)]
pub async fn enqueue_scheduled_dream(
    pool: &PgPool,
    workspace_name: &str,
    observer: &str,
    observed: &str,
    dream_type: &str,
    session_name: &str,
    trigger_reason: &str,
    delay_reason: &str,
    documents_since_last_dream_at_schedule: i64,
    document_threshold: i64,
) -> Result<bool, sqlx::Error> {
    let work_unit_key = format!("dream:{dream_type}:{workspace_name}:{observer}:{observed}");

    // `model_dump(mode="json", exclude_none=True)` — session_name is always set here.
    let payload = serde_json::Map::from_iter([
        ("task_type".to_string(), Value::String("dream".to_string())),
        ("dream_type".to_string(), Value::String(dream_type.to_string())),
        ("observer".to_string(), Value::String(observer.to_string())),
        ("observed".to_string(), Value::String(observed.to_string())),
        ("session_name".to_string(), Value::String(session_name.to_string())),
        ("trigger_reason".to_string(), Value::String(trigger_reason.to_string())),
        ("delay_reason".to_string(), Value::String(delay_reason.to_string())),
        (
            "documents_since_last_dream_at_schedule".to_string(),
            Value::from(documents_since_last_dream_at_schedule),
        ),
        ("document_threshold".to_string(), Value::from(document_threshold)),
    ]);

    insert_dream_if_absent(pool, workspace_name, &work_unit_key, Value::Object(payload)).await
}

/// Insert a `dream` queue item unless one is already in-progress (an
/// `active_queue_sessions` row) or pending (`processed = false`) for the same
/// `work_unit_key` — the dedup shared by [`enqueue_dream`] and
/// [`enqueue_scheduled_dream`], mirroring Python's `enqueue_dream`. Returns
/// whether a row was inserted.
async fn insert_dream_if_absent(
    pool: &PgPool,
    workspace_name: &str,
    work_unit_key: &str,
    payload: Value,
) -> Result<bool, sqlx::Error> {
    let in_progress: bool = sqlx::query_scalar(
        "SELECT EXISTS(SELECT 1 FROM active_queue_sessions WHERE work_unit_key = $1)",
    )
    .bind(work_unit_key)
    .fetch_one(pool)
    .await?;
    if in_progress {
        return Ok(false);
    }

    let pending: bool = sqlx::query_scalar(
        "SELECT EXISTS(SELECT 1 FROM queue WHERE work_unit_key = $1 AND processed = false)",
    )
    .bind(work_unit_key)
    .fetch_one(pool)
    .await?;
    if pending {
        return Ok(false);
    }

    sqlx::query(
        "INSERT INTO queue \
         (work_unit_key, payload, session_id, task_type, workspace_name, message_id) \
         VALUES ($1, $2, NULL, 'dream', $3, NULL)",
    )
    .bind(work_unit_key)
    .bind(payload)
    .bind(workspace_name)
    .execute(pool)
    .await?;
    Ok(true)
}

/// Count live-or-deleted `explicit`-level documents for `(workspace, observer,
/// observed)` — the threshold input for `check_and_schedule_dream`. Matches
/// [`record_dream_guard`]'s count exactly (no `deleted_at` filter) so the
/// schedule baseline and the guard baseline stay consistent.
pub async fn count_explicit_documents(
    pool: &PgPool,
    workspace_name: &str,
    observer: &str,
    observed: &str,
) -> Result<i64, sqlx::Error> {
    sqlx::query_scalar(
        "SELECT COUNT(*) FROM documents \
         WHERE workspace_name = $1 AND observer = $2 AND observed = $3 \
           AND level = 'explicit'",
    )
    .bind(workspace_name)
    .bind(observer)
    .bind(observed)
    .fetch_one(pool)
    .await
}

/// Whether any unprocessed `dream` queue item exists for one of `work_unit_keys`
/// — the in-flight check of `check_and_schedule_dream` (mirrors
/// `uq_queue_dream_pending_work_unit_key`). `false` for an empty key list.
pub async fn any_dream_pending(
    pool: &PgPool,
    work_unit_keys: &[String],
) -> Result<bool, sqlx::Error> {
    if work_unit_keys.is_empty() {
        return Ok(false);
    }
    sqlx::query_scalar(
        "SELECT EXISTS(SELECT 1 FROM queue \
         WHERE task_type = 'dream' AND processed = false \
           AND work_unit_key = ANY($1))",
    )
    .bind(work_unit_keys)
    .fetch_one(pool)
    .await
}

/// Port of `reconciler.scheduler.ReconcilerScheduler._try_enqueue_task`: enqueue a
/// `reconciler` queue item for `work_unit_key`, deduplicated against an in-progress
/// `active_queue_sessions` row or a pending (`processed = false`) queue item with
/// the same key. The payload is just `{"reconciler_type": <name>}` (no
/// `task_type` field, matching the scheduler), `workspace_name`/`message_id`/
/// `session_id` all NULL. Returns whether an item was enqueued.
pub async fn try_enqueue_reconciler_task(
    pool: &PgPool,
    work_unit_key: &str,
    reconciler_type: &str,
) -> Result<bool, sqlx::Error> {
    let in_progress: bool = sqlx::query_scalar(
        "SELECT EXISTS(SELECT 1 FROM active_queue_sessions WHERE work_unit_key = $1)",
    )
    .bind(work_unit_key)
    .fetch_one(pool)
    .await?;
    if in_progress {
        return Ok(false);
    }

    let pending: bool = sqlx::query_scalar(
        "SELECT EXISTS(SELECT 1 FROM queue WHERE work_unit_key = $1 AND processed = false)",
    )
    .bind(work_unit_key)
    .fetch_one(pool)
    .await?;
    if pending {
        return Ok(false);
    }

    let payload = serde_json::json!({ "reconciler_type": reconciler_type });
    sqlx::query(
        "INSERT INTO queue \
         (work_unit_key, payload, session_id, task_type, workspace_name, message_id) \
         VALUES ($1, $2, NULL, 'reconciler', NULL, NULL)",
    )
    .bind(work_unit_key)
    .bind(payload)
    .execute(pool)
    .await?;
    Ok(true)
}

/// Enqueue a `webhook` queue item carrying a pre-built webhook payload (Python
/// `webhooks.events.publish_webhook_event`). Used by the worker's `queue.empty`
/// notification. Unconditional insert (no dedup, matching Python — it just adds a
/// row); `work_unit_key` is the `webhook:{workspace}` form, session_id/message_id
/// are NULL.
pub async fn enqueue_webhook_event(
    pool: &PgPool,
    workspace_name: &str,
    payload: &Value,
) -> Result<(), sqlx::Error> {
    let work_unit_key = format!("webhook:{workspace_name}");
    sqlx::query(
        "INSERT INTO queue \
         (work_unit_key, payload, session_id, task_type, workspace_name, message_id) \
         VALUES ($1, $2, NULL, 'webhook', $3, NULL)",
    )
    .bind(&work_unit_key)
    .bind(payload)
    .bind(workspace_name)
    .execute(pool)
    .await?;
    Ok(())
}

/// Failure modes of `enqueue_workspace_deletion`, mapped to the same HTTP
/// statuses Python's `delete_workspace` produces.
#[derive(Debug)]
pub enum WorkspaceDeleteError {
    /// Workspace does not exist → 404 (`ResourceNotFoundException`).
    NotFound,
    /// Workspace still has active sessions → 409 (`ConflictException`).
    ActiveSessions,
    Database(sqlx::Error),
}

impl From<sqlx::Error> for WorkspaceDeleteError {
    fn from(error: sqlx::Error) -> Self {
        WorkspaceDeleteError::Database(error)
    }
}

/// Port of `delete_workspace`'s accept-and-enqueue path: verify the workspace
/// exists, refuse if any active session remains, then enqueue a single
/// `deletion` queue item (the deriver performs the cascading delete). All three
/// steps share one transaction, matching the Python handler's single commit.
pub async fn enqueue_workspace_deletion(
    pool: &PgPool,
    workspace_name: &str,
) -> Result<(), WorkspaceDeleteError> {
    let mut transaction = pool.begin().await?;

    let exists: bool =
        sqlx::query_scalar("SELECT EXISTS(SELECT 1 FROM workspaces WHERE name = $1)")
            .bind(workspace_name)
            .fetch_one(&mut *transaction)
            .await?;
    if !exists {
        return Err(WorkspaceDeleteError::NotFound);
    }

    let has_active_sessions: bool = sqlx::query_scalar(
        "SELECT EXISTS(SELECT 1 FROM sessions \
         WHERE workspace_name = $1 AND is_active = true)",
    )
    .bind(workspace_name)
    .fetch_one(&mut *transaction)
    .await?;
    if has_active_sessions {
        return Err(WorkspaceDeleteError::ActiveSessions);
    }

    // Mirror of `create_deletion_record`: payload is the `DeletionPayload`
    // model_dump (exclude_none — all fields present), work_unit_key is the
    // `construct_work_unit_key` deletion form, session_id/message_id are NULL.
    let payload = json!({
        "task_type": "deletion",
        "deletion_type": "workspace",
        "resource_id": workspace_name
    });
    let work_unit_key = format!("deletion:{workspace_name}:workspace:{workspace_name}");
    sqlx::query(
        "INSERT INTO queue \
         (work_unit_key, payload, session_id, task_type, workspace_name, message_id) \
         VALUES ($1, $2, NULL, 'deletion', $3, NULL)",
    )
    .bind(&work_unit_key)
    .bind(&payload)
    .bind(workspace_name)
    .execute(&mut *transaction)
    .await?;

    transaction.commit().await?;
    Ok(())
}

/// Cascade counts from [`hard_delete_workspace`] (Python `WorkspaceDeletionResult`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkspaceDeletionCounts {
    pub peers_deleted: i64,
    pub sessions_deleted: i64,
    pub messages_deleted: i64,
    pub conclusions_deleted: i64,
}

/// Cascade counts from [`hard_delete_session`] (Python `SessionDeletionResult`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SessionDeletionCounts {
    pub messages_deleted: i64,
    pub conclusions_deleted: i64,
}

/// Failure modes of the worker hard-delete crud.
#[derive(Debug)]
pub enum HardDeleteError {
    /// The target resource does not exist (Python `ResourceNotFoundException`).
    NotFound,
    Database(sqlx::Error),
}

impl From<sqlx::Error> for HardDeleteError {
    fn from(error: sqlx::Error) -> Self {
        HardDeleteError::Database(error)
    }
}

/// Port of the worker-side `crud.delete_workspace` (`src/crud/workspace.py`): the
/// hard, cascading workspace delete the deriver's deletion task performs (distinct
/// from the API's [`enqueue_workspace_deletion`] accept-and-enqueue). Snapshots the
/// cascade counts, then deletes every child in FK-safe order within one
/// transaction. The external-vector-store namespace and cache invalidations are
/// no-ops on the default pgvector path and are not ported.
pub async fn hard_delete_workspace(
    pool: &PgPool,
    workspace_name: &str,
) -> Result<WorkspaceDeletionCounts, HardDeleteError> {
    let mut tx = pool.begin().await?;

    let exists: bool =
        sqlx::query_scalar("SELECT EXISTS(SELECT 1 FROM workspaces WHERE name = $1)")
            .bind(workspace_name)
            .fetch_one(&mut *tx)
            .await?;
    if !exists {
        return Err(HardDeleteError::NotFound);
    }

    // Counts captured before deletion (telemetry snapshot).
    let count_in = |sql: &'static str| async move {
        sqlx::query_scalar::<_, i64>(sql)
            .bind(workspace_name)
            .fetch_one(pool)
            .await
    };
    let peers_deleted = count_in("SELECT COUNT(*) FROM peers WHERE workspace_name = $1").await?;
    let sessions_deleted =
        count_in("SELECT COUNT(*) FROM sessions WHERE workspace_name = $1").await?;
    let messages_deleted =
        count_in("SELECT COUNT(*) FROM messages WHERE workspace_name = $1").await?;
    let conclusions_deleted =
        count_in("SELECT COUNT(*) FROM documents WHERE workspace_name = $1").await?;

    // Cascade deletes in dependency order (mirrors the Python sequence).
    for sql in [
        "DELETE FROM active_queue_sessions WHERE split_part(work_unit_key, ':', 2) = $1",
        "DELETE FROM queue WHERE workspace_name = $1",
        "DELETE FROM queue WHERE message_id IN (SELECT id FROM messages WHERE workspace_name = $1)",
        "DELETE FROM message_embeddings WHERE workspace_name = $1",
        "DELETE FROM documents WHERE workspace_name = $1",
        "DELETE FROM collections WHERE workspace_name = $1",
        "DELETE FROM messages WHERE workspace_name = $1",
        "DELETE FROM webhook_endpoints WHERE workspace_name = $1",
        "DELETE FROM session_peers WHERE workspace_name = $1",
        "DELETE FROM sessions WHERE workspace_name = $1",
        "DELETE FROM peers WHERE workspace_name = $1",
        "DELETE FROM workspaces WHERE name = $1",
    ] {
        sqlx::query(sql).bind(workspace_name).execute(&mut *tx).await?;
    }

    tx.commit().await?;
    Ok(WorkspaceDeletionCounts {
        peers_deleted,
        sessions_deleted,
        messages_deleted,
        conclusions_deleted,
    })
}

/// Port of the worker-side `crud.delete_session` (`src/crud/session.py`): the hard,
/// cascading session delete (distinct from the API's [`delete_session`] soft
/// is_active=false + enqueue). Resolves the session id (including inactive), then
/// cascade-deletes in FK-safe order within one transaction, returning the message
/// and conclusion counts. External-vector-store deletes are no-ops on pgvector,
/// and the per-5000 batching is collapsed into single DELETEs (semantically identical).
pub async fn hard_delete_session(
    pool: &PgPool,
    workspace_name: &str,
    session_name: &str,
) -> Result<SessionDeletionCounts, HardDeleteError> {
    let mut tx = pool.begin().await?;

    let session_id: Option<String> =
        sqlx::query_scalar("SELECT id FROM sessions WHERE workspace_name = $1 AND name = $2")
            .bind(workspace_name)
            .bind(session_name)
            .fetch_optional(&mut *tx)
            .await?;
    let session_id = session_id.ok_or(HardDeleteError::NotFound)?;

    sqlx::query(
        "DELETE FROM active_queue_sessions \
         WHERE split_part(work_unit_key, ':', 2) = $1 AND split_part(work_unit_key, ':', 3) = $2",
    )
    .bind(workspace_name)
    .bind(session_name)
    .execute(&mut *tx)
    .await?;

    sqlx::query("DELETE FROM queue WHERE session_id = $1")
        .bind(&session_id)
        .execute(&mut *tx)
        .await?;

    sqlx::query("DELETE FROM message_embeddings WHERE session_name = $1 AND workspace_name = $2")
        .bind(session_name)
        .bind(workspace_name)
        .execute(&mut *tx)
        .await?;

    let conclusions_deleted =
        sqlx::query("DELETE FROM documents WHERE session_name = $1 AND workspace_name = $2")
            .bind(session_name)
            .bind(workspace_name)
            .execute(&mut *tx)
            .await?
            .rows_affected() as i64;

    let messages_deleted =
        sqlx::query("DELETE FROM messages WHERE session_name = $1 AND workspace_name = $2")
            .bind(session_name)
            .bind(workspace_name)
            .execute(&mut *tx)
            .await?
            .rows_affected() as i64;

    sqlx::query("DELETE FROM session_peers WHERE session_name = $1 AND workspace_name = $2")
        .bind(session_name)
        .bind(workspace_name)
        .execute(&mut *tx)
        .await?;

    sqlx::query("DELETE FROM sessions WHERE workspace_name = $1 AND name = $2")
        .bind(workspace_name)
        .bind(session_name)
        .execute(&mut *tx)
        .await?;

    tx.commit().await?;
    Ok(SessionDeletionCounts {
        messages_deleted,
        conclusions_deleted,
    })
}

/// Port of `crud.delete_document_by_id` (`src/crud/document.py`): SOFT-delete a
/// document (set `deleted_at = now()`); the reconciler later hard-deletes it.
/// Returns `true` when a row was marked, `false` when none matched (already
/// deleted / wrong workspace) — the caller treats the latter as idempotent
/// success, matching how `process_deletion` swallows `ResourceNotFoundException`.
pub async fn mark_document_deleted(
    pool: &PgPool,
    workspace_name: &str,
    document_id: &str,
) -> Result<bool, sqlx::Error> {
    let affected = sqlx::query(
        "UPDATE documents SET deleted_at = now() \
         WHERE id = $1 AND workspace_name = $2 AND deleted_at IS NULL",
    )
    .bind(document_id)
    .bind(workspace_name)
    .execute(pool)
    .await?
    .rows_affected();
    Ok(affected > 0)
}

/// Port of `crud.delete_documents` (document.py:672): soft-delete multiple
/// documents in one `UPDATE ... RETURNING`, scoped to `(workspace, observer,
/// observed)` and (optionally) session. Returns `(id, level)` for the rows that
/// were actually deleted (matched the filter and weren't already soft-deleted);
/// callers diff against the input ids to detect misses. `level` is nullable in
/// the schema, so it surfaces as `Option<String>`.
pub async fn delete_documents(
    pool: &PgPool,
    workspace_name: &str,
    document_ids: &[String],
    observer: &str,
    observed: &str,
    session_name: Option<&str>,
) -> Result<Vec<(String, Option<String>)>, sqlx::Error> {
    if document_ids.is_empty() {
        return Ok(Vec::new());
    }
    let session_clause = if session_name.is_some() {
        " AND session_name = $5"
    } else {
        ""
    };
    let sql = format!(
        "UPDATE documents SET deleted_at = now() \
         WHERE id = ANY($1) AND workspace_name = $2 AND observer = $3 AND observed = $4 \
           AND deleted_at IS NULL{session_clause} \
         RETURNING id, level"
    );
    let mut query = sqlx::query_as::<_, (String, Option<String>)>(&sql)
        .bind(document_ids)
        .bind(workspace_name)
        .bind(observer)
        .bind(observed);
    if let Some(session) = session_name {
        query = query.bind(session);
    }
    query.fetch_all(pool).await
}

async fn fetch_count(pool: &PgPool, sql: &str, bindings: &[Value]) -> Result<i64, sqlx::Error> {
    fetch_count_with_tail(pool, sql, bindings, &[]).await
}

async fn fetch_count_with_tail(
    pool: &PgPool,
    sql: &str,
    bindings: &[Value],
    tail_bindings: &[&str],
) -> Result<i64, sqlx::Error> {
    let mut query = bind_values(sqlx::query(sql), bindings);
    for value in tail_bindings {
        query = query.bind(*value);
    }
    let row = query.fetch_one(pool).await?;
    row.try_get("count")
}

fn bind_values<'q>(
    mut query: Query<'q, Postgres, PgArguments>,
    bindings: &[Value],
) -> Query<'q, Postgres, PgArguments> {
    for value in bindings {
        query = bind_value(query, value);
    }
    query
}

pub(crate) fn bind_values_as<'q, O>(
    mut query: QueryAs<'q, Postgres, O, PgArguments>,
    bindings: &[Value],
) -> QueryAs<'q, Postgres, O, PgArguments>
where
    O: Send + Unpin + for<'r> FromRow<'r, PgRow>,
{
    for value in bindings {
        query = bind_value_as(query, value);
    }
    query
}

fn bind_value<'q>(
    query: Query<'q, Postgres, PgArguments>,
    value: &Value,
) -> Query<'q, Postgres, PgArguments> {
    match value {
        Value::Null => query.bind(Option::<String>::None),
        Value::Bool(value) => query.bind(*value),
        Value::Number(value) => {
            if let Some(value) = value.as_i64() {
                query.bind(value)
            } else if let Some(value) = value.as_f64() {
                query.bind(value)
            } else {
                query.bind(value.to_string())
            }
        }
        Value::String(value) => query.bind(value.clone()),
        Value::Array(_) | Value::Object(_) => query.bind(sqlx::types::Json(value.clone())),
    }
}

fn bind_value_as<'q, O>(
    query: QueryAs<'q, Postgres, O, PgArguments>,
    value: &Value,
) -> QueryAs<'q, Postgres, O, PgArguments>
where
    O: Send + Unpin + for<'r> FromRow<'r, PgRow>,
{
    match value {
        Value::Null => query.bind(Option::<String>::None),
        Value::Bool(value) => query.bind(*value),
        Value::Number(value) => {
            if let Some(value) = value.as_i64() {
                query.bind(value)
            } else if let Some(value) = value.as_f64() {
                query.bind(value)
            } else {
                query.bind(value.to_string())
            }
        }
        Value::String(value) => query.bind(value.clone()),
        Value::Array(_) | Value::Object(_) => query.bind(sqlx::types::Json(value.clone())),
    }
}

fn direction(reverse: bool) -> &'static str {
    if reverse { "DESC" } else { "ASC" }
}

fn workspace_json(row: WorkspaceRow) -> Value {
    json!({
        "id": row.name,
        "metadata": row.metadata,
        "configuration": row.configuration,
        "created_at": row.created_at
    })
}

fn peer_json(row: PeerRow) -> Value {
    json!({
        "id": row.name,
        "workspace_id": row.workspace_name,
        "created_at": row.created_at,
        "metadata": row.metadata,
        "configuration": row.configuration
    })
}

fn session_json(row: SessionRow) -> Value {
    json!({
        "id": row.name,
        "workspace_id": row.workspace_name,
        "is_active": row.is_active,
        "metadata": row.metadata,
        "configuration": row.configuration,
        "created_at": row.created_at
    })
}

fn message_json(row: MessageRow) -> Value {
    json!({
        "id": row.public_id,
        "content": row.content,
        "peer_id": row.peer_name,
        "session_id": row.session_name,
        "metadata": row.metadata,
        "created_at": row.created_at,
        "workspace_id": row.workspace_name,
        "token_count": row.token_count
    })
}

fn conclusion_json(row: ConclusionRow) -> Value {
    json!({
        "id": row.id,
        "content": row.content,
        "observer_id": row.observer,
        "observed_id": row.observed,
        "session_id": row.session_name,
        "created_at": row.created_at
    })
}

/// Mirror of Python's `settings.WEBHOOK.MAX_WORKSPACE_LIMIT` default (10).
pub const WEBHOOK_MAX_WORKSPACE_LIMIT: i64 = 10;

#[derive(Debug)]
pub enum WebhookCreateError {
    WorkspaceNotFound,
    LimitReached,
    Database(sqlx::Error),
}

impl From<sqlx::Error> for WebhookCreateError {
    fn from(error: sqlx::Error) -> Self {
        WebhookCreateError::Database(error)
    }
}

/// Get-or-create a webhook endpoint, porting `crud.get_or_create_webhook_endpoint`.
///
/// The workspace must already exist (Python uses `get_workspace`, not the
/// get-or-create variant). An endpoint with a matching URL is returned as-is
/// (`created = false`); otherwise a new one is inserted unless the workspace is
/// already at `WEBHOOK_MAX_WORKSPACE_LIMIT`.
pub async fn get_or_create_webhook_endpoint(
    pool: &PgPool,
    workspace_name: &str,
    url: &str,
) -> Result<(Value, bool), WebhookCreateError> {
    let exists = sqlx::query("SELECT EXISTS(SELECT 1 FROM workspaces WHERE name = $1) AS exists")
        .bind(workspace_name)
        .fetch_one(pool)
        .await?;
    if !exists.try_get::<bool, _>("exists")? {
        return Err(WebhookCreateError::WorkspaceNotFound);
    }

    let existing = sqlx::query_as::<_, WebhookEndpointRow>(
        "SELECT id, workspace_name, url, created_at \
         FROM webhook_endpoints \
         WHERE workspace_name = $1",
    )
    .bind(workspace_name)
    .fetch_all(pool)
    .await?;

    if let Some(row) = existing.iter().find(|row| row.url == url) {
        return Ok((
            webhook_endpoint_json(
                row.id.clone(),
                row.workspace_name.clone(),
                row.url.clone(),
                row.created_at,
            ),
            false,
        ));
    }

    if existing.len() as i64 >= WEBHOOK_MAX_WORKSPACE_LIMIT {
        return Err(WebhookCreateError::LimitReached);
    }

    let row = sqlx::query_as::<_, WebhookEndpointRow>(
        "INSERT INTO webhook_endpoints (id, workspace_name, url) \
         VALUES ($1, $2, $3) \
         RETURNING id, workspace_name, url, created_at",
    )
    .bind(generate_nanoid())
    .bind(workspace_name)
    .bind(url)
    .fetch_one(pool)
    .await?;

    Ok((
        webhook_endpoint_json(row.id, row.workspace_name, row.url, row.created_at),
        true,
    ))
}

/// Delete a webhook endpoint, porting `crud.delete_webhook_endpoint`. Returns
/// `false` when no matching endpoint exists in the workspace.
pub async fn delete_webhook_endpoint(
    pool: &PgPool,
    workspace_name: &str,
    endpoint_id: &str,
) -> Result<bool, sqlx::Error> {
    let result = sqlx::query("DELETE FROM webhook_endpoints WHERE id = $1 AND workspace_name = $2")
        .bind(endpoint_id)
        .bind(workspace_name)
        .execute(pool)
        .await?;
    Ok(result.rows_affected() > 0)
}

pub fn webhook_endpoint_json(
    id: String,
    workspace_name: String,
    url: String,
    created_at: DateTime<Utc>,
) -> Value {
    json!({
        "id": id,
        "workspace_id": workspace_name,
        "url": url,
        "created_at": created_at
    })
}

/// Set the peer card the `observer` holds for `observed`, porting
/// `crud.set_peer_card`. The card is merged into the observer peer's
/// `internal_metadata` under the `peer_card` / `{observed}_peer_card` label. The
/// observer is get-or-created first (so the row always exists), matching Python.
pub async fn set_peer_card(
    pool: &PgPool,
    workspace_name: &str,
    observer: &str,
    observed: &str,
    peer_card: &[String],
) -> Result<(), sqlx::Error> {
    get_or_create_peer(pool, workspace_name, observer, None, None).await?;

    let label = if observer == observed {
        "peer_card".to_string()
    } else {
        format!("{observed}_peer_card")
    };
    let patch = json!({ label: peer_card });

    sqlx::query(
        "UPDATE peers SET internal_metadata = internal_metadata || $3::jsonb \
         WHERE workspace_name = $1 AND name = $2",
    )
    .bind(workspace_name)
    .bind(observer)
    .bind(patch)
    .execute(pool)
    .await?;
    Ok(())
}

pub fn peer_card_json(observer: &str, observed: &str, internal_metadata: &Value) -> Value {
    let key = if observer == observed {
        "peer_card".to_string()
    } else {
        format!("{observed}_peer_card")
    };
    let peer_card = internal_metadata
        .get(key)
        .and_then(|value| match value {
            Value::Array(items) if items.iter().all(Value::is_string) => Some(value.clone()),
            _ => None,
        })
        .unwrap_or(Value::Null);

    json!({ "peer_card": peer_card })
}

pub fn session_summaries_json(
    session_name: &str,
    internal_metadata: &Value,
) -> Result<Value, sqlx::Error> {
    let summaries = internal_metadata
        .get("summaries")
        .and_then(Value::as_object);

    Ok(json!({
        "id": session_name,
        "short_summary": summaries
            .and_then(|items| items.get("honcho_chat_summary_short"))
            .map(schema_summary_json)
            .transpose()?,
        "long_summary": summaries
            .and_then(|items| items.get("honcho_chat_summary_long"))
            .map(schema_summary_json)
            .transpose()?
    }))
}

fn schema_summary_json(summary: &Value) -> Result<Value, sqlx::Error> {
    let summary = summary
        .as_object()
        .ok_or_else(|| invalid_summary_error("summary entry must be an object"))?;

    require_i64(summary, "message_id")?;
    let content = require_string(summary, "content")?;
    let summary_type = require_string(summary, "summary_type")?;
    let created_at = require_string(summary, "created_at")?;
    let token_count = require_i64(summary, "token_count")?;
    let message_public_id = match summary.get("message_public_id") {
        Some(value) => value
            .as_str()
            .ok_or_else(|| invalid_summary_error("summary.message_public_id must be a string"))?,
        None => "",
    };

    Ok(json!({
        "content": content,
        "message_id": message_public_id,
        "summary_type": summary_type,
        "created_at": created_at,
        "token_count": token_count
    }))
}

fn require_string<'a>(
    object: &'a serde_json::Map<String, Value>,
    key: &str,
) -> Result<&'a str, sqlx::Error> {
    object
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| invalid_summary_error(format!("summary.{key} must be a string")))
}

fn require_i64(object: &serde_json::Map<String, Value>, key: &str) -> Result<i64, sqlx::Error> {
    object
        .get(key)
        .and_then(Value::as_i64)
        .ok_or_else(|| invalid_summary_error(format!("summary.{key} must be an integer")))
}

fn invalid_summary_error(message: impl Into<String>) -> sqlx::Error {
    sqlx::Error::Decode(Box::new(io::Error::new(
        io::ErrorKind::InvalidData,
        message.into(),
    )))
}
