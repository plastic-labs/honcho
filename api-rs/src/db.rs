use chrono::{DateTime, Utc};
use rand::Rng;
use serde_json::{Value, json};
use sqlx::postgres::{PgArguments, PgRow};
use sqlx::query::{Query, QueryAs};
use sqlx::{FromRow, PgPool, Postgres, Row};
use std::collections::BTreeMap;
use std::io;

use crate::filters::FilterClause;
use crate::pagination::{Pagination, page_response};
use crate::producer::{PeerConfigEntry, QueueRecord};
use crate::queue_status::{QueueStatusCounts, build_queue_status};

const NANOID_ALPHABET: &[u8] = b"_-0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
const NANOID_LENGTH: usize = 21;
pub const SESSION_OBSERVERS_LIMIT: i64 = 10;

pub fn quote_identifier(value: &str) -> String {
    format!("\"{}\"", value.replace('"', "\"\""))
}

fn generate_nanoid() -> String {
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
    if token_limit <= 0 {
        return Ok(session_context_json(session_name, Value::Null, Vec::new()));
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

    Ok(session_context_json(session_name, summary_json, messages))
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

    let in_progress: bool = sqlx::query_scalar(
        "SELECT EXISTS(SELECT 1 FROM active_queue_sessions WHERE work_unit_key = $1)",
    )
    .bind(&work_unit_key)
    .fetch_one(pool)
    .await?;
    if in_progress {
        return Ok(());
    }

    let pending: bool = sqlx::query_scalar(
        "SELECT EXISTS(SELECT 1 FROM queue WHERE work_unit_key = $1 AND processed = false)",
    )
    .bind(&work_unit_key)
    .fetch_one(pool)
    .await?;
    if pending {
        return Ok(());
    }

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

    sqlx::query(
        "INSERT INTO queue \
         (work_unit_key, payload, session_id, task_type, workspace_name, message_id) \
         VALUES ($1, $2, NULL, 'dream', $3, NULL)",
    )
    .bind(&work_unit_key)
    .bind(Value::Object(payload))
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
