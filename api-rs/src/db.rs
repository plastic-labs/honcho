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

#[derive(Debug, FromRow)]
struct ConclusionRow {
    id: String,
    content: String,
    observer: String,
    observed: String,
    session_name: Option<String>,
    created_at: DateTime<Utc>,
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

fn bind_values_as<'q, O>(
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
