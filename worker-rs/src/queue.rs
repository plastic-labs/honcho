//! Queue state-transition primitives, ported from
//! `src/deriver/queue_manager.py`. See
//! `docs/superpowers/plans/2026-06-16-rust-queue-schema-reference.md` for the
//! authoritative contract.
//!
//! State is modelled exactly as Python's: a `queue` row's `(processed, error)`
//! plus the existence of an `active_queue_sessions` row keyed by `work_unit_key`.
//! There is no status column. Claiming a work unit is `INSERT … ON CONFLICT DO
//! NOTHING` on the unique `work_unit_key`; that uniqueness is the entire
//! cross-worker mutex.

use chrono::{DateTime, Duration as ChronoDuration, Utc};
use rand::Rng;
use serde_json::Value;
use sqlx::{FromRow, PgPool, Row};
use std::time::Duration;

const NANOID_ALPHABET: &[u8] = b"_-0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
const NANOID_LENGTH: usize = 21;

/// TEXT column limit Python truncates errors to (`error[:65535]`).
const ERROR_MAX_LEN: usize = 65535;

fn generate_nanoid() -> String {
    let mut rng = rand::thread_rng();
    (0..NANOID_LENGTH)
        .map(|_| {
            let index = rng.gen_range(0..NANOID_ALPHABET.len());
            NANOID_ALPHABET[index] as char
        })
        .collect()
}

/// A claimed work unit: the lane key plus the `active_queue_sessions.id` token
/// the worker now owns.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClaimedWorkUnit {
    pub work_unit_key: String,
    pub aqs_id: String,
}

/// A single queue row the consumer processes.
#[derive(Debug, Clone, FromRow)]
pub struct QueueItem {
    pub id: i64,
    pub work_unit_key: String,
    pub task_type: String,
    pub payload: Value,
    pub message_id: Option<i64>,
    pub session_id: Option<String>,
}

/// Claim up to `limit` work units that have unprocessed items and no active
/// session row.
///
/// NOTE: the representation token-threshold gate from Python's
/// `get_and_claim_work_units` (`SUM(token_count) >= REPRESENTATION_BATCH_MAX_TOKENS`
/// unless `FLUSH_ENABLED`) is intentionally NOT applied here — representation
/// batching lands with the deriver port (Phase 5). For the no-op milestone every
/// work unit with pending items is eligible. Non-representation tasks were always
/// unconditionally eligible in Python, so their behavior already matches.
pub async fn claim_work_units(
    pool: &PgPool,
    limit: i64,
) -> Result<Vec<ClaimedWorkUnit>, sqlx::Error> {
    if limit <= 0 {
        return Ok(Vec::new());
    }

    let mut transaction = pool.begin().await?;

    let candidate_rows = sqlx::query(
        "SELECT wu.work_unit_key AS work_unit_key \
         FROM ( \
             SELECT work_unit_key FROM queue \
             WHERE NOT processed \
             GROUP BY work_unit_key \
         ) wu \
         WHERE NOT EXISTS ( \
             SELECT 1 FROM active_queue_sessions a \
             WHERE a.work_unit_key = wu.work_unit_key \
         ) \
         LIMIT $1",
    )
    .bind(limit)
    .fetch_all(&mut *transaction)
    .await?;

    if candidate_rows.is_empty() {
        transaction.commit().await?;
        return Ok(Vec::new());
    }

    let keys: Vec<String> = candidate_rows
        .iter()
        .map(|row| row.get::<String, _>("work_unit_key"))
        .collect();
    let ids: Vec<String> = keys.iter().map(|_| generate_nanoid()).collect();

    // Bulk INSERT … ON CONFLICT DO NOTHING is the cross-worker mutex: only rows
    // that actually insert are "won".
    let claimed_rows = sqlx::query(
        "INSERT INTO active_queue_sessions (id, work_unit_key) \
         SELECT * FROM UNNEST($1::text[], $2::text[]) \
         ON CONFLICT (work_unit_key) DO NOTHING \
         RETURNING work_unit_key, id",
    )
    .bind(&ids)
    .bind(&keys)
    .fetch_all(&mut *transaction)
    .await?;

    transaction.commit().await?;

    Ok(claimed_rows
        .into_iter()
        .map(|row| ClaimedWorkUnit {
            work_unit_key: row.get("work_unit_key"),
            aqs_id: row.get("id"),
        })
        .collect())
}

/// Fetch the next unprocessed queue item for a work unit the caller owns
/// (verified by joining on the `active_queue_sessions` row id), lowest `id`
/// first. Returns `None` when the unit is drained or ownership was lost.
pub async fn next_unprocessed_item(
    pool: &PgPool,
    work_unit_key: &str,
    aqs_id: &str,
) -> Result<Option<QueueItem>, sqlx::Error> {
    sqlx::query_as::<_, QueueItem>(
        "SELECT q.id, q.work_unit_key, q.task_type, q.payload, q.message_id, q.session_id \
         FROM queue q \
         JOIN active_queue_sessions a ON q.work_unit_key = a.work_unit_key \
         WHERE q.work_unit_key = $1 \
         AND NOT q.processed \
         AND a.id = $2 \
         ORDER BY q.id \
         LIMIT 1",
    )
    .bind(work_unit_key)
    .bind(aqs_id)
    .fetch_optional(pool)
    .await
}

/// Mark items processed and heartbeat the work unit's active session.
pub async fn mark_items_processed(
    pool: &PgPool,
    item_ids: &[i64],
    work_unit_key: &str,
) -> Result<(), sqlx::Error> {
    if item_ids.is_empty() {
        return Ok(());
    }
    let mut transaction = pool.begin().await?;
    sqlx::query(
        "UPDATE queue SET processed = true \
         WHERE id = ANY($1) AND work_unit_key = $2",
    )
    .bind(item_ids)
    .bind(work_unit_key)
    .execute(&mut *transaction)
    .await?;
    heartbeat(&mut transaction, work_unit_key).await?;
    transaction.commit().await?;
    Ok(())
}

/// Mark a single item terminally errored (processed=true + error), heartbeating
/// the active session. Mirrors Python: only the first failing item of a batch is
/// errored; remaining items stay pending and are retried on the next loop. There
/// is no retry counter — the errored item itself is terminal.
pub async fn mark_item_errored(
    pool: &PgPool,
    item_id: i64,
    work_unit_key: &str,
    error: &str,
) -> Result<(), sqlx::Error> {
    let truncated: String = error.chars().take(ERROR_MAX_LEN).collect();
    let mut transaction = pool.begin().await?;
    sqlx::query(
        "UPDATE queue SET processed = true, error = $3 \
         WHERE id = $1 AND work_unit_key = $2",
    )
    .bind(item_id)
    .bind(work_unit_key)
    .bind(&truncated)
    .execute(&mut *transaction)
    .await?;
    heartbeat(&mut transaction, work_unit_key).await?;
    transaction.commit().await?;
    Ok(())
}

async fn heartbeat(
    transaction: &mut sqlx::Transaction<'_, sqlx::Postgres>,
    work_unit_key: &str,
) -> Result<(), sqlx::Error> {
    sqlx::query(
        "UPDATE active_queue_sessions SET last_updated = now() WHERE work_unit_key = $1",
    )
    .bind(work_unit_key)
    .execute(&mut **transaction)
    .await?;
    Ok(())
}

/// Release a work unit by deleting its active session row. Returns whether a row
/// was actually deleted (false when another worker/stale-sweep already removed it).
pub async fn release_work_unit(
    pool: &PgPool,
    aqs_id: &str,
    work_unit_key: &str,
) -> Result<bool, sqlx::Error> {
    let result = sqlx::query(
        "DELETE FROM active_queue_sessions WHERE id = $1 AND work_unit_key = $2",
    )
    .bind(aqs_id)
    .bind(work_unit_key)
    .execute(pool)
    .await?;
    Ok(result.rows_affected() > 0)
}

/// Recover crashed/stalled work units: delete active sessions whose heartbeat is
/// older than `timeout`, using `FOR UPDATE SKIP LOCKED` so concurrent sweepers do
/// not contend. The queue rows keep `processed=false`, so the work unit returns to
/// the pending pool. Returns the number of sessions reclaimed.
pub async fn cleanup_stale_work_units(
    pool: &PgPool,
    timeout: Duration,
) -> Result<u64, sqlx::Error> {
    let cutoff: DateTime<Utc> = Utc::now()
        - ChronoDuration::from_std(timeout).unwrap_or_else(|_| ChronoDuration::zero());
    let result = sqlx::query(
        "DELETE FROM active_queue_sessions \
         WHERE id IN ( \
             SELECT id FROM active_queue_sessions \
             WHERE last_updated < $1 \
             ORDER BY last_updated \
             FOR UPDATE SKIP LOCKED \
         )",
    )
    .bind(cutoff)
    .execute(pool)
    .await?;
    Ok(result.rows_affected())
}
