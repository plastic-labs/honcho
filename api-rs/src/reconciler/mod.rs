//! Port of `src/reconciler/` (pgvector mode): the queue-driven reconciler task
//! that embeds pending `message_embeddings` rows, cleans up soft-deleted
//! documents, and removes expired queue items.
//!
//! Honcho defers message embedding from `create_messages` to the reconciler —
//! rows are inserted `sync_state='pending'` with a NULL vector, then this task
//! re-embeds them. Only the default pgvector mode is ported; the external
//! vector-store branches (`turbopuffer`/`lancedb`) are not.
//!
//! NOTE (deliberate deviation from the "never hold a DB session during an
//! external call" guideline): the embed call runs while the claimed rows are
//! still locked under `FOR UPDATE SKIP LOCKED`, mirroring Python's
//! single-session cycle. Holding the lock through the embed is what prevents two
//! concurrent reconcilers from double-embedding the same chunks.

use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};
use sqlx::PgPool;

use crate::db::{self, PendingMessageEmbedding};
use crate::deriver::payload::{ReconcilerPayload, ReconcilerType};
use crate::dialectic::Embedder;
use crate::telemetry::Emitter;
use crate::telemetry::events::{CleanupStaleItemsCompletedEvent, SyncVectorsCompletedEvent};

/// After this many failed sync attempts, a row is marked `failed` (Python
/// `MAX_SYNC_ATTEMPTS`).
pub const MAX_SYNC_ATTEMPTS: i32 = 20;
/// Flat wait between sync attempts (Python `SYNC_BACKOFF` = 10 minutes).
pub const SYNC_BACKOFF_MINUTES: i64 = 10;
/// Rows claimed per batch (Python `RECONCILIATION_BATCH_SIZE`).
pub const RECONCILIATION_BATCH_SIZE: i64 = 50;
/// Wall-clock budget for one cycle (Python `RECONCILIATION_TIME_BUDGET_SECONDS`).
pub const RECONCILIATION_TIME_BUDGET_SECONDS: u64 = 240;
/// Soft-deleted documents older than this are cleaned up (Python default).
pub const SOFT_DELETE_CLEANUP_MINUTES: i64 = 5;

/// Port of `ReconciliationMetrics` (the pgvector subset — document *sync* counts
/// stay 0 since pgvector needs no external upsert).
#[derive(Debug, Default, Clone)]
pub struct ReconciliationMetrics {
    pub documents_synced: i64,
    pub documents_failed: i64,
    pub documents_cleaned: i64,
    pub message_embeddings_synced: i64,
    pub message_embeddings_failed: i64,
}

impl ReconciliationMetrics {
    pub fn total_synced(&self) -> i64 {
        self.documents_synced + self.message_embeddings_synced
    }
    pub fn total_failed(&self) -> i64 {
        self.documents_failed + self.message_embeddings_failed
    }
    pub fn total_cleaned(&self) -> i64 {
        self.documents_cleaned
    }
}

/// Port of `_reconcile_message_embeddings_batch` (pgvector branch of
/// `_sync_message_embeddings`): claim a batch of pending rows, re-embed those
/// missing a vector, persist the vector + mark `synced`, and bump failed rows'
/// attempt counters. Returns `(synced, failed, did_work)`; `did_work` is false
/// only when nothing was claimed (drives the cycle's break condition).
pub async fn reconcile_message_embeddings_batch<E: Embedder>(
    pool: &PgPool,
    embedder: &E,
) -> Result<(i64, i64, bool), sqlx::Error> {
    let mut tx = pool.begin().await?;
    let embs =
        db::claim_pending_message_embeddings(&mut tx, RECONCILIATION_BATCH_SIZE, SYNC_BACKOFF_MINUTES)
            .await?;
    if embs.is_empty() {
        tx.commit().await?;
        return Ok((0, 0, false));
    }

    let mut synced = 0i64;
    let mut failed = 0i64;

    // Step 1: re-embed rows missing a vector. The lock is held across this call
    // (see module note). On embed error we leave `fresh` empty so every
    // missing-vector row is counted as failed below.
    let needing: Vec<&PendingMessageEmbedding> =
        embs.iter().filter(|emb| !emb.has_embedding).collect();
    let mut fresh: std::collections::HashMap<i64, Vec<f32>> = std::collections::HashMap::new();
    if !needing.is_empty() {
        let contents: Vec<String> = needing.iter().map(|emb| emb.content.clone()).collect();
        match embedder.batch_embed(&contents).await {
            Ok(vectors) => {
                if vectors.len() != needing.len() {
                    tracing::warn!(
                        "Re-embedded {}/{} message embeddings; remaining will be retried",
                        vectors.len(),
                        needing.len()
                    );
                }
                // strict=False zip: surplus rows (no vector) drop to the retry path.
                for (emb, vector) in needing.iter().zip(vectors) {
                    fresh.insert(emb.id, vector);
                }
            }
            Err(error) => {
                tracing::error!(
                    "Failed to re-embed {} message embeddings: {error}",
                    needing.len()
                );
            }
        }
    }

    // Rows that failed to get a vector → bump attempts (may flip to `failed`).
    for emb in embs.iter().filter(|emb| !emb.has_embedding && !fresh.contains_key(&emb.id)) {
        db::bump_message_embedding_sync_attempts(
            &mut tx,
            emb.id,
            emb.sync_attempts,
            MAX_SYNC_ATTEMPTS,
        )
        .await?;
        failed += 1;
    }

    // Rows that now have an embedding (pre-existing or fresh) → mark synced.
    for emb in &embs {
        let fresh_emb = fresh.get(&emb.id);
        if fresh_emb.is_none() && !emb.has_embedding {
            continue;
        }
        db::mark_message_embedding_synced(&mut tx, emb.id, fresh_emb.map(Vec::as_slice)).await?;
        synced += 1;
    }

    tx.commit().await?;
    Ok((synced, failed, true))
}

/// Port of `_cleanup_pgvector_batch`: hard-delete one batch of soft-deleted
/// documents. Returns `(cleaned, did_work)`.
pub async fn cleanup_pgvector_batch(pool: &PgPool) -> Result<(i64, bool), sqlx::Error> {
    let cleaned =
        db::cleanup_soft_deleted_documents_pgvector(pool, RECONCILIATION_BATCH_SIZE, SOFT_DELETE_CLEANUP_MINUTES)
            .await?;
    Ok((cleaned as i64, cleaned > 0))
}

/// Port of `run_vector_reconciliation_cycle` (pgvector mode): loop embedding +
/// cleanup batches until the time budget elapses or a full round does no work.
/// Each batch uses its own transaction so connections aren't held cycle-long.
pub async fn run_vector_reconciliation_cycle<E: Embedder>(
    pool: &PgPool,
    embedder: &E,
) -> Result<ReconciliationMetrics, sqlx::Error> {
    let mut metrics = ReconciliationMetrics::default();
    let deadline = Instant::now() + Duration::from_secs(RECONCILIATION_TIME_BUDGET_SECONDS);

    while Instant::now() < deadline {
        let (synced, failed, embs_work) = reconcile_message_embeddings_batch(pool, embedder).await?;
        metrics.message_embeddings_synced += synced;
        metrics.message_embeddings_failed += failed;

        if Instant::now() >= deadline {
            break;
        }

        let (cleaned, cleanup_work) = cleanup_pgvector_batch(pool).await?;
        metrics.documents_cleaned += cleaned;

        if !(embs_work || cleanup_work) {
            break;
        }
    }

    Ok(metrics)
}

/// Port of `consumer.process_reconciler`: dispatch a reconciler queue item by
/// type, emitting the completion event only when the cycle did something
/// (matching Python's `> 0` gates). `now` stamps the event timestamp;
/// `error_retention_seconds` is `settings.DERIVER.QUEUE_ERROR_RETENTION_SECONDS`.
pub async fn process_reconciler<E: Embedder>(
    pool: &PgPool,
    emitter: &dyn Emitter,
    embedder: &E,
    payload: &ReconcilerPayload,
    error_retention_seconds: i64,
    now: DateTime<Utc>,
) -> Result<(), sqlx::Error> {
    let start = Instant::now();

    match payload.reconciler_type {
        ReconcilerType::SyncVectors => {
            let metrics = run_vector_reconciliation_cycle(pool, embedder).await?;
            let duration_ms = start.elapsed().as_secs_f64() * 1000.0;

            if metrics.total_synced() > 0 || metrics.total_failed() > 0 || metrics.total_cleaned() > 0
            {
                emitter.emit(&SyncVectorsCompletedEvent {
                    timestamp: now,
                    documents_synced: metrics.documents_synced,
                    documents_failed: metrics.documents_failed,
                    documents_cleaned: metrics.documents_cleaned,
                    message_embeddings_synced: metrics.message_embeddings_synced,
                    message_embeddings_failed: metrics.message_embeddings_failed,
                    total_duration_ms: duration_ms,
                });
            }
        }
        ReconcilerType::CleanupQueue => {
            let deleted = db::cleanup_queue_items(pool, error_retention_seconds).await?;
            let duration_ms = start.elapsed().as_secs_f64() * 1000.0;

            if deleted > 0 {
                emitter.emit(&CleanupStaleItemsCompletedEvent {
                    timestamp: now,
                    documents_cleaned: 0,
                    queue_items_cleaned: deleted as i64,
                    total_duration_ms: duration_ms,
                });
            }
        }
    }

    Ok(())
}
