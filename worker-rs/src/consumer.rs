//! No-op queue consumer — the first executable worker milestone (Phase 4 Step 3).
//!
//! It exercises the real claim → process → complete → release state machine
//! against fixture data, but the per-item "work" is a no-op: items are simply
//! marked processed. Real task handlers (deriver, summarizer, deletion, dream,
//! reconciler) land in later phases; this proves the queue plumbing in isolation.

use sqlx::PgPool;

use crate::queue::{
    self, ClaimedWorkUnit,
};

/// Drain one claimed work unit by processing every pending item as a no-op,
/// then release the active session. Returns the number of items processed.
///
/// Mirrors `QueueManager.process_work_unit` for the non-representation,
/// one-item-at-a-time path: fetch the next unprocessed item, "process" it, mark
/// it processed, repeat until the unit drains, then delete the active session.
pub async fn drain_work_unit(
    pool: &PgPool,
    claimed: &ClaimedWorkUnit,
) -> Result<u32, sqlx::Error> {
    let mut processed = 0u32;
    loop {
        let Some(item) =
            queue::next_unprocessed_item(pool, &claimed.work_unit_key, &claimed.aqs_id).await?
        else {
            break;
        };

        // No-op "processing": real handlers run here in later phases.
        tracing::debug!(
            task_type = %item.task_type,
            work_unit_key = %claimed.work_unit_key,
            item_id = item.id,
            "no-op processing queue item"
        );

        queue::mark_items_processed(pool, &[item.id], &claimed.work_unit_key).await?;
        processed += 1;
    }

    queue::release_work_unit(pool, &claimed.aqs_id, &claimed.work_unit_key).await?;
    Ok(processed)
}

/// One poll cycle: claim up to `worker_limit` work units and drain each as no-ops.
/// Returns the total number of items processed across all claimed units.
pub async fn run_once(pool: &PgPool, worker_limit: i64) -> Result<u32, sqlx::Error> {
    let claimed = queue::claim_work_units(pool, worker_limit).await?;
    let mut total = 0u32;
    for unit in &claimed {
        total += drain_work_unit(pool, unit).await?;
    }
    Ok(total)
}
