//! Port of the representation slice of `src/deriver/consumer.py` +
//! `QueueManager.process_work_unit` (`src/deriver/queue_manager.py`).
//!
//! `process_work_unit`'s representation branch is, per loop iteration: fetch a
//! batch ([`db::get_queue_item_batch`]), extract the observer list from the
//! lead item's payload, run [`process_representation_tasks_batch`], then mark
//! the items processed. [`process_representation_work_unit_once`] ports that one
//! iteration; the surrounding ownership/shutdown/semaphore loop and the
//! non-representation `process_item` arms (webhook/summary/dream/deletion/
//! reconciler) land with the worker binary as their dependencies are ported.

use serde_json::Value;
use sqlx::PgPool;

use crate::db::{self, QueueBatchError};
use crate::dialectic::Embedder;
use crate::llm::http::LlmHttp;
use crate::producer::ParsedWorkUnit;

use super::deriver::{DeriverBatchContext, process_representation_tasks_batch};

/// Port of the observer-extraction in `process_work_unit`: the new payload shape
/// carries an `observers` array; the legacy shape carries a single `observer`
/// string (lifted into a one-element list). Anything else yields an empty list.
pub fn extract_observers(payload: &Value) -> Vec<String> {
    if let Some(observers) = payload.get("observers").and_then(Value::as_array) {
        return observers
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_string)
            .collect();
    }
    // Legacy format: a single observer string.
    match payload.get("observer").and_then(Value::as_str) {
        Some(observer) if !observer.is_empty() => vec![observer.to_string()],
        _ => Vec::new(),
    }
}

/// Failure modes of [`process_representation_work_unit_once`].
#[derive(Debug)]
pub enum WorkUnitError {
    /// The batch query failed (DB or malformed configuration payload).
    Batch(QueueBatchError),
    /// `observed`/`observers` missing for a non-empty batch — Python's
    /// `process_representation_batch` raises `ValueError` here.
    MissingObservers,
    /// A downstream DB op (the batch processor or the processed-mark) failed.
    Database(sqlx::Error),
}

impl From<QueueBatchError> for WorkUnitError {
    fn from(error: QueueBatchError) -> Self {
        WorkUnitError::Batch(error)
    }
}

impl From<sqlx::Error> for WorkUnitError {
    fn from(error: sqlx::Error) -> Self {
        WorkUnitError::Database(error)
    }
}

/// Port of one iteration of `process_work_unit`'s representation branch.
///
/// Fetches the next batch for `work_unit_key`, and:
/// - returns `Ok(None)` when there is nothing left to process (the caller's loop
///   breaks),
/// - runs [`process_representation_tasks_batch`] over the batch (skipping the
///   LLM call when the context window is empty, mirroring
///   `process_representation_batch`'s early return), then marks every fetched
///   item processed and returns `Ok(Some(count))`.
///
/// Deviation from Python: when the batch carries no resolved `configuration`
/// (`None`), this uses [`ResolvedConfiguration::default`] rather than the
/// orchestrator's DB-fallback fetch (which is not ported — the worker resolves
/// configuration upstream at enqueue time).
pub async fn process_representation_work_unit_once<H, E>(
    ctx: &DeriverBatchContext<'_, H, E>,
    work_unit: &ParsedWorkUnit,
    work_unit_key: &str,
    aqs_id: &str,
    batch_max_tokens: i64,
    flush_enabled: bool,
) -> Result<Option<usize>, WorkUnitError>
where
    H: LlmHttp + Sync,
    E: Embedder + Sync,
{
    let pool: &PgPool = ctx.pool;
    let batch =
        db::get_queue_item_batch(pool, work_unit_key, aqs_id, batch_max_tokens, flush_enabled)
            .await?;

    if batch.items_to_process.is_empty() {
        return Ok(None);
    }

    let observers = extract_observers(&batch.items_to_process[0].payload);
    let item_ids: Vec<i64> = batch.items_to_process.iter().map(|q| q.id).collect();
    let queue_item_message_ids: Vec<i64> = batch
        .items_to_process
        .iter()
        .filter_map(|q| q.message_id)
        .collect();

    // process_representation_batch: empty context window is a no-op (the items
    // are still marked processed); a non-empty window requires observed +
    // observers.
    if !batch.messages_context.is_empty() {
        let observed = work_unit
            .observed
            .as_deref()
            .filter(|s| !s.is_empty())
            .ok_or(WorkUnitError::MissingObservers)?;
        if observers.is_empty() {
            return Err(WorkUnitError::MissingObservers);
        }
        let configuration = batch.configuration.clone().unwrap_or_default();
        process_representation_tasks_batch(
            ctx,
            &batch.messages_context,
            &configuration,
            &observers,
            observed,
            &queue_item_message_ids,
            batch.hit_batch_token_cap,
            batch.was_flush_enabled,
            batch.batch_max_tokens,
        )
        .await?;
    }

    db::mark_queue_items_as_processed(pool, &item_ids, work_unit_key).await?;
    Ok(Some(item_ids.len()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn extract_observers_new_array_format() {
        let payload = json!({"observers": ["alice", "carol"], "observer": "ignored"});
        assert_eq!(extract_observers(&payload), vec!["alice", "carol"]);
    }

    #[test]
    fn extract_observers_legacy_single_string() {
        let payload = json!({"observer": "bob"});
        assert_eq!(extract_observers(&payload), vec!["bob"]);
    }

    #[test]
    fn extract_observers_empty_legacy_is_empty() {
        assert!(extract_observers(&json!({"observer": ""})).is_empty());
        assert!(extract_observers(&json!({})).is_empty());
        // Empty array stays empty.
        assert!(extract_observers(&json!({"observers": []})).is_empty());
    }

    #[test]
    fn extract_observers_array_wins_over_legacy_even_when_empty() {
        // Presence of an (empty) observers array short-circuits the legacy path.
        let payload = json!({"observers": [], "observer": "bob"});
        assert!(extract_observers(&payload).is_empty());
    }
}
