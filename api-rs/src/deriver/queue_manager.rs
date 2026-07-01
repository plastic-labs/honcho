//! Port of the worker runtime in `src/deriver/queue_manager.py`
//! (`QueueManager.polling_loop` / `process_work_unit` / `main`).
//!
//! The deterministic substrate is already ported elsewhere — the poll-interval
//! backoff + stale-cleanup gate ([`super::poll::PollScheduler`]), every DB op
//! (`db::get_and_claim_work_units`, `db::cleanup_work_unit`,
//! `db::cleanup_stale_work_units`, …), and the per-iteration representation
//! processing ([`super::consumer::process_representation_work_unit_once`]). This
//! module is the async glue: a [`DeriverWorker`] that claims work units and
//! drives them to completion, concurrency-capped at `DERIVER_WORKERS`, with a
//! graceful-shutdown drain.
//!
//! Scope note: all task types — `representation`, `deletion`, `summary`,
//! `reconciler`, `webhook`, and `dream` — are processed by the worker, and the
//! `queue.empty` webhook from Python's `process_work_unit` finally is emitted
//! (representation/summary work units, after a successful drain).

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use tokio::sync::Semaphore;
use tokio::task::JoinSet;

use crate::db;
use crate::deriver::consumer::{process_item, process_representation_work_unit_once};
use crate::deriver::deriver::{DeriverBatchContext, DeriverModelSettings};
use crate::deriver::payload::SummaryPayload;
use crate::deriver::poll::PollScheduler;
use crate::deriver::settings::DeriverSettings;
use crate::dialectic::Embedder;
use crate::llm::credentials::TransportApiKeys;
use crate::llm::http::LlmHttp;
use crate::producer::parse_work_unit_key;
use crate::summarizer::{SummaryGlobalSettings, SummaryModelSettings, summarize_if_needed};
use crate::webhooks::{WebhookSender, deliver_webhook};

/// Build the `queue.empty` webhook queue-item payload, mirroring Python's
/// `publish_webhook_event(QueueEmptyEvent(...))` → `create_webhook_payload`. The
/// stored shape is `{event_type, data}` where `data` carries the event fields
/// (`workspace_id`, `queue_type`, `session_id`, `observer`, `observed`).
///
/// Parity notes: Python's `event.model_dump(exclude={"type"})` does NOT exclude
/// `None`, so genuinely-absent `observer`/`observed` serialize as JSON `null`.
/// And `parse_work_unit_key` keeps the literal `"None"` string for an absent
/// session segment (both languages), so `session_id` is the string `"None"` —
/// not `null` — for a session-less representation/summary work unit.
fn build_queue_empty_payload(
    workspace_name: &str,
    work_unit: &crate::producer::ParsedWorkUnit,
) -> serde_json::Value {
    serde_json::json!({
        "event_type": "queue.empty",
        "data": {
            "workspace_id": workspace_name,
            "queue_type": work_unit.task_type,
            "session_id": work_unit.session_name,
            "observer": work_unit.observer,
            "observed": work_unit.observed,
        },
    })
}

/// The deriver worker: owns the collaborators behind `Arc` so per-work-unit
/// tasks can be spawned, and drives the claim → process → release cycle.
pub struct DeriverWorker<H, E>
where
    H: LlmHttp + Send + Sync + 'static,
    E: Embedder + Send + Sync + 'static,
{
    pool: sqlx::PgPool,
    http: Arc<H>,
    embedder: Arc<E>,
    emitter: Arc<dyn crate::telemetry::Emitter>,
    keys: TransportApiKeys,
    model_settings: DeriverModelSettings,
    summary_settings: SummaryGlobalSettings,
    dream_settings: crate::dreamer::orchestrator::DreamModelSettings,
    dream_schedule_settings: crate::dreamer::scheduler::DreamScheduleSettings,
    poll_settings: DeriverSettings,
    webhook_sender: Arc<dyn WebhookSender>,
    /// `settings.WEBHOOK.SECRET` — required to sign deliveries; `None` skips them.
    webhook_secret: Option<String>,
}

impl<H, E> DeriverWorker<H, E>
where
    H: LlmHttp + Send + Sync + 'static,
    E: Embedder + Send + Sync + 'static,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        pool: sqlx::PgPool,
        http: Arc<H>,
        embedder: Arc<E>,
        emitter: Arc<dyn crate::telemetry::Emitter>,
        keys: TransportApiKeys,
        model_settings: DeriverModelSettings,
        summary_settings: SummaryGlobalSettings,
        dream_settings: crate::dreamer::orchestrator::DreamModelSettings,
        dream_schedule_settings: crate::dreamer::scheduler::DreamScheduleSettings,
        poll_settings: DeriverSettings,
        webhook_sender: Arc<dyn WebhookSender>,
        webhook_secret: Option<String>,
    ) -> Self {
        Self {
            pool,
            http,
            embedder,
            emitter,
            keys,
            model_settings,
            summary_settings,
            dream_settings,
            dream_schedule_settings,
            poll_settings,
            webhook_sender,
            webhook_secret,
        }
    }

    /// Build a borrow-scoped batch context off the `Arc`-held collaborators. Cheap
    /// to recreate per iteration; avoids holding it across the claim query.
    fn batch_context(&self) -> DeriverBatchContext<'_, H, E> {
        DeriverBatchContext {
            pool: &self.pool,
            http: &self.http,
            keys: self.keys.clone(),
            embedder: &self.embedder,
            settings: self.model_settings.clone(),
            emitter: self.emitter.as_ref(),
            dream_schedule_settings: self.dream_schedule_settings.clone(),
        }
    }

    /// Port of `process_work_unit`'s body for one claimed work unit: loop the
    /// representation iteration until the queue drains, then release the claim.
    /// Returns the number of queue items processed. Errors are logged and end the
    /// loop (the items stay unprocessed for a later claim — a simplification of
    /// Python's `_handle_processing_error`, which marks them errored).
    pub async fn process_work_unit(self: Arc<Self>, work_unit_key: String, aqs_id: String) -> usize {
        let work_unit = match parse_work_unit_key(&work_unit_key) {
            Ok(parsed) => parsed,
            Err(error) => {
                tracing::error!(work_unit_key = %work_unit_key, "unparseable work unit key: {error}");
                let _ = db::cleanup_work_unit(&self.pool, &aqs_id, &work_unit_key).await;
                return 0;
            }
        };

        tracing::info!(
            task_type = %work_unit.task_type,
            work_unit_key = %work_unit_key,
            "processing work unit"
        );
        let processed = match work_unit.task_type.as_str() {
            "representation" => self.drain_representation(&work_unit, &work_unit_key, &aqs_id).await,
            "deletion" | "summary" | "reconciler" | "webhook" | "dream" => {
                self.drain_via_process_item(&work_unit_key, &aqs_id).await
            }
            other => {
                tracing::warn!(
                    task_type = %other,
                    work_unit_key = %work_unit_key,
                    "task type not yet ported in the Rust worker; releasing claim, leaving queue items"
                );
                0
            }
        };

        // Release the claim (Python's finally `_cleanup_work_unit`).
        let removed = db::cleanup_work_unit(&self.pool, &aqs_id, &work_unit_key)
            .await
            .unwrap_or(false);

        // Publish a `queue.empty` webhook event when this worker actually removed
        // the active session and drained at least one item (Python's finally
        // block: `if removed and queue_item_count > 0`). Only representation and
        // summary work units notify, and only when scoped to a workspace.
        if removed
            && processed > 0
            && matches!(work_unit.task_type.as_str(), "representation" | "summary")
            && let Some(workspace_name) = work_unit.workspace_name.as_deref()
        {
            let payload = build_queue_empty_payload(workspace_name, &work_unit);
            if let Err(error) =
                db::enqueue_webhook_event(&self.pool, workspace_name, &payload).await
            {
                tracing::error!(
                    work_unit_key = %work_unit_key,
                    "error triggering queue_empty webhook: {error}"
                );
            }
        }
        processed
    }

    /// Drain a representation work unit by looping the batch iteration.
    async fn drain_representation(
        &self,
        work_unit: &crate::producer::ParsedWorkUnit,
        work_unit_key: &str,
        aqs_id: &str,
    ) -> usize {
        let batch_max_tokens = self.poll_settings.representation_batch_max_tokens;
        let flush_enabled = self.poll_settings.flush_enabled;

        let mut processed = 0usize;
        loop {
            let ctx = self.batch_context();
            match process_representation_work_unit_once(
                &ctx,
                work_unit,
                work_unit_key,
                aqs_id,
                batch_max_tokens,
                flush_enabled,
            )
            .await
            {
                Ok(Some(count)) => processed += count,
                Ok(None) => break,
                Err(error) => {
                    tracing::error!(work_unit_key = %work_unit_key, "error processing representation batch: {error:?}");
                    break;
                }
            }
        }
        processed
    }

    /// Drain a non-representation work unit one item at a time via
    /// [`process_item`] (Python's `get_next_queue_item` → `process_item` →
    /// `mark_queue_items_as_processed` loop). On a per-item error the item is
    /// marked errored and the loop stops (mirrors `_handle_processing_error`).
    async fn drain_via_process_item(&self, work_unit_key: &str, aqs_id: &str) -> usize {
        let mut processed = 0usize;
        loop {
            let item = match db::get_next_queue_item(&self.pool, work_unit_key, aqs_id).await {
                Ok(Some(item)) => item,
                Ok(None) => break,
                Err(error) => {
                    tracing::error!(work_unit_key = %work_unit_key, "error fetching queue item: {error}");
                    break;
                }
            };
            tracing::info!(
                task_type = %item.task_type,
                item_id = %item.id,
                message_id = ?item.message_id,
                "processing queue item"
            );
            // Dispatch by task type: summary + reconciler need the worker's LLM /
            // embedding collaborators; the other arms go through process_item.
            let outcome = match item.task_type.as_str() {
                "summary" => self.process_summary_item(&item).await,
                "reconciler" => self.process_reconciler_item(&item).await,
                "webhook" => self.process_webhook_item(&item).await,
                "dream" => self.process_dream_item(&item).await,
                _ => process_item(&self.pool, self.emitter.as_ref(), &item)
                    .await
                    .map_err(|error| error.to_string()),
            };
            match outcome {
                Ok(()) => {
                    let _ =
                        db::mark_queue_items_as_processed(&self.pool, &[item.id], work_unit_key).await;
                    processed += 1;
                }
                Err(error) => {
                    tracing::error!(work_unit_key = %work_unit_key, "error processing queue item: {error}");
                    let _ =
                        db::mark_queue_item_as_errored(&self.pool, item.id, work_unit_key, &error)
                            .await;
                    break;
                }
            }
        }
        processed
    }

    /// Handle a `summary` queue item: build the per-call summary settings from
    /// the worker's global config + the payload's interval configuration, then
    /// run [`summarize_if_needed`].
    async fn process_summary_item(&self, item: &db::QueueItem) -> Result<(), String> {
        let workspace = item
            .workspace_name
            .as_deref()
            .ok_or_else(|| "summary task requires a workspace_name".to_string())?;
        let message_id = item
            .message_id
            .ok_or_else(|| "summary task requires a message_id".to_string())?;
        let payload = SummaryPayload::from_value(&item.payload).map_err(|e| e.to_string())?;
        let config = &payload.configuration;

        // Resolve the message public id, falling back to a DB lookup when the
        // payload omits it (Python `consumer.py` `summary_fallback`). A missing
        // message is an idempotent no-op success, matching Python's early return.
        let message_public_id = match payload
            .message_public_id
            .as_deref()
            .filter(|id| !id.is_empty())
        {
            Some(id) => id.to_string(),
            None => {
                match db::get_message_public_id(
                    &self.pool,
                    workspace,
                    &payload.session_name,
                    message_id,
                )
                .await
                .map_err(|error| error.to_string())?
                {
                    Some(id) => id,
                    None => {
                        tracing::error!(
                            message_id,
                            "Failed to fetch message for process_summary_task"
                        );
                        return Ok(());
                    }
                }
            }
        };

        let settings = SummaryModelSettings {
            model_config: self.summary_settings.model_config.clone(),
            max_tokens_short: self.summary_settings.max_tokens_short,
            max_tokens_long: self.summary_settings.max_tokens_long,
            messages_per_short_summary: config.messages_per_short_summary,
            messages_per_long_summary: config.messages_per_long_summary,
        };
        let now_iso = crate::telemetry::python_isoformat_utc(chrono::Utc::now());

        summarize_if_needed(
            &self.pool,
            self.http.as_ref(),
            &self.keys,
            &settings,
            config.summary_enabled,
            workspace,
            &payload.session_name,
            message_id,
            payload.message_seq_in_session,
            &message_public_id,
            &now_iso,
        )
        .await
        .map_err(|error| error.to_string())
    }

    /// Handle a `reconciler` queue item (Python `process_reconciler`): run the
    /// pgvector vector-reconciliation cycle or the queue-cleanup pass, using the
    /// worker's embedder + emitter. Reconciler items carry no workspace_name, so
    /// (unlike summary) nothing here reads `item.workspace_name`.
    async fn process_reconciler_item(&self, item: &db::QueueItem) -> Result<(), String> {
        let payload =
            crate::deriver::payload::ReconcilerPayload::from_value(&item.payload)
                .map_err(|e| e.to_string())?;
        crate::reconciler::process_reconciler(
            &self.pool,
            self.emitter.as_ref(),
            self.embedder.as_ref(),
            &payload,
            self.poll_settings.queue_error_retention_seconds,
            chrono::Utc::now(),
        )
        .await
        .map_err(|error| error.to_string())
    }

    /// Handle a `webhook` queue item (Python `process_item` webhook arm →
    /// `webhook_delivery.deliver_webhook`). Delivery is best-effort: a malformed
    /// payload is the one hard error (Python raises `ValueError`); everything else
    /// — endpoint lookup, signing, transport — is logged and swallowed inside
    /// [`deliver_webhook`], so this returns `Ok(())` once the payload validates.
    async fn process_webhook_item(&self, item: &db::QueueItem) -> Result<(), String> {
        let workspace = item
            .workspace_name
            .as_deref()
            .ok_or_else(|| "webhook task requires a workspace_name".to_string())?;
        let payload = crate::deriver::payload::WebhookPayload::from_value(&item.payload)
            .map_err(|error| error.to_string())?;
        let now_iso = crate::telemetry::python_isoformat_utc(chrono::Utc::now());
        deliver_webhook(
            &self.pool,
            self.webhook_sender.as_ref(),
            &payload,
            workspace,
            self.webhook_secret.as_deref(),
            &now_iso,
        )
        .await;
        Ok(())
    }

    /// Handle a `dream` queue item (Python `process_item` dream arm →
    /// `process_dream`): run the OMNI dream cycle (deduction + induction) and the
    /// guard-pair collection write, using the worker's LLM + embedding + emitter
    /// collaborators. `process_dream` swallows its own errors (Python's
    /// non-re-raising `except`), so this returns `Ok(())` once the payload validates.
    ///
    /// Deviation: per-resource configuration overrides are not threaded — the
    /// dream payload carries no configuration, and the worker does not refetch the
    /// session/workspace config here, so `ResolvedConfiguration::default()` is used
    /// (deploy-global `DREAM.ENABLED` still gates via `dream_settings.enabled`).
    async fn process_dream_item(&self, item: &db::QueueItem) -> Result<(), String> {
        let workspace = item
            .workspace_name
            .as_deref()
            .ok_or_else(|| "dream task requires a workspace_name".to_string())?;
        let payload = crate::deriver::payload::DreamPayload::from_value(&item.payload)
            .map_err(|error| error.to_string())?;
        let configuration = crate::producer::ResolvedConfiguration::default();
        let now_iso = crate::telemetry::python_isoformat_utc(chrono::Utc::now());
        crate::dreamer::orchestrator::process_dream(
            &self.pool,
            self.http.as_ref(),
            self.keys.clone(),
            self.embedder.as_ref(),
            &payload,
            workspace,
            &configuration,
            &self.dream_settings,
            self.emitter.as_ref(),
            None,
            &now_iso,
        )
        .await;
        Ok(())
    }

    /// One non-spawning poll cycle: claim available work units (with no owned
    /// units) and drive each to completion sequentially. Returns the total queue
    /// items processed. This is the deterministic core the integration test
    /// exercises; [`Self::run`] uses the same [`Self::process_work_unit`] under a
    /// concurrency cap.
    pub async fn poll_once(self: &Arc<Self>) -> Result<usize, sqlx::Error> {
        let claimed = db::get_and_claim_work_units(
            &self.pool,
            self.poll_settings.workers as i64,
            0,
            self.poll_settings.representation_batch_max_tokens,
            self.poll_settings.representation_batch_max_age_seconds,
            self.poll_settings.flush_enabled,
        )
        .await?;

        let mut total = 0usize;
        for (work_unit_key, aqs_id) in claimed {
            total += Arc::clone(self).process_work_unit(work_unit_key, aqs_id).await;
        }
        Ok(total)
    }

    /// Port of `polling_loop`: claim → spawn per work unit (capped by a
    /// `DERIVER_WORKERS`-permit semaphore) → adaptive backoff when idle. Runs
    /// until `shutdown` is set, then drains in-flight tasks. Startup jitter and
    /// the stale-cleanup gate use [`PollScheduler`]; jitter randomness comes from
    /// a cheap process-local PRNG (scheduling scatter, not crypto).
    pub async fn run(self: Arc<Self>, shutdown: Arc<AtomicBool>) {
        let workers = self.poll_settings.workers.max(1) as usize;
        let semaphore = Arc::new(Semaphore::new(workers));
        let mut scheduler = PollScheduler::new(&self.poll_settings);
        let started = Instant::now();
        // xorshift64 seeded from the start instant's nanos; advanced per sample.
        let mut rng_state: u64 = (started.elapsed().subsec_nanos() as u64) | 1;
        let mut sample = |lo: f64, hi: f64| -> f64 {
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            let frac = (rng_state >> 11) as f64 / (1u64 << 53) as f64;
            lo + (hi - lo) * frac
        };

        // Startup jitter so co-launched instances don't poll in lockstep.
        let startup = self.poll_settings.polling_startup_jitter_seconds;
        if startup > 0.0 && !shutdown.load(Ordering::Relaxed) {
            let delay = sample(0.0, startup);
            tokio::time::sleep(std::time::Duration::from_secs_f64(delay)).await;
        }

        let mut tasks: JoinSet<usize> = JoinSet::new();

        while !shutdown.load(Ordering::Relaxed) {
            // Reap finished tasks so permits are released and ownership frees.
            while tasks.try_join_next().is_some() {}

            let available = semaphore.available_permits();
            if available == 0 {
                // All workers busy: short base sleep for fast pickup.
                let base = self.poll_settings.polling_sleep_interval_seconds;
                let nap = crate::deriver::poll::jitter(
                    base,
                    self.poll_settings.polling_jitter_ratio,
                    &mut sample,
                );
                tokio::time::sleep(std::time::Duration::from_secs_f64(nap)).await;
                continue;
            }

            // Gate stale-work-unit cleanup to at most once per (jittered) interval.
            let now = started.elapsed().as_secs_f64();
            if scheduler.try_begin_stale_cleanup(now, &mut sample) {
                if let Err(error) = db::cleanup_stale_work_units(
                    &self.pool,
                    self.poll_settings.stale_session_timeout_minutes as i64,
                )
                .await
                {
                    tracing::error!("stale work-unit cleanup failed: {error}");
                }
            }

            let owned = (workers - available) as i64;
            let claimed = match db::get_and_claim_work_units(
                &self.pool,
                workers as i64,
                owned,
                self.poll_settings.representation_batch_max_tokens,
                self.poll_settings.representation_batch_max_age_seconds,
                self.poll_settings.flush_enabled,
            )
            .await
            {
                Ok(claimed) => claimed,
                Err(error) => {
                    tracing::error!("error claiming work units: {error}");
                    let backoff = scheduler.advance_poll_interval(&mut sample);
                    tokio::time::sleep(std::time::Duration::from_secs_f64(backoff)).await;
                    continue;
                }
            };

            if claimed.is_empty() {
                let backoff = scheduler.advance_poll_interval(&mut sample);
                tokio::time::sleep(std::time::Duration::from_secs_f64(backoff)).await;
                continue;
            }

            scheduler.reset_poll_interval();
            for (work_unit_key, aqs_id) in claimed {
                if shutdown.load(Ordering::Relaxed) {
                    break;
                }
                let permit = match Arc::clone(&semaphore).try_acquire_owned() {
                    Ok(permit) => permit,
                    Err(_) => break, // no capacity (claimed more than permits — shouldn't happen)
                };
                let worker = Arc::clone(&self);
                tasks.spawn(async move {
                    let _permit = permit; // released when the task ends
                    worker.process_work_unit(work_unit_key, aqs_id).await
                });
            }
        }

        // Graceful drain.
        tracing::info!("deriver worker shutting down; draining in-flight work units");
        while tasks.join_next().await.is_some() {}
    }
}

#[cfg(test)]
mod tests {
    use super::build_queue_empty_payload;
    use crate::producer::parse_work_unit_key;
    use serde_json::json;

    #[test]
    fn queue_empty_payload_for_summary_carries_all_fields() {
        // summary:{ws}:{session}:{observer}:{observed}
        let work_unit = parse_work_unit_key("summary:ws1:sess1:alice:bob").unwrap();
        let payload = build_queue_empty_payload("ws1", &work_unit);
        assert_eq!(
            payload,
            json!({
                "event_type": "queue.empty",
                "data": {
                    "workspace_id": "ws1",
                    "queue_type": "summary",
                    "session_id": "sess1",
                    "observer": "alice",
                    "observed": "bob",
                },
            })
        );
    }

    #[test]
    fn queue_empty_payload_for_representation_nulls_absent_observer() {
        // New 4-segment representation: representation:{ws}:{session}:{observed}.
        // observer is genuinely None → JSON null (Python `exclude={"type"}` keeps
        // None); session keeps its literal value.
        let work_unit = parse_work_unit_key("representation:ws1:sess1:bob").unwrap();
        let payload = build_queue_empty_payload("ws1", &work_unit);
        assert_eq!(
            payload,
            json!({
                "event_type": "queue.empty",
                "data": {
                    "workspace_id": "ws1",
                    "queue_type": "representation",
                    "session_id": "sess1",
                    "observer": null,
                    "observed": "bob",
                },
            })
        );
    }

    #[test]
    fn queue_empty_payload_keeps_literal_none_session() {
        // A session-less representation serializes its session segment as the
        // literal "None" string (parse keeps it), so session_id is "None", not null.
        let work_unit = parse_work_unit_key("representation:ws1:None:bob").unwrap();
        let payload = build_queue_empty_payload("ws1", &work_unit);
        assert_eq!(payload["data"]["session_id"], json!("None"));
    }
}
