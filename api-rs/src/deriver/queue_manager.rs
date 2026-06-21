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
//! Scope note: only `representation` work units are fully processed today. The
//! non-representation `process_item` arms (webhook/summary/dream/deletion/
//! reconciler) are unported — a claimed work unit of those types is released via
//! `cleanup_work_unit` (its queue items are left intact for when those arms
//! land) so the worker neither spins nor drops data. The `queue.empty` webhook
//! emitted in Python's `process_work_unit` finally is likewise deferred.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use tokio::sync::Semaphore;
use tokio::task::JoinSet;

use crate::db;
use crate::deriver::consumer::process_representation_work_unit_once;
use crate::deriver::deriver::{DeriverBatchContext, DeriverModelSettings};
use crate::deriver::poll::PollScheduler;
use crate::deriver::settings::DeriverSettings;
use crate::dialectic::Embedder;
use crate::llm::credentials::TransportApiKeys;
use crate::llm::http::LlmHttp;
use crate::producer::parse_work_unit_key;

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
    poll_settings: DeriverSettings,
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
        poll_settings: DeriverSettings,
    ) -> Self {
        Self {
            pool,
            http,
            embedder,
            emitter,
            keys,
            model_settings,
            poll_settings,
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

        if work_unit.task_type != "representation" {
            tracing::warn!(
                task_type = %work_unit.task_type,
                work_unit_key = %work_unit_key,
                "task type not yet ported in the Rust worker; releasing claim, leaving queue items"
            );
            let _ = db::cleanup_work_unit(&self.pool, &aqs_id, &work_unit_key).await;
            return 0;
        }

        let batch_max_tokens = self.poll_settings.representation_batch_max_tokens;
        let flush_enabled = self.poll_settings.flush_enabled;

        let mut processed = 0usize;
        loop {
            let ctx = self.batch_context();
            match process_representation_work_unit_once(
                &ctx,
                &work_unit,
                &work_unit_key,
                &aqs_id,
                batch_max_tokens,
                flush_enabled,
            )
            .await
            {
                Ok(Some(count)) => processed += count,
                Ok(None) => break,
                Err(error) => {
                    tracing::error!(
                        work_unit_key = %work_unit_key,
                        "error processing representation batch: {error:?}"
                    );
                    break;
                }
            }
        }

        // Release the claim (Python's finally `_cleanup_work_unit`). The
        // queue.empty webhook is deferred.
        let _ = db::cleanup_work_unit(&self.pool, &aqs_id, &work_unit_key).await;
        processed
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
