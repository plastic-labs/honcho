//! Port of `reconciler.scheduler.ReconcilerScheduler`: the in-process interval
//! loop (hosted by the deriver worker) that periodically enqueues `reconciler`
//! tasks. Coordination across instances is via the queue itself — an enqueue is
//! skipped when the same `work_unit_key` is already in-progress (an
//! `active_queue_sessions` row) or pending.
//!
//! The Python class is a singleton with an asyncio task + shutdown event; here
//! the schedule bookkeeping lives in a testable [`SchedulerState`] and the async
//! driver is [`run_reconciler_scheduler`], wired into the worker binary.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use sqlx::PgPool;

use crate::db;

/// `QUEUE_CLEANUP_INTERVAL_SECONDS` — 12 hours.
pub const QUEUE_CLEANUP_INTERVAL_SECONDS: u64 = 12 * 3600;
/// Default `VECTOR_STORE.RECONCILIATION_INTERVAL_SECONDS` — 5 minutes.
pub const DEFAULT_SYNC_VECTORS_INTERVAL_SECONDS: u64 = 300;

/// One entry of Python's `RECONCILER_TASKS` registry.
#[derive(Debug, Clone)]
pub struct ReconcilerScheduledTask {
    /// The `reconciler_type` (queue payload value), e.g. `"sync_vectors"`.
    pub name: String,
    /// The `work_unit_key`, e.g. `"reconciler:sync_vectors"`.
    pub work_unit_key: String,
    pub interval_seconds: u64,
}

/// Resolve the sync_vectors interval from the process environment, reading
/// `VECTOR_STORE_RECONCILIATION_INTERVAL_SECONDS` (pydantic `env_prefix="VECTOR_STORE_"`).
/// Falls back to [`DEFAULT_SYNC_VECTORS_INTERVAL_SECONDS`] when absent, unparseable,
/// or out of range — Python enforces `gt=0`, so a non-positive value is rejected
/// in favor of the default (matching how the worker settings treat invalid env vars).
pub fn sync_vectors_interval_from_env() -> u64 {
    sync_vectors_interval_from_pairs(std::env::vars())
}

/// [`sync_vectors_interval_from_env`] over an arbitrary key/value source (testable).
pub fn sync_vectors_interval_from_pairs<I, K, V>(pairs: I) -> u64
where
    I: IntoIterator<Item = (K, V)>,
    K: AsRef<str>,
    V: AsRef<str>,
{
    pairs
        .into_iter()
        .find(|(key, _)| key.as_ref() == "VECTOR_STORE_RECONCILIATION_INTERVAL_SECONDS")
        .and_then(|(_, value)| value.as_ref().trim().parse::<u64>().ok())
        .filter(|seconds| *seconds > 0)
        .unwrap_or(DEFAULT_SYNC_VECTORS_INTERVAL_SECONDS)
}

/// The default registry: `sync_vectors` (configurable interval) + `cleanup_queue`
/// (12h). Mirrors `RECONCILER_TASKS`.
pub fn default_reconciler_tasks(sync_vectors_interval_seconds: u64) -> Vec<ReconcilerScheduledTask> {
    vec![
        ReconcilerScheduledTask {
            name: "sync_vectors".to_string(),
            work_unit_key: "reconciler:sync_vectors".to_string(),
            interval_seconds: sync_vectors_interval_seconds,
        },
        ReconcilerScheduledTask {
            name: "cleanup_queue".to_string(),
            work_unit_key: "reconciler:cleanup_queue".to_string(),
            interval_seconds: QUEUE_CLEANUP_INTERVAL_SECONDS,
        },
    ]
}

/// Per-task next-run bookkeeping, parameterized on a monotonic `now` in seconds so
/// it's deterministic to test (no wall clock). Mirrors `_next_run` + the due/sleep
/// logic of `_scheduler_loop`.
#[derive(Debug, Clone)]
pub struct SchedulerState {
    /// `next_run[i]` aligns with the task list passed to [`due_tasks`].
    next_run: Vec<f64>,
    intervals: Vec<f64>,
}

impl SchedulerState {
    /// Initialize next-run times to `now + interval` for each task (Python's
    /// `start()` seeds the first run one interval out).
    pub fn new(tasks: &[ReconcilerScheduledTask], now: f64) -> Self {
        let intervals: Vec<f64> = tasks.iter().map(|task| task.interval_seconds as f64).collect();
        let next_run = intervals.iter().map(|interval| now + interval).collect();
        Self { next_run, intervals }
    }

    /// Indices of tasks due at `now` (`now >= next_run`). Each returned task's
    /// next run is advanced to `now + interval` (Python reschedules regardless of
    /// whether the enqueue actually happened).
    pub fn due_tasks(&mut self, now: f64) -> Vec<usize> {
        let mut due = Vec::new();
        for index in 0..self.next_run.len() {
            if now >= self.next_run[index] {
                due.push(index);
                self.next_run[index] = now + self.intervals[index];
            }
        }
        due
    }

    /// Seconds to sleep until the next task is due, floored at 1.0 to avoid a busy
    /// loop (Python's `max(1.0, ...)`). Returns 60.0 when there are no tasks.
    pub fn sleep_seconds(&self, now: f64) -> f64 {
        match self.next_run.iter().copied().fold(None, |acc: Option<f64>, value| {
            Some(acc.map_or(value, |current| current.min(value)))
        }) {
            Some(next) => (next - now).max(1.0),
            None => 60.0,
        }
    }
}

/// Run the scheduler loop until `shutdown` is set. Each iteration enqueues any due
/// task (skipping duplicates via [`db::try_enqueue_reconciler_task`]) then sleeps.
/// Sleep is chunked to at most 1s so shutdown is observed promptly without a
/// dedicated wakeup channel (the worker drives shutdown via an `AtomicBool`).
pub async fn run_reconciler_scheduler(
    pool: PgPool,
    tasks: Vec<ReconcilerScheduledTask>,
    shutdown: Arc<AtomicBool>,
) {
    let started = Instant::now();
    let mut state = SchedulerState::new(&tasks, 0.0);
    tracing::info!(
        task_count = tasks.len(),
        "ReconcilerScheduler started"
    );

    while !shutdown.load(Ordering::Relaxed) {
        let now = started.elapsed().as_secs_f64();
        for index in state.due_tasks(now) {
            let task = &tasks[index];
            match db::try_enqueue_reconciler_task(&pool, &task.work_unit_key, &task.name).await {
                Ok(true) => tracing::info!(task = %task.name, "enqueued reconciler task"),
                Ok(false) => {
                    tracing::debug!(task = %task.name, "reconciler task already queued, skipping")
                }
                Err(error) => {
                    tracing::error!(task = %task.name, "error enqueueing reconciler task: {error}")
                }
            }
        }

        let target = state.sleep_seconds(started.elapsed().as_secs_f64());
        // Chunk the sleep so shutdown is noticed within ~1s.
        let step = target.min(1.0).max(0.05);
        tokio::time::sleep(Duration::from_secs_f64(step)).await;
    }

    tracing::info!("ReconcilerScheduler stopped");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tasks() -> Vec<ReconcilerScheduledTask> {
        default_reconciler_tasks(300)
    }

    #[test]
    fn sync_vectors_interval_env_parsing() {
        // Absent → default.
        assert_eq!(
            sync_vectors_interval_from_pairs(Vec::<(String, String)>::new()),
            DEFAULT_SYNC_VECTORS_INTERVAL_SECONDS
        );
        // Valid override.
        assert_eq!(
            sync_vectors_interval_from_pairs([(
                "VECTOR_STORE_RECONCILIATION_INTERVAL_SECONDS",
                "600"
            )]),
            600
        );
        // Non-positive (Python `gt=0`) and unparseable → default.
        assert_eq!(
            sync_vectors_interval_from_pairs([(
                "VECTOR_STORE_RECONCILIATION_INTERVAL_SECONDS",
                "0"
            )]),
            DEFAULT_SYNC_VECTORS_INTERVAL_SECONDS
        );
        assert_eq!(
            sync_vectors_interval_from_pairs([(
                "VECTOR_STORE_RECONCILIATION_INTERVAL_SECONDS",
                "not-a-number"
            )]),
            DEFAULT_SYNC_VECTORS_INTERVAL_SECONDS
        );
    }

    #[test]
    fn default_registry_matches_python() {
        let t = tasks();
        assert_eq!(t.len(), 2);
        assert_eq!(t[0].name, "sync_vectors");
        assert_eq!(t[0].work_unit_key, "reconciler:sync_vectors");
        assert_eq!(t[0].interval_seconds, 300);
        assert_eq!(t[1].name, "cleanup_queue");
        assert_eq!(t[1].work_unit_key, "reconciler:cleanup_queue");
        assert_eq!(t[1].interval_seconds, QUEUE_CLEANUP_INTERVAL_SECONDS);
    }

    #[test]
    fn nothing_due_before_first_interval() {
        let t = tasks();
        let mut state = SchedulerState::new(&t, 0.0);
        // Both seeded to now+interval, so nothing is due at t=0 or t=299.
        assert!(state.due_tasks(0.0).is_empty());
        assert!(state.due_tasks(299.0).is_empty());
    }

    #[test]
    fn sync_vectors_due_at_its_interval_and_reschedules() {
        let t = tasks();
        let mut state = SchedulerState::new(&t, 0.0);
        // At t=300 the sync_vectors task (idx 0) is due; cleanup_queue (12h) isn't.
        assert_eq!(state.due_tasks(300.0), vec![0]);
        // Immediately after, it's no longer due (rescheduled to 600).
        assert!(state.due_tasks(300.0).is_empty());
        // Due again one interval later.
        assert_eq!(state.due_tasks(600.0), vec![0]);
    }

    #[test]
    fn both_due_when_both_intervals_elapse() {
        let t = tasks();
        let mut state = SchedulerState::new(&t, 0.0);
        let now = QUEUE_CLEANUP_INTERVAL_SECONDS as f64;
        assert_eq!(state.due_tasks(now), vec![0, 1]);
    }

    #[test]
    fn sleep_floored_at_one_second() {
        let t = tasks();
        let state = SchedulerState::new(&t, 0.0);
        // Next due is sync_vectors at 300; from t=0 sleep ~300s.
        assert!((state.sleep_seconds(0.0) - 300.0).abs() < 1e-9);
        // Past the due time, floored at 1.0 (never zero/negative).
        assert_eq!(state.sleep_seconds(1000.0), 1.0);
    }
}
