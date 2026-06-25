//! Adaptive polling + stale-cleanup gating for the deriver queue manager.
//!
//! Ports the deterministic scheduling core of Python `QueueManager`
//! (`src/deriver/queue_manager.py`): the `_current_poll_interval` backoff
//! state machine (`_reset_poll_interval` / `_advance_poll_interval`), the
//! `_jitter` scatter, and the once-per-interval gate in
//! `_maybe_cleanup_stale_work_units`.
//!
//! Randomness is injected as a `sample_uniform(lo, hi) -> f64` closure rather
//! than calling a global RNG, so the schedule and the gate decision are exactly
//! testable (and the production loop passes a real uniform sampler). The
//! deterministic backoff schedule (`next_backoff_interval`) and the jitter are
//! kept separable so each can be asserted on its own.

use super::settings::DeriverSettings;

/// Scatter a sleep by +/- `ratio` to avoid lockstep polling.
///
/// Returns `seconds` unchanged when `ratio <= 0.0`; otherwise
/// `seconds * sample_uniform(1 - ratio, 1 + ratio)`. Mirrors
/// `QueueManager._jitter`.
pub fn jitter(seconds: f64, ratio: f64, sample_uniform: impl FnOnce(f64, f64) -> f64) -> f64 {
    if ratio <= 0.0 {
        return seconds;
    }
    seconds * sample_uniform(1.0 - ratio, 1.0 + ratio)
}

/// Deterministic scheduling state extracted from `QueueManager`.
#[derive(Debug, Clone, PartialEq)]
pub struct PollScheduler {
    // Backoff schedule.
    current_poll_interval: f64,
    base_interval: f64,
    max_interval: f64,
    backoff_enabled: bool,
    backoff_multiplier: f64,
    jitter_ratio: f64,

    // Stale-cleanup gate.
    stale_cleanup_interval: f64,
    last_stale_cleanup_attempt: Option<f64>,
    stale_cleanup_gate_seconds: f64,
}

impl PollScheduler {
    /// Build from settings, starting the interval at the base (matching
    /// `QueueManager.__init__`: `_current_poll_interval = POLLING_SLEEP_INTERVAL_SECONDS`).
    pub fn new(settings: &DeriverSettings) -> Self {
        Self {
            current_poll_interval: settings.polling_sleep_interval_seconds,
            base_interval: settings.polling_sleep_interval_seconds,
            max_interval: settings.polling_sleep_max_interval_seconds,
            backoff_enabled: settings.polling_backoff_enabled,
            backoff_multiplier: settings.polling_backoff_multiplier,
            jitter_ratio: settings.polling_jitter_ratio,
            stale_cleanup_interval: settings.stale_work_unit_cleanup_interval_seconds,
            last_stale_cleanup_attempt: None,
            stale_cleanup_gate_seconds: 0.0,
        }
    }

    /// The current (un-jittered) idle interval, for inspection.
    pub fn current_poll_interval(&self) -> f64 {
        self.current_poll_interval
    }

    /// The base poll interval used for the busy-but-has-work branch.
    pub fn base_interval(&self) -> f64 {
        self.base_interval
    }

    /// The configured jitter ratio (so callers can pass it to [`jitter`]).
    pub fn jitter_ratio(&self) -> f64 {
        self.jitter_ratio
    }

    /// Snap the interval back to the base after finding work
    /// (`_reset_poll_interval`).
    pub fn reset_poll_interval(&mut self) {
        self.current_poll_interval = self.base_interval;
    }

    /// Return the current interval, then grow it toward the cap by the
    /// multiplier (deterministic core of `_advance_poll_interval`, before
    /// jitter is applied).
    pub fn next_backoff_interval(&mut self) -> f64 {
        let interval = self.current_poll_interval;
        if self.backoff_enabled {
            self.current_poll_interval =
                (self.current_poll_interval * self.backoff_multiplier).min(self.max_interval);
        }
        interval
    }

    /// Return the current idle/backoff sleep (jittered), then grow the schedule
    /// (`_advance_poll_interval`).
    pub fn advance_poll_interval(&mut self, sample_uniform: impl FnOnce(f64, f64) -> f64) -> f64 {
        let interval = self.next_backoff_interval();
        jitter(interval, self.jitter_ratio, sample_uniform)
    }

    /// Decide whether stale-work-unit cleanup should run now, recording the
    /// attempt and fixing the next (jittered) gate deadline when it does.
    /// Mirrors `_maybe_cleanup_stale_work_units`: skips only when the interval
    /// is positive, a prior attempt exists, and not enough time has elapsed.
    /// `now` is a monotonic clock in seconds.
    pub fn try_begin_stale_cleanup(
        &mut self,
        now: f64,
        sample_uniform: impl FnOnce(f64, f64) -> f64,
    ) -> bool {
        if self.stale_cleanup_interval > 0.0 {
            if let Some(last) = self.last_stale_cleanup_attempt {
                if now - last < self.stale_cleanup_gate_seconds {
                    return false;
                }
            }
        }
        self.last_stale_cleanup_attempt = Some(now);
        self.stale_cleanup_gate_seconds =
            jitter(self.stale_cleanup_interval, self.jitter_ratio, sample_uniform);
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settings() -> DeriverSettings {
        DeriverSettings::default()
    }

    #[test]
    fn jitter_disabled_returns_input() {
        // ratio <= 0 -> unchanged, sampler never consulted.
        assert_eq!(jitter(5.0, 0.0, |_, _| panic!("should not sample")), 5.0);
        assert_eq!(jitter(5.0, -1.0, |_, _| panic!("should not sample")), 5.0);
    }

    #[test]
    fn jitter_scales_by_sampled_factor() {
        // ratio 0.5 -> sampler is asked for uniform(0.5, 1.5).
        let out = jitter(10.0, 0.5, |lo, hi| {
            assert_eq!((lo, hi), (0.5, 1.5));
            1.2
        });
        assert!((out - 12.0).abs() < 1e-9);
    }

    #[test]
    fn backoff_schedule_grows_then_caps() {
        let mut s = settings();
        s.polling_sleep_interval_seconds = 1.0;
        s.polling_backoff_multiplier = 2.0;
        s.polling_sleep_max_interval_seconds = 8.0;
        s.polling_jitter_ratio = 0.0; // isolate the schedule from jitter
        let mut sched = PollScheduler::new(&s);

        // Returns current value, then doubles toward the cap.
        assert_eq!(sched.next_backoff_interval(), 1.0);
        assert_eq!(sched.next_backoff_interval(), 2.0);
        assert_eq!(sched.next_backoff_interval(), 4.0);
        assert_eq!(sched.next_backoff_interval(), 8.0);
        // Capped: stays at the max.
        assert_eq!(sched.next_backoff_interval(), 8.0);
        assert_eq!(sched.current_poll_interval(), 8.0);
    }

    #[test]
    fn reset_snaps_back_to_base() {
        let mut s = settings();
        s.polling_sleep_interval_seconds = 1.0;
        s.polling_backoff_multiplier = 2.0;
        let mut sched = PollScheduler::new(&s);
        sched.next_backoff_interval();
        sched.next_backoff_interval();
        assert!(sched.current_poll_interval() > 1.0);
        sched.reset_poll_interval();
        assert_eq!(sched.current_poll_interval(), 1.0);
    }

    #[test]
    fn backoff_disabled_holds_base() {
        let mut s = settings();
        s.polling_sleep_interval_seconds = 1.0;
        s.polling_backoff_enabled = false;
        let mut sched = PollScheduler::new(&s);
        assert_eq!(sched.next_backoff_interval(), 1.0);
        assert_eq!(sched.next_backoff_interval(), 1.0);
        assert_eq!(sched.current_poll_interval(), 1.0);
    }

    #[test]
    fn advance_applies_jitter_over_schedule() {
        let mut s = settings();
        s.polling_sleep_interval_seconds = 2.0;
        s.polling_backoff_multiplier = 2.0;
        s.polling_jitter_ratio = 0.5;
        let mut sched = PollScheduler::new(&s);
        // First advance: interval 2.0 jittered by factor 1.0 -> 2.0; schedule -> 4.0.
        let out = sched.advance_poll_interval(|_, _| 1.0);
        assert!((out - 2.0).abs() < 1e-9);
        assert_eq!(sched.current_poll_interval(), 4.0);
    }

    #[test]
    fn stale_cleanup_runs_first_time_then_gates() {
        let mut s = settings();
        s.stale_work_unit_cleanup_interval_seconds = 60.0;
        s.polling_jitter_ratio = 0.0; // gate == interval
        let mut sched = PollScheduler::new(&s);

        // First call always runs (no prior attempt).
        assert!(sched.try_begin_stale_cleanup(0.0, |_, _| 1.0));
        // Within the gate window -> skip.
        assert!(!sched.try_begin_stale_cleanup(30.0, |_, _| 1.0));
        assert!(!sched.try_begin_stale_cleanup(59.999, |_, _| 1.0));
        // At/after the gate -> run again, resetting the deadline.
        assert!(sched.try_begin_stale_cleanup(60.0, |_, _| 1.0));
        assert!(!sched.try_begin_stale_cleanup(119.0, |_, _| 1.0));
        assert!(sched.try_begin_stale_cleanup(120.0, |_, _| 1.0));
    }

    #[test]
    fn stale_cleanup_interval_zero_runs_every_time() {
        let mut s = settings();
        s.stale_work_unit_cleanup_interval_seconds = 0.0;
        let mut sched = PollScheduler::new(&s);
        assert!(sched.try_begin_stale_cleanup(0.0, |_, _| 1.0));
        assert!(sched.try_begin_stale_cleanup(0.0, |_, _| 1.0));
        assert!(sched.try_begin_stale_cleanup(0.0, |_, _| 1.0));
    }

    #[test]
    fn stale_cleanup_gate_uses_jitter() {
        let mut s = settings();
        s.stale_work_unit_cleanup_interval_seconds = 60.0;
        s.polling_jitter_ratio = 0.5;
        let mut sched = PollScheduler::new(&s);
        // First run samples uniform(0.5, 1.5) -> factor 1.5 -> gate 90s.
        assert!(sched.try_begin_stale_cleanup(0.0, |lo, hi| {
            assert_eq!((lo, hi), (0.5, 1.5));
            1.5
        }));
        // 89s elapsed: still inside the 90s gate.
        assert!(!sched.try_begin_stale_cleanup(89.0, |_, _| 1.0));
        // 90s elapsed: gate expired.
        assert!(sched.try_begin_stale_cleanup(90.0, |_, _| 1.0));
    }
}
