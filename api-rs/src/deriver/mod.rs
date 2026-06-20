//! Background deriver worker (Python `src/deriver/`).
//!
//! Port in progress. The deterministic scheduling core lands first; the
//! DB-touching queue-claim/batch queries, the batch-config resolution, and the
//! consumer/processing pipeline follow as separate units.

pub mod payload;
pub mod poll;
pub mod prompts;
pub mod settings;

pub use poll::PollScheduler;
pub use settings::DeriverSettings;
