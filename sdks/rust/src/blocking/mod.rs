//! Blocking facade over the async Honcho SDK.
//!
//! Enable with the `blocking` feature flag. Uses an internal tokio runtime.

mod client;
mod conclusion;
mod iter;
mod peer;
mod runtime;
mod session;

pub use client::Honcho;
pub use conclusion::{Conclusion, ConclusionScope};
pub use peer::{ChatStreamIterator, Peer};
pub use session::{BlockingSessionRepresentationBuilder, BlockingUploadFileBuilder, Session};
