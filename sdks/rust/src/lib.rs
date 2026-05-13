//! # Honcho Rust SDK
//!
//! Rust SDK for [Honcho](https://github.com/plastic-labs/honcho) — AI agent memory
//! and social cognition infrastructure.
//!
//! ## Status
//!
//! **Alpha** — this SDK is under active development and not yet ready for production use.
//!
//! See the [porting plan](../rust-port-tdd-plan.md) for current progress.

#![forbid(unsafe_code)]
#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::todo,
    missing_docs
)]
#![warn(clippy::pedantic, clippy::cargo)]
#![allow(
    clippy::module_name_repetitions,
    clippy::missing_errors_doc,
    clippy::multiple_crate_versions
)]

/// High-level Honcho client.
pub mod client;
/// Conclusion wrapper type.
pub mod conclusion;
/// Stream adapter for dialectic responses.
pub mod dialectic_stream;
pub mod error;
pub mod http;
/// Message wrapper type.
pub mod message;
/// Peer wrapper type.
pub mod peer;
/// Session wrapper type.
pub mod session;
/// Shared types for the Honcho SDK.
pub mod types;
/// File source abstraction for uploads.
pub mod upload;

pub use client::Honcho;
pub use conclusion::{Conclusion, ConclusionCreateParams, ConclusionScope};
pub use dialectic_stream::DialecticStream;
pub use message::Message;
pub use peer::Peer;
pub use session::{Session, UploadFileBuilder};
pub use upload::FileSource;

pub use types::dialectic::DialecticOptions;
pub use types::message::{MessageCreate, MessageResponse, MessageSearchOptions};
pub use types::peer::PeerContext;
pub use types::session::{
    SessionContext, SessionContextOptions, SessionPeerConfig, SessionSummaries,
};

#[cfg(feature = "blocking")]
pub mod blocking;
