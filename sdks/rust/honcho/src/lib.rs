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

pub mod error;
pub mod http;
/// Shared types for the Honcho SDK.
pub mod types;
