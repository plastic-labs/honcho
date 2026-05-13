//! HTTP client internals.

/// Low-level HTTP client wrapper around `reqwest`.
pub mod client;
/// Response decoding helpers.
pub mod decode;
/// API route path builders.
pub mod routes;
/// Server-Sent Events (SSE) stream parsing.
pub mod sse;
