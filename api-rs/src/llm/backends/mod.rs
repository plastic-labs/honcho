//! Provider backends, ported from `src/llm/backends/`.
//!
//! The live request/response transport is owned by the provider SDK clients,
//! which aren't available here. What *is* portable and testable is each
//! backend's deterministic request shaping (system extraction, tool-choice
//! conversion, model capability checks) and its response parsing
//! (provider JSON → [`super::CompletionResult`]). Those land per provider.

pub mod anthropic;
pub mod gemini;
pub mod openai;
