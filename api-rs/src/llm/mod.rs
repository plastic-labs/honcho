//! Provider-agnostic LLM layer, ported from `src/llm/`.
//!
//! This is the start of the Rust port of the LLM subsystem. It begins with the
//! deterministic, network-free core: the normalized completion/tool-call types
//! ([`CompletionResult`], [`ToolCallResult`]) and the provider history adapters
//! (`history_adapters`) that turn an assistant turn + tool results into the
//! provider-specific message JSON. The provider SDK backends and the streaming
//! tool loop sit on top of these and require live API access, so they land
//! separately.

pub mod backends;
pub mod conversation;
pub mod history_adapters;
pub mod http;
pub mod request_builder;
pub mod runtime;
pub mod tool_loop;

use serde_json::{Map, Value};

/// The request-shaping subset of Python's `ModelConfig`. Only the fields the
/// deterministic request builder reads are modeled here; provider credentials,
/// fallback chains, and cache policy are layered in as the backends land.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelConfig {
    pub model: String,
    pub transport: Provider,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<i64>,
    pub frequency_penalty: Option<f64>,
    pub presence_penalty: Option<f64>,
    pub seed: Option<i64>,
    pub thinking_budget_tokens: Option<i64>,
    pub max_output_tokens: Option<i64>,
    pub stop_sequences: Option<Vec<String>>,
    /// Free-form provider passthrough merged last into `extra_params`.
    pub provider_params: Map<String, Value>,
    /// Resolved backup config used on the final retry attempt. The fallback's own
    /// `fallback` is always `None` (single-level), matching Python.
    pub fallback: Option<Box<ModelConfig>>,
}

impl ModelConfig {
    /// Minimal config for a model on a transport, all tuning knobs unset.
    pub fn new(model: impl Into<String>, transport: Provider) -> Self {
        Self {
            model: model.into(),
            transport,
            temperature: None,
            top_p: None,
            top_k: None,
            frequency_penalty: None,
            presence_penalty: None,
            seed: None,
            thinking_budget_tokens: None,
            max_output_tokens: None,
            stop_sequences: None,
            provider_params: Map::new(),
            fallback: None,
        }
    }
}

/// A normalized tool call extracted from any provider's response. Mirrors
/// `backend.ToolCallResult`. `input` is the decoded arguments object.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCallResult {
    pub id: String,
    pub name: String,
    pub input: Value,
    pub thought_signature: Option<String>,
}

/// A normalized completion result returned by provider backends, mirroring
/// `backend.CompletionResult`. `content` is `Any` in Python (usually a string);
/// we keep it as a [`Value`] so the adapters can apply the same
/// `isinstance(content, str)` checks. `raw_response` is intentionally omitted —
/// it is provider-SDK-specific and never consumed by the deterministic layer.
#[derive(Debug, Clone)]
pub struct CompletionResult {
    pub content: Value,
    pub input_tokens: i64,
    pub output_tokens: i64,
    pub cache_creation_input_tokens: i64,
    pub cache_read_input_tokens: i64,
    pub finish_reason: String,
    pub tool_calls: Vec<ToolCallResult>,
    pub thinking_content: Option<String>,
    pub thinking_blocks: Vec<Value>,
    pub reasoning_details: Vec<Value>,
}

impl Default for CompletionResult {
    fn default() -> Self {
        Self {
            content: Value::String(String::new()),
            input_tokens: 0,
            output_tokens: 0,
            cache_creation_input_tokens: 0,
            cache_read_input_tokens: 0,
            finish_reason: "stop".to_string(),
            tool_calls: Vec::new(),
            thinking_content: None,
            thinking_blocks: Vec::new(),
            reasoning_details: Vec::new(),
        }
    }
}

/// One tool execution result to fold back into the conversation. The Python
/// adapters read different keys per provider (`tool_id`, `tool_name`, `result`,
/// `is_error`); this unified struct carries all of them. `result` is the
/// already-stringified tool output (the Python adapters call `str(result)`,
/// and tool outputs in this system are strings).
#[derive(Debug, Clone)]
pub struct ToolResult {
    pub tool_id: String,
    pub tool_name: String,
    pub result: String,
    pub is_error: bool,
}

/// The LLM transport providers, mirroring `ModelTransport`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provider {
    Anthropic,
    Openai,
    Gemini,
}
