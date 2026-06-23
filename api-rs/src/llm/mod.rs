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
pub mod credentials;
pub mod executor;
pub mod history_adapters;
pub mod http;
pub mod request_builder;
pub mod runtime;
pub mod tool_loop;
pub mod types;

use serde_json::{Map, Value};

/// The request-shaping subset of Python's `ModelConfig`. Only the fields the
/// deterministic request builder reads are modeled here; provider credentials,
/// fallback chains, and cache policy are layered in as the backends land.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelConfig {
    pub model: String,
    pub transport: Provider,
    /// Per-config API key override; `None` falls back to the global per-transport
    /// key during credential resolution (`credentials::resolve_credentials`).
    pub api_key: Option<String>,
    /// Per-config base-URL override (e.g. an OpenRouter relay); `None` uses the
    /// backend's default endpoint.
    pub base_url: Option<String>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<i64>,
    pub frequency_penalty: Option<f64>,
    pub presence_penalty: Option<f64>,
    pub seed: Option<i64>,
    pub thinking_budget_tokens: Option<i64>,
    /// The reasoning-effort knob (`thinking_effort` in Python — one of
    /// `none`/`minimal`/`low`/`medium`/`high`/`xhigh`/`max`, or unset). Modeled
    /// as a plain string, matching the backend request builders which thread it
    /// through to the provider param verbatim.
    pub thinking_effort: Option<String>,
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
            api_key: None,
            base_url: None,
            temperature: None,
            top_p: None,
            top_k: None,
            frequency_penalty: None,
            presence_penalty: None,
            seed: None,
            thinking_budget_tokens: None,
            thinking_effort: None,
            max_output_tokens: None,
            stop_sequences: None,
            provider_params: Map::new(),
            fallback: None,
        }
    }

    /// Apply nested `{prefix}__{FIELD}` environment overrides onto this config,
    /// returning the updated config. Mirrors Python's `env_nested_delimiter="__"`
    /// parsing of `ConfiguredModelSettings` (e.g. `DERIVER_MODEL_CONFIG__MODEL`,
    /// `SUMMARY_MODEL_CONFIG__TRANSPORT`). Recognized fields: `TRANSPORT`, `MODEL`,
    /// `MAX_OUTPUT_TOKENS`, `TEMPERATURE`, `TOP_P`, `TOP_K`, `FREQUENCY_PENALTY`,
    /// `PRESENCE_PENALTY`, `SEED`, `THINKING_BUDGET_TOKENS`, `THINKING_EFFORT`
    /// (alias `REASONING_EFFORT`), `BASE_URL`, `API_KEY`. Blank or unparseable
    /// values leave the existing value untouched.
    ///
    /// Deviation: fallback chains (`…__FALLBACK__*`), `STOP_SEQUENCES`,
    /// `CACHE_POLICY`, and free-form provider params are not parsed here —
    /// `fallback` and `stop_sequences` keep their existing values.
    pub fn with_env_overrides(
        mut self,
        values: &std::collections::HashMap<String, String>,
        prefix: &str,
    ) -> Self {
        let get = |field: &str| -> Option<String> {
            values
                .get(&format!("{prefix}__{field}"))
                .map(|value| value.trim())
                .filter(|value| !value.is_empty())
                .map(str::to_string)
        };

        if let Some(transport) = get("TRANSPORT").and_then(|t| Provider::from_transport(&t)) {
            self.transport = transport;
        }
        if let Some(model) = get("MODEL") {
            self.model = model;
        }
        if let Some(base_url) = get("BASE_URL") {
            self.base_url = Some(base_url);
        }
        if let Some(api_key) = get("API_KEY") {
            self.api_key = Some(api_key);
        }
        if let Some(effort) = get("THINKING_EFFORT").or_else(|| get("REASONING_EFFORT")) {
            self.thinking_effort = Some(effort);
        }
        if let Some(value) = get("MAX_OUTPUT_TOKENS").and_then(|v| v.parse::<i64>().ok()) {
            self.max_output_tokens = Some(value);
        }
        if let Some(value) = get("THINKING_BUDGET_TOKENS").and_then(|v| v.parse::<i64>().ok()) {
            self.thinking_budget_tokens = Some(value);
        }
        if let Some(value) = get("TEMPERATURE").and_then(|v| v.parse::<f64>().ok()) {
            self.temperature = Some(value);
        }
        if let Some(value) = get("TOP_P").and_then(|v| v.parse::<f64>().ok()) {
            self.top_p = Some(value);
        }
        if let Some(value) = get("TOP_K").and_then(|v| v.parse::<i64>().ok()) {
            self.top_k = Some(value);
        }
        if let Some(value) = get("FREQUENCY_PENALTY").and_then(|v| v.parse::<f64>().ok()) {
            self.frequency_penalty = Some(value);
        }
        if let Some(value) = get("PRESENCE_PENALTY").and_then(|v| v.parse::<f64>().ok()) {
            self.presence_penalty = Some(value);
        }
        if let Some(value) = get("SEED").and_then(|v| v.parse::<i64>().ok()) {
            self.seed = Some(value);
        }

        self
    }
}

impl Provider {
    /// Parse a `ModelTransport` string (`anthropic`/`openai`/`gemini`,
    /// case-insensitive). Returns `None` for anything else (the caller keeps its
    /// current transport).
    pub fn from_transport(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "anthropic" => Some(Provider::Anthropic),
            "openai" => Some(Provider::Openai),
            "gemini" => Some(Provider::Gemini),
            _ => None,
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

#[cfg(test)]
mod env_override_tests {
    use super::*;
    use std::collections::HashMap;

    fn map(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn provider_from_transport_parses_known_values() {
        assert_eq!(Provider::from_transport("OpenAI"), Some(Provider::Openai));
        assert_eq!(Provider::from_transport("anthropic"), Some(Provider::Anthropic));
        assert_eq!(Provider::from_transport(" gemini "), Some(Provider::Gemini));
        assert_eq!(Provider::from_transport("bedrock"), None);
    }

    #[test]
    fn env_overrides_apply_nested_fields() {
        let values = map(&[
            ("DERIVER_MODEL_CONFIG__TRANSPORT", "anthropic"),
            ("DERIVER_MODEL_CONFIG__MODEL", "claude-x"),
            ("DERIVER_MODEL_CONFIG__MAX_OUTPUT_TOKENS", "4096"),
            ("DERIVER_MODEL_CONFIG__THINKING_EFFORT", "high"),
            ("DERIVER_MODEL_CONFIG__TEMPERATURE", "0.7"),
            ("DERIVER_MODEL_CONFIG__BASE_URL", "https://relay.example/v1"),
        ]);
        let config = ModelConfig::new("gpt-5.4-mini", Provider::Openai)
            .with_env_overrides(&values, "DERIVER_MODEL_CONFIG");
        assert_eq!(config.transport, Provider::Anthropic);
        assert_eq!(config.model, "claude-x");
        assert_eq!(config.max_output_tokens, Some(4096));
        assert_eq!(config.thinking_effort.as_deref(), Some("high"));
        assert_eq!(config.temperature, Some(0.7));
        assert_eq!(config.base_url.as_deref(), Some("https://relay.example/v1"));
    }

    #[test]
    fn reasoning_effort_alias_and_blanks_ignored() {
        let values = map(&[
            ("X__REASONING_EFFORT", "low"),
            ("X__MODEL", "   "),
            ("X__MAX_OUTPUT_TOKENS", "not-a-number"),
        ]);
        let config =
            ModelConfig::new("base-model", Provider::Openai).with_env_overrides(&values, "X");
        assert_eq!(config.thinking_effort.as_deref(), Some("low"));
        assert_eq!(config.model, "base-model"); // blank ignored
        assert_eq!(config.max_output_tokens, None); // unparseable ignored
    }

    #[test]
    fn no_overrides_keeps_base() {
        let config = ModelConfig::new("base-model", Provider::Openai)
            .with_env_overrides(&HashMap::new(), "DERIVER_MODEL_CONFIG");
        assert_eq!(config.model, "base-model");
        assert_eq!(config.transport, Provider::Openai);
    }
}
