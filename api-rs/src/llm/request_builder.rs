//! Deterministic request assembly, ported from `src/llm/request_builder.py`.
//!
//! The async `execute_completion` / `execute_stream` wrappers there need a live
//! provider backend, so only the pure helpers are ported here:
//! [`build_config_extra_params`] (flatten tuning knobs + `provider_params`) and
//! [`effective_max_tokens`] (the `config.max_output_tokens or max_tokens` rule).

use std::fmt;

use serde_json::{Map, Value, json};

use super::ModelConfig;

/// Operator escape-hatch keys recognized inside `ModelConfig.provider_params`.
pub const PASSTHROUGH_KEYS: [&str; 3] = ["extra_body", "extra_headers", "extra_query"];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PassthroughError {
    key: String,
    actual_type: &'static str,
}

impl PassthroughError {
    pub(crate) fn new(key: &str, value: &Value) -> Self {
        Self {
            key: key.to_string(),
            actual_type: value_type_name(value),
        }
    }
}

impl fmt::Display for PassthroughError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "provider_params.{} must be a mapping, got {}",
            self.key, self.actual_type
        )
    }
}

impl std::error::Error for PassthroughError {}

fn value_type_name(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "bool",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

/// Validate an operator-supplied passthrough value is a JSON object.
pub fn coerce_passthrough_mapping<'a>(
    key: &str,
    value: &'a Value,
) -> Result<&'a Map<String, Value>, PassthroughError> {
    value
        .as_object()
        .ok_or_else(|| PassthroughError::new(key, value))
}

pub fn passthrough_mapping<'a>(
    key: &str,
    extra_params: &'a Map<String, Value>,
) -> Result<Option<&'a Map<String, Value>>, PassthroughError> {
    extra_params
        .get(key)
        .map(|value| coerce_passthrough_mapping(key, value))
        .transpose()
}

pub fn passthrough_value_to_string(value: &Value) -> String {
    value
        .as_str()
        .map(str::to_string)
        .unwrap_or_else(|| value.to_string())
}

pub fn merge_header_mapping(headers: &mut Vec<(String, String)>, mapping: &Map<String, Value>) {
    for (name, value) in mapping {
        let value = passthrough_value_to_string(value);
        if let Some((_, existing_value)) = headers
            .iter_mut()
            .find(|(existing_name, _)| existing_name.eq_ignore_ascii_case(name))
        {
            *existing_value = value;
        } else {
            headers.push((name.clone(), value));
        }
    }
}

/// Forward provider SDK passthroughs onto a request/options map.
///
/// Values shallow-merge and operator-supplied entries win over existing entries.
pub fn apply_sdk_passthroughs(
    params: &mut Map<String, Value>,
    extra_params: &Map<String, Value>,
) -> Result<(), PassthroughError> {
    for passthrough_key in PASSTHROUGH_KEYS {
        let Some(operator_value) = extra_params.get(passthrough_key) else {
            continue;
        };
        let operator_mapping = coerce_passthrough_mapping(passthrough_key, operator_value)?;
        if operator_mapping.is_empty() {
            continue;
        }
        let existing = params
            .entry(passthrough_key.to_string())
            .or_insert_with(|| Value::Object(Map::new()));
        if !existing.is_object() {
            return Err(PassthroughError::new(passthrough_key, existing));
        }
        let existing_mapping = existing.as_object_mut().expect("object checked above");
        for (key, value) in operator_mapping {
            existing_mapping.insert(key.clone(), value.clone());
        }
    }
    Ok(())
}

/// Flatten a [`ModelConfig`]'s optional tuning knobs and `provider_params` into
/// the `extra_params` map backends read. Ports `build_config_extra_params`: the
/// known knobs are inserted first (when set), then `provider_params` is merged
/// last so it can override them.
pub fn build_config_extra_params(config: &ModelConfig) -> Map<String, Value> {
    let mut extra_params = Map::new();
    if let Some(top_p) = config.top_p {
        extra_params.insert("top_p".to_string(), json!(top_p));
    }
    if let Some(top_k) = config.top_k {
        extra_params.insert("top_k".to_string(), json!(top_k));
    }
    if let Some(frequency_penalty) = config.frequency_penalty {
        extra_params.insert("frequency_penalty".to_string(), json!(frequency_penalty));
    }
    if let Some(presence_penalty) = config.presence_penalty {
        extra_params.insert("presence_penalty".to_string(), json!(presence_penalty));
    }
    if let Some(seed) = config.seed {
        extra_params.insert("seed".to_string(), json!(seed));
    }
    if let Some(mode) = config.structured_output_mode {
        extra_params.insert("structured_output_mode".to_string(), json!(mode.as_str()));
    }
    for (key, value) in &config.provider_params {
        extra_params.insert(key.clone(), value.clone());
    }
    extra_params
}

/// Resolve the effective output-token budget, porting `config.max_output_tokens
/// or max_tokens`: a set, non-zero `max_output_tokens` wins; `None` or `0` falls
/// back to the caller's `max_tokens` (Python treats `0` as falsy here).
pub fn effective_max_tokens(config: &ModelConfig, max_tokens: i64) -> i64 {
    match config.max_output_tokens {
        Some(value) if value != 0 => value,
        _ => max_tokens,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::Provider;

    #[test]
    fn extra_params_empty_when_no_knobs_set() {
        let config = ModelConfig::new("gpt-x", Provider::Openai);
        assert_eq!(build_config_extra_params(&config), Map::new());
    }

    #[test]
    fn extra_params_includes_set_knobs() {
        let mut config = ModelConfig::new("gpt-x", Provider::Openai);
        config.top_p = Some(0.9);
        config.seed = Some(7);
        config.structured_output_mode = Some(crate::llm::StructuredOutputMode::JsonObject);
        let params = build_config_extra_params(&config);
        assert_eq!(params.get("top_p"), Some(&json!(0.9)));
        assert_eq!(params.get("seed"), Some(&json!(7)));
        assert_eq!(
            params.get("structured_output_mode"),
            Some(&json!("json_object"))
        );
        assert_eq!(params.get("top_k"), None);
    }

    #[test]
    fn provider_params_override_known_knobs() {
        let mut config = ModelConfig::new("gpt-x", Provider::Openai);
        config.top_p = Some(0.9);
        config
            .provider_params
            .insert("top_p".to_string(), json!(0.1));
        config
            .provider_params
            .insert("custom".to_string(), json!("v"));
        let params = build_config_extra_params(&config);
        // provider_params is merged last, so it wins.
        assert_eq!(params.get("top_p"), Some(&json!(0.1)));
        assert_eq!(params.get("custom"), Some(&json!("v")));
    }

    #[test]
    fn apply_sdk_passthroughs_shallow_merges_operator_wins() {
        let mut params = Map::new();
        params.insert(
            "extra_body".to_string(),
            json!({"reasoning": {"max_tokens": 100}, "keep": true}),
        );
        let mut extra = Map::new();
        extra.insert(
            "extra_body".to_string(),
            json!({"reasoning": {"max_tokens": 200}, "custom": "v"}),
        );
        extra.insert("extra_headers".to_string(), json!({"X-Test": "yes"}));

        apply_sdk_passthroughs(&mut params, &extra).unwrap();

        assert_eq!(
            params["extra_body"],
            json!({"reasoning": {"max_tokens": 200}, "keep": true, "custom": "v"})
        );
        assert_eq!(params["extra_headers"], json!({"X-Test": "yes"}));
    }

    #[test]
    fn coerce_passthrough_mapping_rejects_non_mapping() {
        let err = coerce_passthrough_mapping("extra_headers", &json!(["bad"])).unwrap_err();
        assert_eq!(
            err.to_string(),
            "provider_params.extra_headers must be a mapping, got array"
        );
    }

    #[test]
    fn effective_max_tokens_prefers_nonzero_config() {
        let mut config = ModelConfig::new("gpt-x", Provider::Openai);
        assert_eq!(effective_max_tokens(&config, 1024), 1024);
        config.max_output_tokens = Some(0); // falsy -> fall back
        assert_eq!(effective_max_tokens(&config, 1024), 1024);
        config.max_output_tokens = Some(256);
        assert_eq!(effective_max_tokens(&config, 1024), 256);
    }
}
