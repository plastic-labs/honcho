//! Deterministic request assembly, ported from `src/llm/request_builder.py`.
//!
//! The async `execute_completion` / `execute_stream` wrappers there need a live
//! provider backend, so only the pure helpers are ported here:
//! [`build_config_extra_params`] (flatten tuning knobs + `provider_params`) and
//! [`effective_max_tokens`] (the `config.max_output_tokens or max_tokens` rule).

use serde_json::{Map, Value, json};

use super::ModelConfig;

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
        let params = build_config_extra_params(&config);
        assert_eq!(params.get("top_p"), Some(&json!(0.9)));
        assert_eq!(params.get("seed"), Some(&json!(7)));
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
    fn effective_max_tokens_prefers_nonzero_config() {
        let mut config = ModelConfig::new("gpt-x", Provider::Openai);
        assert_eq!(effective_max_tokens(&config, 1024), 1024);
        config.max_output_tokens = Some(0); // falsy -> fall back
        assert_eq!(effective_max_tokens(&config, 1024), 1024);
        config.max_output_tokens = Some(256);
        assert_eq!(effective_max_tokens(&config, 1024), 256);
    }
}
