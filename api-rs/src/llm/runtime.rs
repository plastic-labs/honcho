//! Per-attempt planning, ported from `src/llm/runtime.py`.
//!
//! Only the deterministic selection logic is ported: which config a retry
//! attempt uses (primary vs. resolved fallback) and the retry temperature bump.
//! The parts of `plan_attempt` that resolve a live provider client / backend
//! stay with the SDK layer.

use super::{ModelConfig, Provider};

/// Whether `attempt` is the fallback attempt: the final retry, when a fallback
/// config exists. Mirrors the branch condition in `select_model_config_for_attempt`.
pub fn is_fallback_attempt(config: &ModelConfig, attempt: u32, retry_attempts: u32) -> bool {
    attempt == retry_attempts && config.fallback.is_some()
}

/// Pick the effective config for `attempt`, porting `select_model_config_for_attempt`:
/// the primary config on every attempt except the last, which swaps to the
/// resolved fallback (whose own `fallback` is already `None`).
pub fn select_model_config_for_attempt(
    config: &ModelConfig,
    attempt: u32,
    retry_attempts: u32,
) -> ModelConfig {
    match &config.fallback {
        Some(fallback) if attempt == retry_attempts => (**fallback).clone(),
        _ => config.clone(),
    }
}

/// Bump temperature `0.0 → 0.2` on retry attempts (attempt > 1) for output
/// variety, porting `effective_temperature`. The Python version reads the
/// current attempt from a context var; here it is an explicit parameter.
pub fn effective_temperature(temperature: Option<f64>, attempt: u32) -> Option<f64> {
    if temperature == Some(0.0) && attempt > 1 {
        Some(0.2)
    } else {
        temperature
    }
}

/// Build the `ModelConfig` passed to the request builder, porting
/// `effective_config_for_call`. Per-call values (temperature, stop sequences,
/// thinking budget/effort) win when set; otherwise the `selected_config`'s
/// values are kept. `max_output_tokens` is always forced to `None` so the
/// per-call `max_tokens` kwarg is authoritative. With no `selected_config`
/// (test-only callers passing provider+model directly) a minimal config is
/// synthesized carrying just the per-call values.
#[allow(clippy::too_many_arguments)]
pub fn effective_config_for_call(
    selected_config: Option<&ModelConfig>,
    provider: Provider,
    model: &str,
    temperature: Option<f64>,
    stop_seqs: Option<&[String]>,
    thinking_budget_tokens: Option<i64>,
    reasoning_effort: Option<&str>,
) -> ModelConfig {
    let Some(selected) = selected_config else {
        let mut config = ModelConfig::new(model, provider);
        config.temperature = temperature;
        config.stop_sequences = stop_seqs.map(<[String]>::to_vec);
        config.thinking_budget_tokens = thinking_budget_tokens;
        config.thinking_effort = reasoning_effort.map(str::to_string);
        return config;
    };

    let mut config = selected.clone();
    config.max_output_tokens = None;
    if temperature.is_some() {
        config.temperature = temperature;
    }
    if let Some(stop_seqs) = stop_seqs {
        config.stop_sequences = Some(stop_seqs.to_vec());
    }
    if thinking_budget_tokens.is_some() {
        config.thinking_budget_tokens = thinking_budget_tokens;
    }
    if let Some(reasoning_effort) = reasoning_effort {
        config.thinking_effort = Some(reasoning_effort.to_string());
    }
    config
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::Provider;

    fn config_with_fallback() -> ModelConfig {
        let mut primary = ModelConfig::new("primary-model", Provider::Anthropic);
        primary.fallback = Some(Box::new(ModelConfig::new("backup-model", Provider::Openai)));
        primary
    }

    #[test]
    fn primary_used_until_final_attempt() {
        let config = config_with_fallback();
        // retry_attempts = 3; attempts 1 and 2 stay on primary.
        for attempt in [1, 2] {
            let selected = select_model_config_for_attempt(&config, attempt, 3);
            assert_eq!(selected.model, "primary-model");
            assert_eq!(selected.transport, Provider::Anthropic);
            assert!(!is_fallback_attempt(&config, attempt, 3));
        }
    }

    #[test]
    fn fallback_used_on_final_attempt() {
        let config = config_with_fallback();
        let selected = select_model_config_for_attempt(&config, 3, 3);
        assert_eq!(selected.model, "backup-model");
        assert_eq!(selected.transport, Provider::Openai);
        assert!(selected.fallback.is_none());
        assert!(is_fallback_attempt(&config, 3, 3));
    }

    #[test]
    fn no_fallback_stays_primary_even_on_final_attempt() {
        let config = ModelConfig::new("only-model", Provider::Gemini);
        let selected = select_model_config_for_attempt(&config, 3, 3);
        assert_eq!(selected.model, "only-model");
        assert!(!is_fallback_attempt(&config, 3, 3));
    }

    #[test]
    fn effective_config_synthesizes_when_no_selected() {
        let stop = vec!["STOP".to_string()];
        let config = effective_config_for_call(
            None,
            Provider::Openai,
            "gpt-x",
            Some(0.3),
            Some(&stop),
            Some(512),
            Some("high"),
        );
        assert_eq!(config.model, "gpt-x");
        assert_eq!(config.transport, Provider::Openai);
        assert_eq!(config.temperature, Some(0.3));
        assert_eq!(config.stop_sequences, Some(stop));
        assert_eq!(config.thinking_budget_tokens, Some(512));
        assert_eq!(config.thinking_effort.as_deref(), Some("high"));
        assert_eq!(config.max_output_tokens, None);
    }

    #[test]
    fn effective_config_overrides_only_set_values_and_forces_max_tokens_none() {
        let mut selected = ModelConfig::new("base-model", Provider::Anthropic);
        selected.temperature = Some(0.9);
        selected.thinking_budget_tokens = Some(2048);
        selected.thinking_effort = Some("low".to_string());
        selected.stop_sequences = Some(vec!["BASE".to_string()]);
        selected.max_output_tokens = Some(4096);

        // All per-call values None -> selected values preserved, max_tokens forced None.
        let kept = effective_config_for_call(
            Some(&selected),
            Provider::Anthropic,
            "base-model",
            None,
            None,
            None,
            None,
        );
        assert_eq!(kept.temperature, Some(0.9));
        assert_eq!(kept.thinking_budget_tokens, Some(2048));
        assert_eq!(kept.thinking_effort.as_deref(), Some("low"));
        assert_eq!(kept.stop_sequences, Some(vec!["BASE".to_string()]));
        assert_eq!(kept.max_output_tokens, None);

        // Per-call values present -> they win.
        let overridden = effective_config_for_call(
            Some(&selected),
            Provider::Anthropic,
            "base-model",
            Some(0.1),
            Some(&["CALL".to_string()]),
            Some(128),
            Some("max"),
        );
        assert_eq!(overridden.temperature, Some(0.1));
        assert_eq!(overridden.thinking_budget_tokens, Some(128));
        assert_eq!(overridden.thinking_effort.as_deref(), Some("max"));
        assert_eq!(overridden.stop_sequences, Some(vec!["CALL".to_string()]));
        assert_eq!(overridden.max_output_tokens, None);
    }

    #[test]
    fn temperature_bumped_only_on_retries_from_zero() {
        assert_eq!(effective_temperature(Some(0.0), 1), Some(0.0));
        assert_eq!(effective_temperature(Some(0.0), 2), Some(0.2));
        assert_eq!(effective_temperature(Some(0.7), 3), Some(0.7));
        assert_eq!(effective_temperature(None, 2), None);
    }
}
