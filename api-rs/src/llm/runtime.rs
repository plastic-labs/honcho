//! Per-attempt planning, ported from `src/llm/runtime.py`.
//!
//! Only the deterministic selection logic is ported: which config a retry
//! attempt uses (primary vs. resolved fallback) and the retry temperature bump.
//! The parts of `plan_attempt` that resolve a live provider client / backend
//! stay with the SDK layer.

use super::ModelConfig;

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
    fn temperature_bumped_only_on_retries_from_zero() {
        assert_eq!(effective_temperature(Some(0.0), 1), Some(0.0));
        assert_eq!(effective_temperature(Some(0.0), 2), Some(0.2));
        assert_eq!(effective_temperature(Some(0.7), 3), Some(0.7));
        assert_eq!(effective_temperature(None, 2), None);
    }
}
