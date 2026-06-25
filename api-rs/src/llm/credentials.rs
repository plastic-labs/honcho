//! Per-transport credential resolution, ported from `src/llm/credentials.py`.
//!
//! Python reads the global keys off `settings.LLM`; to keep `llm/` decoupled from
//! the config layer (as the existing [`Credentials`] type already is), the global
//! per-transport keys are passed in as [`TransportApiKeys`] rather than imported.

use super::http::Credentials;
use super::{ModelConfig, Provider};

/// The global per-transport API keys (Python `settings.LLM.{ANTHROPIC,OPENAI,
/// GEMINI}_API_KEY`), used as the fallback when a `ModelConfig` has no `api_key`.
#[derive(Debug, Clone, Default)]
pub struct TransportApiKeys {
    pub anthropic: Option<String>,
    pub openai: Option<String>,
    pub gemini: Option<String>,
}

impl TransportApiKeys {
    /// The global key for `transport`, porting `default_transport_api_key`.
    /// (Python raises on an unknown transport string; the Rust [`Provider`] enum
    /// makes that case unrepresentable.)
    pub fn for_transport(&self, transport: Provider) -> Option<&str> {
        match transport {
            Provider::Anthropic => self.anthropic.as_deref(),
            Provider::Openai => self.openai.as_deref(),
            Provider::Gemini => self.gemini.as_deref(),
        }
    }
}

/// Resolve the [`Credentials`] for a config's transport, porting
/// `resolve_credentials`: the config's own `api_key` wins, else the global
/// per-transport key. The `base_url` override passes through. An unconfigured key
/// resolves to an empty string (Python passes `None` down to the provider SDK,
/// which then fails auth — the empty-key request fails the same way).
pub fn resolve_credentials(config: &ModelConfig, keys: &TransportApiKeys) -> Credentials {
    let api_key = config
        .api_key
        .clone()
        .or_else(|| keys.for_transport(config.transport).map(str::to_string))
        .unwrap_or_default();
    Credentials {
        api_key,
        base_url: config.base_url.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn keys() -> TransportApiKeys {
        TransportApiKeys {
            anthropic: Some("anthropic-global".to_string()),
            openai: Some("openai-global".to_string()),
            gemini: None,
        }
    }

    #[test]
    fn config_api_key_wins_over_global() {
        let mut config = ModelConfig::new("claude-x", Provider::Anthropic);
        config.api_key = Some("per-config".to_string());
        let creds = resolve_credentials(&config, &keys());
        assert_eq!(creds.api_key, "per-config");
        assert_eq!(creds.base_url, None);
    }

    #[test]
    fn falls_back_to_global_transport_key() {
        let config = ModelConfig::new("gpt-x", Provider::Openai);
        let creds = resolve_credentials(&config, &keys());
        assert_eq!(creds.api_key, "openai-global");
    }

    #[test]
    fn missing_key_resolves_to_empty_string() {
        let config = ModelConfig::new("gemini-x", Provider::Gemini);
        let creds = resolve_credentials(&config, &keys());
        assert_eq!(creds.api_key, "");
    }

    #[test]
    fn base_url_override_passes_through() {
        let mut config = ModelConfig::new("gpt-x", Provider::Openai);
        config.base_url = Some("https://relay.example/v1".to_string());
        let creds = resolve_credentials(&config, &keys());
        assert_eq!(creds.base_url.as_deref(), Some("https://relay.example/v1"));
    }
}
