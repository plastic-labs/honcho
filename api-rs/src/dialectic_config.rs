//! Dialectic reasoning-level configuration, ported from `config.py`
//! (`DialecticLevelSettings`, `DialecticSettings`, `_default_dialectic_levels`).
//!
//! Only the built-in defaults are ported here. Operator overrides (the
//! `DIALECTIC_LEVELS__<level>__MODEL_CONFIG__*` env merge) are not yet wired; the
//! read-only sidecar runs on these defaults.

use crate::llm::{ModelConfig, Provider};

/// The five dialectic reasoning tiers (Python `ReasoningLevel`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReasoningLevel {
    Minimal,
    Low,
    Medium,
    High,
    Max,
}

impl ReasoningLevel {
    /// Parse the wire string (`"minimal"`..`"max"`); unknown values yield `None`.
    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "minimal" => Some(Self::Minimal),
            "low" => Some(Self::Low),
            "medium" => Some(Self::Medium),
            "high" => Some(Self::High),
            "max" => Some(Self::Max),
            _ => None,
        }
    }

    /// The wire string for this level.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Minimal => "minimal",
            Self::Low => "low",
            Self::Medium => "medium",
            Self::High => "high",
            Self::Max => "max",
        }
    }
}

/// Per-level dialectic settings, porting `DialecticLevelSettings`. `max_output_tokens`
/// `None` means "use the global `DialecticSettings::max_output_tokens`"; `tool_choice`
/// `None`/`"auto"` lets the model decide.
#[derive(Debug, Clone, PartialEq)]
pub struct DialecticLevelSettings {
    pub model_config: ModelConfig,
    pub max_tool_iterations: u32,
    pub max_output_tokens: Option<i64>,
    pub tool_choice: Option<String>,
}

/// The whole dialectic config, porting `DialecticSettings`.
#[derive(Debug, Clone, PartialEq)]
pub struct DialecticSettings {
    minimal: DialecticLevelSettings,
    low: DialecticLevelSettings,
    medium: DialecticLevelSettings,
    high: DialecticLevelSettings,
    max: DialecticLevelSettings,
    pub max_output_tokens: i64,
    pub max_input_tokens: i64,
    pub history_token_limit: i64,
    pub session_history_max_tokens: i64,
}

impl DialecticSettings {
    /// The settings for `level`.
    pub fn level(&self, level: ReasoningLevel) -> &DialecticLevelSettings {
        match level {
            ReasoningLevel::Minimal => &self.minimal,
            ReasoningLevel::Low => &self.low,
            ReasoningLevel::Medium => &self.medium,
            ReasoningLevel::High => &self.high,
            ReasoningLevel::Max => &self.max,
        }
    }

    /// The effective output-token budget for `level`: the level's own override
    /// when set, else the global `max_output_tokens` (Python's "None means use
    /// global DIALECTIC.MAX_OUTPUT_TOKENS").
    pub fn effective_max_output_tokens(&self, level: ReasoningLevel) -> i64 {
        self.level(level)
            .max_output_tokens
            .unwrap_or(self.max_output_tokens)
    }
}

impl Default for DialecticSettings {
    /// The built-in defaults from `_default_dialectic_levels` + the
    /// `DialecticSettings` field defaults. Every level defaults to the same model
    /// (`openai/gpt-5.4-mini`); only the per-level tuning differs.
    fn default() -> Self {
        let default_model = || ModelConfig::new("gpt-5.4-mini", Provider::Openai);
        Self {
            minimal: DialecticLevelSettings {
                model_config: default_model(),
                max_tool_iterations: 1,
                max_output_tokens: Some(250),
                tool_choice: Some("auto".to_string()),
            },
            low: DialecticLevelSettings {
                model_config: default_model(),
                max_tool_iterations: 5,
                max_output_tokens: None,
                tool_choice: Some("auto".to_string()),
            },
            medium: DialecticLevelSettings {
                model_config: default_model(),
                max_tool_iterations: 2,
                max_output_tokens: None,
                tool_choice: None,
            },
            high: DialecticLevelSettings {
                model_config: default_model(),
                max_tool_iterations: 4,
                max_output_tokens: None,
                tool_choice: None,
            },
            max: DialecticLevelSettings {
                model_config: default_model(),
                max_tool_iterations: 10,
                max_output_tokens: None,
                tool_choice: None,
            },
            max_output_tokens: 8192,
            max_input_tokens: 100_000,
            history_token_limit: 8192,
            session_history_max_tokens: 4_096,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reasoning_level_roundtrips() {
        for level in [
            ReasoningLevel::Minimal,
            ReasoningLevel::Low,
            ReasoningLevel::Medium,
            ReasoningLevel::High,
            ReasoningLevel::Max,
        ] {
            assert_eq!(ReasoningLevel::parse(level.as_str()), Some(level));
        }
        assert_eq!(ReasoningLevel::parse("bogus"), None);
    }

    #[test]
    fn default_level_tuning_matches_python() {
        let settings = DialecticSettings::default();
        // (max_tool_iterations, max_output_tokens, tool_choice)
        assert_eq!(settings.level(ReasoningLevel::Minimal).max_tool_iterations, 1);
        assert_eq!(
            settings.level(ReasoningLevel::Minimal).max_output_tokens,
            Some(250)
        );
        assert_eq!(
            settings.level(ReasoningLevel::Minimal).tool_choice.as_deref(),
            Some("auto")
        );
        assert_eq!(settings.level(ReasoningLevel::Low).max_tool_iterations, 5);
        assert_eq!(
            settings.level(ReasoningLevel::Low).tool_choice.as_deref(),
            Some("auto")
        );
        assert_eq!(settings.level(ReasoningLevel::Medium).max_tool_iterations, 2);
        assert_eq!(settings.level(ReasoningLevel::Medium).tool_choice, None);
        assert_eq!(settings.level(ReasoningLevel::High).max_tool_iterations, 4);
        assert_eq!(settings.level(ReasoningLevel::Max).max_tool_iterations, 10);
    }

    #[test]
    fn default_model_is_openai_gpt54_mini() {
        let settings = DialecticSettings::default();
        let model = &settings.level(ReasoningLevel::Medium).model_config;
        assert_eq!(model.model, "gpt-5.4-mini");
        assert_eq!(model.transport, Provider::Openai);
    }

    #[test]
    fn effective_output_tokens_uses_global_when_level_unset() {
        let settings = DialecticSettings::default();
        // minimal has its own 250; others fall back to the global 8192.
        assert_eq!(
            settings.effective_max_output_tokens(ReasoningLevel::Minimal),
            250
        );
        assert_eq!(
            settings.effective_max_output_tokens(ReasoningLevel::High),
            8192
        );
    }

    #[test]
    fn global_token_budgets_match_python() {
        let settings = DialecticSettings::default();
        assert_eq!(settings.max_output_tokens, 8192);
        assert_eq!(settings.max_input_tokens, 100_000);
        assert_eq!(settings.history_token_limit, 8192);
        assert_eq!(settings.session_history_max_tokens, 4_096);
    }
}
