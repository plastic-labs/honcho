//! The completion executor, porting the dispatch core of `request_builder.py`
//! (`execute_completion`). Maps a [`ModelConfig`] plus per-call inputs into the
//! right provider backend's `RequestParams` and runs the call over an
//! [`LlmHttp`] transport.
//!
//! **Gemini is not yet routable here.** Its backend has no `complete()` send
//! path (only the deterministic config/response helpers are ported), and Python
//! delegates Gemini to the genai SDK — so there is no REST request body to port
//! byte-identically. [`execute_completion`] returns
//! [`ExecutorError::UnsupportedProvider`] for Gemini until that send path lands.

use serde_json::{Map, Value};

use super::backends::{anthropic, openai};
use super::http::{Credentials, LlmHttp, LlmHttpError};
use super::request_builder::{build_config_extra_params, effective_max_tokens};
use super::{CompletionResult, ModelConfig, Provider};

/// An error from [`execute_completion`].
#[derive(Debug)]
pub enum ExecutorError {
    /// The underlying provider HTTP call failed.
    Http(LlmHttpError),
    /// The config's transport has no ported send path (currently Gemini).
    UnsupportedProvider(Provider),
}

impl std::fmt::Display for ExecutorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExecutorError::Http(error) => write!(f, "{error}"),
            ExecutorError::UnsupportedProvider(provider) => {
                write!(f, "no send path for provider {provider:?}")
            }
        }
    }
}

impl std::error::Error for ExecutorError {}

impl From<LlmHttpError> for ExecutorError {
    fn from(error: LlmHttpError) -> Self {
        ExecutorError::Http(error)
    }
}

/// Run one completion, porting `execute_completion`: resolve the effective output
/// budget (`config.max_output_tokens or max_tokens`), merge the config's tuning
/// knobs with the caller's `extra_params` (caller wins), apply the stop-sequence
/// precedence (explicit `stop` else `config.stop_sequences`), then dispatch to
/// the transport's backend. Reasoning params are threaded per provider: Anthropic
/// takes only `thinking_budget_tokens`; OpenAI takes both the budget and
/// `thinking_effort`.
#[allow(clippy::too_many_arguments)]
pub async fn execute_completion<H: LlmHttp>(
    http: &H,
    credentials: &Credentials,
    config: &ModelConfig,
    messages: &[Value],
    max_tokens: i64,
    tools: Option<&[Value]>,
    tool_choice: Option<&Value>,
    stop: Option<&[String]>,
    extra_params: Option<&Map<String, Value>>,
) -> Result<CompletionResult, ExecutorError> {
    let max_tokens = effective_max_tokens(config, max_tokens);

    // Config-derived knobs first, caller-supplied extra params override.
    let mut merged_extra_params = build_config_extra_params(config);
    if let Some(extra) = extra_params {
        for (key, value) in extra {
            merged_extra_params.insert(key.clone(), value.clone());
        }
    }

    let effective_stop: Option<&[String]> = stop.or(config.stop_sequences.as_deref());

    match config.transport {
        Provider::Anthropic => {
            let params = anthropic::RequestParams {
                model: &config.model,
                messages,
                max_tokens,
                temperature: config.temperature,
                stop: effective_stop,
                tools,
                tool_choice,
                thinking_budget_tokens: config.thinking_budget_tokens,
                extra_params: &merged_extra_params,
            };
            Ok(anthropic::complete(http, credentials, &params).await?)
        }
        Provider::Openai => {
            let params = openai::RequestParams {
                model: &config.model,
                messages,
                max_tokens,
                temperature: config.temperature,
                stop: effective_stop,
                tools,
                tool_choice,
                thinking_effort: config.thinking_effort.as_deref(),
                thinking_budget_tokens: config.thinking_budget_tokens,
                extra_params: &merged_extra_params,
            };
            Ok(openai::complete(http, credentials, &params).await?)
        }
        Provider::Gemini => Err(ExecutorError::UnsupportedProvider(Provider::Gemini)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::http::mock::MockHttp;
    use serde_json::json;

    fn ok_http() -> MockHttp {
        MockHttp::ok(json!({
            "content": [{"type": "text", "text": "hi"}],
            "usage": {"input_tokens": 1, "output_tokens": 2},
            "stop_reason": "end_turn",
            "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}]
        }))
    }

    fn messages() -> Vec<Value> {
        vec![json!({"role": "user", "content": "hi"})]
    }

    #[tokio::test]
    async fn anthropic_routes_to_messages_endpoint() {
        let http = ok_http();
        let creds = Credentials::new("key");
        let config = ModelConfig::new("claude-x", Provider::Anthropic);
        execute_completion(
            &http, &creds, &config, &messages(), 1024, None, None, None, None,
        )
        .await
        .expect("completion");
        assert!(http.last_url().ends_with("/v1/messages"));
        assert_eq!(http.last_body()["model"], json!("claude-x"));
        assert_eq!(http.last_body()["max_tokens"], json!(1024));
    }

    #[tokio::test]
    async fn openai_routes_to_chat_completions_endpoint() {
        let http = ok_http();
        let creds = Credentials::new("key");
        let config = ModelConfig::new("gpt-x", Provider::Openai);
        execute_completion(
            &http, &creds, &config, &messages(), 512, None, None, None, None,
        )
        .await
        .expect("completion");
        assert!(http.last_url().ends_with("/chat/completions"));
        assert_eq!(http.last_body()["model"], json!("gpt-x"));
    }

    #[tokio::test]
    async fn gemini_is_unsupported() {
        let http = ok_http();
        let creds = Credentials::new("key");
        let config = ModelConfig::new("gemini-x", Provider::Gemini);
        let error = execute_completion(
            &http, &creds, &config, &messages(), 512, None, None, None, None,
        )
        .await
        .expect_err("gemini has no send path");
        assert!(matches!(
            error,
            ExecutorError::UnsupportedProvider(Provider::Gemini)
        ));
    }

    #[tokio::test]
    async fn effective_max_tokens_and_extra_params_flow_into_body() {
        let http = ok_http();
        let creds = Credentials::new("key");
        let mut config = ModelConfig::new("claude-x", Provider::Anthropic);
        config.max_output_tokens = Some(256); // wins over the per-call 1024
        config.top_p = Some(0.9); // flows in via build_config_extra_params
        execute_completion(
            &http, &creds, &config, &messages(), 1024, None, None, None, None,
        )
        .await
        .expect("completion");
        assert_eq!(http.last_body()["max_tokens"], json!(256));
        assert_eq!(http.last_body()["top_p"], json!(0.9));
    }

    #[tokio::test]
    async fn caller_stop_overrides_config_stop() {
        let http = ok_http();
        let creds = Credentials::new("key");
        let mut config = ModelConfig::new("claude-x", Provider::Anthropic);
        config.stop_sequences = Some(vec!["CONFIG".to_string()]);
        let stop = vec!["CALL".to_string()];
        execute_completion(
            &http,
            &creds,
            &config,
            &messages(),
            1024,
            None,
            None,
            Some(&stop),
            None,
        )
        .await
        .expect("completion");
        assert_eq!(http.last_body()["stop_sequences"], json!(["CALL"]));
    }
}
