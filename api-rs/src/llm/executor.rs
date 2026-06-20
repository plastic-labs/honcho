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

use serde_json::{Map, Value, json};

use super::backends::{anthropic, openai};
use super::http::{Credentials, LlmHttp, LlmHttpError};
use super::request_builder::{build_config_extra_params, effective_max_tokens};
use super::runtime::effective_config_for_call;
use super::types::{HonchoLLMCallResponse, completion_result_to_response};
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

/// One backend call returning the public response, porting the non-stream path
/// of `executor.py::honcho_llm_call_inner`: resolve the effective config
/// ([`effective_config_for_call`]), thread the per-call `json_mode`/`verbosity`
/// toggles through `extra_params`, run [`execute_completion`], and map the
/// backend result with [`completion_result_to_response`]. Retry, fallback, and
/// the tool loop sit *above* this (the `api.py::honcho_llm_call` layer).
///
/// `stop_seqs` is folded into the effective config here, so `execute_completion`
/// is invoked with no separate `stop` (it reads `config.stop_sequences`).
#[allow(clippy::too_many_arguments)]
pub async fn honcho_llm_call_inner<H: LlmHttp>(
    http: &H,
    credentials: &Credentials,
    provider: Provider,
    model: &str,
    messages: &[Value],
    max_tokens: i64,
    selected_config: Option<&ModelConfig>,
    temperature: Option<f64>,
    stop_seqs: Option<&[String]>,
    thinking_budget_tokens: Option<i64>,
    reasoning_effort: Option<&str>,
    tools: Option<&[Value]>,
    tool_choice: Option<&Value>,
    json_mode: bool,
    verbosity: Option<&str>,
) -> Result<HonchoLLMCallResponse, ExecutorError> {
    let effective_config = effective_config_for_call(
        selected_config,
        provider,
        model,
        temperature,
        stop_seqs,
        thinking_budget_tokens,
        reasoning_effort,
    );

    // Per-call transport toggles, not ModelConfig knobs (Python always sets both
    // keys, verbosity as null when unset).
    let mut call_extras = Map::new();
    call_extras.insert("json_mode".to_string(), json!(json_mode));
    call_extras.insert(
        "verbosity".to_string(),
        verbosity.map_or(Value::Null, |value| json!(value)),
    );

    let result = execute_completion(
        http,
        credentials,
        &effective_config,
        messages,
        max_tokens,
        tools,
        tool_choice,
        None,
        Some(&call_extras),
    )
    .await?;
    Ok(completion_result_to_response(&result))
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
    async fn inner_call_maps_backend_result_to_response() {
        let http = ok_http();
        let creds = Credentials::new("key");
        let response = honcho_llm_call_inner(
            &http,
            &creds,
            Provider::Anthropic,
            "claude-x",
            &messages(),
            1024,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            false,
            None,
        )
        .await
        .expect("inner call");
        assert_eq!(response.content, json!("hi"));
        assert_eq!(response.input_tokens, 1);
        assert_eq!(response.output_tokens, 2);
        assert_eq!(response.finish_reasons, vec!["end_turn".to_string()]);
        assert!(http.last_url().ends_with("/v1/messages"));
    }

    #[tokio::test]
    async fn inner_call_folds_stop_seqs_and_reasoning_effort() {
        let http = ok_http();
        let creds = Credentials::new("key");
        let stop = vec!["DONE".to_string()];
        honcho_llm_call_inner(
            &http,
            &creds,
            Provider::Openai,
            "gpt-x",
            &messages(),
            1024,
            None,
            None,
            Some(&stop),
            None,
            Some("high"),
            None,
            None,
            false,
            None,
        )
        .await
        .expect("inner call");
        let body = http.last_body();
        assert_eq!(body["stop"], json!(["DONE"]));
        assert_eq!(body["reasoning_effort"], json!("high"));
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
