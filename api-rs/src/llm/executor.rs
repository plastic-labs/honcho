//! The completion executor, porting the dispatch core of `request_builder.py`
//! (`execute_completion`). Maps a [`ModelConfig`] plus per-call inputs into the
//! right provider backend's `RequestParams` and runs the call over an
//! [`LlmHttp`] transport. All three providers (Anthropic, OpenAI, Gemini) are
//! routable; Gemini speaks the `generateContent` REST endpoint directly (Python
//! uses the genai SDK — see `backends::gemini` for the REST-casing notes).

use std::sync::Mutex;
use std::time::Duration;

use serde_json::{Map, Value, json};

use super::backends::{anthropic, gemini, openai};
use super::credentials::{TransportApiKeys, resolve_credentials};
use super::http::{Credentials, LlmHttp, LlmHttpError};
use super::request_builder::{build_config_extra_params, effective_max_tokens};
use super::runtime::{effective_config_for_call, effective_temperature, plan_attempt};
use super::tool_loop::ToolLoopCaller;
use super::types::{HonchoLLMCallResponse, completion_result_to_response};
use super::{CompletionResult, ModelConfig, Provider};

/// An error from [`execute_completion`].
#[derive(Debug)]
pub enum ExecutorError {
    /// The underlying provider HTTP call failed.
    Http(LlmHttpError),
}

impl std::fmt::Display for ExecutorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExecutorError::Http(error) => write!(f, "{error}"),
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
        Provider::Gemini => {
            let params = gemini::RequestParams {
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
            Ok(gemini::complete(http, credentials, &params).await?)
        }
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

/// Exponential-backoff retry policy, porting tenacity's
/// `wait_exponential(multiplier=1, min=4, max=10)` + `stop_after_attempt`. The
/// durations are fields so tests can zero them out (no real sleeping).
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    pub attempts: u32,
    pub backoff_min: Duration,
    pub backoff_max: Duration,
}

impl Default for RetryPolicy {
    /// The production default: 3 attempts, 4s→10s exponential backoff.
    fn default() -> Self {
        Self {
            attempts: 3,
            backoff_min: Duration::from_secs(4),
            backoff_max: Duration::from_secs(10),
        }
    }
}

impl RetryPolicy {
    /// Sleep duration before re-attempting after `failed_attempt` (1-indexed):
    /// `clamp(min * 2^(failed_attempt-1), .., max)`, mirroring tenacity's
    /// exponential schedule.
    fn backoff_for(&self, failed_attempt: u32) -> Duration {
        let scaled = self
            .backoff_min
            .saturating_mul(1u32 << (failed_attempt.saturating_sub(1)).min(16));
        scaled.min(self.backoff_max)
    }
}

/// The provider-backed [`ToolLoopCaller`], porting the retry + fallback
/// orchestration of `api.py::honcho_llm_call` (the tool path that threads
/// `get_attempt_plan` into the loop). Each [`ToolLoopCaller::complete`] runs its
/// own attempt budget: per attempt it plans the config ([`plan_attempt`] — the
/// final attempt swaps to the fallback), resolves credentials, builds the
/// effective config, and dispatches via [`execute_completion`]. On exhaustion the
/// last error is returned as the caller error the tool loop surfaces.
///
/// Deviation from Python: the attempt budget resets per `complete()` call here,
/// rather than persisting across tool-loop iterations via a ContextVar (a quirk
/// of the Python implementation, not intended behavior).
pub struct HonchoCaller<'a, H: LlmHttp> {
    pub http: &'a H,
    pub keys: TransportApiKeys,
    pub runtime_config: ModelConfig,
    pub max_tokens: i64,
    pub retry: RetryPolicy,
    pub temperature: Option<f64>,
    pub stop_seqs: Option<Vec<String>>,
    pub thinking_budget_tokens: Option<i64>,
    pub reasoning_effort: Option<String>,
    pub json_mode: bool,
    pub verbosity: Option<String>,
    /// The provider of the most recently dispatched attempt, for `provider()`
    /// (the tool loop reads it to shape the next assistant turn). Starts at the
    /// runtime config's transport before the first call.
    last_provider: Mutex<Provider>,
}

impl<'a, H: LlmHttp> HonchoCaller<'a, H> {
    /// Construct a caller for `runtime_config`, defaulting the per-call knobs to
    /// unset and the retry policy to [`RetryPolicy::default`].
    pub fn new(
        http: &'a H,
        keys: TransportApiKeys,
        runtime_config: ModelConfig,
        max_tokens: i64,
    ) -> Self {
        let transport = runtime_config.transport;
        Self {
            http,
            keys,
            runtime_config,
            max_tokens,
            retry: RetryPolicy::default(),
            temperature: None,
            stop_seqs: None,
            thinking_budget_tokens: None,
            reasoning_effort: None,
            json_mode: false,
            verbosity: None,
            last_provider: Mutex::new(transport),
        }
    }

    /// The shared retry + fallback attempt loop (used by both the tool-loop
    /// `complete` and the single-call `complete_single`). Per attempt: plan the
    /// config (final attempt swaps to fallback), resolve credentials, build the
    /// effective config, dispatch via [`execute_completion`]. On exhaustion the
    /// last error string is returned.
    async fn run_attempts(
        &self,
        messages: &[Value],
        tools_opt: Option<&[Value]>,
        tool_choice: Option<&Value>,
    ) -> Result<CompletionResult, String> {
        let mut last_error: Option<String> = None;

        for attempt in 1..=self.retry.attempts {
            let plan = plan_attempt(
                &self.runtime_config,
                attempt,
                self.retry.attempts,
                self.thinking_budget_tokens,
                self.reasoning_effort.as_deref(),
            );
            *self.last_provider.lock().unwrap() = plan.provider;

            let credentials = resolve_credentials(&plan.selected_config, &self.keys);
            let temperature = effective_temperature(self.temperature, attempt);
            let effective_config = effective_config_for_call(
                Some(&plan.selected_config),
                plan.provider,
                &plan.model,
                temperature,
                self.stop_seqs.as_deref(),
                plan.thinking_budget_tokens,
                plan.reasoning_effort.as_deref(),
            );

            let mut call_extras = Map::new();
            call_extras.insert("json_mode".to_string(), json!(self.json_mode));
            call_extras.insert(
                "verbosity".to_string(),
                self.verbosity
                    .as_deref()
                    .map_or(Value::Null, |value| json!(value)),
            );

            match execute_completion(
                self.http,
                &credentials,
                &effective_config,
                messages,
                self.max_tokens,
                tools_opt,
                tool_choice,
                None,
                Some(&call_extras),
            )
            .await
            {
                Ok(result) => return Ok(result),
                Err(error) => {
                    last_error = Some(error.to_string());
                    if attempt < self.retry.attempts {
                        let backoff = self.retry.backoff_for(attempt);
                        if !backoff.is_zero() {
                            tokio::time::sleep(backoff).await;
                        }
                    }
                }
            }
        }

        Err(last_error.unwrap_or_else(|| "completion failed with no attempts".to_string()))
    }

    /// Single (toolless) completion with retry + fallback, mapped to the public
    /// [`HonchoLLMCallResponse`]. This is the structured-output / minimal-deriver
    /// path: the caller applies structured-output validation to `content`
    /// afterwards (the Python backend does the equivalent inline via
    /// `response_format`).
    pub async fn complete_single(&self, messages: &[Value]) -> Result<HonchoLLMCallResponse, String> {
        let result = self.run_attempts(messages, None, None).await?;
        Ok(completion_result_to_response(&result))
    }
}

impl<H: LlmHttp + Sync> ToolLoopCaller for HonchoCaller<'_, H> {
    fn provider(&self) -> Provider {
        *self.last_provider.lock().unwrap()
    }

    async fn complete(
        &self,
        messages: &[Value],
        tools: &[Value],
        tool_choice: Option<&Value>,
    ) -> Result<CompletionResult, String> {
        let tools_opt = if tools.is_empty() { None } else { Some(tools) };
        self.run_attempts(messages, tools_opt, tool_choice).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::llm::http::mock::MockHttp;
    use serde_json::json;

    fn no_backoff() -> RetryPolicy {
        RetryPolicy {
            attempts: 2,
            backoff_min: Duration::ZERO,
            backoff_max: Duration::ZERO,
        }
    }

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
    async fn gemini_routes_to_generate_content_endpoint() {
        // A camelCase REST-shaped reply, parsed via the dual-casing path.
        let http = MockHttp::ok(json!({
            "candidates": [{
                "content": {"parts": [{"text": "hi there"}]},
                "finishReason": "STOP"
            }],
            "usageMetadata": {"promptTokenCount": 3, "candidatesTokenCount": 2}
        }));
        let creds = Credentials::new("key");
        let config = ModelConfig::new("gemini-x", Provider::Gemini);
        let result = execute_completion(
            &http, &creds, &config, &messages(), 512, None, None, None, None,
        )
        .await
        .expect("completion");
        assert_eq!(
            http.last_url(),
            "https://generativelanguage.googleapis.com/v1beta/models/gemini-x:generateContent"
        );
        assert_eq!(
            http.last_headers(),
            vec![("x-goog-api-key".to_string(), "key".to_string())]
        );
        // contents carry the user turn; generation_config carries max tokens.
        let body = http.last_body();
        assert_eq!(body["contents"][0]["parts"][0]["text"], json!("hi"));
        assert_eq!(body["generation_config"]["max_output_tokens"], json!(512));
        // The camelCase response was parsed.
        assert_eq!(result.content, json!("hi there"));
        assert_eq!(result.input_tokens, 3);
        assert_eq!(result.output_tokens, 2);
        assert_eq!(result.finish_reason, "STOP");
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
    async fn complete_single_feeds_structured_output_finalize() {
        // The LLM returns a JSON representation as its text content.
        let json_text = "{\"explicit\": [{\"content\": \"likes coffee\"}]}";
        let http = MockHttp::ok(json!({
            "content": [{"type": "text", "text": json_text}],
            "usage": {"input_tokens": 5, "output_tokens": 7},
            "stop_reason": "end_turn"
        }));
        let keys = TransportApiKeys {
            anthropic: Some("k".to_string()),
            openai: None,
            gemini: None,
        };
        let config = ModelConfig::new("claude-x", Provider::Anthropic);
        let mut caller = HonchoCaller::new(&http, keys, config, 1024);
        caller.retry = no_backoff();
        caller.json_mode = true;

        let response = caller.complete_single(&messages()).await.expect("single call");
        assert_eq!(response.content, json!(json_text));
        assert_eq!(response.output_tokens, 7);

        // The caller-side structured-output step (what the deriver runs) turns the
        // JSON text content into a validated PromptRepresentation.
        let pr = crate::structured_output::finalize_structured_output(
            &response.content,
            crate::structured_output::FailurePolicy::RepairThenEmpty,
        )
        .expect("finalize");
        assert_eq!(pr.explicit, vec!["likes coffee".to_string()]);
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

    fn caller_keys() -> TransportApiKeys {
        TransportApiKeys {
            anthropic: Some("a-key".to_string()),
            openai: Some("o-key".to_string()),
            gemini: None,
        }
    }

    #[tokio::test]
    async fn caller_succeeds_and_reports_provider() {
        let http = ok_http();
        let config = ModelConfig::new("claude-x", Provider::Anthropic);
        let mut caller = HonchoCaller::new(&http, caller_keys(), config, 1024);
        caller.retry = no_backoff();

        let result = caller
            .complete(&messages(), &[json!({"name": "t"})], None)
            .await
            .expect("completion");
        assert_eq!(result.content, json!("hi"));
        assert_eq!(caller.provider(), Provider::Anthropic);
        // The config api_key was resolved from the global anthropic key.
        assert!(
            http.last_headers()
                .iter()
                .any(|(name, value)| name == "x-api-key" && value == "a-key")
        );
    }

    #[tokio::test]
    async fn caller_propagates_error_after_exhausting_attempts() {
        let http = MockHttp::err(LlmHttpError::Status {
            status: 500,
            body: "boom".to_string(),
        });
        let config = ModelConfig::new("claude-x", Provider::Anthropic);
        let mut caller = HonchoCaller::new(&http, caller_keys(), config, 1024);
        caller.retry = no_backoff();

        let error = caller
            .complete(&messages(), &[], None)
            .await
            .expect_err("all attempts fail");
        assert!(error.contains("500"));
    }

    #[tokio::test]
    async fn caller_uses_fallback_on_final_attempt() {
        // Anthropic primary with an OpenAI fallback; every call errs, so after the
        // 2-attempt budget the LAST dispatch (captured by the mock) is the
        // fallback provider's endpoint.
        let http = MockHttp::err(LlmHttpError::Status {
            status: 500,
            body: "boom".to_string(),
        });
        let mut config = ModelConfig::new("claude-x", Provider::Anthropic);
        config.fallback = Some(Box::new(ModelConfig::new("gpt-x", Provider::Openai)));
        let mut caller = HonchoCaller::new(&http, caller_keys(), config, 1024);
        caller.retry = no_backoff();

        let _ = caller.complete(&messages(), &[], None).await;
        assert!(http.last_url().ends_with("/chat/completions"));
        assert_eq!(caller.provider(), Provider::Openai);
    }

    #[test]
    fn backoff_grows_then_caps() {
        let policy = RetryPolicy {
            attempts: 5,
            backoff_min: Duration::from_secs(4),
            backoff_max: Duration::from_secs(10),
        };
        assert_eq!(policy.backoff_for(1), Duration::from_secs(4));
        assert_eq!(policy.backoff_for(2), Duration::from_secs(8));
        assert_eq!(policy.backoff_for(3), Duration::from_secs(10)); // 16 capped to 10
    }
}
