//! Tool-loop helpers, ported from `src/llm/tool_loop.py`.
//!
//! The two history-adapter wrappers (`format_assistant_tool_message`,
//! `append_tool_results`) grow the conversation each iteration; the synthesis
//! prompt closes a maxed-out loop. [`execute_tool_loop`] ports the non-streaming
//! core orchestration over two seams — [`ToolLoopCaller`] (one completion) and
//! [`ToolExecutor`] (one tool run) — so the whole control flow is exercisable
//! with mocks. The streaming-final, telemetry, tenacity-retry, and structured-
//! output paths are infrastructure concerns layered on top and are not ported
//! here.

use std::future::Future;

use serde_json::{Value, json};

use super::conversation::{count_message_tokens, truncate_messages_to_fit};
use super::history_adapters;
use super::{CompletionResult, Provider, ToolCallResult, ToolResult};

/// Lower bound on `max_tool_iterations` (`MIN_TOOL_ITERATIONS`).
pub const MIN_TOOL_ITERATIONS: usize = 1;
/// Upper bound on `max_tool_iterations` (`MAX_TOOL_ITERATIONS`).
pub const MAX_TOOL_ITERATIONS: usize = 100;

/// The nudge appended to the conversation when the loop hits its tool-call cap,
/// asking the model to synthesize a final answer (`synthesis_prompt`).
pub const SYNTHESIS_PROMPT: &str = "You have reached the maximum number of tool calls. \
Based on all the information you have gathered, provide your final response now. \
Do not attempt to call any more tools.";

/// Format an assistant turn (text + tool calls + provider reasoning blocks) in
/// the provider's native shape, porting `format_assistant_tool_message`.
pub fn format_assistant_tool_message(
    provider: Provider,
    content: Value,
    tool_calls: Vec<ToolCallResult>,
    thinking_blocks: Vec<Value>,
    reasoning_details: Vec<Value>,
) -> Value {
    let result = CompletionResult {
        content,
        tool_calls,
        thinking_blocks,
        reasoning_details,
        ..CompletionResult::default()
    };
    history_adapters::for_provider(provider).format_assistant_tool_message(&result)
}

/// Append tool results to `conversation_messages` in the provider's native
/// shape, porting `append_tool_results`.
pub fn append_tool_results(
    provider: Provider,
    tool_results: &[ToolResult],
    conversation_messages: &mut Vec<Value>,
) {
    conversation_messages
        .extend(history_adapters::for_provider(provider).format_tool_results(tool_results));
}

/// The completion seam the loop drives. In production this is the full executor
/// (`honcho_llm_call_inner`, with fallback/retry/structured-output underneath);
/// in tests it's a scripted mock. `provider()` reports the transport whose
/// native message shaping to use for the *next* assistant turn — read after each
/// call so cross-provider fallback is honored, like Python's
/// `get_attempt_plan().provider`.
pub trait ToolLoopCaller {
    fn provider(&self) -> Provider;
    /// One completion over `messages` with `tools` available (empty `tools`
    /// means a tool-free call, as the synthesis/final call uses). Returns `Err`
    /// when the call fails after the caller's own retry/fallback is exhausted —
    /// the loop surfaces it as [`ToolLoopError::Caller`] (Python raises out of
    /// the tool loop on retry exhaustion).
    fn complete(
        &self,
        messages: &[Value],
        tools: &[Value],
        tool_choice: Option<&Value>,
    ) -> impl Future<Output = Result<CompletionResult, String>> + Send;
}

/// Executes one tool by name, returning the stringified result or an error
/// message. An `Err` maps to an `is_error` tool result fed back to the model,
/// porting the `except` branch in the Python loop (the call is *not* recorded in
/// the returned tool-call history).
pub trait ToolExecutor {
    fn execute(
        &self,
        name: &str,
        input: &Value,
    ) -> impl Future<Output = Result<String, String>> + Send;
}

/// One executed tool call recorded in the loop's history, mirroring the dict
/// pushed onto `all_tool_calls` (minus the telemetry-only metadata).
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCallRecord {
    pub tool_name: String,
    pub tool_input: Value,
    pub tool_result: String,
}

/// The non-streaming result of [`execute_tool_loop`], mirroring the fields of
/// `HonchoLLMCallResponse` the loop populates.
#[derive(Debug, Clone)]
pub struct ToolLoopResponse {
    pub content: Value,
    pub input_tokens: i64,
    pub output_tokens: i64,
    pub cache_creation_input_tokens: i64,
    pub cache_read_input_tokens: i64,
    pub finish_reason: String,
    pub thinking_content: Option<String>,
    pub tool_calls_made: Vec<ToolCallRecord>,
    pub iterations: usize,
    pub hit_input_token_cap: bool,
    /// The conversation that produced the final (no-tool-call) response — the
    /// messages to re-issue for `stream_final` streaming (`tool_loop.py`
    /// `stream_final_response`, which re-runs them with `tools=None`).
    pub final_conversation: Vec<Value>,
}

/// Errors from [`execute_tool_loop`]. Only the iteration-bound check is a
/// pre-flight validation error (Python's `ValidationException`); provider/tool
/// failures are folded into the conversation rather than surfaced here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ToolLoopError {
    InvalidIterations(usize),
    /// The completion call failed after the caller exhausted its retry/fallback.
    Caller(String),
}

impl std::fmt::Display for ToolLoopError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ToolLoopError::InvalidIterations(got) => write!(
                f,
                "max_tool_iterations must be in [{MIN_TOOL_ITERATIONS}, {MAX_TOOL_ITERATIONS}]; got {got}"
            ),
            ToolLoopError::Caller(message) => write!(f, "completion call failed: {message}"),
        }
    }
}

impl std::error::Error for ToolLoopError {}

/// The empty-response nudge appended when the model returns blank content with
/// no tool calls and a retry is still available.
const EMPTY_RESPONSE_NUDGE: &str =
    "Your last response was empty. Provide a concise answer to the original query \
using the available context.";

/// Run the iterative tool-calling loop (non-streaming), porting the core of
/// `execute_tool_loop`. Each iteration optionally truncates to `max_input_tokens`
/// (latching `hit_input_token_cap`), makes one tool-enabled call, executes any
/// requested tools, and folds their results back into the conversation; the loop
/// ends when the model stops calling tools (with a one-shot empty-response retry)
/// or, at `max_tool_iterations`, after a tool-free synthesis call.
#[allow(clippy::too_many_arguments)]
pub async fn execute_tool_loop<C: ToolLoopCaller, E: ToolExecutor>(
    caller: &C,
    executor: &E,
    prompt: &str,
    messages: Option<&[Value]>,
    tools: &[Value],
    tool_choice: Option<&Value>,
    max_tool_iterations: usize,
    max_input_tokens: Option<usize>,
) -> Result<ToolLoopResponse, ToolLoopError> {
    if !(MIN_TOOL_ITERATIONS..=MAX_TOOL_ITERATIONS).contains(&max_tool_iterations) {
        return Err(ToolLoopError::InvalidIterations(max_tool_iterations));
    }

    // `if messages` is falsy for both None and an empty list, so both fall back
    // to the prompt-seeded conversation.
    let mut conversation: Vec<Value> = match messages {
        Some(messages) if !messages.is_empty() => messages.to_vec(),
        _ => vec![json!({"role": "user", "content": prompt})],
    };

    let mut iteration = 0usize;
    let mut all_tool_calls: Vec<ToolCallRecord> = Vec::new();
    let mut total_input_tokens = 0i64;
    let mut total_output_tokens = 0i64;
    let mut total_cache_creation_tokens = 0i64;
    let mut total_cache_read_tokens = 0i64;
    let mut empty_response_retries = 0u32;
    let mut hit_input_token_cap = false;
    let mut effective_tool_choice: Option<Value> = tool_choice.cloned();

    while iteration < max_tool_iterations {
        if let Some(cap) = max_input_tokens {
            if count_message_tokens(&conversation) > cap {
                hit_input_token_cap = true;
            }
            conversation = truncate_messages_to_fit(&conversation, cap, true);
        }

        let response = caller
            .complete(&conversation, tools, effective_tool_choice.as_ref())
            .await
            .map_err(ToolLoopError::Caller)?;

        total_input_tokens += response.input_tokens;
        total_output_tokens += response.output_tokens;
        total_cache_creation_tokens += response.cache_creation_input_tokens;
        total_cache_read_tokens += response.cache_read_input_tokens;

        if response.tool_calls.is_empty() {
            // One empty-response retry: blank string content, retry budget left,
            // and not already on the last allowed iteration.
            let content_is_blank = response
                .content
                .as_str()
                .is_some_and(|content| content.trim().is_empty());
            if content_is_blank
                && empty_response_retries < 1
                && iteration < max_tool_iterations - 1
            {
                empty_response_retries += 1;
                conversation.push(json!({"role": "user", "content": EMPTY_RESPONSE_NUDGE}));
                iteration += 1;
                continue;
            }

            return Ok(ToolLoopResponse {
                content: response.content,
                input_tokens: total_input_tokens,
                output_tokens: total_output_tokens,
                cache_creation_input_tokens: total_cache_creation_tokens,
                cache_read_input_tokens: total_cache_read_tokens,
                finish_reason: response.finish_reason,
                thinking_content: response.thinking_content,
                tool_calls_made: all_tool_calls,
                iterations: iteration + 1,
                hit_input_token_cap,
                final_conversation: conversation.clone(),
            });
        }

        let current_provider = caller.provider();
        let assistant_message = format_assistant_tool_message(
            current_provider,
            response.content.clone(),
            response.tool_calls.clone(),
            response.thinking_blocks.clone(),
            response.reasoning_details.clone(),
        );
        conversation.push(assistant_message);

        let mut tool_results: Vec<ToolResult> = Vec::new();
        for tool_call in &response.tool_calls {
            match executor.execute(&tool_call.name, &tool_call.input).await {
                Ok(result) => {
                    tool_results.push(ToolResult {
                        tool_id: tool_call.id.clone(),
                        tool_name: tool_call.name.clone(),
                        result: result.clone(),
                        is_error: false,
                    });
                    all_tool_calls.push(ToolCallRecord {
                        tool_name: tool_call.name.clone(),
                        tool_input: tool_call.input.clone(),
                        tool_result: result,
                    });
                }
                Err(message) => {
                    // Failures are reported back to the model but not recorded in
                    // the tool-call history, matching the Python except branch.
                    tool_results.push(ToolResult {
                        tool_id: tool_call.id.clone(),
                        tool_name: tool_call.name.clone(),
                        result: format!("Error: {message}"),
                        is_error: true,
                    });
                }
            }
        }

        append_tool_results(current_provider, &tool_results, &mut conversation);

        // After the first iteration, relax a forced choice so the model can stop.
        if iteration == 0
            && let Some(Value::String(choice)) = &effective_tool_choice
            && (choice == "required" || choice == "any")
        {
            effective_tool_choice = Some(Value::String("auto".to_string()));
        }

        iteration += 1;
    }

    // Max iterations reached: nudge for a final synthesis, re-truncate (the
    // appended prompt may have nudged us back over the cap), then call once more
    // with no tools.
    conversation.push(json!({"role": "user", "content": SYNTHESIS_PROMPT}));
    if let Some(cap) = max_input_tokens {
        if count_message_tokens(&conversation) > cap {
            hit_input_token_cap = true;
        }
        conversation = truncate_messages_to_fit(&conversation, cap, true);
    }

    let final_response = caller
        .complete(&conversation, &[], None)
        .await
        .map_err(ToolLoopError::Caller)?;

    Ok(ToolLoopResponse {
        content: final_response.content,
        input_tokens: total_input_tokens + final_response.input_tokens,
        output_tokens: total_output_tokens + final_response.output_tokens,
        cache_creation_input_tokens: total_cache_creation_tokens
            + final_response.cache_creation_input_tokens,
        cache_read_input_tokens: total_cache_read_tokens + final_response.cache_read_input_tokens,
        finish_reason: final_response.finish_reason,
        thinking_content: final_response.thinking_content,
        tool_calls_made: all_tool_calls,
        iterations: iteration + 1,
        hit_input_token_cap,
        final_conversation: conversation.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn tool_call() -> ToolCallResult {
        ToolCallResult {
            id: "t1".to_string(),
            name: "search".to_string(),
            input: json!({"q": "x"}),
            thought_signature: None,
        }
    }

    #[test]
    fn assistant_message_delegates_to_provider_adapter() {
        let message = format_assistant_tool_message(
            Provider::Openai,
            Value::String("hi".to_string()),
            vec![tool_call()],
            Vec::new(),
            Vec::new(),
        );
        assert_eq!(message["role"], json!("assistant"));
        assert_eq!(message["content"], json!("hi"));
        assert_eq!(
            message["tool_calls"][0]["function"]["name"],
            json!("search")
        );
    }

    #[test]
    fn append_tool_results_extends_conversation() {
        let mut conversation = vec![json!({"role": "user", "content": "hi"})];
        let results = vec![ToolResult {
            tool_id: "t1".to_string(),
            tool_name: "search".to_string(),
            result: "done".to_string(),
            is_error: false,
        }];
        append_tool_results(Provider::Anthropic, &results, &mut conversation);
        assert_eq!(conversation.len(), 2);
        // Anthropic wraps tool results in a single user message.
        assert_eq!(conversation[1]["role"], json!("user"));
        assert_eq!(conversation[1]["content"][0]["type"], json!("tool_result"));
        assert_eq!(conversation[1]["content"][0]["tool_use_id"], json!("t1"));
    }

    #[test]
    fn synthesis_prompt_is_a_single_paragraph() {
        assert!(SYNTHESIS_PROMPT.starts_with("You have reached the maximum"));
        assert!(SYNTHESIS_PROMPT.ends_with("call any more tools."));
        assert!(!SYNTHESIS_PROMPT.contains('\n'));
    }

    // ----- execute_tool_loop -----

    use std::collections::VecDeque;
    use std::sync::Mutex;

    /// A captured `complete()` invocation: the messages, tools, and tool_choice
    /// the loop passed in.
    #[derive(Debug, Clone)]
    struct Call {
        messages: Vec<Value>,
        tools: Vec<Value>,
        tool_choice: Option<Value>,
    }

    /// A caller that replays a scripted queue of [`CompletionResult`]s and
    /// records every invocation.
    struct ScriptedCaller {
        provider: Provider,
        responses: Mutex<VecDeque<CompletionResult>>,
        calls: Mutex<Vec<Call>>,
    }

    impl ScriptedCaller {
        fn new(responses: Vec<CompletionResult>) -> Self {
            Self {
                provider: Provider::Anthropic,
                responses: Mutex::new(responses.into()),
                calls: Mutex::new(Vec::new()),
            }
        }

        fn calls(&self) -> Vec<Call> {
            self.calls.lock().unwrap().clone()
        }
    }

    impl ToolLoopCaller for ScriptedCaller {
        fn provider(&self) -> Provider {
            self.provider
        }

        async fn complete(
            &self,
            messages: &[Value],
            tools: &[Value],
            tool_choice: Option<&Value>,
        ) -> Result<CompletionResult, String> {
            self.calls.lock().unwrap().push(Call {
                messages: messages.to_vec(),
                tools: tools.to_vec(),
                tool_choice: tool_choice.cloned(),
            });
            Ok(self
                .responses
                .lock()
                .unwrap()
                .pop_front()
                .expect("ScriptedCaller ran out of scripted responses"))
        }
    }

    /// A caller whose completion always fails, to exercise error propagation.
    struct FailingCaller;

    impl ToolLoopCaller for FailingCaller {
        fn provider(&self) -> Provider {
            Provider::Anthropic
        }

        async fn complete(
            &self,
            _messages: &[Value],
            _tools: &[Value],
            _tool_choice: Option<&Value>,
        ) -> Result<CompletionResult, String> {
            Err("provider unavailable".to_string())
        }
    }

    /// A tool executor that echoes its input, or always fails when `fail` is set.
    struct EchoExecutor {
        fail: Option<String>,
    }

    impl ToolExecutor for EchoExecutor {
        async fn execute(&self, name: &str, input: &Value) -> Result<String, String> {
            match &self.fail {
                Some(message) => Err(message.clone()),
                None => Ok(format!("ran {name}({input})")),
            }
        }
    }

    fn answer(text: &str) -> CompletionResult {
        CompletionResult {
            content: json!(text),
            input_tokens: 1,
            output_tokens: 1,
            ..CompletionResult::default()
        }
    }

    fn tool_call_response(name: &str) -> CompletionResult {
        CompletionResult {
            content: json!(""),
            input_tokens: 1,
            output_tokens: 1,
            tool_calls: vec![ToolCallResult {
                id: format!("call_{name}"),
                name: name.to_string(),
                input: json!({"q": "x"}),
                thought_signature: None,
            }],
            ..CompletionResult::default()
        }
    }

    fn echo() -> EchoExecutor {
        EchoExecutor { fail: None }
    }

    #[tokio::test]
    async fn caller_failure_surfaces_as_tool_loop_error() {
        let error = execute_tool_loop(
            &FailingCaller,
            &echo(),
            "hi",
            None,
            &[json!({"name": "t"})],
            None,
            5,
            None,
        )
        .await
        .unwrap_err();
        assert_eq!(error, ToolLoopError::Caller("provider unavailable".to_string()));
    }

    #[tokio::test]
    async fn returns_immediately_when_no_tools_requested() {
        let caller = ScriptedCaller::new(vec![answer("hello")]);
        let response =
            execute_tool_loop(&caller, &echo(), "hi", None, &[json!({"name": "t"})], None, 5, None)
                .await
                .unwrap();

        assert_eq!(response.content, json!("hello"));
        assert_eq!(response.iterations, 1);
        assert!(response.tool_calls_made.is_empty());
        // The single call seeded the conversation from the prompt and forwarded
        // the tools.
        let calls = caller.calls();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0].messages, vec![json!({"role": "user", "content": "hi"})]);
        assert_eq!(calls[0].tools, vec![json!({"name": "t"})]);
    }

    #[tokio::test]
    async fn executes_tool_then_answers_and_accumulates_tokens() {
        let caller = ScriptedCaller::new(vec![tool_call_response("search"), answer("done")]);
        let response =
            execute_tool_loop(&caller, &echo(), "hi", None, &[json!({"name": "search"})], None, 5, None)
                .await
                .unwrap();

        assert_eq!(response.content, json!("done"));
        assert_eq!(response.iterations, 2);
        assert_eq!(response.tool_calls_made.len(), 1);
        assert_eq!(response.tool_calls_made[0].tool_name, "search");
        assert_eq!(
            response.tool_calls_made[0].tool_result,
            "ran search({\"q\":\"x\"})"
        );
        // Two calls, tokens summed (1+1 in, 1+1 out).
        assert_eq!(response.input_tokens, 2);
        assert_eq!(response.output_tokens, 2);
        // The second call saw the grown conversation: user, assistant(tool_use),
        // tool_result.
        let calls = caller.calls();
        assert_eq!(calls[1].messages.len(), 3);
        assert_eq!(calls[1].messages[1]["role"], json!("assistant"));
    }

    #[tokio::test]
    async fn reaching_max_iterations_runs_a_tool_free_synthesis_call() {
        // max=1: one tool iteration, then the synthesis call.
        let caller = ScriptedCaller::new(vec![tool_call_response("search"), answer("final")]);
        let response =
            execute_tool_loop(&caller, &echo(), "hi", None, &[json!({"name": "search"})], None, 1, None)
                .await
                .unwrap();

        assert_eq!(response.content, json!("final"));
        assert_eq!(response.iterations, 2);
        let calls = caller.calls();
        assert_eq!(calls.len(), 2);
        // The synthesis call carries no tools and the synthesis nudge is the last
        // user message.
        assert!(calls[1].tools.is_empty());
        assert_eq!(calls[1].tool_choice, None);
        let last = calls[1].messages.last().unwrap();
        assert_eq!(last["content"], json!(SYNTHESIS_PROMPT));
    }

    #[tokio::test]
    async fn empty_response_triggers_one_retry_nudge() {
        let caller = ScriptedCaller::new(vec![answer("   "), answer("recovered")]);
        let response =
            execute_tool_loop(&caller, &echo(), "hi", None, &[], None, 3, None)
                .await
                .unwrap();

        assert_eq!(response.content, json!("recovered"));
        assert_eq!(response.iterations, 2);
        // The retry call saw the appended nudge as the second message.
        let calls = caller.calls();
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[1].messages[1]["content"], json!(EMPTY_RESPONSE_NUDGE));
    }

    #[tokio::test]
    async fn forced_tool_choice_relaxes_to_auto_after_first_iteration() {
        let caller = ScriptedCaller::new(vec![tool_call_response("search"), answer("ok")]);
        let required = json!("required");
        execute_tool_loop(
            &caller,
            &echo(),
            "hi",
            None,
            &[json!({"name": "search"})],
            Some(&required),
            5,
            None,
        )
        .await
        .unwrap();

        let calls = caller.calls();
        assert_eq!(calls[0].tool_choice, Some(json!("required")));
        assert_eq!(calls[1].tool_choice, Some(json!("auto")));
    }

    #[tokio::test]
    async fn failed_tool_is_reported_but_not_recorded() {
        let caller = ScriptedCaller::new(vec![tool_call_response("search"), answer("ok")]);
        let executor = EchoExecutor {
            fail: Some("boom".to_string()),
        };
        let response =
            execute_tool_loop(&caller, &executor, "hi", None, &[json!({"name": "search"})], None, 5, None)
                .await
                .unwrap();

        // The failed call is fed back as an is_error tool_result but kept out of
        // the recorded history.
        assert!(response.tool_calls_made.is_empty());
        let calls = caller.calls();
        let tool_result_message = &calls[1].messages[2];
        assert_eq!(tool_result_message["content"][0]["is_error"], json!(true));
        assert_eq!(
            tool_result_message["content"][0]["content"],
            json!("Error: boom")
        );
    }

    #[tokio::test]
    async fn token_cap_latches_when_conversation_exceeds_limit() {
        let caller = ScriptedCaller::new(vec![answer("hi")]);
        // A 1-token cap is exceeded by even the seed message.
        let response =
            execute_tool_loop(&caller, &echo(), "a longer prompt here", None, &[], None, 5, Some(1))
                .await
                .unwrap();
        assert!(response.hit_input_token_cap);
    }

    #[tokio::test]
    async fn rejects_out_of_range_iteration_bounds() {
        let caller = ScriptedCaller::new(vec![answer("x")]);
        assert_eq!(
            execute_tool_loop(&caller, &echo(), "hi", None, &[], None, 0, None)
                .await
                .unwrap_err(),
            ToolLoopError::InvalidIterations(0)
        );
        assert_eq!(
            execute_tool_loop(&caller, &echo(), "hi", None, &[], None, 101, None)
                .await
                .unwrap_err(),
            ToolLoopError::InvalidIterations(101)
        );
    }
}
