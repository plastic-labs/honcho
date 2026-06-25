//! Deterministic Anthropic backend helpers, ported from
//! `src/llm/backends/anthropic.py`.
//!
//! Covers the network-free parts: splitting system messages out of the request,
//! converting the canonical `tool_choice` to Anthropic's shape, the assistant-
//! prefill capability check, and parsing a Messages API response into a
//! [`CompletionResult`]. The structured-output (`response_format`) repair path
//! is not ported here — without it, `content` is the joined text, which is the
//! common (no-`response_format`) case.

use serde_json::{Map, Value, json};

use crate::llm::http::{Credentials, LlmHttp, LlmHttpError, LlmStreamHttp, TextStream};
use crate::llm::{CompletionResult, ToolCallResult};

/// The Anthropic SDK's default API host, used when no `base_url` override is set.
pub const DEFAULT_BASE_URL: &str = "https://api.anthropic.com";

/// The `anthropic-version` header the SDK pins by default.
pub const ANTHROPIC_VERSION: &str = "2023-06-01";

/// Run one Messages API completion: build the request body, POST it through the
/// [`LlmHttp`] transport with Anthropic's auth headers, and parse the response.
/// This is the deterministic orchestration around the single non-deterministic
/// step (`post_json`); with a mock transport it is exercisable without network.
pub async fn complete<H: LlmHttp>(
    http: &H,
    credentials: &Credentials,
    params: &RequestParams<'_>,
) -> Result<CompletionResult, LlmHttpError> {
    let body = build_request(params);
    let url = format!("{}/v1/messages", credentials.effective_base_url(DEFAULT_BASE_URL));
    let headers = [
        ("x-api-key".to_string(), credentials.api_key.clone()),
        ("anthropic-version".to_string(), ANTHROPIC_VERSION.to_string()),
    ];
    let response = http.post_json(&url, &headers, &body).await?;
    Ok(parse_response(&response))
}

/// Open a streaming Messages API completion: the same request as [`complete`]
/// with `stream: true`, returning the raw SSE [`TextStream`] (decoded by
/// [`crate::llm::streaming::decode_stream`]).
pub async fn stream<H: LlmStreamHttp>(
    http: &H,
    credentials: &Credentials,
    params: &RequestParams<'_>,
) -> Result<TextStream, LlmHttpError> {
    let mut body = build_request(params);
    if let Some(object) = body.as_object_mut() {
        object.insert("stream".to_string(), json!(true));
    }
    let url = format!("{}/v1/messages", credentials.effective_base_url(DEFAULT_BASE_URL));
    let headers = [
        ("x-api-key".to_string(), credentials.api_key.clone()),
        ("anthropic-version".to_string(), ANTHROPIC_VERSION.to_string()),
    ];
    http.post_json_stream(&url, &headers, &body).await
}

/// Inputs for [`build_request`], mirroring the portable subset of the
/// `complete()` signature (the `response_format` JSON-prefill path is excluded).
#[derive(Debug, Clone)]
pub struct RequestParams<'a> {
    pub model: &'a str,
    pub messages: &'a [Value],
    pub max_tokens: i64,
    pub temperature: Option<f64>,
    pub stop: Option<&'a [String]>,
    pub tools: Option<&'a [Value]>,
    pub tool_choice: Option<&'a Value>,
    pub thinking_budget_tokens: Option<i64>,
    /// Flattened tuning params (`top_p`/`top_k` are forwarded), as produced by
    /// `request_builder::build_config_extra_params`.
    pub extra_params: &'a Map<String, Value>,
}

/// Build the JSON body for the Anthropic Messages API, porting the deterministic
/// param assembly in `complete()` (system extraction, temperature, stop
/// sequences, tools + converted tool_choice, thinking config, `top_p`/`top_k`
/// passthrough). The `response_format` assistant-prefill path is not included.
pub fn build_request(params: &RequestParams<'_>) -> Value {
    let (request_messages, system_messages) = extract_system(params.messages);

    let mut body = Map::new();
    body.insert("model".to_string(), json!(params.model));
    body.insert("max_tokens".to_string(), json!(params.max_tokens));
    body.insert("messages".to_string(), json!(request_messages));

    if let Some(temperature) = params.temperature {
        body.insert("temperature".to_string(), json!(temperature));
    }
    if let Some(stop) = params.stop.filter(|stop| !stop.is_empty()) {
        body.insert("stop_sequences".to_string(), json!(stop));
    }
    if !system_messages.is_empty() {
        body.insert(
            "system".to_string(),
            json!([{
                "type": "text",
                "text": system_messages.join("\n\n"),
                "cache_control": {"type": "ephemeral"},
            }]),
        );
    }
    if let Some(tools) = params.tools.filter(|tools| !tools.is_empty()) {
        body.insert("tools".to_string(), json!(tools));
        if let Some(tool_choice) = convert_tool_choice(params.tool_choice) {
            body.insert("tool_choice".to_string(), tool_choice);
        }
    }
    // Python treats 0 as falsy, so a zero budget disables thinking.
    if let Some(budget) = params.thinking_budget_tokens.filter(|budget| *budget != 0) {
        body.insert(
            "thinking".to_string(),
            json!({"type": "enabled", "budget_tokens": budget}),
        );
    }
    for key in ["top_p", "top_k"] {
        if let Some(value) = params.extra_params.get(key) {
            body.insert(key.to_string(), value.clone());
        }
    }

    Value::Object(body)
}

/// Split system messages out of a request, porting `_extract_system`. A message
/// is "system" only when `role == "system"` and its `content` is a string;
/// everything else is kept (in order) as a non-system message.
pub fn extract_system(messages: &[Value]) -> (Vec<Value>, Vec<String>) {
    let mut non_system = Vec::new();
    let mut system = Vec::new();
    for message in messages {
        let is_system = message.get("role").and_then(Value::as_str) == Some("system");
        match (is_system, message.get("content").and_then(Value::as_str)) {
            (true, Some(content)) => system.push(content.to_string()),
            _ => non_system.push(message.clone()),
        }
    }
    (non_system, system)
}

/// Convert the canonical `tool_choice` to Anthropic's object form, porting
/// `_convert_tool_choice`. An object passes through; the known strings map to
/// `auto`/`any`/`none`; any other string names a specific tool.
pub fn convert_tool_choice(tool_choice: Option<&Value>) -> Option<Value> {
    match tool_choice {
        None => None,
        Some(Value::Object(_)) => tool_choice.cloned(),
        Some(Value::String(choice)) => Some(match choice.as_str() {
            "auto" => json!({"type": "auto"}),
            "any" | "required" => json!({"type": "any"}),
            "none" => json!({"type": "none"}),
            name => json!({"type": "tool", "name": name}),
        }),
        // Non-string, non-object inputs are not valid tool_choice values.
        Some(_) => None,
    }
}

/// Whether the model accepts assistant-prefill, porting
/// `_supports_assistant_prefill`. Claude 4-class models reject it.
pub fn supports_assistant_prefill(model: &str) -> bool {
    !(model.starts_with("claude-opus-4")
        || model.starts_with("claude-sonnet-4")
        || model.starts_with("claude-haiku-4"))
}

/// Parse a Messages API response into a [`CompletionResult`], porting the
/// content-block loop and usage math (no `response_format`). Text blocks are
/// joined with `\n`; thinking blocks contribute both joined text and the full
/// `{type,thinking,signature}` blocks; `tool_use` blocks become tool calls.
/// `input_tokens` is the uncached + cache-creation + cache-read sum.
pub fn parse_response(response: &Value) -> CompletionResult {
    let mut text_blocks: Vec<&str> = Vec::new();
    let mut thinking_text: Vec<&str> = Vec::new();
    let mut thinking_blocks: Vec<Value> = Vec::new();
    let mut tool_calls: Vec<ToolCallResult> = Vec::new();

    if let Some(blocks) = response.get("content").and_then(Value::as_array) {
        for block in blocks {
            match block.get("type").and_then(Value::as_str) {
                Some("text") => {
                    if let Some(text) = block.get("text").and_then(Value::as_str) {
                        text_blocks.push(text);
                    }
                }
                Some("thinking") => {
                    let thinking = block.get("thinking").and_then(Value::as_str).unwrap_or("");
                    thinking_text.push(thinking);
                    thinking_blocks.push(json!({
                        "type": "thinking",
                        "thinking": thinking,
                        "signature": block.get("signature").cloned().unwrap_or(Value::Null),
                    }));
                }
                Some("tool_use") => {
                    tool_calls.push(ToolCallResult {
                        id: block
                            .get("id")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string(),
                        name: block
                            .get("name")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string(),
                        input: block.get("input").cloned().unwrap_or_else(|| json!({})),
                        thought_signature: None,
                    });
                }
                _ => {}
            }
        }
    }

    let usage = response.get("usage");
    let usage_int = |key: &str| -> i64 {
        usage
            .and_then(|u| u.get(key))
            .and_then(Value::as_i64)
            .unwrap_or(0)
    };
    let cache_creation = usage_int("cache_creation_input_tokens");
    let cache_read = usage_int("cache_read_input_tokens");
    let input_tokens = usage_int("input_tokens") + cache_creation + cache_read;

    let thinking_content = if thinking_text.is_empty() {
        None
    } else {
        Some(thinking_text.join("\n"))
    };

    CompletionResult {
        content: Value::String(text_blocks.join("\n")),
        input_tokens,
        output_tokens: usage_int("output_tokens"),
        cache_creation_input_tokens: cache_creation,
        cache_read_input_tokens: cache_read,
        finish_reason: response
            .get("stop_reason")
            .and_then(Value::as_str)
            .unwrap_or("stop")
            .to_string(),
        tool_calls,
        thinking_content,
        thinking_blocks,
        reasoning_details: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_system_splits_string_system_messages() {
        let messages = vec![
            json!({"role": "system", "content": "be nice"}),
            json!({"role": "user", "content": "hi"}),
            // role=system but non-string content is NOT treated as system.
            json!({"role": "system", "content": [{"type": "text", "text": "x"}]}),
        ];
        let (non_system, system) = extract_system(&messages);
        assert_eq!(system, vec!["be nice".to_string()]);
        assert_eq!(non_system.len(), 2);
        assert_eq!(non_system[0]["role"], json!("user"));
        assert_eq!(non_system[1]["role"], json!("system"));
    }

    #[test]
    fn convert_tool_choice_maps_known_values() {
        assert_eq!(convert_tool_choice(None), None);
        assert_eq!(
            convert_tool_choice(Some(&json!("auto"))),
            Some(json!({"type": "auto"}))
        );
        assert_eq!(
            convert_tool_choice(Some(&json!("any"))),
            Some(json!({"type": "any"}))
        );
        assert_eq!(
            convert_tool_choice(Some(&json!("required"))),
            Some(json!({"type": "any"}))
        );
        assert_eq!(
            convert_tool_choice(Some(&json!("none"))),
            Some(json!({"type": "none"}))
        );
        assert_eq!(
            convert_tool_choice(Some(&json!("search"))),
            Some(json!({"type": "tool", "name": "search"}))
        );
        let passthrough = json!({"type": "tool", "name": "x"});
        assert_eq!(
            convert_tool_choice(Some(&passthrough)),
            Some(passthrough.clone())
        );
    }

    #[test]
    fn assistant_prefill_rejected_for_claude_4_class() {
        assert!(!supports_assistant_prefill("claude-opus-4-8"));
        assert!(!supports_assistant_prefill("claude-sonnet-4-6"));
        assert!(!supports_assistant_prefill("claude-haiku-4-5"));
        assert!(supports_assistant_prefill("claude-3-5-sonnet"));
    }

    #[test]
    fn parse_response_collects_blocks_and_usage() {
        let response = json!({
            "content": [
                {"type": "text", "text": "Hello"},
                {"type": "thinking", "thinking": "hmm", "signature": "sig"},
                {"type": "text", "text": "world"},
                {"type": "tool_use", "id": "toolu_1", "name": "search", "input": {"q": "x"}},
            ],
            "stop_reason": "tool_use",
            "usage": {
                "input_tokens": 10,
                "output_tokens": 5,
                "cache_creation_input_tokens": 2,
                "cache_read_input_tokens": 3,
            },
        });
        let result = parse_response(&response);
        assert_eq!(result.content, json!("Hello\nworld"));
        assert_eq!(result.thinking_content.as_deref(), Some("hmm"));
        assert_eq!(
            result.thinking_blocks,
            vec![json!({"type": "thinking", "thinking": "hmm", "signature": "sig"})]
        );
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].id, "toolu_1");
        assert_eq!(result.tool_calls[0].name, "search");
        assert_eq!(result.tool_calls[0].input, json!({"q": "x"}));
        assert_eq!(result.input_tokens, 15); // 10 + 2 + 3
        assert_eq!(result.output_tokens, 5);
        assert_eq!(result.cache_creation_input_tokens, 2);
        assert_eq!(result.cache_read_input_tokens, 3);
        assert_eq!(result.finish_reason, "tool_use");
    }

    #[test]
    fn build_request_assembles_messages_body() {
        let messages = vec![
            json!({"role": "system", "content": "be concise"}),
            json!({"role": "user", "content": "hi"}),
        ];
        let tools = vec![json!({"name": "search", "description": "d", "input_schema": {}})];
        let mut extra = Map::new();
        extra.insert("top_p".to_string(), json!(0.9));
        extra.insert("frequency_penalty".to_string(), json!(0.5)); // not forwarded
        let stop = vec!["STOP".to_string()];

        let body = build_request(&RequestParams {
            model: "claude-opus-4-8",
            messages: &messages,
            max_tokens: 1024,
            temperature: Some(0.7),
            stop: Some(&stop),
            tools: Some(&tools),
            tool_choice: Some(&json!("auto")),
            thinking_budget_tokens: Some(2048),
            extra_params: &extra,
        });

        assert_eq!(body["model"], json!("claude-opus-4-8"));
        assert_eq!(body["max_tokens"], json!(1024));
        // System message is pulled out into the `system` field.
        assert_eq!(body["messages"], json!([{"role": "user", "content": "hi"}]));
        assert_eq!(body["system"][0]["text"], json!("be concise"));
        assert_eq!(
            body["system"][0]["cache_control"],
            json!({"type": "ephemeral"})
        );
        assert_eq!(body["temperature"], json!(0.7));
        assert_eq!(body["stop_sequences"], json!(["STOP"]));
        assert_eq!(body["tools"], json!(tools));
        assert_eq!(body["tool_choice"], json!({"type": "auto"}));
        assert_eq!(
            body["thinking"],
            json!({"type": "enabled", "budget_tokens": 2048})
        );
        assert_eq!(body["top_p"], json!(0.9));
        assert!(body.get("top_k").is_none());
        assert!(body.get("frequency_penalty").is_none());
    }

    #[test]
    fn build_request_omits_optional_fields_when_unset() {
        let messages = vec![json!({"role": "user", "content": "hi"})];
        let body = build_request(&RequestParams {
            model: "claude-3-5-sonnet",
            messages: &messages,
            max_tokens: 256,
            temperature: None,
            stop: Some(&[]),  // empty -> omitted
            tools: Some(&[]), // empty -> omitted
            tool_choice: Some(&json!("auto")),
            thinking_budget_tokens: Some(0), // falsy -> omitted
            extra_params: &Map::new(),
        });
        assert!(body.get("temperature").is_none());
        assert!(body.get("stop_sequences").is_none());
        assert!(body.get("tools").is_none());
        assert!(body.get("tool_choice").is_none()); // no tools -> no tool_choice
        assert!(body.get("thinking").is_none());
        assert!(body.get("system").is_none());
    }

    #[test]
    fn parse_response_defaults_when_usage_and_stop_absent() {
        let response = json!({"content": [{"type": "text", "text": "ok"}]});
        let result = parse_response(&response);
        assert_eq!(result.content, json!("ok"));
        assert_eq!(result.input_tokens, 0);
        assert_eq!(result.output_tokens, 0);
        assert_eq!(result.finish_reason, "stop");
        assert!(result.thinking_content.is_none());
        assert!(result.tool_calls.is_empty());
    }

    #[tokio::test]
    async fn complete_posts_to_messages_endpoint_and_parses() {
        use crate::llm::http::mock::MockHttp;

        let http = MockHttp::ok(json!({
            "content": [{"type": "text", "text": "hi there"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 4, "output_tokens": 2},
        }));
        let messages = vec![json!({"role": "user", "content": "hi"})];
        let params = RequestParams {
            model: "claude-opus-4-8",
            messages: &messages,
            max_tokens: 128,
            temperature: None,
            stop: None,
            tools: None,
            tool_choice: None,
            thinking_budget_tokens: None,
            extra_params: &Map::new(),
        };
        let credentials = Credentials::new("sk-test");

        let result = complete(&http, &credentials, &params).await.unwrap();

        assert_eq!(http.last_url(), "https://api.anthropic.com/v1/messages");
        assert_eq!(
            http.last_headers(),
            vec![
                ("x-api-key".to_string(), "sk-test".to_string()),
                ("anthropic-version".to_string(), "2023-06-01".to_string()),
            ]
        );
        // The posted body is exactly what build_request produces.
        assert_eq!(http.last_body(), build_request(&params));
        assert_eq!(result.content, json!("hi there"));
        assert_eq!(result.input_tokens, 4);
        assert_eq!(result.finish_reason, "end_turn");
    }

    #[tokio::test]
    async fn complete_honors_base_url_override_and_propagates_errors() {
        use crate::llm::http::mock::MockHttp;
        use crate::llm::http::LlmHttpError;

        let http = MockHttp::ok(json!({"content": []}));
        let messages = vec![json!({"role": "user", "content": "hi"})];
        let params = RequestParams {
            model: "claude-opus-4-8",
            messages: &messages,
            max_tokens: 128,
            temperature: None,
            stop: None,
            tools: None,
            tool_choice: None,
            thinking_budget_tokens: None,
            extra_params: &Map::new(),
        };
        // Trailing slash on the override is trimmed before the path is appended.
        let credentials = Credentials::with_base_url("k", Some("https://proxy.test/".to_string()));
        complete(&http, &credentials, &params).await.unwrap();
        assert_eq!(http.last_url(), "https://proxy.test/v1/messages");

        let failing = MockHttp::err(LlmHttpError::Status {
            status: 429,
            body: "rate limited".to_string(),
        });
        let err = complete(&failing, &Credentials::new("k"), &params)
            .await
            .unwrap_err();
        assert!(matches!(err, LlmHttpError::Status { status: 429, .. }));
    }
}
