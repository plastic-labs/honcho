//! Deterministic OpenAI backend helpers, ported from
//! `src/llm/backends/openai.py`.
//!
//! Covers the network-free parts: cache-token extraction, reasoning
//! content/details extraction, canonical→OpenAI tool conversion, structured-output
//! enforcement (`response_format` / `json_mode`, threaded via `extra_params`), and
//! parsing a Chat Completions response into a [`CompletionResult`].

use serde_json::{Map, Value, json};

use crate::llm::http::{Credentials, LlmHttp, LlmHttpError, LlmStreamHttp, TextStream};
use crate::llm::request_builder::{
    PassthroughError, apply_sdk_passthroughs, merge_header_mapping, passthrough_mapping,
    passthrough_value_to_string,
};
use crate::llm::{CompletionResult, ToolCallResult};

/// The OpenAI SDK's default API base. Unlike Anthropic this already includes the
/// `/v1` version segment, so the path appended below is just `/chat/completions`.
pub const DEFAULT_BASE_URL: &str = "https://api.openai.com/v1";

/// Run one Chat Completions request: build the body, POST it through the
/// [`LlmHttp`] transport with a Bearer auth header, and parse the response.
pub async fn complete<H: LlmHttp>(
    http: &H,
    credentials: &Credentials,
    params: &RequestParams<'_>,
) -> Result<CompletionResult, LlmHttpError> {
    let body = build_request(params).map_err(passthrough_error)?;
    let url = apply_extra_query(
        format!(
            "{}/chat/completions",
            credentials.effective_base_url(DEFAULT_BASE_URL)
        ),
        params.extra_params,
    )?;
    let headers = build_headers(credentials, params.extra_params)?;
    let response = http.post_json(&url, &headers, &body).await?;
    Ok(parse_response(&response))
}

fn passthrough_error(error: PassthroughError) -> LlmHttpError {
    LlmHttpError::Transport(error.to_string())
}

fn build_headers(
    credentials: &Credentials,
    extra_params: &Map<String, Value>,
) -> Result<Vec<(String, String)>, LlmHttpError> {
    let mut headers = vec![(
        "Authorization".to_string(),
        format!("Bearer {}", credentials.api_key),
    )];
    if let Some(mapping) =
        passthrough_mapping("extra_headers", extra_params).map_err(passthrough_error)?
    {
        merge_header_mapping(&mut headers, mapping);
    }
    Ok(headers)
}

fn apply_extra_query(
    url: String,
    extra_params: &Map<String, Value>,
) -> Result<String, LlmHttpError> {
    let Some(mapping) =
        passthrough_mapping("extra_query", extra_params).map_err(passthrough_error)?
    else {
        return Ok(url);
    };
    let mut parsed = reqwest::Url::parse(&url)
        .map_err(|error| LlmHttpError::Transport(format!("invalid request URL: {error}")))?;
    {
        let mut pairs = parsed.query_pairs_mut();
        for (key, value) in mapping {
            pairs.append_pair(key, &passthrough_value_to_string(value));
        }
    }
    Ok(parsed.to_string())
}

fn merge_extra_body(
    body: &mut Map<String, Value>,
    extra_params: &Map<String, Value>,
) -> Result<(), PassthroughError> {
    let mut passthroughs = Map::new();
    apply_sdk_passthroughs(&mut passthroughs, extra_params)?;
    let Some(extra_body) = passthroughs.remove("extra_body") else {
        return Ok(());
    };
    let existing = body
        .entry("extra_body".to_string())
        .or_insert_with(|| Value::Object(Map::new()));
    if !existing.is_object() {
        return Err(PassthroughError::new("extra_body", existing));
    }
    let existing_mapping = existing.as_object_mut().expect("object checked above");
    let extra_body = extra_body
        .as_object()
        .ok_or_else(|| PassthroughError::new("extra_body", &extra_body))?;
    for (key, value) in extra_body {
        existing_mapping.insert(key.clone(), value.clone());
    }
    Ok(())
}

/// Open a streaming Chat Completions request: the same body as [`complete`] with
/// `stream: true` and `stream_options.include_usage` (so the terminal chunk
/// carries usage), returning the raw SSE [`TextStream`].
pub async fn stream<H: LlmStreamHttp>(
    http: &H,
    credentials: &Credentials,
    params: &RequestParams<'_>,
) -> Result<TextStream, LlmHttpError> {
    let mut body = build_request(params).map_err(passthrough_error)?;
    if let Some(object) = body.as_object_mut() {
        object.insert("stream".to_string(), json!(true));
        object.insert("stream_options".to_string(), json!({"include_usage": true}));
    }
    let url = apply_extra_query(
        format!(
            "{}/chat/completions",
            credentials.effective_base_url(DEFAULT_BASE_URL)
        ),
        params.extra_params,
    )?;
    let headers = build_headers(credentials, params.extra_params)?;
    http.post_json_stream(&url, &headers, &body).await
}

/// Whether the model uses `max_completion_tokens` instead of `max_tokens`,
/// porting `_uses_max_completion_tokens`: GPT-5 family and the o1/o3/o4
/// reasoning models.
pub fn uses_max_completion_tokens(model: &str) -> bool {
    if model == "gpt-5" || model.starts_with("gpt-5-") || model.starts_with("gpt-5.") {
        return true;
    }
    ["o1", "o3", "o4"]
        .iter()
        .any(|prefix| model == *prefix || model.starts_with(&format!("{prefix}-")))
}

/// Inputs for [`build_request`], mirroring the portable subset of `_build_params`.
/// `response_format` and `json_mode` are not separate fields here — they ride in
/// `extra_params` (see [`build_request`]'s structured-output handling).
#[derive(Debug, Clone)]
pub struct RequestParams<'a> {
    pub model: &'a str,
    pub messages: &'a [Value],
    pub max_tokens: i64,
    pub temperature: Option<f64>,
    pub stop: Option<&'a [String]>,
    pub tools: Option<&'a [Value]>,
    pub tool_choice: Option<&'a Value>,
    pub thinking_effort: Option<&'a str>,
    pub thinking_budget_tokens: Option<i64>,
    pub extra_params: &'a Map<String, Value>,
}

/// Build the JSON body for the Chat Completions API, porting the deterministic
/// `_build_params` (max-token field selection, temperature, reasoning effort,
/// proxy `extra_body.reasoning.max_tokens`, stop, converted tools + tool_choice,
/// `top_p`/`frequency_penalty`/`presence_penalty`/`seed` passthrough, and the
/// `response_format`/`json_mode` structured-output enforcement).
pub fn build_request(params: &RequestParams<'_>) -> Result<Value, PassthroughError> {
    let mut body = Map::new();
    body.insert("model".to_string(), json!(params.model));
    body.insert("messages".to_string(), json!(params.messages));

    if uses_max_completion_tokens(params.model) {
        body.insert(
            "max_completion_tokens".to_string(),
            json!(params.max_tokens),
        );
        if let Some(verbosity) = params
            .extra_params
            .get("verbosity")
            .filter(|value| !value.is_null())
        {
            body.insert("verbosity".to_string(), verbosity.clone());
        }
    } else {
        body.insert("max_tokens".to_string(), json!(params.max_tokens));
    }

    if let Some(temperature) = params.temperature {
        body.insert("temperature".to_string(), json!(temperature));
    }
    if let Some(effort) = params.thinking_effort.filter(|effort| !effort.is_empty()) {
        body.insert("reasoning_effort".to_string(), json!(effort));
    }
    if let Some(budget) = params.thinking_budget_tokens.filter(|budget| *budget > 0) {
        body.insert(
            "extra_body".to_string(),
            json!({"reasoning": {"max_tokens": budget}}),
        );
    }
    if let Some(stop) = params.stop.filter(|stop| !stop.is_empty()) {
        body.insert("stop".to_string(), json!(stop));
    }
    if let Some(tools) = params.tools.filter(|tools| !tools.is_empty()) {
        body.insert("tools".to_string(), json!(convert_tools(tools)));
        if let Some(tool_choice) = params.tool_choice {
            body.insert("tool_choice".to_string(), convert_tool_choice(tool_choice));
        }
    }
    for key in ["top_p", "frequency_penalty", "presence_penalty", "seed"] {
        if let Some(value) = params.extra_params.get(key) {
            body.insert(key.to_string(), value.clone());
        }
    }

    // Structured-output enforcement, porting the response_format branch of
    // openai.py `complete`/`stream`: an explicit `response_format` (e.g. the
    // deriver's strict `json_schema`, mirroring Python's `response_model`) wins;
    // otherwise `json_mode` maps to `{"type": "json_object"}`. Dropping this is
    // what let the deriver model reply with prose.
    if let Some(response_format) = params
        .extra_params
        .get("response_format")
        .filter(|value| !value.is_null())
    {
        body.insert("response_format".to_string(), response_format.clone());
    } else if params
        .extra_params
        .get("json_mode")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        body.insert(
            "response_format".to_string(),
            json!({"type": "json_object"}),
        );
    }

    merge_extra_body(&mut body, params.extra_params)?;

    Ok(Value::Object(body))
}

/// Extract `(cache_creation, cache_read)` tokens, porting
/// `extract_openai_cache_tokens`. `cache_read` prefers
/// `usage.prompt_tokens_details.cached_tokens`, then `cache_read_input_tokens`,
/// then top-level `cached_tokens`; `cache_creation` comes from
/// `cache_creation_input_tokens`. Zero/absent values are treated as unset.
pub fn extract_cache_tokens(usage: Option<&Value>) -> (i64, i64) {
    let Some(usage) = usage else {
        return (0, 0);
    };
    let truthy_int = |value: Option<&Value>| value.and_then(Value::as_i64).filter(|v| *v != 0);

    let mut cache_read = truthy_int(
        usage
            .get("prompt_tokens_details")
            .and_then(|details| details.get("cached_tokens")),
    );
    if cache_read.is_none() {
        cache_read = truthy_int(usage.get("cache_read_input_tokens"))
            .or_else(|| truthy_int(usage.get("cached_tokens")));
    }
    let cache_creation = truthy_int(usage.get("cache_creation_input_tokens"));
    (cache_creation.unwrap_or(0), cache_read.unwrap_or(0))
}

/// Join the non-empty `content` of each `message.reasoning_details` entry (then
/// fall back to `message.reasoning_content`), porting
/// `extract_openai_reasoning_content`.
pub fn extract_reasoning_content(message: &Value) -> Option<String> {
    if let Some(details) = message.get("reasoning_details").and_then(Value::as_array) {
        let parts: Vec<&str> = details
            .iter()
            .filter_map(|detail| detail.get("content").and_then(Value::as_str))
            .filter(|content| !content.is_empty())
            .collect();
        if !parts.is_empty() {
            return Some(parts.join("\n"));
        }
    }
    message
        .get("reasoning_content")
        .and_then(Value::as_str)
        .filter(|content| !content.is_empty())
        .map(str::to_string)
}

/// Return the `message.reasoning_details` list as-is, porting
/// `extract_openai_reasoning_details` (each detail dict round-trips unchanged).
pub fn extract_reasoning_details(message: &Value) -> Vec<Value> {
    message
        .get("reasoning_details")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default()
}

/// Convert canonical tool schemas to OpenAI's function shape, porting
/// `_convert_tools`. Already-converted tools (first entry `type == "function"`)
/// and empty lists pass through unchanged.
/// Translate Honcho's canonical `tool_choice` vocabulary to OpenAI's, mirroring
/// the Anthropic/Gemini backends so a single `TOOL_CHOICE` value resolves
/// correctly regardless of which provider a fallback chain lands on. OpenAI has
/// no `"any"` — it spells the same intent `"required"`. A bare tool-name string
/// or a `{"name": ...}` object becomes a function selection; `auto`/`none` pass
/// through. (Port of #850.)
pub fn convert_tool_choice(tool_choice: &Value) -> Value {
    match tool_choice {
        Value::Object(map) => match map.get("name") {
            Some(name) => json!({"type": "function", "function": {"name": name}}),
            None => tool_choice.clone(),
        },
        Value::String(choice) => match choice.as_str() {
            "any" | "required" => json!("required"),
            "auto" | "none" => tool_choice.clone(),
            name => json!({"type": "function", "function": {"name": name}}),
        },
        other => other.clone(),
    }
}

pub fn convert_tools(tools: &[Value]) -> Vec<Value> {
    let already_converted = tools
        .first()
        .and_then(|tool| tool.get("type"))
        .and_then(Value::as_str)
        == Some("function");
    if tools.is_empty() || already_converted {
        return tools.to_vec();
    }
    tools
        .iter()
        .map(|tool| {
            json!({
                "type": "function",
                "function": {
                    "name": tool.get("name").cloned().unwrap_or(Value::Null),
                    "description": tool.get("description").cloned().unwrap_or(Value::Null),
                    "parameters": tool.get("input_schema").cloned().unwrap_or(Value::Null),
                },
            })
        })
        .collect()
}

/// Parse a Chat Completions response into a [`CompletionResult`], porting
/// `_normalize_response` (no `response_format`). `content` defaults to `""`;
/// tool-call `arguments` are JSON-decoded (malformed → `{}`); `input_tokens` is
/// `prompt_tokens` (unlike Anthropic, not summed with cache tokens).
pub fn parse_response(response: &Value) -> CompletionResult {
    let choice = response
        .get("choices")
        .and_then(Value::as_array)
        .and_then(|choices| choices.first());
    let message = choice.and_then(|choice| choice.get("message"));

    let content = message
        .and_then(|message| message.get("content"))
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();

    let mut tool_calls = Vec::new();
    if let Some(calls) = message
        .and_then(|message| message.get("tool_calls"))
        .and_then(Value::as_array)
    {
        for call in calls {
            let function = call.get("function");
            let arguments = function
                .and_then(|function| function.get("arguments"))
                .and_then(Value::as_str)
                .unwrap_or("");
            // Malformed arguments degrade to an empty object, as in Python.
            let input = if arguments.is_empty() {
                json!({})
            } else {
                serde_json::from_str(arguments).unwrap_or_else(|_| json!({}))
            };
            tool_calls.push(ToolCallResult {
                id: call
                    .get("id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                name: function
                    .and_then(|function| function.get("name"))
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                input,
                thought_signature: None,
            });
        }
    }

    let usage = response.get("usage");
    let usage_int = |key: &str| -> i64 {
        usage
            .and_then(|usage| usage.get(key))
            .and_then(Value::as_i64)
            .unwrap_or(0)
    };
    let (cache_creation, cache_read) = extract_cache_tokens(usage);

    CompletionResult {
        content: Value::String(content),
        input_tokens: usage_int("prompt_tokens"),
        output_tokens: usage_int("completion_tokens"),
        cache_creation_input_tokens: cache_creation,
        cache_read_input_tokens: cache_read,
        finish_reason: choice
            .and_then(|choice| choice.get("finish_reason"))
            .and_then(Value::as_str)
            .unwrap_or("stop")
            .to_string(),
        tool_calls,
        thinking_content: message.and_then(extract_reasoning_content),
        thinking_blocks: Vec::new(),
        reasoning_details: message.map(extract_reasoning_details).unwrap_or_default(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_tokens_prefer_prompt_tokens_details() {
        let usage = json!({"prompt_tokens_details": {"cached_tokens": 4}});
        assert_eq!(extract_cache_tokens(Some(&usage)), (0, 4));

        let fallback = json!({"cache_read_input_tokens": 7, "cache_creation_input_tokens": 2});
        assert_eq!(extract_cache_tokens(Some(&fallback)), (2, 7));

        assert_eq!(extract_cache_tokens(None), (0, 0));
        // Zero cached_tokens is treated as unset and falls through.
        let zero = json!({"prompt_tokens_details": {"cached_tokens": 0}, "cached_tokens": 5});
        assert_eq!(extract_cache_tokens(Some(&zero)), (0, 5));
    }

    #[test]
    fn convert_tools_maps_canonical_schema() {
        let tools = vec![json!({
            "name": "search",
            "description": "find things",
            "input_schema": {"type": "object", "properties": {}},
        })];
        assert_eq!(
            convert_tools(&tools),
            vec![json!({
                "type": "function",
                "function": {
                    "name": "search",
                    "description": "find things",
                    "parameters": {"type": "object", "properties": {}},
                },
            })]
        );
        // Already-converted tools pass through.
        let converted = vec![json!({"type": "function", "function": {"name": "x"}})];
        assert_eq!(convert_tools(&converted), converted);
        assert_eq!(convert_tools(&[]), Vec::<Value>::new());
    }

    #[test]
    fn convert_tool_choice_maps_canonical_vocabulary() {
        // "any" (canonical, used by Anthropic/Gemini) maps to OpenAI's "required".
        assert_eq!(convert_tool_choice(&json!("any")), json!("required"));
        assert_eq!(convert_tool_choice(&json!("required")), json!("required"));
        // auto/none pass through unchanged.
        assert_eq!(convert_tool_choice(&json!("auto")), json!("auto"));
        assert_eq!(convert_tool_choice(&json!("none")), json!("none"));
        // A bare tool-name string becomes a function selection.
        assert_eq!(
            convert_tool_choice(&json!("search")),
            json!({"type": "function", "function": {"name": "search"}})
        );
        // A {"name": ...} object becomes a function selection.
        assert_eq!(
            convert_tool_choice(&json!({"name": "search"})),
            json!({"type": "function", "function": {"name": "search"}})
        );
        // An already-OpenAI-shaped object passes through.
        let native = json!({"type": "function", "function": {"name": "x"}});
        assert_eq!(convert_tool_choice(&native), native);
    }

    #[test]
    fn uses_max_completion_tokens_for_reasoning_models() {
        assert!(uses_max_completion_tokens("gpt-5"));
        assert!(uses_max_completion_tokens("gpt-5-mini"));
        assert!(uses_max_completion_tokens("o1"));
        assert!(uses_max_completion_tokens("o3-mini"));
        assert!(uses_max_completion_tokens("o4-preview"));
        assert!(!uses_max_completion_tokens("gpt-4o"));
        assert!(!uses_max_completion_tokens("o10")); // not o1-prefixed
    }

    #[test]
    fn build_request_selects_token_field_and_forwards_knobs() {
        let messages = vec![json!({"role": "user", "content": "hi"})];
        let tools = vec![json!({"name": "search", "description": "d", "input_schema": {}})];
        let mut extra = Map::new();
        extra.insert("top_p".to_string(), json!(0.8));
        extra.insert("seed".to_string(), json!(7));
        let body = build_request(&RequestParams {
            model: "gpt-5",
            messages: &messages,
            max_tokens: 512,
            temperature: Some(0.3),
            stop: None,
            tools: Some(&tools),
            tool_choice: Some(&json!("auto")),
            thinking_effort: Some("high"),
            thinking_budget_tokens: Some(1000),
            extra_params: &extra,
        })
        .unwrap();
        // gpt-5 -> max_completion_tokens, not max_tokens.
        assert_eq!(body["max_completion_tokens"], json!(512));
        assert!(body.get("max_tokens").is_none());
        assert_eq!(body["temperature"], json!(0.3));
        assert_eq!(body["reasoning_effort"], json!("high"));
        assert_eq!(
            body["extra_body"],
            json!({"reasoning": {"max_tokens": 1000}})
        );
        // Tools are converted to function shape.
        assert_eq!(body["tools"][0]["type"], json!("function"));
        assert_eq!(body["tool_choice"], json!("auto"));
        assert_eq!(body["top_p"], json!(0.8));
        assert_eq!(body["seed"], json!(7));
        assert!(body.get("stop").is_none());
    }

    #[test]
    fn build_request_threads_response_format_and_json_mode() {
        let messages = vec![json!({"role": "user", "content": "hi"})];

        // An explicit response_format (the deriver's json_schema) is passed through.
        let schema = json!({"type": "json_schema", "json_schema": {"name": "X"}});
        let mut extra = Map::new();
        extra.insert("response_format".to_string(), schema.clone());
        extra.insert("json_mode".to_string(), json!(true));
        let body = build_request(&RequestParams {
            model: "gpt-5.4-mini",
            messages: &messages,
            max_tokens: 256,
            temperature: None,
            stop: None,
            tools: None,
            tool_choice: None,
            thinking_effort: None,
            thinking_budget_tokens: None,
            extra_params: &extra,
        })
        .unwrap();
        // Explicit response_format wins over json_mode.
        assert_eq!(body["response_format"], schema);

        // json_mode alone falls back to the json_object response format.
        let mut json_only = Map::new();
        json_only.insert("json_mode".to_string(), json!(true));
        let body = build_request(&RequestParams {
            model: "gpt-4o",
            messages: &messages,
            max_tokens: 256,
            temperature: None,
            stop: None,
            tools: None,
            tool_choice: None,
            thinking_effort: None,
            thinking_budget_tokens: None,
            extra_params: &json_only,
        })
        .unwrap();
        assert_eq!(body["response_format"], json!({"type": "json_object"}));

        // json_mode=false emits nothing.
        let mut json_off = Map::new();
        json_off.insert("json_mode".to_string(), json!(false));
        let body = build_request(&RequestParams {
            model: "gpt-4o",
            messages: &messages,
            max_tokens: 256,
            temperature: None,
            stop: None,
            tools: None,
            tool_choice: None,
            thinking_effort: None,
            thinking_budget_tokens: None,
            extra_params: &json_off,
        })
        .unwrap();
        assert!(body.get("response_format").is_none());
    }

    #[test]
    fn build_request_uses_max_tokens_for_classic_models() {
        let messages = vec![json!({"role": "user", "content": "hi"})];
        let body = build_request(&RequestParams {
            model: "gpt-4o",
            messages: &messages,
            max_tokens: 256,
            temperature: None,
            stop: Some(&["END".to_string()]),
            tools: None,
            tool_choice: None,
            thinking_effort: None,
            thinking_budget_tokens: None,
            extra_params: &Map::new(),
        })
        .unwrap();
        assert_eq!(body["max_tokens"], json!(256));
        assert!(body.get("max_completion_tokens").is_none());
        assert_eq!(body["stop"], json!(["END"]));
        assert!(body.get("reasoning_effort").is_none());
        assert!(body.get("extra_body").is_none());
    }

    #[test]
    fn build_request_merges_provider_extra_body_operator_wins() {
        let messages = vec![json!({"role": "user", "content": "hi"})];
        let mut extra = Map::new();
        extra.insert(
            "extra_body".to_string(),
            json!({"reasoning": {"max_tokens": 200}, "custom": "v"}),
        );
        let body = build_request(&RequestParams {
            model: "gpt-4o",
            messages: &messages,
            max_tokens: 256,
            temperature: None,
            stop: None,
            tools: None,
            tool_choice: None,
            thinking_effort: None,
            thinking_budget_tokens: Some(100),
            extra_params: &extra,
        })
        .unwrap();

        assert_eq!(
            body["extra_body"],
            json!({"reasoning": {"max_tokens": 200}, "custom": "v"})
        );
    }

    #[test]
    fn build_request_rejects_non_mapping_passthrough() {
        let messages = vec![json!({"role": "user", "content": "hi"})];
        let mut extra = Map::new();
        extra.insert("extra_headers".to_string(), json!("bad"));
        let err = build_request(&RequestParams {
            model: "gpt-4o",
            messages: &messages,
            max_tokens: 256,
            temperature: None,
            stop: None,
            tools: None,
            tool_choice: None,
            thinking_effort: None,
            thinking_budget_tokens: None,
            extra_params: &extra,
        })
        .unwrap_err();

        assert_eq!(
            err.to_string(),
            "provider_params.extra_headers must be a mapping, got string"
        );
    }

    #[test]
    fn parse_response_decodes_tool_arguments_and_reasoning() {
        let response = json!({
            "choices": [{
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "search", "arguments": "{\"q\": \"x\"}"},
                    }],
                    "reasoning_details": [{"type": "reasoning.text", "content": "thinking"}],
                },
            }],
            "usage": {
                "prompt_tokens": 12,
                "completion_tokens": 8,
                "prompt_tokens_details": {"cached_tokens": 4},
            },
        });
        let result = parse_response(&response);
        assert_eq!(result.content, json!(""));
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].id, "call_1");
        assert_eq!(result.tool_calls[0].name, "search");
        assert_eq!(result.tool_calls[0].input, json!({"q": "x"}));
        assert_eq!(result.input_tokens, 12); // prompt_tokens, not summed
        assert_eq!(result.output_tokens, 8);
        assert_eq!(result.cache_read_input_tokens, 4);
        assert_eq!(result.finish_reason, "tool_calls");
        assert_eq!(result.thinking_content.as_deref(), Some("thinking"));
        assert_eq!(
            result.reasoning_details,
            vec![json!({"type": "reasoning.text", "content": "thinking"})]
        );
    }

    #[test]
    fn parse_response_handles_plain_text_and_malformed_args() {
        let response = json!({
            "choices": [{
                "finish_reason": "stop",
                "message": {
                    "content": "hello",
                    "tool_calls": [{
                        "id": "c1",
                        "function": {"name": "f", "arguments": "not json"},
                    }],
                },
            }],
        });
        let result = parse_response(&response);
        assert_eq!(result.content, json!("hello"));
        assert_eq!(result.tool_calls[0].input, json!({})); // malformed -> {}
        assert_eq!(result.finish_reason, "stop");
        assert_eq!(result.input_tokens, 0);
        assert!(result.thinking_content.is_none());
    }

    #[tokio::test]
    async fn complete_posts_to_chat_completions_with_bearer_auth() {
        use crate::llm::http::mock::MockHttp;

        let http = MockHttp::ok(json!({
            "choices": [{"finish_reason": "stop", "message": {"content": "hi"}}],
            "usage": {"prompt_tokens": 3, "completion_tokens": 1},
        }));
        let messages = vec![json!({"role": "user", "content": "hi"})];
        let params = RequestParams {
            model: "gpt-4o",
            messages: &messages,
            max_tokens: 64,
            temperature: None,
            stop: None,
            tools: None,
            tool_choice: None,
            thinking_effort: None,
            thinking_budget_tokens: None,
            extra_params: &Map::new(),
        };

        let result = complete(&http, &Credentials::new("sk-openai"), &params)
            .await
            .unwrap();

        // The default base already carries `/v1`; only the path is appended.
        assert_eq!(
            http.last_url(),
            "https://api.openai.com/v1/chat/completions"
        );
        assert_eq!(
            http.last_headers(),
            vec![("Authorization".to_string(), "Bearer sk-openai".to_string())]
        );
        assert_eq!(http.last_body(), build_request(&params).unwrap());
        assert_eq!(result.content, json!("hi"));
        assert_eq!(result.input_tokens, 3);
    }

    #[tokio::test]
    async fn complete_forwards_provider_extra_headers_and_query() {
        use crate::llm::http::mock::MockHttp;

        let http = MockHttp::ok(json!({
            "choices": [{"finish_reason": "stop", "message": {"content": "hi"}}],
        }));
        let messages = vec![json!({"role": "user", "content": "hi"})];
        let mut extra = Map::new();
        extra.insert("extra_headers".to_string(), json!({"X-Trace": "abc"}));
        extra.insert("extra_query".to_string(), json!({"route": "openrouter"}));
        let params = RequestParams {
            model: "gpt-4o",
            messages: &messages,
            max_tokens: 64,
            temperature: None,
            stop: None,
            tools: None,
            tool_choice: None,
            thinking_effort: None,
            thinking_budget_tokens: None,
            extra_params: &extra,
        };

        complete(&http, &Credentials::new("sk-openai"), &params)
            .await
            .unwrap();

        assert_eq!(
            http.last_url(),
            "https://api.openai.com/v1/chat/completions?route=openrouter"
        );
        assert_eq!(
            http.last_headers(),
            vec![
                ("Authorization".to_string(), "Bearer sk-openai".to_string()),
                ("X-Trace".to_string(), "abc".to_string()),
            ]
        );
    }
}
