//! Deterministic Gemini backend helpers, ported from
//! `src/llm/backends/gemini.py`.
//!
//! Covers parsing a `generateContent` response into a [`CompletionResult`]
//! (no `response_format`). The "blocked finish reason" escalation that Python
//! raises as an `LLMError` is left to callers — [`is_blocked_finish_reason`] and
//! [`BLOCKED_FINISH_REASONS`] expose the same set, and the parsed
//! `finish_reason` is preserved so callers can decide.

use serde_json::{Map, Value, json};

use crate::llm::{CompletionResult, ToolCallResult};

/// Finish reasons that mean Gemini blocked the response (`GEMINI_BLOCKED_FINISH_REASONS`).
pub const BLOCKED_FINISH_REASONS: &[&str] =
    &["SAFETY", "RECITATION", "PROHIBITED_CONTENT", "BLOCKLIST"];

/// JSON-Schema keywords Gemini's `function_declarations` validator accepts
/// (`_GEMINI_ALLOWED_SCHEMA_KEYS`). Anything else triggers a 400, so it's stripped.
const ALLOWED_SCHEMA_KEYS: &[&str] = &[
    "type",
    "format",
    "description",
    "nullable",
    "enum",
    "properties",
    "required",
    "items",
    "minItems",
    "maxItems",
    "minimum",
    "maximum",
    "title",
];

/// Returned when a request sets both thinking-budget and thinking-effort, which
/// Gemini cannot accept together (matches the Python `ValidationException`).
#[derive(Debug, PartialEq, Eq)]
pub struct ConflictingThinkingParams;

/// Inputs for [`build_config`], mirroring the portable subset of `_build_config`
/// (the `response_format`/`response_schema` path is excluded; `json_mode` via
/// `extra_params` is honored).
#[derive(Debug, Clone)]
pub struct ConfigParams<'a> {
    pub max_tokens: i64,
    pub temperature: Option<f64>,
    pub stop: Option<&'a [String]>,
    pub tools: Option<&'a [Value]>,
    pub tool_choice: Option<&'a Value>,
    pub thinking_budget_tokens: Option<i64>,
    pub thinking_effort: Option<&'a str>,
    pub extra_params: &'a Map<String, Value>,
}

/// Recursively strip JSON-Schema keywords Gemini rejects, porting
/// `_sanitize_schema`. `properties` keys (user field names) are preserved while
/// their sub-schemas are sanitized; `items` recurses; `required`/`enum` lists
/// pass through; other allowed keywords pass through verbatim.
pub fn sanitize_schema(schema: &Value) -> Value {
    let Some(object) = schema.as_object() else {
        return schema.clone();
    };
    let mut cleaned = Map::new();
    for (key, value) in object {
        if !ALLOWED_SCHEMA_KEYS.contains(&key.as_str()) {
            continue;
        }
        match (key.as_str(), value) {
            ("properties", Value::Object(properties)) => {
                let sanitized = properties
                    .iter()
                    .map(|(name, sub)| (name.clone(), sanitize_schema(sub)))
                    .collect();
                cleaned.insert("properties".to_string(), Value::Object(sanitized));
            }
            ("items", _) => {
                cleaned.insert("items".to_string(), sanitize_schema(value));
            }
            _ => {
                cleaned.insert(key.clone(), value.clone());
            }
        }
    }
    Value::Object(cleaned)
}

/// Convert canonical tool schemas to Gemini's `function_declarations` shape,
/// porting `_convert_tools`. Already-wrapped tools pass through; otherwise each
/// tool's `input_schema` is sanitized.
pub fn convert_tools(tools: &[Value]) -> Vec<Value> {
    if tools
        .first()
        .is_some_and(|tool| tool.get("function_declarations").is_some())
    {
        return tools.to_vec();
    }
    let declarations = tools
        .iter()
        .map(|tool| {
            json!({
                "name": tool.get("name").cloned().unwrap_or(Value::Null),
                "description": tool.get("description").cloned().unwrap_or(Value::Null),
                "parameters": sanitize_schema(tool.get("input_schema").unwrap_or(&Value::Null)),
            })
        })
        .collect::<Vec<_>>();
    vec![json!({"function_declarations": declarations})]
}

/// Convert the canonical `tool_choice` to Gemini's `function_calling_config`,
/// porting `_convert_tool_choice`.
pub fn convert_tool_choice(tool_choice: &Value) -> Value {
    if let Some(name) = tool_choice.get("name").and_then(Value::as_str) {
        return json!({
            "function_calling_config": {"mode": "ANY", "allowed_function_names": [name]}
        });
    }
    match tool_choice.as_str() {
        Some("auto") => json!({"function_calling_config": {"mode": "AUTO"}}),
        Some("any" | "required") => json!({"function_calling_config": {"mode": "ANY"}}),
        Some("none") => json!({"function_calling_config": {"mode": "NONE"}}),
        Some(name) => json!({
            "function_calling_config": {"mode": "ANY", "allowed_function_names": [name]}
        }),
        None => json!({"function_calling_config": {"mode": "AUTO"}}),
    }
}

/// Whether a JSON value is "truthy" the way Python treats `if tool_choice:` /
/// `extra_params.get("json_mode")`: non-null, non-empty string, non-zero number,
/// non-empty container, `true`.
fn is_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64().is_some_and(|f| f != 0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// Build the `GenerateContentConfig` dict, porting the deterministic
/// `_build_config` (max tokens, temperature, stop, tools, tool_config, thinking
/// config, `json_mode`, and tuning passthrough). Errors when both thinking
/// params are set, matching Python. The `response_schema` path is excluded.
pub fn build_config(params: &ConfigParams<'_>) -> Result<Value, ConflictingThinkingParams> {
    let mut config = Map::new();
    config.insert("max_output_tokens".to_string(), json!(params.max_tokens));

    if let Some(temperature) = params.temperature {
        config.insert("temperature".to_string(), json!(temperature));
    }
    if let Some(stop) = params.stop.filter(|stop| !stop.is_empty()) {
        config.insert("stop_sequences".to_string(), json!(stop));
    }
    let has_tools = params.tools.is_some_and(|tools| !tools.is_empty());
    if let Some(tools) = params.tools.filter(|tools| !tools.is_empty()) {
        config.insert("tools".to_string(), json!(convert_tools(tools)));
    }
    if let Some(tool_choice) = params.tool_choice.filter(|tc| is_truthy(tc)) {
        config.insert("tool_config".to_string(), convert_tool_choice(tool_choice));
    }
    // json_mode (no response_format) only applies when there are no tools.
    if !has_tools && params.extra_params.get("json_mode").is_some_and(is_truthy) {
        config.insert("response_mime_type".to_string(), json!("application/json"));
    }

    let mut thinking_config = Map::new();
    if let Some(budget) = params.thinking_budget_tokens {
        thinking_config.insert("thinking_budget".to_string(), json!(budget));
    }
    if let Some(effort) = params.thinking_effort {
        thinking_config.insert("thinking_level".to_string(), json!(effort));
    }
    if thinking_config.len() > 1 {
        return Err(ConflictingThinkingParams);
    }
    if !thinking_config.is_empty() {
        config.insert(
            "thinking_config".to_string(),
            Value::Object(thinking_config),
        );
    }

    for key in [
        "top_p",
        "top_k",
        "frequency_penalty",
        "presence_penalty",
        "seed",
    ] {
        if let Some(value) = params.extra_params.get(key) {
            config.insert(key.to_string(), value.clone());
        }
    }

    Ok(Value::Object(config))
}

/// Whether a finish reason indicates a blocked response.
pub fn is_blocked_finish_reason(reason: &str) -> bool {
    BLOCKED_FINISH_REASONS.contains(&reason)
}

fn push_function_call(
    tool_calls: &mut Vec<ToolCallResult>,
    function_call: &Value,
    thought_signature: Option<String>,
) {
    let Some(name) = function_call.get("name").and_then(Value::as_str) else {
        return;
    };
    let input = match function_call.get("args") {
        Some(args) if args.is_object() => args.clone(),
        _ => json!({}),
    };
    tool_calls.push(ToolCallResult {
        id: format!("call_{name}_{}", tool_calls.len()),
        name: name.to_string(),
        input,
        thought_signature,
    });
}

/// Parse a `generateContent` response into a [`CompletionResult`], porting
/// `_normalize_response` (no `response_format`). Text parts join with `\n`;
/// `function_call` parts become tool calls with synthetic `call_{name}_{i}` ids
/// and carry any `thought_signature`. Falls back to the top-level `text` /
/// `function_calls` SDK conveniences when the candidate yields none.
pub fn parse_response(response: &Value) -> CompletionResult {
    let candidate = response
        .get("candidates")
        .and_then(Value::as_array)
        .and_then(|candidates| candidates.first());

    let finish_reason = candidate
        .and_then(|candidate| candidate.get("finish_reason"))
        .and_then(Value::as_str)
        .unwrap_or("stop")
        .to_string();

    let mut text_parts: Vec<String> = Vec::new();
    let mut tool_calls: Vec<ToolCallResult> = Vec::new();

    if let Some(parts) = candidate
        .and_then(|candidate| candidate.get("content"))
        .and_then(|content| content.get("parts"))
        .and_then(Value::as_array)
    {
        for part in parts {
            if let Some(text) = part
                .get("text")
                .and_then(Value::as_str)
                .filter(|text| !text.is_empty())
            {
                text_parts.push(text.to_string());
            }
            if let Some(function_call) = part.get("function_call") {
                let signature = part
                    .get("thought_signature")
                    .and_then(Value::as_str)
                    .map(str::to_string);
                push_function_call(&mut tool_calls, function_call, signature);
            }
        }
    }

    // Top-level SDK-convenience fallbacks.
    if text_parts.is_empty()
        && let Some(text) = response
            .get("text")
            .and_then(Value::as_str)
            .filter(|text| !text.is_empty())
    {
        text_parts.push(text.to_string());
    }
    if tool_calls.is_empty()
        && let Some(function_calls) = response.get("function_calls").and_then(Value::as_array)
    {
        for function_call in function_calls {
            push_function_call(&mut tool_calls, function_call, None);
        }
    }

    let usage = response.get("usage_metadata");
    let usage_int = |key: &str| -> i64 {
        usage
            .and_then(|usage| usage.get(key))
            .and_then(Value::as_i64)
            .unwrap_or(0)
    };

    CompletionResult {
        content: Value::String(text_parts.join("\n")),
        input_tokens: usage_int("prompt_token_count"),
        output_tokens: usage_int("candidates_token_count"),
        cache_creation_input_tokens: 0,
        cache_read_input_tokens: usage_int("cached_content_token_count"),
        finish_reason,
        tool_calls,
        thinking_content: None,
        thinking_blocks: Vec::new(),
        reasoning_details: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocked_finish_reasons() {
        assert!(is_blocked_finish_reason("SAFETY"));
        assert!(is_blocked_finish_reason("BLOCKLIST"));
        assert!(!is_blocked_finish_reason("STOP"));
    }

    #[test]
    fn sanitize_schema_strips_disallowed_keys_recursively() {
        let schema = json!({
            "type": "object",
            "additionalProperties": false,           // stripped
            "properties": {
                "q": {"type": "string", "$ref": "#/x"}, // $ref stripped
                "n": {"type": "integer", "minimum": 0, "anyOf": []}, // anyOf stripped
            },
            "required": ["q"],
            "items": {"type": "string", "pattern": "x"}, // pattern stripped
        });
        assert_eq!(
            sanitize_schema(&schema),
            json!({
                "type": "object",
                "properties": {
                    "q": {"type": "string"},
                    "n": {"type": "integer", "minimum": 0},
                },
                "required": ["q"],
                "items": {"type": "string"},
            })
        );
    }

    #[test]
    fn convert_tools_wraps_function_declarations() {
        let tools = vec![json!({
            "name": "search",
            "description": "find",
            "input_schema": {"type": "object", "additionalProperties": false},
        })];
        assert_eq!(
            convert_tools(&tools),
            vec![json!({
                "function_declarations": [{
                    "name": "search",
                    "description": "find",
                    "parameters": {"type": "object"}, // additionalProperties stripped
                }]
            })]
        );
        // Already-wrapped passes through.
        let wrapped = vec![json!({"function_declarations": []})];
        assert_eq!(convert_tools(&wrapped), wrapped);
    }

    #[test]
    fn convert_tool_choice_maps_modes() {
        assert_eq!(
            convert_tool_choice(&json!("auto")),
            json!({"function_calling_config": {"mode": "AUTO"}})
        );
        assert_eq!(
            convert_tool_choice(&json!("required")),
            json!({"function_calling_config": {"mode": "ANY"}})
        );
        assert_eq!(
            convert_tool_choice(&json!("none")),
            json!({"function_calling_config": {"mode": "NONE"}})
        );
        assert_eq!(
            convert_tool_choice(&json!({"name": "search"})),
            json!({"function_calling_config": {"mode": "ANY", "allowed_function_names": ["search"]}})
        );
        assert_eq!(
            convert_tool_choice(&json!("search")),
            json!({"function_calling_config": {"mode": "ANY", "allowed_function_names": ["search"]}})
        );
    }

    #[test]
    fn build_config_assembles_and_rejects_dual_thinking() {
        let mut extra = Map::new();
        extra.insert("top_k".to_string(), json!(40));
        let config = build_config(&ConfigParams {
            max_tokens: 1024,
            temperature: Some(0.5),
            stop: None,
            tools: None,
            tool_choice: Some(&json!("auto")),
            thinking_budget_tokens: Some(2048),
            thinking_effort: None,
            extra_params: &extra,
        })
        .unwrap();
        assert_eq!(config["max_output_tokens"], json!(1024));
        assert_eq!(config["temperature"], json!(0.5));
        assert_eq!(
            config["tool_config"]["function_calling_config"]["mode"],
            json!("AUTO")
        );
        assert_eq!(config["thinking_config"], json!({"thinking_budget": 2048}));
        assert_eq!(config["top_k"], json!(40));

        // Both thinking params set -> error, matching Python's ValidationException.
        let err = build_config(&ConfigParams {
            max_tokens: 1024,
            temperature: None,
            stop: None,
            tools: None,
            tool_choice: None,
            thinking_budget_tokens: Some(2048),
            thinking_effort: Some("high"),
            extra_params: &Map::new(),
        });
        assert_eq!(err, Err(ConflictingThinkingParams));
    }

    #[test]
    fn build_config_json_mode_only_without_tools() {
        let mut extra = Map::new();
        extra.insert("json_mode".to_string(), json!(true));
        let config = build_config(&ConfigParams {
            max_tokens: 100,
            temperature: None,
            stop: None,
            tools: None,
            tool_choice: None,
            thinking_budget_tokens: None,
            thinking_effort: None,
            extra_params: &extra,
        })
        .unwrap();
        assert_eq!(config["response_mime_type"], json!("application/json"));
    }

    #[test]
    fn parse_response_collects_parts_and_function_calls() {
        let response = json!({
            "candidates": [{
                "finish_reason": "STOP",
                "content": {
                    "role": "model",
                    "parts": [
                        {"text": "Hello"},
                        {"text": "world"},
                        {
                            "function_call": {"name": "search", "args": {"q": "x"}},
                            "thought_signature": "sig",
                        },
                    ],
                },
            }],
            "usage_metadata": {
                "prompt_token_count": 10,
                "candidates_token_count": 5,
                "cached_content_token_count": 3,
            },
        });
        let result = parse_response(&response);
        assert_eq!(result.content, json!("Hello\nworld"));
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].id, "call_search_0");
        assert_eq!(result.tool_calls[0].name, "search");
        assert_eq!(result.tool_calls[0].input, json!({"q": "x"}));
        assert_eq!(
            result.tool_calls[0].thought_signature.as_deref(),
            Some("sig")
        );
        assert_eq!(result.input_tokens, 10);
        assert_eq!(result.output_tokens, 5);
        assert_eq!(result.cache_read_input_tokens, 3);
        assert_eq!(result.finish_reason, "STOP");
    }

    #[test]
    fn function_call_ids_increment_and_args_default_to_empty() {
        let response = json!({
            "candidates": [{
                "content": {"parts": [
                    {"function_call": {"name": "a"}},
                    {"function_call": {"name": "a", "args": {"k": 1}}},
                ]},
            }],
        });
        let result = parse_response(&response);
        assert_eq!(result.tool_calls.len(), 2);
        assert_eq!(result.tool_calls[0].id, "call_a_0");
        assert_eq!(result.tool_calls[0].input, json!({})); // missing args -> {}
        assert_eq!(result.tool_calls[1].id, "call_a_1");
        assert_eq!(result.tool_calls[1].input, json!({"k": 1}));
        assert_eq!(result.finish_reason, "stop"); // absent -> default
    }

    #[test]
    fn top_level_fallbacks_used_when_candidate_empty() {
        let response = json!({
            "candidates": [{"content": {"parts": []}}],
            "text": "fallback text",
            "function_calls": [{"name": "f", "args": {"a": 1}}],
        });
        let result = parse_response(&response);
        assert_eq!(result.content, json!("fallback text"));
        assert_eq!(result.tool_calls.len(), 1);
        assert_eq!(result.tool_calls[0].id, "call_f_0");
        assert!(result.tool_calls[0].thought_signature.is_none());
    }
}
