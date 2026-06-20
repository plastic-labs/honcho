//! Public LLM call result types, ported from `src/llm/types.py`.
//!
//! [`HonchoLLMCallResponse`] is the response object returned by the executor and
//! the public `honcho_llm_call` API (distinct from the tool loop's internal
//! [`crate::llm::tool_loop::ToolLoopResponse`]). This module also ports the
//! deterministic backend-result → response converters from `executor.py`
//! (`completion_result_to_response`, `_tool_call_result_to_dict`).

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use super::{CompletionResult, ToolCallResult};

/// Port of `HonchoLLMCallResponse`. `content` is `Any` in Python (a string for
/// text completions, a validated model for structured output); we keep it as a
/// [`Value`] like the rest of the deterministic layer. The token/`iterations`/
/// `hit_input_token_cap` fields default to match the Pydantic defaults.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HonchoLLMCallResponse {
    pub content: Value,
    #[serde(default)]
    pub input_tokens: i64,
    pub output_tokens: i64,
    #[serde(default)]
    pub cache_creation_input_tokens: i64,
    #[serde(default)]
    pub cache_read_input_tokens: i64,
    pub finish_reasons: Vec<String>,
    #[serde(default)]
    pub tool_calls_made: Vec<Value>,
    #[serde(default)]
    pub iterations: i64,
    #[serde(default)]
    pub thinking_content: Option<String>,
    #[serde(default)]
    pub thinking_blocks: Vec<Value>,
    #[serde(default)]
    pub reasoning_details: Vec<Value>,
    #[serde(default)]
    pub hit_input_token_cap: bool,
}

/// Port of `_tool_call_result_to_dict`: a `{id, name, input}` object, with
/// `thought_signature` added only when present (Gemini carries it; the other
/// providers leave it `None`).
pub fn tool_call_result_to_dict(tool_call: &ToolCallResult) -> Value {
    let mut object = Map::new();
    object.insert("id".to_string(), Value::String(tool_call.id.clone()));
    object.insert("name".to_string(), Value::String(tool_call.name.clone()));
    object.insert("input".to_string(), tool_call.input.clone());
    if let Some(signature) = &tool_call.thought_signature {
        object.insert(
            "thought_signature".to_string(),
            Value::String(signature.clone()),
        );
    }
    Value::Object(object)
}

/// Port of `completion_result_to_response`. `finish_reasons` is the single-element
/// list `[finish_reason]` (empty when the backend reported no reason — Python's
/// `[fr] if fr else []` truthiness check, where an empty string is falsy).
/// `iterations` and `hit_input_token_cap` keep their defaults here (the tool loop
/// fills them in on the path that has them).
pub fn completion_result_to_response(result: &CompletionResult) -> HonchoLLMCallResponse {
    HonchoLLMCallResponse {
        content: result.content.clone(),
        input_tokens: result.input_tokens,
        output_tokens: result.output_tokens,
        cache_creation_input_tokens: result.cache_creation_input_tokens,
        cache_read_input_tokens: result.cache_read_input_tokens,
        finish_reasons: if result.finish_reason.is_empty() {
            Vec::new()
        } else {
            vec![result.finish_reason.clone()]
        },
        tool_calls_made: result.tool_calls.iter().map(tool_call_result_to_dict).collect(),
        iterations: 0,
        thinking_content: result.thinking_content.clone(),
        thinking_blocks: result.thinking_blocks.clone(),
        reasoning_details: result.reasoning_details.clone(),
        hit_input_token_cap: false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn tool_call_dict_omits_absent_thought_signature() {
        let call = ToolCallResult {
            id: "call_1".to_string(),
            name: "search_memory".to_string(),
            input: json!({"query": "x"}),
            thought_signature: None,
        };
        assert_eq!(
            tool_call_result_to_dict(&call),
            json!({"id": "call_1", "name": "search_memory", "input": {"query": "x"}})
        );
    }

    #[test]
    fn tool_call_dict_includes_thought_signature_when_present() {
        let call = ToolCallResult {
            id: "call_2".to_string(),
            name: "t".to_string(),
            input: json!({}),
            thought_signature: Some("sig".to_string()),
        };
        assert_eq!(
            tool_call_result_to_dict(&call),
            json!({"id": "call_2", "name": "t", "input": {}, "thought_signature": "sig"})
        );
    }

    #[test]
    fn completion_to_response_maps_fields_and_finish_reason() {
        let result = CompletionResult {
            content: json!("hello"),
            input_tokens: 10,
            output_tokens: 5,
            cache_creation_input_tokens: 2,
            cache_read_input_tokens: 1,
            finish_reason: "stop".to_string(),
            tool_calls: vec![ToolCallResult {
                id: "c".to_string(),
                name: "n".to_string(),
                input: json!({"a": 1}),
                thought_signature: None,
            }],
            thinking_content: Some("thinking".to_string()),
            thinking_blocks: vec![json!({"b": 1})],
            reasoning_details: vec![json!({"r": 2})],
        };

        let response = completion_result_to_response(&result);
        assert_eq!(response.content, json!("hello"));
        assert_eq!(response.input_tokens, 10);
        assert_eq!(response.output_tokens, 5);
        assert_eq!(response.cache_creation_input_tokens, 2);
        assert_eq!(response.cache_read_input_tokens, 1);
        assert_eq!(response.finish_reasons, vec!["stop".to_string()]);
        assert_eq!(response.tool_calls_made.len(), 1);
        assert_eq!(response.tool_calls_made[0]["name"], json!("n"));
        assert_eq!(response.thinking_content.as_deref(), Some("thinking"));
        assert_eq!(response.thinking_blocks, vec![json!({"b": 1})]);
        assert_eq!(response.reasoning_details, vec![json!({"r": 2})]);
        assert_eq!(response.iterations, 0);
        assert!(!response.hit_input_token_cap);
    }

    #[test]
    fn completion_to_response_empty_finish_reason_yields_empty_list() {
        let result = CompletionResult {
            finish_reason: String::new(),
            ..CompletionResult::default()
        };
        let response = completion_result_to_response(&result);
        assert!(response.finish_reasons.is_empty());
    }
}
