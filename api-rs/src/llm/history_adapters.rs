//! Provider history adapters, ported from `src/llm/history_adapters.py`.
//!
//! Each adapter turns an assistant [`CompletionResult`] into the provider's
//! assistant-message JSON and a list of [`ToolResult`]s into the provider's
//! tool-result messages. Pure data → data; no network.

use serde_json::{Map, Value, json};

use super::{CompletionResult, Provider, ToolResult};

/// Shared behavior of the per-provider history adapters
/// (`HistoryAdapter` Protocol in Python).
pub trait HistoryAdapter {
    /// Format the assistant turn (text + tool calls + provider reasoning blocks)
    /// as a single provider message.
    fn format_assistant_tool_message(&self, result: &CompletionResult) -> Value;

    /// Format executed tool results as the provider's follow-up message(s).
    fn format_tool_results(&self, tool_results: &[ToolResult]) -> Vec<Value>;
}

/// Select the provider-appropriate adapter (`history_adapter_for_provider`).
/// Anthropic and Gemini get their own; everything else uses OpenAI's shape.
pub fn for_provider(provider: Provider) -> Box<dyn HistoryAdapter> {
    match provider {
        Provider::Anthropic => Box::new(AnthropicHistoryAdapter),
        Provider::Gemini => Box::new(GeminiHistoryAdapter),
        Provider::Openai => Box::new(OpenAIHistoryAdapter),
    }
}

/// `result.content` as a non-empty string, mirroring Python's
/// `isinstance(result.content, str) and result.content`.
fn nonempty_content(result: &CompletionResult) -> Option<&str> {
    match &result.content {
        Value::String(text) if !text.is_empty() => Some(text),
        _ => None,
    }
}

pub struct AnthropicHistoryAdapter;

impl HistoryAdapter for AnthropicHistoryAdapter {
    fn format_assistant_tool_message(&self, result: &CompletionResult) -> Value {
        let mut content_blocks: Vec<Value> = Vec::new();
        content_blocks.extend(result.thinking_blocks.iter().cloned());
        if let Some(text) = nonempty_content(result) {
            content_blocks.push(json!({"type": "text", "text": text}));
        }
        for tool_call in &result.tool_calls {
            content_blocks.push(json!({
                "type": "tool_use",
                "id": tool_call.id,
                "name": tool_call.name,
                "input": tool_call.input,
            }));
        }
        json!({"role": "assistant", "content": content_blocks})
    }

    fn format_tool_results(&self, tool_results: &[ToolResult]) -> Vec<Value> {
        let content = tool_results
            .iter()
            .map(|tr| {
                json!({
                    "type": "tool_result",
                    "tool_use_id": tr.tool_id,
                    "content": tr.result,
                    "is_error": tr.is_error,
                })
            })
            .collect::<Vec<_>>();
        vec![json!({"role": "user", "content": content})]
    }
}

pub struct GeminiHistoryAdapter;

impl HistoryAdapter for GeminiHistoryAdapter {
    fn format_assistant_tool_message(&self, result: &CompletionResult) -> Value {
        let mut parts: Vec<Value> = Vec::new();
        if let Some(text) = nonempty_content(result) {
            parts.push(json!({"text": text}));
        }
        for tool_call in &result.tool_calls {
            let mut part = Map::new();
            part.insert(
                "function_call".to_string(),
                json!({"name": tool_call.name, "args": tool_call.input}),
            );
            if let Some(signature) = &tool_call.thought_signature {
                part.insert(
                    "thought_signature".to_string(),
                    Value::String(signature.clone()),
                );
            }
            parts.push(Value::Object(part));
        }
        json!({"role": "model", "parts": parts})
    }

    fn format_tool_results(&self, tool_results: &[ToolResult]) -> Vec<Value> {
        let parts = tool_results
            .iter()
            .map(|tr| {
                json!({
                    "function_response": {
                        "name": tr.tool_name,
                        "response": {"result": tr.result},
                    }
                })
            })
            .collect::<Vec<_>>();
        vec![json!({"role": "user", "parts": parts})]
    }
}

pub struct OpenAIHistoryAdapter;

impl HistoryAdapter for OpenAIHistoryAdapter {
    fn format_assistant_tool_message(&self, result: &CompletionResult) -> Value {
        // Python keeps a string `content` as-is (including ""), else null.
        let content = match &result.content {
            Value::String(text) => Value::String(text.clone()),
            _ => Value::Null,
        };
        let tool_calls = result
            .tool_calls
            .iter()
            .map(|tool_call| {
                json!({
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.name,
                        // Python uses json.dumps(input); compact serialization is
                        // semantically identical (the provider re-parses it).
                        "arguments": serde_json::to_string(&tool_call.input)
                            .unwrap_or_else(|_| "{}".to_string()),
                    },
                })
            })
            .collect::<Vec<_>>();
        let mut message = json!({
            "role": "assistant",
            "content": content,
            "tool_calls": tool_calls,
        });
        if !result.reasoning_details.is_empty() {
            message["reasoning_details"] = Value::Array(result.reasoning_details.clone());
        }
        message
    }

    fn format_tool_results(&self, tool_results: &[ToolResult]) -> Vec<Value> {
        tool_results
            .iter()
            .map(|tr| {
                json!({
                    "role": "tool",
                    "tool_call_id": tr.tool_id,
                    "content": tr.result,
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool_call(id: &str, name: &str, input: Value) -> ToolCallResultStub {
        ToolCallResultStub {
            id: id.to_string(),
            name: name.to_string(),
            input,
            thought_signature: None,
        }
    }

    // Local alias so the test reads naturally.
    use super::super::ToolCallResult as ToolCallResultStub;

    fn result_with(content: Value, tool_calls: Vec<ToolCallResultStub>) -> CompletionResult {
        CompletionResult {
            content,
            tool_calls,
            ..CompletionResult::default()
        }
    }

    #[test]
    fn anthropic_assistant_message_orders_thinking_text_tooluse() {
        let mut result = result_with(
            Value::String("hi".to_string()),
            vec![tool_call("t1", "search", json!({"q": "x"}))],
        );
        result.thinking_blocks = vec![json!({"type": "thinking", "thinking": "hmm"})];
        let message = AnthropicHistoryAdapter.format_assistant_tool_message(&result);
        assert_eq!(
            message,
            json!({
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "hmm"},
                    {"type": "text", "text": "hi"},
                    {"type": "tool_use", "id": "t1", "name": "search", "input": {"q": "x"}},
                ]
            })
        );
    }

    #[test]
    fn anthropic_empty_content_omits_text_block() {
        let result = result_with(Value::String(String::new()), vec![]);
        let message = AnthropicHistoryAdapter.format_assistant_tool_message(&result);
        assert_eq!(message, json!({"role": "assistant", "content": []}));
    }

    #[test]
    fn anthropic_tool_results_wrap_in_single_user_message() {
        let results = vec![ToolResult {
            tool_id: "t1".to_string(),
            tool_name: "search".to_string(),
            result: "found it".to_string(),
            is_error: false,
        }];
        assert_eq!(
            AnthropicHistoryAdapter.format_tool_results(&results),
            vec![json!({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": "t1",
                    "content": "found it",
                    "is_error": false,
                }]
            })]
        );
    }

    #[test]
    fn gemini_assistant_message_includes_thought_signature() {
        let mut call = tool_call("t1", "search", json!({"q": "x"}));
        call.thought_signature = Some("sig".to_string());
        let result = result_with(Value::String("hi".to_string()), vec![call]);
        let message = GeminiHistoryAdapter.format_assistant_tool_message(&result);
        assert_eq!(
            message,
            json!({
                "role": "model",
                "parts": [
                    {"text": "hi"},
                    {
                        "function_call": {"name": "search", "args": {"q": "x"}},
                        "thought_signature": "sig",
                    },
                ]
            })
        );
    }

    #[test]
    fn gemini_tool_results_use_function_response() {
        let results = vec![ToolResult {
            tool_id: "ignored".to_string(),
            tool_name: "search".to_string(),
            result: "42".to_string(),
            is_error: false,
        }];
        assert_eq!(
            GeminiHistoryAdapter.format_tool_results(&results),
            vec![json!({
                "role": "user",
                "parts": [{"function_response": {"name": "search", "response": {"result": "42"}}}]
            })]
        );
    }

    #[test]
    fn openai_assistant_message_serializes_arguments_and_keeps_empty_content() {
        let result = result_with(
            Value::String(String::new()),
            vec![tool_call("t1", "search", json!({"q": "x"}))],
        );
        let message = OpenAIHistoryAdapter.format_assistant_tool_message(&result);
        assert_eq!(message["role"], json!("assistant"));
        assert_eq!(message["content"], json!("")); // empty string preserved
        assert_eq!(message["tool_calls"][0]["id"], json!("t1"));
        assert_eq!(message["tool_calls"][0]["type"], json!("function"));
        assert_eq!(
            message["tool_calls"][0]["function"]["name"],
            json!("search")
        );
        // arguments is a JSON string; compare structurally (separator-agnostic).
        let arguments = message["tool_calls"][0]["function"]["arguments"]
            .as_str()
            .unwrap();
        assert_eq!(
            serde_json::from_str::<Value>(arguments).unwrap(),
            json!({"q": "x"})
        );
        assert!(message.get("reasoning_details").is_none());
    }

    #[test]
    fn openai_non_string_content_becomes_null() {
        let result = result_with(Value::Null, vec![]);
        let message = OpenAIHistoryAdapter.format_assistant_tool_message(&result);
        assert_eq!(message["content"], Value::Null);
    }

    #[test]
    fn openai_tool_results_are_tool_role_messages() {
        let results = vec![
            ToolResult {
                tool_id: "t1".to_string(),
                tool_name: "a".to_string(),
                result: "r1".to_string(),
                is_error: false,
            },
            ToolResult {
                tool_id: "t2".to_string(),
                tool_name: "b".to_string(),
                result: "r2".to_string(),
                is_error: true,
            },
        ];
        assert_eq!(
            OpenAIHistoryAdapter.format_tool_results(&results),
            vec![
                json!({"role": "tool", "tool_call_id": "t1", "content": "r1"}),
                json!({"role": "tool", "tool_call_id": "t2", "content": "r2"}),
            ]
        );
    }

    #[test]
    fn for_provider_selects_expected_adapter_shapes() {
        let result = result_with(Value::String("x".to_string()), vec![]);
        assert_eq!(
            for_provider(Provider::Anthropic).format_assistant_tool_message(&result)["role"],
            json!("assistant")
        );
        assert_eq!(
            for_provider(Provider::Gemini).format_assistant_tool_message(&result)["role"],
            json!("model")
        );
        assert_eq!(
            for_provider(Provider::Openai).format_assistant_tool_message(&result)["role"],
            json!("assistant")
        );
    }
}
