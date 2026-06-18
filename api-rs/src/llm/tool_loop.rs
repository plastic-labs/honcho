//! Tool-loop helpers, ported from `src/llm/tool_loop.py`.
//!
//! The full `execute_tool_loop` is async orchestration over live provider calls
//! and isn't portable here. What *is* deterministic — and is what the loop uses
//! to grow the conversation each iteration — are the two history-adapter
//! wrappers below, plus the max-iteration synthesis prompt.

use serde_json::Value;

use super::history_adapters;
use super::{CompletionResult, Provider, ToolCallResult, ToolResult};

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
}
