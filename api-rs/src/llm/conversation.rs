//! Conversation-shaping helpers, ported from `src/llm/conversation.py`.
//!
//! Pure, deterministic: token counting and tool-aware truncation that keeps
//! `tool_use`/`tool_result` pairs intact. Token counting serializes content
//! blocks the way Python's `json.dumps` does (so the o200k_base estimate lines
//! up); `python_json_dumps` matches `json.dumps(sort_keys=True)` — serde's maps
//! are already key-sorted, and the separator/`ensure_ascii` rules drive the
//! token count more than key order does.

use serde_json::Value;

use crate::tokens::estimate_tokens;

/// Count o200k_base tokens across a message list, porting `count_message_tokens`.
/// String `content` is counted directly; list `content` and any `parts` are
/// counted as their JSON serialization.
pub fn count_message_tokens(messages: &[Value]) -> usize {
    let mut total = 0;
    for message in messages {
        match message.get("content") {
            Some(Value::String(content)) => total += estimate_tokens(content),
            Some(content @ Value::Array(_)) => {
                total += estimate_tokens(&python_json_dumps(content));
            }
            _ => {}
        }
        if let Some(parts) = message.get("parts") {
            total += estimate_tokens(&python_json_dumps(parts));
        }
    }
    total
}

/// Whether a message carries tool calls in any provider format, porting
/// `_is_tool_use_message`.
pub fn is_tool_use_message(message: &Value) -> bool {
    if let Some(blocks) = message.get("content").and_then(Value::as_array)
        && blocks
            .iter()
            .any(|block| block.get("type").and_then(Value::as_str) == Some("tool_use"))
    {
        return true;
    }
    if let Some(parts) = message.get("parts").and_then(Value::as_array)
        && parts.iter().any(|part| part.get("function_call").is_some())
    {
        return true;
    }
    match message.get("tool_calls") {
        Some(Value::Array(calls)) => !calls.is_empty(),
        Some(Value::Null) | None => false,
        Some(_) => true,
    }
}

/// Whether a message carries tool results in any provider format, porting
/// `_is_tool_result_message`.
pub fn is_tool_result_message(message: &Value) -> bool {
    if let Some(blocks) = message.get("content").and_then(Value::as_array)
        && blocks
            .iter()
            .any(|block| block.get("type").and_then(Value::as_str) == Some("tool_result"))
    {
        return true;
    }
    if let Some(parts) = message.get("parts").and_then(Value::as_array)
        && parts
            .iter()
            .any(|part| part.get("function_response").is_some())
    {
        return true;
    }
    message.get("role").and_then(Value::as_str) == Some("tool")
}

/// Group messages into truncation units, porting `_group_into_units`. A
/// `tool_use` message plus all consecutive `tool_result` messages form one unit
/// (kept together); orphaned tool_use / tool_result messages are dropped.
fn group_into_units(messages: &[Value]) -> Vec<Vec<Value>> {
    let mut units: Vec<Vec<Value>> = Vec::new();
    let mut i = 0;
    while i < messages.len() {
        if is_tool_use_message(&messages[i]) {
            let mut j = i + 1;
            while j < messages.len() && is_tool_result_message(&messages[j]) {
                j += 1;
            }
            if j - i > 1 {
                units.push(messages[i..j].to_vec());
                i = j;
            } else {
                // Orphaned tool_use with no results — skip it.
                i += 1;
            }
        } else if is_tool_result_message(&messages[i]) {
            // Orphaned tool_result — skip it.
            i += 1;
        } else {
            units.push(vec![messages[i].clone()]);
            i += 1;
        }
    }
    units
}

/// Truncate a message list to fit `max_tokens`, porting `truncate_messages_to_fit`.
/// System messages are preserved (when `preserve_system`); the conversation is
/// grouped into units and the oldest units are dropped until it fits, always
/// keeping at least one unit.
pub fn truncate_messages_to_fit(
    messages: &[Value],
    max_tokens: usize,
    preserve_system: bool,
) -> Vec<Value> {
    if count_message_tokens(messages) <= max_tokens {
        return messages.to_vec();
    }

    let mut system_messages: Vec<Value> = Vec::new();
    let mut conversation: Vec<Value> = Vec::new();
    for message in messages {
        let is_system = message.get("role").and_then(Value::as_str) == Some("system");
        if is_system && preserve_system {
            system_messages.push(message.clone());
        } else {
            conversation.push(message.clone());
        }
    }

    let system_tokens = count_message_tokens(&system_messages);
    if system_tokens >= max_tokens {
        // System messages alone exceed the budget — return the original list.
        return messages.to_vec();
    }
    let available_tokens = max_tokens - system_tokens;

    let mut units = group_into_units(&conversation);
    if units.is_empty() {
        return system_messages;
    }

    // Drop oldest units until the conversation fits, keeping at least one.
    while units.len() > 1 {
        let flat: Vec<Value> = units.iter().flatten().cloned().collect();
        if count_message_tokens(&flat) <= available_tokens {
            break;
        }
        units.remove(0);
    }

    let mut result = system_messages;
    result.extend(units.into_iter().flatten());
    result
}

/// Serialize a [`Value`] the way Python's `json.dumps(x, sort_keys=True)` does:
/// `", "` / `": "` separators and `ensure_ascii` (non-ASCII and DEL escaped as
/// `\uXXXX`, with surrogate pairs above U+FFFF). serde maps are already sorted.
fn python_json_dumps(value: &Value) -> String {
    let mut out = String::new();
    dump_value(value, &mut out);
    out
}

fn dump_value(value: &Value, out: &mut String) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(true) => out.push_str("true"),
        Value::Bool(false) => out.push_str("false"),
        Value::Number(number) => out.push_str(&number.to_string()),
        Value::String(text) => dump_string(text, out),
        Value::Array(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                dump_value(item, out);
            }
            out.push(']');
        }
        Value::Object(entries) => {
            out.push('{');
            for (index, (key, val)) in entries.iter().enumerate() {
                if index > 0 {
                    out.push_str(", ");
                }
                dump_string(key, out);
                out.push_str(": ");
                dump_value(val, out);
            }
            out.push('}');
        }
    }
}

fn dump_string(text: &str, out: &mut String) {
    out.push('"');
    for ch in text.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\u{08}' => out.push_str("\\b"),
            '\u{0c}' => out.push_str("\\f"),
            // Printable ASCII (0x20..=0x7e) passes through verbatim.
            c if ('\u{20}'..='\u{7e}').contains(&c) => out.push(c),
            // Everything else is \uXXXX (surrogate pair above the BMP),
            // matching json.dumps's ensure_ascii.
            c => {
                let code = c as u32;
                if code <= 0xFFFF {
                    out.push_str(&format!("\\u{code:04x}"));
                } else {
                    let v = code - 0x10000;
                    let high = 0xD800 + (v >> 10);
                    let low = 0xDC00 + (v & 0x3FF);
                    out.push_str(&format!("\\u{high:04x}\\u{low:04x}"));
                }
            }
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn python_json_dumps_matches_sort_keys_golden() {
        assert_eq!(
            python_json_dumps(&json!({"type": "text", "text": "hi"})),
            r#"{"text": "hi", "type": "text"}"#
        );
        // ensure_ascii escapes non-ASCII to \uXXXX (golden from json.dumps);
        // written as normal strings so the \u escapes are literal output text.
        assert_eq!(
            python_json_dumps(&json!({"type": "text", "text": "café"})),
            "{\"text\": \"caf\\u00e9\", \"type\": \"text\"}"
        );
        assert_eq!(
            python_json_dumps(&json!(["a", {"x": 1}, true, null])),
            r#"["a", {"x": 1}, true, null]"#
        );
        // Above the BMP -> surrogate pair, matching json.dumps.
        assert_eq!(
            python_json_dumps(&json!("emoji 😀")),
            "\"emoji \\ud83d\\ude00\""
        );
        assert_eq!(
            python_json_dumps(&json!({"a": "x\ty\nz"})),
            r#"{"a": "x\ty\nz"}"#
        );
    }

    #[test]
    fn count_tokens_string_and_block_content() {
        // String content path uses estimate_tokens directly.
        let messages = vec![json!({"role": "user", "content": "hello world"})];
        assert_eq!(count_message_tokens(&messages), 2);
        // Block content is counted via its JSON serialization (nonzero).
        let blocks = vec![json!({
            "role": "assistant",
            "content": [{"type": "text", "text": "hi"}],
        })];
        assert!(count_message_tokens(&blocks) > 0);
    }

    #[test]
    fn tool_use_and_result_detection_all_formats() {
        assert!(is_tool_use_message(
            &json!({"content": [{"type": "tool_use", "id": "t1"}]})
        ));
        assert!(is_tool_use_message(
            &json!({"parts": [{"function_call": {"name": "f"}}]})
        ));
        assert!(is_tool_use_message(&json!({"tool_calls": [{"id": "c1"}]})));
        assert!(!is_tool_use_message(&json!({"tool_calls": []})));
        assert!(!is_tool_use_message(&json!({"content": "plain"})));

        assert!(is_tool_result_message(
            &json!({"content": [{"type": "tool_result", "tool_use_id": "t1"}]})
        ));
        assert!(is_tool_result_message(
            &json!({"parts": [{"function_response": {"name": "f"}}]})
        ));
        assert!(is_tool_result_message(&json!({"role": "tool"})));
        assert!(!is_tool_result_message(&json!({"role": "user"})));
    }

    #[test]
    fn truncation_no_op_when_within_budget() {
        let messages = vec![json!({"role": "user", "content": "hi"})];
        assert_eq!(truncate_messages_to_fit(&messages, 1000, true), messages);
    }

    #[test]
    fn truncation_drops_oldest_units_and_keeps_system() {
        // Many user turns; tiny budget forces dropping oldest, keeping >=1 unit.
        let mut messages = vec![json!({"role": "system", "content": "sys"})];
        for n in 0..20 {
            messages.push(
                json!({"role": "user", "content": format!("message number {n} with some words")}),
            );
        }
        let result = truncate_messages_to_fit(&messages, 30, true);
        // System preserved at the front.
        assert_eq!(result[0]["role"], json!("system"));
        // Truncated below the original length, but not empty.
        assert!(result.len() < messages.len());
        assert!(result.len() >= 2);
        // The kept conversation turns are the most recent ones.
        let last = messages.last().unwrap();
        assert_eq!(result.last().unwrap(), last);
    }

    #[test]
    fn truncation_keeps_tool_pairs_together() {
        // A tool_use + tool_result unit must never be split.
        let messages = vec![
            json!({"role": "user", "content": "old turn one with padding words here"}),
            json!({"role": "user", "content": "old turn two with padding words here"}),
            json!({"content": [{"type": "tool_use", "id": "t1", "name": "f", "input": {}}]}),
            json!({"content": [{"type": "tool_result", "tool_use_id": "t1", "content": "result"}]}),
        ];
        let result = truncate_messages_to_fit(&messages, 25, true);
        // If the tool_use survived, its tool_result must immediately follow.
        let has_tool_use = result.iter().any(is_tool_use_message);
        if has_tool_use {
            let idx = result.iter().position(is_tool_use_message).unwrap();
            assert!(is_tool_result_message(&result[idx + 1]));
        }
    }
}
