//! Deterministic formatting helpers for the dialectic tool layer, ported from
//! `src/utils/agent_tools.py` + `src/utils/formatting.py`.
//!
//! These are the network-free, byte-exact output side of the dialectic tools
//! (the agent loop, tool dispatch, and LLM executor that *call* them are the
//! larger unported phase). All length math is on **Unicode code points** to
//! match Python `str` `len`/slicing semantics.
//!
//! Char-cap defaults are hardcoded constants (Python reads env-overridable
//! `settings.LLM.*`; same simplification as other ported defaults).

use chrono::{DateTime, Utc};
use serde_json::Value;

/// `settings.LLM.MAX_TOOL_OUTPUT_CHARS` default.
pub const MAX_TOOL_OUTPUT_CHARS: usize = 10_000;
/// `settings.LLM.MAX_MESSAGE_CONTENT_CHARS` default.
pub const MAX_MESSAGE_CONTENT_CHARS: usize = 2_000;

/// Port of `_safe_int`: coerce a tool-input JSON value to an integer, returning
/// `default` on failure (LLMs sometimes pass non-numeric strings). Mirrors
/// Python `int(value)`: integers as-is, floats truncated toward zero, booleans
/// to 0/1, plain base-10 integer strings parsed (whitespace-trimmed), everything
/// else (null/array/object/malformed string) → `default`.
pub fn safe_int(value: &Value, default: i64) -> i64 {
    match value {
        Value::Number(number) => number
            .as_i64()
            .or_else(|| number.as_f64().map(|float| float.trunc() as i64))
            .unwrap_or(default),
        Value::Bool(flag) => i64::from(*flag),
        Value::String(text) => text.trim().parse::<i64>().unwrap_or(default),
        _ => default,
    }
}

/// Python `f"{n:,}"` — group digits in threes with commas.
fn comma_grouped(n: usize) -> String {
    let digits = n.to_string();
    let bytes = digits.as_bytes();
    let len = bytes.len();
    let mut out = String::with_capacity(len + len / 3);
    for (index, byte) in bytes.iter().enumerate() {
        if index > 0 && (len - index).is_multiple_of(3) {
            out.push(',');
        }
        out.push(*byte as char);
    }
    out
}

/// Port of `_truncate_tool_output`: clamp `output` to `max_chars` code points,
/// appending a comma-grouped truncation notice. Returns
/// `(text, original_chars, was_truncated)` — the truncation signal the caller
/// threads into the tool-result metadata.
pub fn truncate_tool_output(output: &str, max_chars: usize) -> (String, usize, bool) {
    let chars: Vec<char> = output.chars().collect();
    let original = chars.len();
    if original <= max_chars {
        return (output.to_string(), original, false);
    }
    let head: String = chars[..max_chars].iter().collect();
    let truncated = format!(
        "{head}\n\n[OUTPUT TRUNCATED - showing {} of {} characters]",
        comma_grouped(max_chars),
        comma_grouped(original)
    );
    (truncated, original, true)
}

/// Port of `_truncate_message_content`: simple head truncation of a single
/// message's content to `max_chars` code points (appending `...` when clamped).
pub fn truncate_message_content(content: &str, max_chars: usize) -> String {
    let chars: Vec<char> = content.chars().collect();
    if chars.len() <= max_chars {
        return content.to_string();
    }
    let head: String = chars[..max_chars].iter().collect();
    format!("{head}...")
}

/// First char of a char's lowercase expansion, keeping a 1:1 code-point mapping
/// so match indices stay aligned. Exact for ASCII (the realistic grep case);
/// approximates Python `re.IGNORECASE` only for the rare chars whose case fold
/// expands to multiple code points.
fn lower1(c: char) -> char {
    c.to_lowercase().next().unwrap_or(c)
}

/// Case-insensitive literal search over a code-point slice. Returns the match's
/// `(start_index, char_len)`; a literal pattern matches `pattern.chars().count()`
/// code points.
fn find_case_insensitive(haystack: &[char], pattern: &str) -> Option<(usize, usize)> {
    let needle: Vec<char> = pattern.chars().map(lower1).collect();
    if needle.is_empty() {
        return Some((0, 0));
    }
    if needle.len() > haystack.len() {
        return None;
    }
    let hay: Vec<char> = haystack.iter().map(|&c| lower1(c)).collect();
    (0..=hay.len() - needle.len()).find_map(|start| {
        if hay[start..start + needle.len()] == needle[..] {
            Some((start, needle.len()))
        } else {
            None
        }
    })
}

/// Port of `_extract_pattern_snippet`: for grep/exact-text results, extract a
/// `max_chars`-code-point window centered on the first case-insensitive match
/// of `pattern` (literal), with `...` prefix/suffix when the window doesn't
/// reach the content edges. Content within the cap is returned unchanged; no
/// match → the head of the content. Window math mirrors Python's floor
/// division and boundary adjustment exactly.
pub fn extract_pattern_snippet(content: &str, pattern: &str, max_chars: usize) -> String {
    let chars: Vec<char> = content.chars().collect();
    let n = chars.len() as i64;
    if chars.len() <= max_chars {
        return content.to_string();
    }

    let Some((match_start, match_len)) = find_case_insensitive(&chars, pattern) else {
        let head: String = chars[..max_chars].iter().collect();
        return format!("{head}...");
    };

    let max = max_chars as i64;
    let match_start = match_start as i64;
    let match_end = match_start + match_len as i64;
    let remaining = max - match_len as i64;
    let before = remaining.div_euclid(2);
    let after = remaining - before;

    let mut start = (match_start - before).max(0);
    let mut end = (match_end + after).min(n);
    if start == 0 {
        end = n.min(max);
    } else if end == n {
        start = (n - max).max(0);
    }

    let snippet: String = chars[start as usize..end as usize].iter().collect();
    let prefix = if start > 0 { "..." } else { "" };
    let suffix = if end < n { "..." } else { "" };
    format!("{prefix}{snippet}{suffix}")
}

/// Port of `format_new_turn_with_timestamp`: `"YYYY-MM-DD HH:MM:SS speaker: text"`.
/// The timestamp is rendered in UTC (matching the timezone-aware `created_at`
/// values stored in the DB; Python's `strftime` prints the datetime's own
/// components without conversion).
pub fn format_new_turn_with_timestamp(
    new_turn: &str,
    current_time: DateTime<Utc>,
    speaker: &str,
) -> String {
    format!(
        "{} {speaker}: {new_turn}",
        current_time.format("%Y-%m-%d %H:%M:%S")
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;
    use serde_json::json;

    #[test]
    fn safe_int_coerces_like_python() {
        assert_eq!(safe_int(&json!(7), 10), 7);
        assert_eq!(safe_int(&json!(3.9), 10), 3); // truncate toward zero
        assert_eq!(safe_int(&json!(-3.9), 10), -3);
        assert_eq!(safe_int(&json!("5"), 10), 5);
        assert_eq!(safe_int(&json!("  12 "), 10), 12);
        assert_eq!(safe_int(&json!("Infinity"), 10), 10); // ValueError -> default
        assert_eq!(safe_int(&json!("5.0"), 10), 10); // int("5.0") raises -> default
        assert_eq!(safe_int(&json!(true), 10), 1);
        assert_eq!(safe_int(&json!(false), 10), 0);
        assert_eq!(safe_int(&Value::Null, 10), 10);
        assert_eq!(safe_int(&json!([1]), 10), 10);
    }

    #[test]
    fn comma_grouping_matches_python_format() {
        assert_eq!(comma_grouped(0), "0");
        assert_eq!(comma_grouped(999), "999");
        assert_eq!(comma_grouped(1_000), "1,000");
        assert_eq!(comma_grouped(10_000), "10,000");
        assert_eq!(comma_grouped(1_234_567), "1,234,567");
    }

    #[test]
    fn truncate_tool_output_clamps_with_notice() {
        let (text, original, was) = truncate_tool_output("hello", 100);
        assert_eq!(text, "hello");
        assert_eq!(original, 5);
        assert!(!was);

        let long = "x".repeat(12);
        let (text, original, was) = truncate_tool_output(&long, 10);
        assert!(was);
        assert_eq!(original, 12);
        assert_eq!(
            text,
            "xxxxxxxxxx\n\n[OUTPUT TRUNCATED - showing 10 of 12 characters]"
        );
    }

    #[test]
    fn truncate_tool_output_counts_code_points() {
        // 6 multibyte chars, cap 4 -> head of 4 code points.
        let (text, original, was) = truncate_tool_output("☃☃☃☃☃☃", 4);
        assert!(was);
        assert_eq!(original, 6);
        assert!(text.starts_with("☃☃☃☃\n\n[OUTPUT TRUNCATED"));
    }

    #[test]
    fn truncate_message_content_appends_ellipsis() {
        assert_eq!(truncate_message_content("short", 100), "short");
        assert_eq!(truncate_message_content("abcdef", 3), "abc...");
        assert_eq!(truncate_message_content("café", 4), "café");
    }

    #[test]
    fn extract_pattern_snippet_returns_short_content_unchanged() {
        assert_eq!(extract_pattern_snippet("hello world", "world", 100), "hello world");
    }

    #[test]
    fn extract_pattern_snippet_windows_around_match() {
        // 40-char content, pattern in the middle, cap 10 -> window with both ...
        let content = "0123456789ABCDEFGHIJfindKLMNOPQRSTUVWXYZ"; // len 39
        let snippet = extract_pattern_snippet(content, "find", 10);
        assert!(snippet.starts_with("..."));
        assert!(snippet.ends_with("..."));
        assert!(snippet.contains("find"));
        // 10-char window + both "..." markers.
        assert_eq!(snippet.chars().count(), 10 + 6);
    }

    #[test]
    fn extract_pattern_snippet_no_match_returns_head() {
        let content = "a".repeat(50);
        let snippet = extract_pattern_snippet(&content, "zzz", 10);
        assert_eq!(snippet, format!("{}...", "a".repeat(10)));
    }

    #[test]
    fn extract_pattern_snippet_match_at_start_no_prefix() {
        let content = format!("find{}", "x".repeat(50));
        let snippet = extract_pattern_snippet(&content, "FIND", 10);
        assert!(!snippet.starts_with("..."));
        assert!(snippet.ends_with("..."));
        assert!(snippet.to_lowercase().starts_with("find"));
    }

    #[test]
    fn format_new_turn_with_timestamp_golden() {
        let time = Utc.with_ymd_and_hms(2023, 5, 8, 13, 56, 0).unwrap();
        assert_eq!(
            format_new_turn_with_timestamp("hello", time, "alice"),
            "2023-05-08 13:56:00 alice: hello"
        );
    }
}
