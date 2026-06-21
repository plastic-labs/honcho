//! Port of the `json_repair` library (no-schema path only).
//!
//! Honcho calls `repair_json(json_str)` with no schema, no file descriptor, and
//! all flags default — so only `JSONParser.parse()` (objects/arrays/strings/
//! numbers/literals/comments/parenthesized) is reachable; the ~1000-line schema
//! machinery (`schema_repair.py`, `parser_schema.py`, etc.) is never exercised
//! and is intentionally out of scope.
//!
//! Port status: foundation in place (`JsonContext`, `ObjectComparer`). The
//! recursive parser itself (`JSONParser` + `parse_string`/`parse_object`/
//! `parse_array`/…) is being ported as a dedicated multi-step effort, validated
//! against the real Python library used as a parity oracle. Until that lands,
//! `comprehensive_json_repair` (see [`crate::json_parser`]) remains the only
//! wired repair path.

pub mod array;
pub mod boolean_null;
pub mod comment;
pub mod context;
pub mod llm_block;
pub mod number;
pub mod object;
pub mod object_comparer;
pub mod parenthesized;
pub mod parser;
pub mod string;
pub mod string_helpers;

use serde_json::Value;

/// Port of `STRING_DELIMITERS` (`utils/constants.py`).
pub const STRING_DELIMITERS: [char; 4] = ['"', '\'', '\u{201c}', '\u{201d}'];

/// Python truthiness for a parsed JSON value (used by `_parse_top_level`):
/// `None`/`""`/`[]`/`{}`/`0`/`0.0`/`false` are falsy.
pub fn is_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i != 0
            } else if let Some(u) = n.as_u64() {
                u != 0
            } else if let Some(f) = n.as_f64() {
                f != 0.0
            } else {
                true
            }
        }
    }
}

/// Port of `repair_json(json_str)` returning the parsed value (Python
/// `return_objects=True`). Mirrors the `json.loads` fast path, then the parser.
pub fn repair_json_value(json_str: &str) -> Value {
    if let Ok(value) = serde_json::from_str::<Value>(json_str) {
        return value;
    }
    let mut parser = parser::JsonParser::new(json_str);
    parser.parse()
}

#[cfg(test)]
mod oracle_tests {
    use super::*;

    /// Input/expected-output pairs captured from the real `json_repair` library
    /// (`repair_json(input, return_objects=True)`, JSON-serialized). Comparison
    /// is parsed-value equivalence, which is what the downstream consumer relies
    /// on (it re-parses the repaired string).
    const BATTERY: &str = include_str!("battery.json");

    #[test]
    fn repair_json_value_matches_python_oracle() {
        let cases: Vec<serde_json::Value> = serde_json::from_str(BATTERY).unwrap();
        let mut failures = Vec::new();
        for case in &cases {
            let input = case["in"].as_str().unwrap();
            // Skip any oracle entries that errored in Python (none expected).
            let Some(expected_str) = case.get("out").and_then(|v| v.as_str()) else {
                continue;
            };
            let expected: Value = serde_json::from_str(expected_str).unwrap();
            let got = repair_json_value(input);
            if got != expected {
                failures.push(format!(
                    "input={input:?}\n  expected={expected}\n  got     ={got}"
                ));
            }
        }
        assert!(
            failures.is_empty(),
            "{} / {} oracle mismatches:\n{}",
            failures.len(),
            cases.len(),
            failures.join("\n")
        );
    }
}
