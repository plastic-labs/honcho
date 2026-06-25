//! Faithful port of `src/utils/json_parser.py` (the honcho-authored repair
//! strategies). This module ports `comprehensive_json_repair` and its full
//! helper tree exactly; the top-level `validate_and_repair_json` (which first
//! delegates to the external `json_repair` library) is ported separately.
//!
//! Parity notes:
//! - JSON validation uses `serde_json` in place of Python `json.loads`. The two
//!   agree on the inputs honcho cares about; the one known divergence is that
//!   Python's `json.loads` accepts the non-standard literals `NaN`, `Infinity`,
//!   and `-Infinity`, which `serde_json` rejects. Such literals do not appear in
//!   structured-output JSON, so this does not affect the repair path in practice.
//! - Character indexing mirrors Python's code-point semantics by operating on
//!   `Vec<char>`. `char.isspace()`/`char.isdigit()`/regex `\s` are approximated
//!   with `char::is_whitespace`/`char::is_ascii_digit`, which match Python on all
//!   ASCII (and common Unicode) inputs that occur in JSON.

use std::sync::LazyLock;

use regex::Regex;

/// `json.loads(s)` succeeds — used to validate repair attempts.
fn is_valid_json(s: &str) -> bool {
    serde_json::from_str::<serde_json::Value>(s).is_ok()
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum TokType {
    ObjectStart,
    ObjectEnd,
    ArrayStart,
    ArrayEnd,
    Comma,
    Colon,
    Str,
    Number,
    Boolean,
    Null,
}

struct Token {
    ttype: TokType,
    /// The structural character (`}`, `]`, …) for end tokens; only the first
    /// char is compared downstream, matching the Python `token["value"]`.
    value: String,
}

/// Port of `tokenize_json`: tokenize a JSON string into meaningful components.
fn tokenize_json(s: &[char]) -> Vec<Token> {
    let mut tokens: Vec<Token> = Vec::new();
    let n = s.len();
    let mut i = 0;

    while i < n {
        let ch = s[i];

        // Skip whitespace
        if ch.is_whitespace() {
            i += 1;
            continue;
        }

        // String literals
        if ch == '"' {
            let start = i;
            i += 1;
            while i < n {
                if s[i] == '"' && s[i - 1] != '\\' {
                    break;
                }
                i += 1;
            }
            // Python slice json_str[start:i+1] clamps past the end.
            let end = (i + 1).min(n);
            let value: String = s[start..end].iter().collect();
            tokens.push(Token {
                ttype: TokType::Str,
                value,
            });
        } else if ch.is_ascii_digit() || ch == '-' {
            // Numbers
            let start = i;
            while i < n && (s[i].is_ascii_digit() || matches!(s[i], '.' | '-' | 'e' | 'E')) {
                i += 1;
            }
            let value: String = s[start..i].iter().collect();
            tokens.push(Token {
                ttype: TokType::Number,
                value,
            });
            continue; // Don't increment i again
        } else if matches!(ch, '{' | '}' | '[' | ']' | ',' | ':') {
            // Structural characters
            let (ttype, value) = match ch {
                '{' => (TokType::ObjectStart, "{"),
                '}' => (TokType::ObjectEnd, "}"),
                '[' => (TokType::ArrayStart, "["),
                ']' => (TokType::ArrayEnd, "]"),
                ',' => (TokType::Comma, ","),
                ':' => (TokType::Colon, ":"),
                _ => unreachable!(),
            };
            tokens.push(Token {
                ttype,
                value: value.to_string(),
            });
        } else if matches!(ch, 't' | 'f' | 'n') {
            // Boolean/null literals
            let slice4: String = s[i..(i + 4).min(n)].iter().collect();
            let slice5: String = s[i..(i + 5).min(n)].iter().collect();
            if slice4 == "true" {
                tokens.push(Token {
                    ttype: TokType::Boolean,
                    value: "true".to_string(),
                });
                i += 3;
            } else if slice5 == "false" {
                tokens.push(Token {
                    ttype: TokType::Boolean,
                    value: "false".to_string(),
                });
                i += 4;
            } else if slice4 == "null" {
                tokens.push(Token {
                    ttype: TokType::Null,
                    value: "null".to_string(),
                });
                i += 3;
            }
        }

        i += 1;
    }

    tokens
}

/// Port of `generate_closure_attempts`: different ways to close the structure.
fn generate_closure_attempts(partial_json: &str) -> Vec<String> {
    let chars: Vec<char> = partial_json.chars().collect();

    let mut stack: Vec<char> = Vec::new();
    let mut in_string = false;
    let mut escape_next = false;

    for &ch in &chars {
        if escape_next {
            escape_next = false;
            continue;
        }
        if ch == '\\' {
            escape_next = true;
            continue;
        }
        if ch == '"' {
            in_string = !in_string;
            continue;
        }
        if in_string {
            continue;
        }
        if ch == '(' || ch == '{' || ch == '[' {
            stack.push(ch);
        } else if (ch == ')' || ch == '}' || ch == ']') && !stack.is_empty() {
            let opener = stack.pop().unwrap();
            let matched = (ch == ')' && opener == '(')
                || (ch == '}' && opener == '{')
                || (ch == ']' && opener == '[');
            if !matched {
                // Mismatched - this is likely where corruption started
                break;
            }
        }
    }

    let mut attempts: Vec<String> = Vec::new();

    // base = partial_json.rstrip()
    let mut base = partial_json.trim_end().to_string();

    // Remove trailing comma if present
    if base.ends_with(',') {
        base.pop();
        attempts.push(base.clone());
    }

    // Close based on stack (reversed)
    let mut closures: Vec<char> = Vec::new();
    for &opener in stack.iter().rev() {
        match opener {
            '{' => closures.push('}'),
            '[' => closures.push(']'),
            '(' => closures.push(')'),
            _ => {}
        }
    }
    let closures_str: String = closures.iter().collect();

    // Try different combinations
    attempts.push(format!("{base}{closures_str}"));

    // Try closing just objects/arrays (ignore parentheses)
    let obj_closures: String = closures.iter().filter(|c| **c == ']' || **c == '}').collect();
    attempts.push(format!("{base}{obj_closures}"));

    // Try adding missing quotes if we're in a string
    if in_string {
        attempts.push(format!("{base}\"{closures_str}"));
    }

    attempts
}

/// Port of `try_partial_parse_repair`.
fn try_partial_parse_repair(json_str: &str) -> Option<String> {
    let lines: Vec<&str> = json_str.split('\n').collect();

    for i in (1..=lines.len()).rev() {
        let partial = lines[..i].join("\n");
        for attempt in generate_closure_attempts(&partial) {
            if is_valid_json(&attempt) {
                return Some(attempt);
            }
        }
    }

    None
}

/// Port of `try_close_after_value`.
fn try_close_after_value(json_str: &str, tokens: &[Token]) -> Option<String> {
    let mut nesting_stack: Vec<char> = Vec::new();

    // Exclude the last token (which is the value)
    let upto = &tokens[..tokens.len().saturating_sub(1)];
    for token in upto {
        match token.ttype {
            TokType::ObjectStart => nesting_stack.push('}'),
            TokType::ArrayStart => nesting_stack.push(']'),
            TokType::ObjectEnd | TokType::ArrayEnd => {
                if let Some(&top) = nesting_stack.last() {
                    // `token.value` is the single structural char (`}`/`]`).
                    if token.value.starts_with(top) {
                        nesting_stack.pop();
                    }
                }
            }
            _ => {}
        }
    }

    let closure: String = nesting_stack.iter().rev().collect();
    let candidate = format!("{json_str}{closure}");

    if is_valid_json(&candidate) {
        Some(candidate)
    } else {
        None
    }
}

/// Port of `try_complete_structure`.
fn try_complete_structure(json_str: &str, tokens: &[Token]) -> Option<String> {
    let last_token = tokens.last()?;

    match last_token.ttype {
        TokType::Comma => {
            // After comma, try removing the trailing comma first.
            let trimmed = json_str.trim_end().trim_end_matches(',');
            try_contextual_closure_repair(trimmed)
        }
        TokType::Colon => {
            // After colon, we're missing a value - try adding a placeholder.
            for suffix in ["null", "\"\"", "[]", "{}"] {
                let candidate = format!("{json_str}{suffix}");
                if let Some(repaired) = try_contextual_closure_repair(&candidate) {
                    return Some(repaired);
                }
            }
            None
        }
        _ => None,
    }
}

/// Port of `try_contextual_closure_repair`.
fn try_contextual_closure_repair(json_str: &str) -> Option<String> {
    let chars: Vec<char> = json_str.chars().collect();
    let tokens = tokenize_json(&chars);

    if tokens.is_empty() {
        return None;
    }

    let last_token = tokens.last().unwrap();

    match last_token.ttype {
        TokType::Str | TokType::Number | TokType::Boolean | TokType::Null => {
            try_close_after_value(json_str, &tokens)
        }
        TokType::Comma | TokType::Colon => try_complete_structure(json_str, &tokens),
        _ => None,
    }
}

/// Port of `try_line_reconstruction_repair`.
fn try_line_reconstruction_repair(json_str: &str) -> Option<String> {
    let lines: Vec<&str> = json_str.split('\n').collect();

    for i in (1..=lines.len()).rev() {
        let partial = lines[..i].join("\n");
        if let Some(repaired) = try_contextual_closure_repair(&partial) {
            return Some(repaired);
        }
    }

    None
}

// Remove trailing commas before closing braces/brackets: `,(\s*[}\]])` -> `\1`
static RE_TRAILING_COMMA: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r",(\s*[}\]])").expect("valid regex"));

// Remove incomplete key-value pairs at the end: `,\s*"[^"]*"?\s*:?\s*$` -> ``
static RE_INCOMPLETE_KV: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r#",\s*"[^"]*"?\s*:?\s*$"#).expect("valid regex"));

/// Port of the Python regex `(?<!\\)"(?![,\]\}:\s]|$)` -> `\"`.
///
/// Rust's `regex` crate has no lookbehind/lookahead, so this is a faithful
/// single left-to-right scan over the original string: a `"` is escaped iff the
/// preceding character is not a backslash AND there is a following character
/// that is not one of `,])}:` or whitespace. `re.sub` references the original
/// string for its lookaround, which this mirrors by reading from the source
/// chars while emitting to a separate buffer.
fn escape_unescaped_quotes(s: &str) -> String {
    let chars: Vec<char> = s.chars().collect();
    let n = chars.len();
    let mut out = String::with_capacity(s.len());

    for i in 0..n {
        let c = chars[i];
        if c == '"' {
            let prev_not_backslash = i == 0 || chars[i - 1] != '\\';
            let next_ok = if i + 1 < n {
                let nx = chars[i + 1];
                !(nx == ',' || nx == ']' || nx == '}' || nx == ':' || nx.is_whitespace())
            } else {
                false // `$`: at end of string -> excluded by lookahead
            };
            if prev_not_backslash && next_ok {
                out.push('\\');
                out.push('"');
            } else {
                out.push('"');
            }
        } else {
            out.push(c);
        }
    }

    out
}

/// Port of `try_regex_pattern_repair`.
fn try_regex_pattern_repair(json_str: &str) -> Option<String> {
    // Remove trailing commas before closing braces/brackets.
    let mut fixed = RE_TRAILING_COMMA.replace_all(json_str, "${1}").into_owned();

    // Fix unescaped quotes in strings (basic attempt).
    fixed = escape_unescaped_quotes(&fixed);

    // Remove incomplete key-value pairs at the end.
    fixed = RE_INCOMPLETE_KV.replace(&fixed, "").into_owned();

    // Try to parse the fixed version.
    if is_valid_json(&fixed) {
        return Some(fixed);
    }

    // If that didn't work, try closing it.
    try_contextual_closure_repair(&fixed)
}

/// Port of `simple_bracket_repair`: original simple bracket-counting fallback.
fn simple_bracket_repair(json_str: &str) -> String {
    let count = |target: char| json_str.chars().filter(|c| *c == target).count() as i64;
    let open_braces = count('{');
    let close_braces = count('}');
    let open_brackets = count('[');
    let close_brackets = count(']');

    let missing_brackets = open_brackets - close_brackets;
    let missing_braces = open_braces - close_braces;

    let mut repaired = json_str.to_string();
    for _ in 0..missing_brackets.max(0) {
        repaired.push(']');
    }
    for _ in 0..missing_braces.max(0) {
        repaired.push('}');
    }

    repaired
}

/// Port of `validate_and_repair_json` (`json_parser.py`).
///
/// Mirrors Python: first run `repair_json` (the json_repair library, ported in
/// [`crate::json_repair`]); if it yields a non-empty result, return it. Otherwise
/// fall back to [`comprehensive_json_repair`] and validate it. `Err` corresponds
/// to Python raising `ValueError` ("could not repair").
///
/// Faithfulness: the `repair_json` branch returns a *re-serialized* value
/// (serde's compact form rather than Python `json.dumps`' spaced form). The only
/// caller re-parses the result, so this is parsed-value equivalent — what the
/// downstream `repair_response_model_json` actually depends on. The comprehensive
/// fallback branch is byte-identical to Python.
pub fn validate_and_repair_json(json_str: &str) -> Result<String, String> {
    let trimmed = json_str.trim();

    // repair_json(json_str): "" only when the parsed value is the empty string.
    let parsed = crate::json_repair::repair_json_value(trimmed);
    if parsed != serde_json::Value::String(String::new()) {
        return Ok(serde_json::to_string(&parsed).unwrap_or_default());
    }

    // Comprehensive repair fallback, then validate.
    let repaired = comprehensive_json_repair(trimmed);
    if is_valid_json(&repaired) {
        Ok(repaired)
    } else {
        Err(format!("Could not repair JSON. Original input: {trimmed:?}"))
    }
}

/// Port of `comprehensive_json_repair`: multi-strategy malformed-JSON repair.
///
/// Each strategy's truthy result short-circuits, mirroring Python's `if repaired:`
/// (an empty string is falsy and falls through to the next strategy).
pub fn comprehensive_json_repair(json_str: &str) -> String {
    // Strategy 1: Handle truncated JSON by parsing what we can.
    if let Some(repaired) = try_partial_parse_repair(json_str) {
        if !repaired.is_empty() {
            return repaired;
        }
    }

    // Strategy 2: Smart bracket/brace matching with context awareness.
    if let Some(repaired) = try_contextual_closure_repair(json_str) {
        if !repaired.is_empty() {
            return repaired;
        }
    }

    // Strategy 3: Line-by-line reconstruction.
    if let Some(repaired) = try_line_reconstruction_repair(json_str) {
        if !repaired.is_empty() {
            return repaired;
        }
    }

    // Strategy 4: Regex-based common pattern fixes.
    if let Some(repaired) = try_regex_pattern_repair(json_str) {
        if !repaired.is_empty() {
            return repaired;
        }
    }

    // Fallback: Original simple method.
    simple_bracket_repair(json_str)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Golden vectors captured from the Python `comprehensive_json_repair`.
    const GOLDEN: &str = r#"[{"in": "{\"a\": 1, \"b\": 2", "out": "{\"a\": 1, \"b\": 2}"}, {"in": "[1, 2, 3", "out": "[1, 2, 3]"}, {"in": "{\"a\": [1, 2, {\"b\": 3", "out": "{\"a\": [1, 2, {\"b\": 3}]}"}, {"in": "{\"a\": 1,", "out": "{\"a\": 1}"}, {"in": "{\"a\":", "out": "{\"a\":null}"}, {"in": "{\"a\": \"hello", "out": "{\"a\": \"hello\"}"}, {"in": "not json at all", "out": "not json at all"}, {"in": "", "out": ""}, {"in": "{\"x\": true, \"y\": fal", "out": "{\"x\": true, \"y\": fal}"}, {"in": "{\"items\": [\n  {\"id\": 1},\n  {\"id\": 2}\n", "out": "{\"items\": [\n  {\"id\": 1},\n  {\"id\": 2}]}"}, {"in": "{\"a\": 1} extra", "out": "{\"a\": 1} extra"}, {"in": "{\"nested\": {\"deep\": [1,2,3]", "out": "{\"nested\": {\"deep\": [1,2,3]}}"}, {"in": "[{\"k\": \"v\"}, {\"k\":", "out": "[{\"k\": \"v\"}, {\"k\":null}]"}, {"in": "{\"a\": -12.5e3, \"b\": null", "out": "{\"a\": -12.5e3, \"b\": null}"}, {"in": "[1, 2, 3,]", "out": "[1, 2, 3]"}, {"in": "{\"a\": 1, \"b\": 2,}", "out": "{\"a\": 1, \"b\": 2,}"}, {"in": "{\"a\": 1, \"incomplete\":", "out": "{\"a\": 1, \"incomplete\":null}"}, {"in": "{\"a\": 1, \"key\"", "out": "{\"a\": 1, \"key\"}"}, {"in": "{\"good\": 1,\n\"bad\": \n}", "out": "{\"good\": 1}"}, {"in": "{\"a\": \"x\", }", "out": "{\"a\": \"x\", }"}, {"in": "{\"a\": 1,, \"b\": 2}", "out": "{\"a\": 1,, \"b\": 2}"}, {"in": "   {\"a\": 1}   ", "out": "   {\"a\": 1}"}, {"in": "{\"deductive\": [{\"premises\": [\"p1\"]", "out": "{\"deductive\": [{\"premises\": [\"p1\"]}]}"}, {"in": "{\"explicit\": [\"fact one\", \"fact tw", "out": "{\"explicit\": [\"fact one\", \"fact tw\"]}"}, {"in": "{}", "out": "{}"}, {"in": "[]", "out": "[]"}, {"in": "true", "out": "true"}, {"in": "\"just a string\"", "out": "\"just a string\""}, {"in": "{\"a\": {\"b\": {\"c\": [1,2,", "out": "{\"a\": {\"b\": {\"c\": [1,2]}}}"}, {"in": "{\"unicode\": \"café", "out": "{\"unicode\": \"café\"}"}]"#;

    #[test]
    fn golden_comprehensive_json_repair() {
        let cases: Vec<serde_json::Value> = serde_json::from_str(GOLDEN).unwrap();
        for case in &cases {
            let inp = case["in"].as_str().unwrap();
            let expected = case["out"].as_str().unwrap();
            assert_eq!(
                comprehensive_json_repair(inp),
                expected,
                "comprehensive_json_repair({inp:?})"
            );
        }
    }

    #[test]
    fn simple_bracket_repair_appends_brackets_then_braces() {
        assert_eq!(simple_bracket_repair("{\"a\": [1, 2"), "{\"a\": [1, 2]}");
        assert_eq!(simple_bracket_repair(""), "");
        assert_eq!(simple_bracket_repair("not json at all"), "not json at all");
        // Excess closers -> no negatives appended.
        assert_eq!(simple_bracket_repair("}]"), "}]");
    }

    #[test]
    fn repair_bounds_recursion_on_pathological_nesting() {
        // A runaway-bracket LLM response (the kind that degenerate generation
        // produces) used to recurse unbounded and overflow the deriver worker's
        // 2 MB stack — `fatal runtime error: stack overflow, aborting`. The depth
        // guard must turn that into a graceful, crash-free repair. The test
        // passing at all (no abort) is the assertion; we also confirm the result
        // is valid JSON.
        let deep = "[".repeat(100_000);
        let repaired = validate_and_repair_json(&deep).expect("deep input must not crash");
        assert!(serde_json::from_str::<serde_json::Value>(&repaired).is_ok());
    }

    #[test]
    fn repair_bounds_recursion_on_direct_object_array_paths() {
        // These trigger the recursion that re-enters parse_object / parse_array
        // *directly* (complete_object_parse's trailing key/value pairs, and the
        // object→array continuation), bypassing the parse_json guard. They were
        // the actual deriver crash: a stack overflow on real LLM content. The
        // object/array guards must bound them.
        let trailing = format!("{{}}{}", ",\"k\":1".repeat(50_000));
        assert!(validate_and_repair_json(&trailing).is_ok());
        let continuation = format!("{{\"a\":[{}", "[1],".repeat(50_000));
        assert!(validate_and_repair_json(&continuation).is_ok());
    }

    #[test]
    fn escape_unescaped_quotes_matches_python_lookaround() {
        assert_eq!(escape_unescaped_quotes("[1, 2, 3]"), "[1, 2, 3]");
        assert_eq!(escape_unescaped_quotes("{\"a\": 1}"), "{\\\"a\": 1}");
        assert_eq!(escape_unescaped_quotes("say \"hello\" world"), "say \\\"hello\" world");
        assert_eq!(escape_unescaped_quotes("a\"b\"c"), "a\\\"b\\\"c");
        assert_eq!(escape_unescaped_quotes("\"start"), "\\\"start");
        // Already-escaped quotes (preceded by backslash) are left untouched.
        assert_eq!(escape_unescaped_quotes("\\\"esc\\\""), "\\\"esc\\\"");
        // Trailing quote -> excluded by the `$` branch of the lookahead.
        assert_eq!(escape_unescaped_quotes("abc\""), "abc\"");
        assert_eq!(escape_unescaped_quotes("\"\""), "\\\"\"");
    }

    #[test]
    fn validate_and_repair_json_parsed_equivalence() {
        // (input, expected parsed value as JSON, or None for Python ValueError)
        let ok_cases: Vec<(&str, &str)> = vec![
            ("{\"a\": 1, \"b\": 2", "{\"a\": 1, \"b\": 2}"),
            ("{\"a\": 1}", "{\"a\": 1}"),
            ("[1,2,3,]", "[1, 2, 3]"),
            ("{'x': 1}", "{\"x\": 1}"),
            ("```json\n{\"k\":[1,2}\n```", "{\"k\": [1, 2]}"),
            ("{\"a\":", "{\"a\": \"\"}"),
        ];
        for (input, expected_json) in ok_cases {
            let out = validate_and_repair_json(input).expect("should repair");
            let got: serde_json::Value = serde_json::from_str(&out).unwrap();
            let expected: serde_json::Value = serde_json::from_str(expected_json).unwrap();
            assert_eq!(got, expected, "validate_and_repair_json({input:?})");
        }
        // Unrepairable -> Err (Python ValueError).
        assert!(validate_and_repair_json("not json at all").is_err());
    }

    #[test]
    fn trailing_comma_regex_single_pass() {
        // Mirrors Python re.sub single left-to-right non-overlapping pass.
        assert_eq!(RE_TRAILING_COMMA.replace_all("[1, 2, 3,]", "${1}"), "[1, 2, 3]");
        assert_eq!(RE_TRAILING_COMMA.replace_all(",,]", "${1}"), ",]");
        // Group 1 captures the trailing whitespace, so it is preserved (matches Python).
        assert_eq!(RE_TRAILING_COMMA.replace_all("{\"a\": 1,  }", "${1}"), "{\"a\": 1  }");
    }
}
