//! Port of `parse_string_helpers/parse_boolean_or_null.py`.

use serde_json::Value;

use super::parser::JsonParser;

/// Port of `parse_boolean_or_null`.
///
/// Returns `Some(Value::Bool/Null)` on a full literal match (advancing `index`),
/// or `None` for the Python `""` failure case (with `index` reset).
///
/// Python indexes `value_map[char]` and would `KeyError` if the first char is
/// not `t`/`f`/`n`; its sole caller only invokes it when the first char is one
/// of those, so this returns `None` for any other first char (unreachable in
/// practice) instead of panicking.
pub fn parse_boolean_or_null(parser: &mut JsonParser) -> Option<Value> {
    let first = parser.get_char_at(0).map(|c| c.to_ascii_lowercase());
    let (word, value): (&str, Value) = match first {
        Some('t') => ("true", Value::Bool(true)),
        Some('f') => ("false", Value::Bool(false)),
        Some('n') => ("null", Value::Null),
        _ => return None,
    };

    let word_chars: Vec<char> = word.chars().collect();
    let starting_index = parser.index;

    let mut i = 0usize;
    let mut char_lower = parser.get_char_at(0).map(|c| c.to_ascii_lowercase());
    while let Some(c) = char_lower {
        if i < word_chars.len() && c == word_chars[i] {
            i += 1;
            parser.index += 1;
            char_lower = parser.get_char_at(0).map(|c| c.to_ascii_lowercase());
        } else {
            break;
        }
    }

    if i == word_chars.len() {
        return Some(value);
    }

    // If nothing works reset the index before returning.
    parser.index = starting_index;
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_boolean_or_null_golden() {
        // (input, expected Some(value)/None, expected index after)
        let cases: Vec<(&str, Option<Value>, i64)> = vec![
            ("true", Some(Value::Bool(true)), 4),
            ("false", Some(Value::Bool(false)), 5),
            ("null", Some(Value::Null), 4),
            ("True", Some(Value::Bool(true)), 4),
            ("FALSE", Some(Value::Bool(false)), 5),
            ("Null", Some(Value::Null), 4),
            ("nul", None, 0),
            ("tru", None, 0),
            ("t", None, 0),
            ("n,", None, 0),
            ("f}", None, 0),
        ];
        for (input, expected, expected_index) in cases {
            let mut p = JsonParser::new(input);
            let r = parse_boolean_or_null(&mut p);
            assert_eq!(r, expected, "parse_boolean_or_null({input:?})");
            assert_eq!(p.index, expected_index, "index after {input:?}");
        }
    }
}
