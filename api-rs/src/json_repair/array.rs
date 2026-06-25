//! Port of `parse_array.py` (no-schema path).

use serde_json::Value;

use super::STRING_DELIMITERS;
use super::context::ContextValues;
use super::object::parse_object;
use super::object_comparer::is_strictly_empty;
use super::parser::JsonParser;
use super::string::parse_string;

/// Port of `parse_array` with `closing_delimiter` (default `]`, `)` for tuples).
pub fn parse_array(p: &mut JsonParser, closing_delimiter: char) -> Value {
    let mut arr: Vec<Value> = Vec::new();
    p.context.set(ContextValues::Array);

    p.skip_whitespaces();
    let mut char = p.get_char_at(0);
    while let Some(c) = char {
        if c == closing_delimiter || c == '}' {
            break;
        }

        let value = if STRING_DELIMITERS.contains(&c) {
            // A string followed by ':' is often a missing object start.
            let i = p.skip_to_char(c, 1);
            let i = p.scroll_whitespaces(i + 1);
            if p.get_char_at(i) == Some(':') {
                parse_object(p)
            } else {
                parse_string(p)
            }
        } else {
            p.parse_json()
        };

        if is_strictly_empty(&value)
            && !matches!(p.get_char_at(0), Some(cc) if cc == closing_delimiter || cc == ',')
        {
            p.index += 1;
        } else if value == Value::String("...".to_string()) && p.get_char_at(-1) == Some('.') {
            // stray '...' — ignore it
        } else {
            arr.push(value);
        }

        char = p.get_char_at(0);
        while let Some(cc) = char {
            if cc != closing_delimiter && (cc.is_whitespace() || cc == ',') {
                p.index += 1;
                char = p.get_char_at(0);
            } else {
                break;
            }
        }
    }

    p.index += 1;
    p.context.reset();

    Value::Array(arr)
}
