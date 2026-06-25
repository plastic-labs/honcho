//! Port of `parse_number.py`.

use serde_json::Value;

use super::context::ContextValues;
use super::parser::JsonParser;
use super::string::parse_string;

const NUMBER_CHARS: [char; 16] = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-', '.', 'e', 'E', '/', ',',
];

fn is_number_char(c: char) -> bool {
    // Python NUMBER_CHARS also includes '_' (stripped during accumulation).
    NUMBER_CHARS.contains(&c) || c == '_'
}

/// Port of `parse_number`.
pub fn parse_number(p: &mut JsonParser) -> Value {
    let mut number_str = String::new();
    let mut char = p.get_char_at(0);
    let is_array = p.context.current == Some(ContextValues::Array);

    while let Some(c) = char {
        if is_number_char(c) && (!is_array || c != ',') {
            if c != '_' {
                number_str.push(c);
            }
            p.index += 1;
            char = p.get_char_at(0);
        } else {
            break;
        }
    }

    if let Some(c) = p.get_char_at(0) {
        if c.is_alphabetic() {
            // this was a string instead, sorry
            p.index -= number_str.chars().count() as i64;
            return parse_string(p);
        }
    }

    if let Some(last) = number_str.chars().last() {
        if matches!(last, '-' | 'e' | 'E' | '/' | ',') {
            number_str.pop();
            p.index -= 1;
        }
    }

    if number_str.contains(',') {
        return Value::String(number_str);
    }
    if number_str.contains('.') || number_str.contains('e') || number_str.contains('E') {
        return match number_str.parse::<f64>() {
            Ok(f) => match serde_json::Number::from_f64(f) {
                Some(n) => Value::Number(n),
                None => Value::String(number_str),
            },
            Err(_) => Value::String(number_str),
        };
    }
    match number_str.parse::<i64>() {
        Ok(i) => Value::Number(serde_json::Number::from(i)),
        Err(_) => Value::String(number_str),
    }
}
